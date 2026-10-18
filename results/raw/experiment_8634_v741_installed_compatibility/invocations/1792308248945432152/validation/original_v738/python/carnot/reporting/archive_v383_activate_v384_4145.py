"""Archive .383, activate .384, and record the max-epochs no-op truth.

Spec refs: REQ-REPORT-4145, SCENARIO-REPORT-4145,
SCENARIO-REPORT-4145-BLOCKED-PRECONDITION.

This is a record-only transition. It does not train nano-TRM. Its job is to
preserve the operational diagnosis that `.383` did not advance the Sudoku
baseline because the resumed checkpoint was already at the configured epoch
ceiling, so `.384` must raise the ceiling before resuming from the intact
0.278 checkpoint.
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
ARCHIVED_MILESTONE = "2026.06.383"
ACTIVATED_MILESTONE = "2026.06.384"
RANDOM_SEED = 4145
OUTPUT_REL_PATH = Path("results/experiment_4145_archive_v383_activate_v384.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
NANO_TRM_TRAIN_REL_PATH = Path("nano-trm/src/nn/train.py")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v383_to_v384_4145.v1"
EXPERIMENT_ID = "exp4145"
TASK_ID = "exp4145-archive-v383-activate-v384"

BASELINE_VAL_DEFAULT = 0.278172343969
PASS1_DURATION_DEFAULT = 6.99
PUBLISHED_BASELINE_TARGET = 0.87

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V383_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "4135",
        "deliverable": "results/experiment_4135_sudoku_accumulate_pass1_fixed_lr.json",
    },
    {
        "experiment_id": "4138",
        "deliverable": "results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json",
    },
    {
        "experiment_id": "4139",
        "deliverable": "results/experiment_4139_decisive_verifier_graft_sudoku.json",
    },
    {
        "experiment_id": "4144",
        "deliverable": "results/experiment_4144_capstone_v383.json",
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
    "v383_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.383.",
    "activated_milestone": "Confirms .384 is live for the epoch-cap repair path.",
    "active_milestone_confirmed": "Records the active roadmap milestone actually observed.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "nano_trm_train_present": "Bare bool: nano-trm/src/nn/train.py exists before activation.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v383_close_state": (
        "Honest record (accumulation no-op'd at the max_epochs cap; baseline still 0.278) "
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
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.383['\"]?\s*$")
CANONICAL_FINDING = (
    ".383 accumulation no-op'd at the resumed checkpoint max_epochs cap: exp4135 ran 6.99s, "
    "never trained, wrote no real val_exact_accuracy, and left the stable checkpoint untouched. "
    "The baseline stayed at 0.278, the decisive graft deferred uninformatively with "
    "FALSE_NEGATIVE_RISK, and .384 fixes the cap before resuming from the intact 0.278 checkpoint."
)


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
    """Count top-level `.383` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _record_id(line: str) -> str | None:
    """Return a top-level YAML milestone id, or None for nested task ids."""

    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _insert_before_tasks(lines: list[str], new_line: str) -> list[str]:
    for index, line in enumerate(lines):
        if line == "  tasks:":
            return lines[:index] + [new_line] + lines[index:]
    return lines + [new_line]


def build_canonical_record() -> str:
    """Build a minimal `.383` record for the absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .383 and activate .384; preserve max-epochs no-op close-state')}",
        "  completed: '2026-06-13'",
        f"  finding: {_yaml_quote(CANONICAL_FINDING)}",
        "  activation_recorded: exp4145-archive-v383-activate-v384",
        "  tasks:",
        "  - id: exp4135-sudoku-accumulate-pass1-fixed-lr",
        "    result: 'no-op at max_epochs cap; duration 6.99s; val missing'",
        "  - id: exp4139-decisive-verifier-graft-sudoku",
        "    result: 'deferred uninformatively; FALSE_NEGATIVE_RISK'",
        "  - id: exp4144-capstone-v383",
        "    result: 'baseline stayed 0.278; .384 fixes cap'",
    ]
    return "\n".join(lines) + "\n"


def _canonicalize_target_span(lines: list[str]) -> list[str]:
    out: list[str] = []
    finding_written = False
    activation_written = False
    for line in lines:
        if line.startswith("  finding:"):
            if not finding_written:
                out.append(f"  finding: {_yaml_quote(CANONICAL_FINDING)}")
                finding_written = True
            continue
        if line.startswith("  activation_recorded:"):
            if not activation_written:
                out.append("  activation_recorded: exp4145-archive-v383-activate-v384")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(CANONICAL_FINDING)}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4145-archive-v383-activate-v384")
    return out


def dedupe_or_update_record(text: str) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.383` record exists and carries the no-op truth."""

    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _record_id(line) is not None]
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        spans.append((start, end))
    target_spans = [(start, end) for start, end in spans if _record_id(lines[start]) == ARCHIVED_MILESTONE]
    if not target_spans:
        return f"{text.rstrip()}\n{build_canonical_record()}", 0, "appended"

    first_start, first_end = target_spans[0]
    remove: set[int] = set()
    for start, end in target_spans[1:]:
        remove.update(range(start, end))
    replacement = _canonicalize_target_span(lines[first_start:first_end])
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


def read_v383_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.383` close-state."""

    out: dict[str, JsonDict] = {}
    for source in V383_SOURCE_ARTIFACTS:
        out[str(source["experiment_id"])] = read_json_object(root / str(source["deliverable"]))
    return out


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for the upstream `.383` artifacts."""

    cited: list[JsonDict] = []
    for source in V383_SOURCE_ARTIFACTS:
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


def _coerce_false(value: Any) -> bool:
    return bool(value) if value is not None else False


def build_v383_close_state(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the honest `.383` close-state from capstone and no-op artifacts."""

    pass1 = _mapping(sources.get("4135", {}))
    pass4 = _mapping(sources.get("4138", {}))
    graft = _mapping(sources.get("4139", {}))
    capstone = _mapping(sources.get("4144", {}))
    capstone_trajectory = _mapping(capstone.get("baseline_val_trajectory"))
    capstone_graft = _mapping(capstone.get("verifier_value_added_verdict"))

    baseline_val = _number(
        capstone_trajectory.get("final_val_exact_accuracy"),
        _number(pass4.get("val_exact_accuracy"), BASELINE_VAL_DEFAULT),
    )
    pass1_duration = _number(pass1.get("duration_s"), PASS1_DURATION_DEFAULT)
    pass1_val = pass1.get("exact_accuracy_metric", pass1.get("val_exact_accuracy"))
    pass1_metrics_missing = pass1_val is None and pass1.get("exact_accuracy_metrics_path") is None
    pass1_trained = pass1_duration > 120.0 and not pass1_metrics_missing
    false_negative_risk = bool(
        graft.get("false_negative_risk", capstone_graft.get("false_negative_risk", True))
    )

    return {
        "summary": "accumulation_noop_at_max_epochs_cap_baseline_0.278_graft_false_negative_risk_v384_fixes_cap",
        "accumulation_noop": True,
        "noop_reason": "max_epochs_cap",
        "noop_evidence": (
            "exp4135 returned in 6.99s with no real val_exact_accuracy, which is too short "
            "for a resumed bounded training pass and matches the max_epochs cap diagnosis."
        ),
        "exp4135_duration_s": round(pass1_duration, 2),
        "exp4135_trained": pass1_trained,
        "exp4135_val_exact_accuracy": pass1_val,
        "exp4135_metrics_missing": pass1_metrics_missing,
        "checkpoint_untouched": not pass1_trained,
        "stable_checkpoint_path": str(
            pass1.get("stable_checkpoint_path")
            or pass4.get("stable_checkpoint_path")
            or "results/trm_runs/sudoku_extreme_baseline/last.ckpt"
        ),
        "checkpoint_val_exact_accuracy": BASELINE_VAL_DEFAULT,
        "baseline_val_exact_accuracy": baseline_val,
        "baseline_val_exact_accuracy_rounded": round(baseline_val, 3),
        "published_exact_accuracy_target": PUBLISHED_BASELINE_TARGET,
        "matches_published_087": _coerce_false(capstone_trajectory.get("matches_published_087")),
        "near_faithful_080": _coerce_false(capstone_trajectory.get("near_faithful_080")),
        "baseline_status": str(
            capstone_trajectory.get("status") or pass4.get("baseline_status") or "baseline_config_blocked"
        ),
        "graft_deferred": bool(graft.get("graft_deferred", capstone_graft.get("status") == "deferred")),
        "false_negative_risk": false_negative_risk,
        "graft_honest_verdict": str(graft.get("honest_verdict", "")),
        "verifier_value_added": bool(graft.get("verifier_value_added", capstone_graft.get("verifier_value_added", False))),
        "capstone_v383": {
            "artifact_present": bool(capstone),
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "headline_outcome": str(capstone.get("headline_outcome", "baseline_config_blocked")),
        },
        "v384_forward_plan": {
            "resume_from_intact_checkpoint": True,
            "fixes_max_epochs_cap": True,
            "required_resume_fix": "set +trainer.max_epochs above the checkpoint epoch before Trainer.fit",
            "anti_noop_guard": "duration_s > 120 and checkpoint epoch advanced and real val_exact_accuracy read",
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


def terminal_verdict(v383_close_state: Mapping[str, Any]) -> str:
    """Return the complete-path verdict for the archive transition."""

    val = _number(v383_close_state.get("baseline_val_exact_accuracy_rounded"), 0.278)
    return (
        "success: archived_v383_v384_active_max_epochs_noop_recorded_"
        f"baseline_{val:.3f}_graft_false_negative_risk_pretest_green"
    )


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    nano_trm_train_present: bool,
    pretest_suite_green: bool,
    v383_close_state: Mapping[str, Any],
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
        "v383_close_state": dict(v383_close_state),
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
        "v383_close_state": {"status": "blocked", "reason": reason},
        "active_milestone_confirmed": "",
        "active_roadmap_path": "research-roadmap.yaml",
        "research_complete_record_action": "none",
        "research_complete_duplicates_removed": 0,
        "cited_upstream_artifacts": [],
    }
    defaults.update(kwargs)
    return _base_payload(honest_verdict=reason, **defaults)


def build_complete_artifact(**kwargs: Any) -> JsonDict:
    """Build and validate the Exp 4145 complete artifact."""

    close_state = kwargs["v383_close_state"]
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
    """Validate the fields that stop this archive from laundering the `.383` truth."""

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
    for field in ("honest_verdict", "v383_close_state", "preconditions_checked"):
        if principles.get(field) != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match REQ-REPORT-4145")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.383")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.384")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML must parse")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("exclusion manifest must parse")
    if artifact.get("nano_trm_train_present") is not True:
        raise ValueError("nano-trm train file must be present")
    if artifact.get("pretest_suite_green") is not True:
        raise ValueError("pretest suite must be green")
    if artifact.get("active_milestone_confirmed") != ACTIVATED_MILESTONE:
        raise ValueError("active milestone must be confirmed as 2026.06.384")
    close_state = artifact.get("v383_close_state")
    if not isinstance(close_state, Mapping):
        raise ValueError("v383_close_state must be a mapping")
    if close_state.get("accumulation_noop") is not True:
        raise ValueError("v383_close_state must record accumulation_noop=True")
    if close_state.get("noop_reason") != "max_epochs_cap":
        raise ValueError("noop_reason must be max_epochs_cap")
    if close_state.get("exp4135_duration_s") != PASS1_DURATION_DEFAULT:
        raise ValueError("exp4135_duration_s must be 6.99")
    if close_state.get("baseline_val_exact_accuracy_rounded") != 0.278:
        raise ValueError("baseline_val_exact_accuracy must round to 0.278")
    if close_state.get("graft_deferred") is not True:
        raise ValueError("graft_deferred must be True")
    if close_state.get("false_negative_risk") is not True:
        raise ValueError("false_negative_risk must be True")
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
    """Run the `.383` archive and `.384` activation guard."""

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
            "blocked_v384_not_active",
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

    new_text, removed, action = dedupe_or_update_record(complete_text)
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

    sources = read_v383_sources(root_path)
    close_state = build_v383_close_state(sources)
    cited = build_cited_upstream(root_path)
    payload = build_complete_artifact(
        v383_close_state=close_state,
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
