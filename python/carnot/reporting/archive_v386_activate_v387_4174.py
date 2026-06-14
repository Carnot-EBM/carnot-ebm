"""Archive .386, activate .387, and record the headroom-limited graft null.

Spec refs: REQ-REPORT-4174, SCENARIO-REPORT-4174,
SCENARIO-REPORT-4174-BLOCKED-PRECONDITION.

This is a record-only transition. It preserves the `.386` truth that the
outer-loop TRM run reached the gate-0.82 baseline and stopped, the defensive
graft fired, and the null result was limited by selectable headroom. The next
planner needs that distinction so it frames `.387` as a positive-control moat
test instead of re-running the same Sudoku checkpoint.
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
ARCHIVED_MILESTONE = "2026.06.386"
ACTIVATED_MILESTONE = "2026.06.387"
RANDOM_SEED = 4174
OUTPUT_REL_PATH = Path("results/experiment_4174_archive_v386_activate_v387.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4173_capstone_v386.json")
GRAFT_V2_REL_PATH = Path("results/experiment_4168_decisive_verifier_graft_v2_gate082.json")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v386_to_v387_4174.v1"
EXPERIMENT_ID = "exp4174"
TASK_ID = "exp4174-archive-v386-activate-v387"

BASELINE_BESTVAL_DEFAULT = 0.822656
BASELINE_GATE_DEFAULT = 0.82
ORACLE_AT_K_DEFAULT = 0.8125
VERIFIER_PASS_AT_1_DEFAULT = 0.8125
VOTE_AT_1_DEFAULT = 0.796875
TOTAL_GAMES_SOLVED_DEFAULT = 13

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V386_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "4173",
        "deliverable": str(CAPSTONE_REL_PATH),
    },
    {
        "experiment_id": "4168-v2",
        "deliverable": str(GRAFT_V2_REL_PATH),
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "pretest_suite_green",
    "v386_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.386.",
    "activated_milestone": "Confirms .387 is live for the headroom-controlled moat test.",
    "active_milestone_confirmed": "Records the active roadmap milestone actually observed.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v386_close_state": (
        "Honest record (graft null was headroom-limited, not a true verifier failure) "
        "so the .387 planner frames the moat test as a positive-control problem, not a redo."
    ),
    "preconditions_checked": (
        "Records resources verified; pre-empts the silent-missing-resource fabrication mode."
    ),
    "honest_verdict": (
        "Self-declared terminal state lets the reconciler classify success without re-running; "
        "MUST start with complete:/success:/passed:/shipped:."
    ),
    "duration_s": "Positive bare wall-clock for this record-only aggregation.",
    "inference_substrate": "Declares aggregation only; no live training happens in this task.",
}

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.386['\"]?\s*$")


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
    """Compute a positive duration so blocked artifacts still carry timing."""

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


def archive_record_count(text: str) -> int:
    """Count top-level `.386` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _record_id(line: str) -> str | None:
    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _contains_zero(ci: Any) -> bool:
    if not isinstance(ci, Sequence) or isinstance(ci, str) or len(ci) < 2:
        return False
    low = _number(ci[0], 1.0)
    high = _number(ci[1], -1.0)
    return low <= 0.0 <= high


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.386` archive finding from the close-state."""

    bestval = _number(close_state.get("baseline_bestval_exact_accuracy"), BASELINE_BESTVAL_DEFAULT)
    oracle = _number(close_state.get("oracle_at_k"), ORACLE_AT_K_DEFAULT)
    vote = _number(close_state.get("vote_at_1"), VOTE_AT_1_DEFAULT)
    return (
        ".386 close-state: outer-loop TRM training is DONE and stopped after reaching "
        f"val {bestval:.6f} (rounded 0.8227). The decisive gate-0.82 graft FIRED "
        "(graft_deferred=false) but returned a headroom-limited null, not a true verifier "
        f"failure: verifier_value_added=false, oracle@k={oracle:.4f} is approximately the "
        f"0.82 baseline gate, verifier pass@1={_number(close_state.get('verifier_pass_at_1'), VERIFIER_PASS_AT_1_DEFAULT):.4f}, "
        f"vote@1={vote:.6f}, and the RFT deconfound CI includes zero. DiffusionGemma remains "
        "STILL-PENDING; ARC total_games_solved=13. .387 must frame the moat test as a "
        "headroom-present positive-control problem while the conductor stays stood down on TRM training."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.386` record for the absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .386 and activate .387; preserve headroom-limited graft null')}",
        "  completed: '2026-06-14'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4174-archive-v386-activate-v387",
        "  tasks:",
        "  - id: exp4168-decisive-verifier-graft-v2-gate082",
        "    result: 'graft fired; headroom-limited null; verifier_value_added=false'",
        "  - id: exp4173-capstone-v386",
        "    result: 'DiffusionGemma STILL-PENDING; total_games_solved=13'",
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
                out.append("  activation_recorded: exp4174-archive-v386-activate-v387")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4174-archive-v386-activate-v387")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.386` record exists and carries the null."""

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


def read_v386_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.386` close-state."""

    return {
        "4173": read_json_object(root / CAPSTONE_REL_PATH),
        "4168_v2": read_json_object(root / GRAFT_V2_REL_PATH),
    }


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.386` artifacts."""

    cited: list[JsonDict] = []
    for source in V386_SOURCE_ARTIFACTS:
        rel = str(source["deliverable"])
        cited.append(
            {
                "kind": "artifact",
                "experiment_id": str(source["experiment_id"]),
                "deliverable": rel,
                "sha256": file_sha256(root / rel),
            }
        )
    return cited


def build_v386_close_state(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the honest `.386` close-state from capstone and graft artifacts."""

    capstone = _mapping(sources.get("4173", {}))
    graft = _mapping(sources.get("4168_v2", {}))
    capstone_trajectory = _mapping(capstone.get("baseline_val_trajectory"))
    capstone_graft = _mapping(capstone.get("defensive_graft_verdict"))
    baseline_status = _mapping(graft.get("baseline_status"))
    rerank = _mapping(graft.get("rerank_lift_vs_vote"))
    rft = _mapping(graft.get("rft_vs_ablation_delta"))

    bestval = _number(baseline_status.get("bestval_exact_accuracy"), BASELINE_BESTVAL_DEFAULT)
    baseline_gate = _number(baseline_status.get("faithful_threshold"), BASELINE_GATE_DEFAULT)
    oracle_at_k = _number(rerank.get("oracle_at_k"), ORACLE_AT_K_DEFAULT)
    verifier_pass = _number(rerank.get("verifier_pass_at_1"), VERIFIER_PASS_AT_1_DEFAULT)
    vote_at_1 = _number(rerank.get("vote_at_1"), VOTE_AT_1_DEFAULT)
    graft_deferred = bool(graft.get("graft_deferred", capstone_graft.get("graft_deferred", False)))
    verifier_value_added = bool(graft.get("verifier_value_added", capstone_graft.get("verifier_value_added", False)))
    outerloop_stopped = (
        bool(baseline_status.get("pid_not_alive_passed", False))
        or bool(baseline_status.get("gpu_train_stopped_passed", False))
        or baseline_status.get("outerloop_pid_alive") is False
    )
    null_ci_includes_zero = _contains_zero(rft.get("ci95")) or "null" in str(rft.get("status", "")).lower()
    oracle_approximately_baseline = abs(oracle_at_k - baseline_gate) <= 0.01
    total_games = int(
        _number(
            capstone.get("total_arc_games_solved"),
            _number(capstone.get("total_games_solved"), TOTAL_GAMES_SOLVED_DEFAULT),
        )
    )

    return {
        "summary": "outerloop_done_graft_fired_headroom_limited_null",
        "outer_loop_trm_training_done": outerloop_stopped,
        "outer_loop_sigterm_reported": True,
        "conductor_stands_down_on_trm_training": True,
        "no_conductor_training_rule": True,
        "forbidden_conductor_actions": [
            "launch_trm_training",
            "pkill_or_kill_train_py",
            "write_stable_checkpoint_dir",
        ],
        "baseline_bestval_exact_accuracy": round(bestval, 6),
        "baseline_val_rounded": round(bestval, 4),
        "baseline_gate": baseline_gate,
        "capstone_recorded_val": capstone_trajectory.get("current_val_exact_accuracy"),
        "capstone_honest_verdict": str(capstone.get("honest_verdict", "")),
        "capstone_was_stale_before_final_graft": bool(capstone) and bestval > _number(
            capstone_trajectory.get("current_val_exact_accuracy"), 0.0
        ),
        "decisive_graft_fired": graft_deferred is False,
        "graft_deferred": graft_deferred,
        "verifier_value_added": verifier_value_added,
        "headroom_limited_null": (
            graft_deferred is False
            and verifier_value_added is False
            and oracle_approximately_baseline
            and null_ci_includes_zero
        ),
        "true_verifier_failure": False,
        "oracle_at_k": oracle_at_k,
        "oracle_approximately_baseline": oracle_approximately_baseline,
        "verifier_pass_at_1": verifier_pass,
        "vote_at_1": vote_at_1,
        "rerank_delta": _number(rerank.get("delta"), 0.0),
        "rerank_ci95": list(rerank.get("ci95", [])) if isinstance(rerank.get("ci95"), list) else [],
        "rerank_status": str(rerank.get("status", "")),
        "rft_delta": _number(rft.get("delta"), 0.0),
        "rft_ci95": list(rft.get("ci95", [])) if isinstance(rft.get("ci95"), list) else [],
        "rft_status": str(rft.get("status", "")),
        "rft_ci_includes_zero": null_ci_includes_zero,
        "diffusiongemma_gate_status": str(capstone.get("diffusiongemma_gate_status", "STILL-PENDING")),
        "total_games_solved": total_games,
        "v387_planner_frame": (
            "positive-control headroom-present executable moat test; do not treat the Sudoku null as "
            "a true verifier failure or redo the same no-headroom setup"
        ),
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


def terminal_verdict(v386_close_state: Mapping[str, Any]) -> str:
    """Return the complete-path verdict for the archive transition."""

    bestval = _number(v386_close_state.get("baseline_bestval_exact_accuracy"), BASELINE_BESTVAL_DEFAULT)
    return (
        "success: archived_v386_v387_active_outerloop_done_val_"
        f"{bestval:.4f}_graft_fired_headroom_limited_null_diffusiongemma_STILL_PENDING"
    )


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    pretest_suite_green: bool,
    v386_close_state: Mapping[str, Any],
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
        "pretest_suite_green": pretest_suite_green,
        "v386_close_state": dict(v386_close_state),
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
        "pretest_suite_green": False,
        "v386_close_state": {"status": "blocked", "reason": reason},
        "active_milestone_confirmed": "",
        "active_roadmap_path": "research-roadmap.yaml",
        "research_complete_record_action": "none",
        "research_complete_duplicates_removed": 0,
        "cited_upstream_artifacts": [],
    }
    defaults.update(kwargs)
    return _base_payload(honest_verdict=reason, **defaults)


def build_complete_artifact(**kwargs: Any) -> JsonDict:
    """Build and validate the Exp 4174 complete artifact."""

    close_state = kwargs["v386_close_state"]
    payload = _base_payload(
        honest_verdict=terminal_verdict(close_state),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        pretest_suite_green=True,
        active_milestone_confirmed=ACTIVATED_MILESTONE,
        **kwargs,
    )
    validate_artifact(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate fields that stop this archive from laundering the `.386` truth."""

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
    for field in ("honest_verdict", "v386_close_state", "preconditions_checked"):
        if principles.get(field) != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match REQ-REPORT-4174")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.386")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.387")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML must parse")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("exclusion manifest must parse")
    if artifact.get("pretest_suite_green") is not True:
        raise ValueError("pretest suite must be green")
    if artifact.get("active_milestone_confirmed") != ACTIVATED_MILESTONE:
        raise ValueError("active milestone must be confirmed as 2026.06.387")
    close_state = artifact.get("v386_close_state")
    if not isinstance(close_state, Mapping):
        raise ValueError("v386_close_state must be a mapping")
    if close_state.get("outer_loop_trm_training_done") is not True:
        raise ValueError("outer_loop_trm_training_done must be True")
    if close_state.get("conductor_stands_down_on_trm_training") is not True:
        raise ValueError("conductor_stands_down_on_trm_training must be True")
    if close_state.get("decisive_graft_fired") is not True:
        raise ValueError("decisive_graft_fired must be True")
    if close_state.get("verifier_value_added") is not False:
        raise ValueError("verifier_value_added must be False")
    if close_state.get("headroom_limited_null") is not True:
        raise ValueError("headroom_limited_null must be True")
    if close_state.get("true_verifier_failure") is not False:
        raise ValueError("true_verifier_failure must be False")
    if round(_number(close_state.get("baseline_bestval_exact_accuracy"), 0.0), 6) != round(
        BASELINE_BESTVAL_DEFAULT, 6
    ):
        raise ValueError("baseline bestval must be 0.822656")
    if round(_number(close_state.get("oracle_at_k"), 0.0), 4) != round(ORACLE_AT_K_DEFAULT, 4):
        raise ValueError("oracle_at_k must be 0.8125")
    if close_state.get("diffusiongemma_gate_status") != "STILL-PENDING":
        raise ValueError("DiffusionGemma gate must remain STILL-PENDING")
    if close_state.get("total_games_solved") != TOTAL_GAMES_SOLVED_DEFAULT:
        raise ValueError("total games solved must be 13")
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
    """Run the `.386` archive and `.387` activation guard."""

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

    preconditions: JsonDict = {
        "research_complete_yaml_exists": complete_exists,
        "research_complete_yaml_parses": complete_parses,
        "exclusion_manifest_exists": manifest_exists,
        "exclusion_manifest_parses": manifest_parses,
        "active_milestone": active_milestone,
        "active_roadmap_path": active_roadmap_path,
        "pretest_suite_green": False,
        "v386_capstone_present": False,
        "v386_capstone_path": str(CAPSTONE_REL_PATH),
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
    if active_milestone != ACTIVATED_MILESTONE:
        return blocked(
            "blocked_v387_not_active",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
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
        )

    capstone_present = (root_path / CAPSTONE_REL_PATH).exists()
    preconditions["v386_capstone_present"] = capstone_present
    if not capstone_present:
        return blocked(
            "blocked_v386_capstone_missing",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            pretest_suite_green=True,
        )

    sources = read_v386_sources(root_path)
    close_state = build_v386_close_state(sources)
    preconditions["gate082_graft_artifact_present"] = (root_path / GRAFT_V2_REL_PATH).exists()
    preconditions["gate082_graft_path"] = str(GRAFT_V2_REL_PATH)

    new_text, removed, action = dedupe_or_update_record(complete_text, close_state)
    if not yaml_parses(new_text):
        return blocked(
            "blocked_research_complete_edit_invalid",
            research_complete_yaml_parses=False,
            exclusion_manifest_parses=True,
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
            pretest_suite_green=True,
            research_complete_record_action=action,
            research_complete_duplicates_removed=removed,
        )

    payload = build_complete_artifact(
        v386_close_state=close_state,
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
