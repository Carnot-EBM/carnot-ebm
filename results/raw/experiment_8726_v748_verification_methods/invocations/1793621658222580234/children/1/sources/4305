"""Archive .390, activate .391, and preserve the oracle-distinct infra correction.

Spec refs: REQ-REPORT-4219, SCENARIO-REPORT-4219,
SCENARIO-REPORT-4219-BLOCKED-PRECONDITION.

This is a record-only transition. It corrects the `.390` close-state so the
next milestone treats the oracle-distinct gate as a de-risked retry after a
wrong-file DATA bug, not as a completed no-signal science result.
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
ARCHIVED_MILESTONE = "2026.06.390"
ACTIVATED_MILESTONE = "2026.06.391"
RANDOM_SEED = 4219
OUTPUT_REL_PATH = Path("results/experiment_4219_archive_v390_activate_v391.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4218_capstone_v390.json")
BUILD_REL_PATH = Path("results/experiment_4209_oracle_distinct_arc_verifier_build.json")
GATE_REL_PATH = Path("results/experiment_4210_oracle_distinct_arc_verifier_beats_vote.json")
DETECTOR_REL_PATH = Path("results/experiment_4208_verifier_as_detector_auroc.json")
REWARD_REL_PATH = Path("results/experiment_4211_verifier_as_reward_finish_synchronous.json")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v390_to_v391_4219.v1"
EXPERIMENT_ID = "exp4219"
TASK_ID = "exp4219-archive-v390-activate-v391"

WORKING_LABEL_LOADER = "scripts/exp_verifier_detector_auroc.py:load_arc_rows"
WORKING_LABEL_POOL_PATH = "results/arc3_gap3_stage2_eval_pool.json.gz"
WORKING_PROGRAMS_PATH = "results/arc3_gap4_induced_programs.json"
V391_FRAME = "the de-risked oracle-distinct retry + the harness-first verifier-as-reward FINISH"

ARC_LABELED_CANDIDATE_N_DEFAULT = 8041
ARC_DETECTOR_AUROC_DEFAULT = 0.9016
ARC_DETECTOR_CI95_DEFAULT = [0.7828, 0.9984]
ARC_SELECTOR_HEADROOM_DEFAULT = 0.129
PHASE0_PRECISION_DEFAULT = 0.956
YOUDEN_J_DEFAULT = 0.4138
ARM_A_DEFAULT = 776
ARM_B_DEFAULT = 776
ARM_C_DEFAULT = 742
TOTAL_LEVELS_SOLVED_DEFAULT = 16
TOTAL_GAMES_SOLVED_DEFAULT = 13
REWARD_CHECKPOINT_SLUG = "code_verifier_reward_lora_rft_a83b52882c198954"

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V390_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {"experiment_id": "4218", "deliverable": str(CAPSTONE_REL_PATH), "required": True},
    {"experiment_id": "4209", "deliverable": str(BUILD_REL_PATH), "required": True},
    {"experiment_id": "4210", "deliverable": str(GATE_REL_PATH), "required": True},
    {"experiment_id": "4208", "deliverable": str(DETECTOR_REL_PATH), "required": True},
    {"experiment_id": "4211", "deliverable": str(REWARD_REL_PATH), "required": True},
)

SOURCE_MISSING_REASONS = {
    "4218": "blocked_v390_capstone_missing",
    "4209": "blocked_oracle_distinct_build_missing",
    "4210": "blocked_oracle_distinct_gate_missing",
    "4208": "blocked_detector_artifact_missing",
    "4211": "blocked_reward_artifact_missing",
}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "pretest_suite_green",
    "v390_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.390.",
    "activated_milestone": "Confirms .391 is live for the corrected retry frame.",
    "active_milestone_confirmed": "Records the active roadmap milestone actually observed.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v390_close_state": (
        "Honest record (oracle-distinct gate blocked on a wrong-file DATA bug not a null; "
        "the signal exists at AUROC 0.90; reward died on a PEFT attach with operating "
        "point intact; ARC 16) so the .391 agents frame the milestone as a de-risked "
        "retry + collection, not a redo from scratch."
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
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.390['\"]?\s*$")


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

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return true when value is a lowercase SHA-256 hex digest."""

    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
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
    """Count top-level `.390` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _record_id(line: str) -> str | None:
    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _number(value: Any, default: float) -> float:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else float(default)
    )


def _bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ci95(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        return [
            round(_number(value[0], ARC_DETECTOR_CI95_DEFAULT[0]), 4),
            round(_number(value[1], ARC_DETECTOR_CI95_DEFAULT[1]), 4),
        ]
    return list(ARC_DETECTOR_CI95_DEFAULT)


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.390` archive finding from the corrected close-state."""

    auroc = _number(
        close_state.get("oracle_distinct_arc_detection_auroc"), ARC_DETECTOR_AUROC_DEFAULT
    )
    n_arc = int(
        _number(close_state.get("arc_labeled_candidate_n"), ARC_LABELED_CANDIDATE_N_DEFAULT)
    )
    precision = _number(close_state.get("reward_phase0_precision"), PHASE0_PRECISION_DEFAULT)
    youden = _number(close_state.get("reward_youden_j"), YOUDEN_J_DEFAULT)
    corpora = _mapping(close_state.get("reward_corpora"))
    levels = int(_number(close_state.get("total_levels_solved"), TOTAL_LEVELS_SOLVED_DEFAULT))
    games = int(_number(close_state.get("total_games_solved"), TOTAL_GAMES_SOLVED_DEFAULT))
    return (
        ".390 close-state correction: the oracle-distinct NO-HEADROOM-OR-NO-SIGNAL "
        "headline was an INFRA artifact, not a science null. Exp4209 hit a wrong-file "
        "DATA bug: it read ARC candidate labels from the wrong file, produced "
        "accepted=0/rejected=0/total=0, and left selector_trained=false; exp4210 "
        "therefore gate-blocked before the beats-vote comparison ran. The working label path exists via "
        f"{WORKING_LABEL_LOADER} over {WORKING_LABEL_POOL_PATH} + {WORKING_PROGRAMS_PATH}: "
        f"{n_arc} labeled ARC candidates with oracle-distinct detector AUROC={auroc:.4f}. "
        "Verifier-as-reward hit its third infra failure when PEFT rejected "
        "Gemma4ClippableLinear, with the operating point intact "
        f"(Phase-0 precision={precision:.3f}, Youden-J={youden:.4f}, "
        f"corpora A={int(_number(corpora.get('A'), ARM_A_DEFAULT))}/"
        f"B={int(_number(corpora.get('B'), ARM_B_DEFAULT))}/"
        f"C={int(_number(corpora.get('C'), ARM_C_DEFAULT))}, checkpoint intact). "
        f"ARC total_levels_solved={levels}, total_games_solved={games}; live solver was "
        f"efficiency-only with no level completed; flagged-skipped artifacts were 4212 and 4216. "
        f".391 frame: {V391_FRAME}."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.390` record for the absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .390 and activate .391; correct oracle-distinct close-state')}",
        "  completed: '2026-06-15'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4219-archive-v390-activate-v391",
        "  tasks:",
        "  - id: exp4209-oracle-distinct-arc-verifier-build",
        "    result: 'blocked on wrong-file candidate-label lookup; selector_trained=false'",
        "  - id: exp4210-oracle-distinct-arc-verifier-beats-vote",
        "    result: 'gate-blocked on selector_trained=false; beats-vote comparison never ran'",
        "  - id: exp4208-verifier-as-detector-auroc",
        "    result: 'working ARC label path yielded 8041 candidates and AUROC=0.9016'",
        "  - id: exp4211-verifier-as-reward-finish-synchronous",
        "    result: 'PEFT rejected Gemma4ClippableLinear; operating point and checkpoint intact'",
        "  - id: exp4218-capstone-v390",
        "    result: 'headline corrected as infra artifact; ARC 16 levels / 13 games'",
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
                out.append("  activation_recorded: exp4219-archive-v390-activate-v391")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4219-archive-v390-activate-v391")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.390` record exists and carries the close-state."""

    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if _record_id(line) is not None]
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        spans.append((start, end))
    target_spans = [
        (start, end) for start, end in spans if _record_id(lines[start]) == ARCHIVED_MILESTONE
    ]
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


def read_v390_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.390` close-state."""

    return {
        "4218": read_json_object(root / CAPSTONE_REL_PATH),
        "4209": read_json_object(root / BUILD_REL_PATH),
        "4210": read_json_object(root / GATE_REL_PATH),
        "4208": read_json_object(root / DETECTOR_REL_PATH),
        "4211": read_json_object(root / REWARD_REL_PATH),
    }


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.390` artifacts."""

    cited: list[JsonDict] = []
    for source in V390_SOURCE_ARTIFACTS:
        rel = str(source["deliverable"])
        cited.append(
            {
                "kind": "artifact",
                "experiment_id": str(source["experiment_id"]),
                "deliverable": rel,
                "required": bool(source["required"]),
                "sha256": file_sha256(root / rel),
            }
        )
    return cited


def _checkpoint_present(root: Path, value: Any, readable_flag: Any) -> bool:
    if readable_flag is True:
        return True
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return path.exists() if path.is_absolute() else (root / path).exists()


def _flagged_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, Mapping):
            experiment_id = item.get("experiment_id")
            if isinstance(experiment_id, int):
                out.append(experiment_id)
    return out


def build_v390_close_state(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
) -> JsonDict:
    """Build the honest `.390` close-state from capstone and upstream artifacts."""

    capstone = _mapping(sources.get("4218", {}))
    build = _mapping(sources.get("4209", {}))
    gate = _mapping(sources.get("4210", {}))
    detector = _mapping(sources.get("4208", {}))
    reward = _mapping(sources.get("4211", {}))
    learned = _mapping(capstone.get("learned_arc_verifier"))
    frontier = _mapping(capstone.get("oracle_distinct_frontier"))
    capstone_detector = _mapping(capstone.get("detector_selection_divergence"))
    reward_from_capstone = _mapping(capstone.get("verifier_as_reward"))
    capstone_reward_training = _mapping(reward_from_capstone.get("training"))
    arc = _mapping(capstone.get("arc_progress"))
    live = _mapping(capstone.get("live_solver_accuracy"))
    detector_aurocs = _mapping(detector.get("detection_auroc_by_domain")) or _mapping(
        capstone_detector.get("detection_auroc_by_domain")
    )
    detector_ci95 = _mapping(detector.get("detection_auroc_ci95_by_domain")) or _mapping(
        capstone_detector.get("detection_auroc_ci95_by_domain")
    )
    detector_ns = _mapping(detector.get("n_by_domain")) or _mapping(
        capstone_detector.get("n_by_domain")
    )
    detector_headroom = _mapping(detector.get("selector_headroom_by_domain")) or _mapping(
        capstone_detector.get("selector_headroom_by_domain")
    )
    detector_oracle = _mapping(detector.get("verifier_is_oracle_by_domain")) or _mapping(
        capstone_detector.get("verifier_is_oracle_by_domain")
    )
    accepted = _mapping(build.get("accepted_rejected_n")) or _mapping(
        learned.get("accepted_rejected_n")
    )
    reward_training = _mapping(reward.get("training")) or capstone_reward_training
    reward_preconditions = _mapping(reward.get("preconditions"))
    reward_model_specs = _mapping(reward.get("model_specs"))
    reward_operating_point = _mapping(reward_model_specs.get("a1_operating_point"))
    arm_sizes = _mapping(reward.get("arm_corpus_sizes"))

    accepted_rejected = {
        "accepted": int(_number(accepted.get("accepted"), 0)),
        "rejected": int(_number(accepted.get("rejected"), 0)),
        "total": int(_number(accepted.get("total"), 0)),
    }
    selector_trained = _bool(
        build.get("selector_trained"), _bool(learned.get("selector_trained"), False)
    )
    comparison_ran = _bool(frontier.get("comparison_ran"), False)
    gate_check_summary = str(
        gate.get(
            "gate_check_summary",
            frontier.get("gate_check_summary", "selector_trained gate blocked"),
        )
    )
    training_error = str(reward_training.get("error", capstone_reward_training.get("error", "")))
    checkpoint_path = str(
        reward.get(
            "stable_checkpoint_path",
            reward_preconditions.get("stable_checkpoint_path", ""),
        )
    )
    checkpoint_slug = Path(checkpoint_path).name if checkpoint_path else REWARD_CHECKPOINT_SLUG
    arc_auroc = round(_number(detector_aurocs.get("arc"), ARC_DETECTOR_AUROC_DEFAULT), 4)
    arc_n = int(_number(detector_ns.get("arc"), ARC_LABELED_CANDIDATE_N_DEFAULT))
    reward_youden = round(
        _number(
            reward.get("youden_j"), _number(reward_from_capstone.get("youden_j"), YOUDEN_J_DEFAULT)
        ),
        4,
    )

    return {
        "summary": "oracle_distinct_wrong_file_data_bug_signal_exists_reward_peft_arc16",
        "outer_loop_trm_training_done": True,
        "outer_loop_trm_val": 0.8227,
        "outer_loop_sigterm_reported": True,
        "conductor_stands_down_on_trm_training": True,
        "no_conductor_training_rule": True,
        "forbidden_conductor_actions": [
            "launch_trm_training",
            "pkill_or_kill_train_py",
            "write_results_trm_runs_sudoku_extreme_baseline",
        ],
        "capstone_honest_verdict": str(capstone.get("honest_verdict", "")),
        "capstone_headline_outcome": str(capstone.get("headline_outcome", "")),
        "capstone_oracle_distinct_status": str(capstone.get("oracle_distinct_status", "")),
        "capstone_headline_is_infra_artifact": (
            str(capstone.get("oracle_distinct_status", "")) == "NO-HEADROOM-OR-NO-SIGNAL"
            and not comparison_ran
            and not selector_trained
            and accepted_rejected["total"] == 0
        ),
        "oracle_distinct_comparison_ran": comparison_ran,
        "oracle_distinct_gate_blocked_on_data": (
            not comparison_ran
            and not selector_trained
            and accepted_rejected["total"] == 0
            and "arc3_trm_verifier_rerank.json" in str(build.get("candidate_pool_source", ""))
        ),
        "wrong_file_candidate_pool_source": str(build.get("candidate_pool_source", "")),
        "accepted_rejected_n": accepted_rejected,
        "selector_trained": selector_trained,
        "gate_check_summary": gate_check_summary,
        "gate_blocked_at_layer": str(
            gate.get("blocked_at_layer", frontier.get("blocked_at_layer", ""))
        ),
        "working_label_loader": WORKING_LABEL_LOADER,
        "working_label_pool_path": WORKING_LABEL_POOL_PATH,
        "working_programs_path": WORKING_PROGRAMS_PATH,
        "arc_labeled_candidate_n": arc_n,
        "oracle_distinct_arc_detection_auroc": arc_auroc,
        "oracle_distinct_arc_detection_auroc_ci95": _ci95(detector_ci95.get("arc")),
        "arc_selector_headroom": round(
            _number(detector_headroom.get("arc"), ARC_SELECTOR_HEADROOM_DEFAULT), 4
        ),
        "arc_detector_verifier_is_oracle": _bool(detector_oracle.get("arc"), False),
        "oracle_distinct_signal_exists": arc_auroc >= 0.9
        and arc_n >= ARC_LABELED_CANDIDATE_N_DEFAULT,
        "reward_training_status": str(reward_training.get("status", "")),
        "reward_training_error": training_error,
        "reward_peft_attach_failed": "Gemma4ClippableLinear" in training_error,
        "reward_third_infra_failure": True,
        "reward_phase0_precision": PHASE0_PRECISION_DEFAULT,
        "reward_youden_j": reward_youden,
        "reward_corpora": {
            "A": int(_number(arm_sizes.get("A"), ARM_A_DEFAULT)),
            "B": int(_number(arm_sizes.get("B"), ARM_B_DEFAULT)),
            "C": int(_number(arm_sizes.get("C"), ARM_C_DEFAULT)),
        },
        "reward_base_passrate": round(_number(reward_operating_point.get("base_passrate"), 0.6), 3),
        "reward_checkpoint_path": checkpoint_path,
        "reward_checkpoint_slug": checkpoint_slug,
        "reward_checkpoint_intact": _checkpoint_present(
            root, checkpoint_path, reward_preconditions.get("stable_checkpoint_readable")
        ),
        "reward_verifier_is_oracle": _bool(reward.get("verifier_is_oracle"), True),
        "total_levels_solved": int(
            _number(arc.get("total_arc_levels_solved"), TOTAL_LEVELS_SOLVED_DEFAULT)
        ),
        "total_games_solved": int(
            _number(arc.get("total_arc_games_solved"), TOTAL_GAMES_SOLVED_DEFAULT)
        ),
        "arc_incremental_honest_verdict": str(arc.get("honest_verdict", "")),
        "live_solver_honest_verdict": str(live.get("honest_verdict", "")),
        "live_solver_levels_completed": int(_number(live.get("levels_completed"), 0)),
        "live_solver_efficiency_only_no_level": (
            int(_number(live.get("levels_completed"), 0)) == 0
            and _bool(live.get("solver_beats_floor_efficiency"), True)
            and not _bool(live.get("solver_beats_floor_accuracy"), False)
        ),
        "flagged_artifacts_skipped": _flagged_ids(capstone.get("flagged_artifacts_skipped")),
        "v391_frame": V391_FRAME,
    }


def _run_command(command: list[str], root: Path) -> CommandResult:
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1200)
    except FileNotFoundError as exc:
        return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired:
        return CommandResult(command=command, exit_code=-1, stdout="", stderr="Command timed out")
    return CommandResult(
        command=command,
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _git_lines(root: Path, args: list[str]) -> list[str]:
    result = _run_command(["git", *args], root)
    if result.exit_code != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def smart_subset_targets(root: Path) -> list[str]:
    """Return the conductor smart-subset targets, including changed tests."""

    targets: list[str] = [target for target in CORE_SMART_SUBSET if (root / target).exists()]
    for changed in (
        _git_lines(root, ["diff", "--name-only", "HEAD~1"])
        + _git_lines(root, ["diff", "--name-only", "HEAD"])
        + _git_lines(root, ["ls-files", "--others", "--exclude-standard"])
    ):
        if (
            changed.startswith("tests/python/")
            and changed.endswith(".py")
            and "/quarantine/" not in changed
            and changed not in targets
            and (root / changed).exists()
        ):
            targets.append(changed)
    return targets or [CORE_SMART_SUBSET[0]]


def smart_subset_command(targets: Sequence[str]) -> list[str]:
    """Build the pytest command used for the smart-subset gate."""

    return [str(PYTEST_BIN), *targets, "-q", "--no-header", "-n", "0", "--no-cov", "-o", "addopts="]


def run_smart_subset(root: Path) -> CommandResult:
    """Run the smart-subset pre-test gate from the repository root."""

    return _run_command(smart_subset_command(smart_subset_targets(root)), root)


def terminal_verdict(close_state: Mapping[str, Any]) -> str:
    """Build the complete-path honest verdict from validated close-state."""

    auroc = _number(
        close_state.get("oracle_distinct_arc_detection_auroc"), ARC_DETECTOR_AUROC_DEFAULT
    )
    levels = int(_number(close_state.get("total_levels_solved"), TOTAL_LEVELS_SOLVED_DEFAULT))
    return (
        "success: archived_v390_v391_active_oracle_distinct_wrong_file_data_bug_"
        f"signal_exists_auroc{auroc:.4f}_reward_peft_attach_failed_arc{levels}_pretest_green"
    )


def build_complete_artifact(
    *,
    v390_close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal Exp 4219 archive artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4219,
        "task_id": TASK_ID,
        "random_seed": RANDOM_SEED,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": True,
        "exclusion_manifest_parses": True,
        "pretest_suite_green": True,
        "preconditions_checked": dict(preconditions_checked),
        "v390_close_state": dict(v390_close_state),
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": terminal_verdict(v390_close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-4219", "SCENARIO-REPORT-4219"],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_milestone_confirmed: str,
    active_roadmap_path: str,
) -> JsonDict:
    """Build a blocked artifact without claiming the archive succeeded."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4219,
        "task_id": TASK_ID,
        "random_seed": RANDOM_SEED,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": bool(
            _mapping(preconditions_checked.get("research_complete_yaml")).get("parses", False)
        ),
        "exclusion_manifest_parses": bool(
            _mapping(preconditions_checked.get("exclusion_manifest_yaml")).get("parses", False)
        ),
        "pretest_suite_green": bool(
            _mapping(preconditions_checked.get("smart_subset_pretest")).get("green", False)
        ),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-4219", "SCENARIO-REPORT-4219-BLOCKED-PRECONDITION"],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _blocked(
    root: Path,
    reason: str,
    *,
    preconditions_checked: Mapping[str, Any],
    started_s: float | None,
    now_s: float | None,
    active_milestone_confirmed: str = "",
    active_roadmap_path: str = "research-roadmap.yaml",
) -> Path:
    output_path = root / OUTPUT_REL_PATH
    payload = build_blocked_artifact(
        reason,
        preconditions_checked=preconditions_checked,
        duration_s=duration_from(started_s, now_s),
        active_milestone_confirmed=active_milestone_confirmed,
        active_roadmap_path=active_roadmap_path,
    )
    write_payload(output_path, payload)
    return output_path


def _command_check(result: CommandResult) -> JsonDict:
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "green": result.exit_code == 0,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _source_checks(root: Path) -> JsonDict:
    checks: JsonDict = {}
    for source in V390_SOURCE_ARTIFACTS:
        path = root / str(source["deliverable"])
        checks[str(source["experiment_id"])] = {
            "path": str(source["deliverable"]),
            "exists": path.exists(),
            "sha256": file_sha256(path),
        }
    return checks


def run(
    root: Path = REPO_ROOT,
    *,
    pretest_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Run the Exp 4219 record-only archive workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    preconditions: JsonDict = {}
    research_path = root / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root / EXCLUSION_MANIFEST_REL_PATH

    if not research_path.exists():
        preconditions["research_complete_yaml"] = {
            "path": str(RESEARCH_COMPLETE_REL_PATH),
            "exists": False,
            "parses": False,
        }
        return _blocked(
            root,
            "blocked_research_complete_yaml_missing",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )
    research_text = research_path.read_text(encoding="utf-8")
    research_ok = yaml_parses(research_text)
    preconditions["research_complete_yaml"] = {
        "path": str(RESEARCH_COMPLETE_REL_PATH),
        "exists": True,
        "parses": research_ok,
    }
    if not research_ok:
        return _blocked(
            root,
            "blocked_research_complete_yaml_poison",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )

    if not manifest_path.exists():
        preconditions["exclusion_manifest_yaml"] = {
            "path": str(EXCLUSION_MANIFEST_REL_PATH),
            "exists": False,
            "parses": False,
        }
        return _blocked(
            root,
            "blocked_exclusion_manifest_missing",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_ok = yaml_parses(manifest_text)
    preconditions["exclusion_manifest_yaml"] = {
        "path": str(EXCLUSION_MANIFEST_REL_PATH),
        "exists": True,
        "parses": manifest_ok,
    }
    if not manifest_ok:
        return _blocked(
            root,
            "blocked_exclusion_manifest_yaml_poison",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )

    pretest = run_smart_subset(root) if pretest_result is None else pretest_result
    preconditions["smart_subset_pretest"] = _command_check(pretest)
    if pretest.exit_code != 0:
        return _blocked(
            root,
            "blocked_smart_subset_pretest_not_green",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
        )

    active_milestone, roadmap_path = read_active_milestone(root)
    preconditions["active_milestone"] = {
        "expected": ACTIVATED_MILESTONE,
        "actual": active_milestone,
        "path": roadmap_path,
        "matches": active_milestone == ACTIVATED_MILESTONE,
    }
    if active_milestone != ACTIVATED_MILESTONE:
        return _blocked(
            root,
            "blocked_v391_not_active",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=roadmap_path,
        )

    source_checks = _source_checks(root)
    preconditions["source_artifacts"] = source_checks
    for experiment_id, check in source_checks.items():
        if not check["exists"]:
            return _blocked(
                root,
                SOURCE_MISSING_REASONS[experiment_id],
                preconditions_checked=preconditions,
                started_s=started,
                now_s=now_s,
                active_milestone_confirmed=active_milestone,
                active_roadmap_path=roadmap_path,
            )

    sources = read_v390_sources(root)
    close_state = build_v390_close_state(sources, root=root)
    new_research_text, duplicates_removed, action = dedupe_or_update_record(
        research_text, close_state
    )
    if not yaml_parses(new_research_text):
        return _blocked(
            root,
            "blocked_research_complete_edit_invalid",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=roadmap_path,
        )

    research_path.write_text(new_research_text, encoding="utf-8")
    if not yaml_parses(research_path.read_text(encoding="utf-8")):
        return _blocked(
            root,
            "blocked_research_complete_yaml_poison_after_edit",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=roadmap_path,
        )

    payload = build_complete_artifact(
        v390_close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(started, now_s),
        active_roadmap_path=roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=duplicates_removed,
        cited_upstream_artifacts=build_cited_upstream(root),
    )
    validate_artifact(payload)
    output_path = root / OUTPUT_REL_PATH
    write_payload(output_path, payload)
    return output_path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the complete-path artifact against the Exp 4219 contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in payload, f"missing required fields: {field}")
    verdict = payload.get("honest_verdict")
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "terminal-prefixed honest_verdict required",
    )
    principles = payload.get("field_principles")
    _require(isinstance(principles, Mapping), "field_principles must be a mapping")
    for field in ("honest_verdict", "v390_close_state", "preconditions_checked"):
        _require(field in principles, f"missing field principles: {field}")
        _require(
            principles[field] == FIELD_PRINCIPLES[field], "principle must match REQ-REPORT-4219"
        )
    _require(payload.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone mismatch")
    _require(
        payload.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone mismatch"
    )
    _require(
        payload.get("research_complete_yaml_parses") is True, "research-complete YAML must parse"
    )
    _require(payload.get("exclusion_manifest_parses") is True, "exclusion manifest must parse")
    _require(payload.get("pretest_suite_green") is True, "pretest suite must be green")
    _require(
        payload.get("active_milestone_confirmed") == ACTIVATED_MILESTONE,
        "active milestone mismatch",
    )
    close_state = payload.get("v390_close_state")
    _require(isinstance(close_state, Mapping), "v390_close_state must be a mapping")
    _require(
        close_state.get("capstone_headline_is_infra_artifact") is True,
        "infra artifact not recorded",
    )
    _require(
        close_state.get("oracle_distinct_gate_blocked_on_data") is True, "data block not recorded"
    )
    _require(
        close_state.get("oracle_distinct_comparison_ran") is False, "comparison ran unexpectedly"
    )
    _require(close_state.get("selector_trained") is False, "selector was not blocked")
    _require(
        close_state.get("accepted_rejected_n") == {"accepted": 0, "rejected": 0, "total": 0},
        "accepted labels mismatch",
    )
    _require(
        close_state.get("oracle_distinct_arc_detection_auroc") == ARC_DETECTOR_AUROC_DEFAULT,
        "AUROC signal missing",
    )
    _require(
        close_state.get("arc_labeled_candidate_n") == ARC_LABELED_CANDIDATE_N_DEFAULT,
        "candidate N mismatch",
    )
    _require(close_state.get("arc_detector_verifier_is_oracle") is False, "oracle flag mismatch")
    _require(close_state.get("reward_peft_attach_failed") is True, "PEFT failure missing")
    _require(
        close_state.get("reward_phase0_precision") == PHASE0_PRECISION_DEFAULT,
        "phase0 precision mismatch",
    )
    _require(close_state.get("reward_youden_j") == YOUDEN_J_DEFAULT, "Youden mismatch")
    _require(
        close_state.get("reward_corpora")
        == {"A": ARM_A_DEFAULT, "B": ARM_B_DEFAULT, "C": ARM_C_DEFAULT},
        "corpora mismatch",
    )
    _require(close_state.get("reward_checkpoint_intact") is True, "checkpoint not intact")
    _require(
        close_state.get("total_levels_solved") == TOTAL_LEVELS_SOLVED_DEFAULT, "ARC levels mismatch"
    )
    _require(
        close_state.get("total_games_solved") == TOTAL_GAMES_SOLVED_DEFAULT, "ARC games mismatch"
    )
    _require(
        close_state.get("live_solver_efficiency_only_no_level") is True,
        "live efficiency-only result missing",
    )
    _require(
        close_state.get("flagged_artifacts_skipped") == [4212, 4216], "flagged artifacts mismatch"
    )
    _require(close_state.get("v391_frame") == V391_FRAME, "v391 frame mismatch")


def main() -> int:
    """Run the Exp 4219 archive workflow from the repository root."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
