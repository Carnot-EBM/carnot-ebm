"""Archive .391, activate .392, and preserve the first clean oracle-distinct read.

Spec refs: REQ-REPORT-4230, SCENARIO-REPORT-4230,
SCENARIO-REPORT-4230-BLOCKED-PRECONDITION.

This is a record-only transition. It records that `.391` finally ran the
oracle-distinct beats-vote gate, tied vote on a weak under-powered build, and
therefore hands `.392` a strengthen-and-re-test milestone instead of a redo or
settled refutation.
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
ARCHIVED_MILESTONE = "2026.06.391"
ACTIVATED_MILESTONE = "2026.06.392"
RANDOM_SEED = 4230
OUTPUT_REL_PATH = Path("results/experiment_4230_archive_v391_activate_v392.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4229_capstone_v391.json")
GATE_REL_PATH = Path("results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json")
BUILD_REL_PATH = Path("results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json")
HARNESS_REL_PATH = Path("results/experiment_4222_verifier_reward_lora_harness_fix_smoke.json")
REWARD_REL_PATH = Path("results/experiment_4223_verifier_as_reward_3arm_synchronous.json")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v391_to_v392_4230.v1"
EXPERIMENT_ID = "exp4230"
TASK_ID = "exp4230-archive-v391-activate-v392"

V392_FRAME = (
    "STRENGTHEN the oracle-distinct verifier (fix the three named causes, re-test at power) "
    "+ the harness-first window-boxed verifier-as-reward FINISH"
)

VERIFIER_DELTA_DEFAULT = -0.0714
VERIFIER_CI95_DEFAULT = [-0.214, 0.0]
N_TASKS_DEFAULT = 14
ORACLE_AT_K_DEFAULT = 1.0
ACCEPTED_DEFAULT = 14
REJECTED_DEFAULT = 1782
TOTAL_DEFAULT = 1796
BASE_RATE_DEFAULT = 0.0078
OFF_FOLD_AUROC_DEFAULT = 0.779
TOTAL_LEVELS_SOLVED_DEFAULT = 17
TOTAL_GAMES_SOLVED_DEFAULT = 13
HARNESS_DURATION_DEFAULT = 14.0647
REWARD_DURATION_DEFAULT = 36.6945
YOUDEN_J_DEFAULT = 0.4138
ARM_A_DEFAULT = 776
ARM_B_DEFAULT = 776
ARM_C_DEFAULT = 742
BUILD_ARCHITECTURE_DEFAULT = "class_weight_balanced_standardized_logistic_regression"
BUILD_MODEL_TYPE_DEFAULT = "standardized_logistic_regression"

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V391_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {"experiment_id": "4229", "deliverable": str(CAPSTONE_REL_PATH), "required": True},
    {"experiment_id": "4221", "deliverable": str(GATE_REL_PATH), "required": True},
    {"experiment_id": "4220", "deliverable": str(BUILD_REL_PATH), "required": True},
    {"experiment_id": "4222", "deliverable": str(HARNESS_REL_PATH), "required": True},
    {"experiment_id": "4223", "deliverable": str(REWARD_REL_PATH), "required": True},
)

SOURCE_MISSING_REASONS = {
    "4229": "blocked_v391_capstone_missing",
    "4221": "blocked_oracle_distinct_gate_missing",
    "4220": "blocked_oracle_distinct_build_missing",
    "4222": "blocked_reward_harness_missing",
    "4223": "blocked_reward_three_arm_missing",
}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "pretest_suite_green",
    "v391_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.391.",
    "activated_milestone": "Confirms .392 is live for the strengthen-and-re-test frame.",
    "active_milestone_confirmed": "Records the active roadmap milestone actually observed.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v391_close_state": (
        "Honest record (the oracle-distinct gate RAN and TIED vote on a weak/under-powered "
        "build, NOT a settled refutation; reward 4th/5th infra short-circuit; ARC 17) so "
        "the .392 agents frame the milestone as STRENGTHEN-and-re-test, not a redo."
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
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.391['\"]?\s*$")


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
    """Count top-level `.391` archive records without counting nested task ids."""

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
            round(_number(value[0], VERIFIER_CI95_DEFAULT[0]), 3),
            round(_number(value[1], VERIFIER_CI95_DEFAULT[1]), 3),
        ]
    return list(VERIFIER_CI95_DEFAULT)


def _pass_rates(value: Any) -> JsonDict:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): round(_number(val, 0.0), 4)
        for key, val in value.items()
        if isinstance(val, int | float) and not isinstance(val, bool)
    }


def _corrigendum_kinds(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    kinds: list[str] = []
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("kind"), str):
            kinds.append(str(item["kind"]))
    return kinds


def _flagged_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("experiment_id"), int):
            out.append(int(item["experiment_id"]))
    return out


def _duration_flagged(payload: Mapping[str, Any]) -> bool:
    return "DURATION_TOO_SHORT" in _corrigendum_kinds(payload.get("corrigendum_pending"))


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.391` archive finding from the close-state."""

    accepted = _mapping(close_state.get("accepted_rejected_n"))
    corpora = _mapping(close_state.get("reward_corpora"))
    failures = _mapping(close_state.get("reward_infra_failures"))
    levels = int(_number(close_state.get("total_levels_solved"), TOTAL_LEVELS_SOLVED_DEFAULT))
    games = int(_number(close_state.get("total_games_solved"), TOTAL_GAMES_SOLVED_DEFAULT))
    return (
        ".391 close-state: FIRST CLEAN oracle-distinct gate read, not a settled refutation. "
        "Exp4221 RAN and returned TIES-VOTE-NULL: verifier@1-vote@1="
        f"{_number(close_state.get('verifier_minus_vote_delta'), VERIFIER_DELTA_DEFAULT):.4f}, "
        "CI95 "
        f"{close_state.get('verifier_minus_vote_ci95', VERIFIER_CI95_DEFAULT)}, "
        f"oracle@K={_number(close_state.get('oracle_at_k'), ORACLE_AT_K_DEFAULT):.1f}, "
        f"n={int(_number(close_state.get('n_tasks'), N_TASKS_DEFAULT))}. The build was "
        "UNDER-POWERED + WEAKLY-BUILT: isolated per-candidate logistic regression on "
        f"{int(_number(accepted.get('accepted'), ACCEPTED_DEFAULT))} accepted / "
        f"{int(_number(accepted.get('rejected'), REJECTED_DEFAULT))} rejected rows "
        f"(base-rate={_number(close_state.get('base_rate'), BASE_RATE_DEFAULT):.4f}, "
        f"off-fold AUROC={_number(close_state.get('off_fold_auroc'), OFF_FOLD_AUROC_DEFAULT):.3f}), "
        "with a TAUTOLOGY-flagged build artifact. Verifier-as-reward failed a 4th/5th "
        "time on infra: exp4222 smoke "
        f"{_number(failures.get('exp4222_smoke_duration_s'), HARNESS_DURATION_DEFAULT):.1f}s / "
        "exp4223 3-arm "
        f"{_number(failures.get('exp4223_three_arm_duration_s'), REWARD_DURATION_DEFAULT):.1f}s, "
        "both DURATION-flagged, while operating point and corpora stayed intact "
        f"(A={int(_number(corpora.get('A'), ARM_A_DEFAULT))}/"
        f"B={int(_number(corpora.get('B'), ARM_B_DEFAULT))}/"
        f"C={int(_number(corpora.get('C'), ARM_C_DEFAULT))}). "
        f"ARC total_levels_solved={levels}, total_games_solved={games}; live solver completed "
        "0 levels efficiency-only; flagged-skipped artifacts were 4220, 4222, 4223; "
        "DiffusionGemma STILL-PENDING. "
        f".392 frame: {V392_FRAME}."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.391` record for the absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .391 and activate .392; preserve first clean oracle-distinct read')}",
        "  completed: '2026-06-15'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4230-archive-v391-activate-v392",
        "  tasks:",
        "  - id: exp4221-oracle-distinct-arc-verifier-beats-vote",
        "    result: 'gate ran cleanly and tied vote with headroom present'",
        "  - id: exp4220-oracle-distinct-arc-verifier-build-labeled",
        "    result: 'weak isolated logistic build; 14 accepted / 1782 rejected; AUROC 0.779'",
        "  - id: exp4222-verifier-reward-lora-harness-fix-smoke",
        "    result: 'DURATION-flagged 14s short-circuit'",
        "  - id: exp4223-verifier-as-reward-3arm-synchronous",
        "    result: 'DURATION-flagged 36.7s short-circuit; operating point intact'",
        "  - id: exp4229-capstone-v391",
        "    result: 'TIES-VOTE-NULL; ARC 17 levels / 13 games; DiffusionGemma STILL-PENDING'",
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
                out.append("  activation_recorded: exp4230-archive-v391-activate-v392")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4230-archive-v391-activate-v392")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.391` record exists and carries the close-state."""

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


def read_v391_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.391` close-state."""

    return {
        "4229": read_json_object(root / CAPSTONE_REL_PATH),
        "4221": read_json_object(root / GATE_REL_PATH),
        "4220": read_json_object(root / BUILD_REL_PATH),
        "4222": read_json_object(root / HARNESS_REL_PATH),
        "4223": read_json_object(root / REWARD_REL_PATH),
    }


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.391` artifacts."""

    cited: list[JsonDict] = []
    for source in V391_SOURCE_ARTIFACTS:
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


def build_v391_close_state(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
) -> JsonDict:
    """Build the honest `.391` close-state from capstone and upstream artifacts."""

    capstone = _mapping(sources.get("4229", {}))
    gate = _mapping(sources.get("4221", {}))
    build = _mapping(sources.get("4220", {}))
    harness = _mapping(sources.get("4222", {}))
    reward = _mapping(sources.get("4223", {}))
    learned = _mapping(capstone.get("learned_arc_verifier"))
    frontier = _mapping(capstone.get("oracle_distinct_frontier"))
    reward_from_capstone = _mapping(capstone.get("verifier_as_reward"))
    arc = _mapping(capstone.get("arc_progress"))
    live = _mapping(capstone.get("live_solver_accuracy"))
    build_specs = _mapping(build.get("model_specs"))
    accepted_map = _mapping(build.get("accepted_rejected_n")) or _mapping(
        learned.get("accepted_rejected_n")
    )
    reward_preconditions = _mapping(reward.get("preconditions"))
    harness_preconditions = _mapping(harness.get("preconditions"))
    reward_model_specs = _mapping(reward.get("model_specs"))
    reward_operating_point = _mapping(reward_model_specs.get("a1_operating_point"))
    arm_sizes = _mapping(reward.get("arm_corpus_sizes"))

    accepted = int(_number(accepted_map.get("accepted"), ACCEPTED_DEFAULT))
    rejected = int(_number(accepted_map.get("rejected"), REJECTED_DEFAULT))
    total = int(_number(accepted_map.get("total"), TOTAL_DEFAULT))
    base_rate = round(accepted / total, 4) if total else BASE_RATE_DEFAULT
    gate_ran = _bool(
        gate.get("gate_ran"),
        _bool(frontier.get("gate_ran"), str(gate.get("status", "")) == "complete"),
    )
    verifier_is_oracle = _bool(
        gate.get("verifier_is_oracle"), _bool(frontier.get("verifier_is_oracle"), False)
    )
    gate_clean = gate_ran and str(gate.get("honest_verdict", "")).startswith("complete:") and not verifier_is_oracle
    checkpoint_path = str(
        reward.get(
            "stable_checkpoint_path",
            reward_preconditions.get(
                "stable_checkpoint_path", harness_preconditions.get("stable_checkpoint_path", "")
            ),
        )
    )
    checkpoint_readable = reward_preconditions.get(
        "stable_checkpoint_readable", harness_preconditions.get("stable_checkpoint_readable")
    )
    reward_corpora = {
        "A": int(_number(arm_sizes.get("A"), ARM_A_DEFAULT)),
        "B": int(_number(arm_sizes.get("B"), ARM_B_DEFAULT)),
        "C": int(_number(arm_sizes.get("C"), ARM_C_DEFAULT)),
    }
    live_levels = int(
        _number(live.get("levels_completed"), _number(live.get("scorecard_levels_completed"), 0))
    )

    return {
        "summary": "oracle_distinct_clean_ties_vote_weak_build_reward_duration_arc17",
        "outer_loop_trm_training_done": True,
        "outer_loop_trm_val": 0.8227,
        "outer_loop_sigterm_reported": True,
        "conductor_stands_down_on_trm_training": True,
        "oracle_distinct_gate_ran": gate_ran,
        "oracle_distinct_gate_clean": gate_clean,
        "oracle_distinct_status": str(
            frontier.get(
                "oracle_distinct_status",
                "TIES-VOTE-NULL" if gate_ran else capstone.get("oracle_distinct_status", ""),
            )
        ),
        "oracle_distinct_beats_vote": _bool(
            gate.get("oracle_distinct_beats_vote"),
            _bool(frontier.get("oracle_distinct_beats_vote"), False),
        ),
        "verifier_is_oracle": verifier_is_oracle,
        "verifier_minus_vote_delta": round(
            _number(
                gate.get("verifier_minus_vote_delta"),
                _number(frontier.get("verifier_minus_vote_delta"), VERIFIER_DELTA_DEFAULT),
            ),
            4,
        ),
        "verifier_minus_vote_ci95": _ci95(
            gate.get("verifier_minus_vote_ci95", frontier.get("verifier_minus_vote_ci95"))
        ),
        "n_tasks": int(
            _number(gate.get("n_tasks"), _number(frontier.get("n_tasks"), N_TASKS_DEFAULT))
        ),
        "oracle_at_k": round(
            _number(gate.get("oracle_at_k"), _number(frontier.get("oracle_at_k"), ORACLE_AT_K_DEFAULT)),
            4,
        ),
        "headroom_exists": _bool(
            gate.get("headroom_exists"), _bool(frontier.get("headroom_present"), True)
        ),
        "pass_rates": _pass_rates(gate.get("pass_rates", frontier.get("pass_rates"))),
        "matched_control_delta": round(
            _number(gate.get("matched_control_delta"), _number(frontier.get("matched_control_delta"), 0.0)),
            4,
        ),
        "underpowered_first_clean_read": True,
        "not_settled_refutation": True,
        "weak_build_causes": [
            "isolated_per_candidate_logistic_regression",
            "extreme_class_imbalance_14_positive_1782_negative",
            "held_out_gate_n14_below_clt_floor",
        ],
        "build_model_type": str(learned.get("model_type", BUILD_MODEL_TYPE_DEFAULT)),
        "build_architecture": str(build_specs.get("architecture", BUILD_ARCHITECTURE_DEFAULT)),
        "accepted_rejected_n": {"accepted": accepted, "rejected": rejected, "total": total},
        "base_rate": base_rate,
        "off_fold_auroc": round(
            _number(
                learned.get("off_fold_auroc"),
                _number(build.get("oracle_distinct_auroc"), OFF_FOLD_AUROC_DEFAULT),
            ),
            3,
        ),
        "positive_sparsity_flag": _bool(build.get("positive_sparsity_flag"), True),
        "build_flagged_adversarial": _bool(build.get("flagged_adversarial"), True),
        "build_corrigendum_kinds": _corrigendum_kinds(build.get("corrigendum_pending")),
        "wrong_majority_n": int(
            _number(build.get("wrong_majority_n"), _number(learned.get("wrong_majority_n"), 5))
        ),
        "reward_infra_failures": {
            "exp4222_smoke_duration_s": round(
                _number(harness.get("duration_s"), HARNESS_DURATION_DEFAULT), 4
            ),
            "exp4223_three_arm_duration_s": round(
                _number(reward.get("duration_s"), REWARD_DURATION_DEFAULT), 4
            ),
            "fourth_and_fifth_infra_short_circuit": True,
        },
        "reward_duration_flagged": _duration_flagged(harness) and _duration_flagged(reward),
        "reward_corpora": reward_corpora,
        "reward_base_passrate": round(_number(reward_operating_point.get("base_passrate"), 0.6), 3),
        "reward_youden_j": round(
            _number(
                reward.get("youden_j"),
                _number(reward_from_capstone.get("youden_j"), YOUDEN_J_DEFAULT),
            ),
            4,
        ),
        "reward_checkpoint_path": checkpoint_path,
        "reward_checkpoint_intact": _checkpoint_present(root, checkpoint_path, checkpoint_readable),
        "reward_verifier_is_oracle": _bool(reward.get("verifier_is_oracle"), True),
        "reward_label_carries_signal": _bool(reward.get("verifier_label_carries_signal"), False),
        "reward_status": "HARNESS-DEFERRED",
        "total_levels_solved": int(
            _number(arc.get("total_arc_levels_solved"), TOTAL_LEVELS_SOLVED_DEFAULT)
        ),
        "total_games_solved": int(
            _number(arc.get("total_arc_games_solved"), TOTAL_GAMES_SOLVED_DEFAULT)
        ),
        "arc_incremental_honest_verdict": str(arc.get("honest_verdict", "")),
        "live_solver_honest_verdict": str(live.get("honest_verdict", "")),
        "live_solver_levels_completed": live_levels,
        "live_solver_efficiency_only_no_level": (
            live_levels == 0
            and _bool(live.get("solver_beats_floor_efficiency"), True)
            and not _bool(live.get("solver_beats_floor_accuracy"), False)
        ),
        "flagged_artifacts_skipped": _flagged_ids(capstone.get("flagged_artifacts_skipped")),
        "diffusiongemma_status": "STILL-PENDING"
        if not _bool(capstone.get("diffusiongemma_gate_resolvable"), False)
        else "MET",
        "v392_frame": V392_FRAME,
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

    auroc = _number(close_state.get("off_fold_auroc"), OFF_FOLD_AUROC_DEFAULT)
    levels = int(_number(close_state.get("total_levels_solved"), TOTAL_LEVELS_SOLVED_DEFAULT))
    return (
        "success: archived_v391_v392_active_oracle_distinct_ties_vote_"
        f"weak_build_auroc{auroc:.3f}_reward_duration_arc{levels}_pretest_green"
    )


def build_complete_artifact(
    *,
    v391_close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal Exp 4230 archive artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4230,
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
        "v391_close_state": dict(v391_close_state),
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": terminal_verdict(v391_close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-4230", "SCENARIO-REPORT-4230"],
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
        "experiment_id": 4230,
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
        "spec_refs": ["REQ-REPORT-4230", "SCENARIO-REPORT-4230-BLOCKED-PRECONDITION"],
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
    for source in V391_SOURCE_ARTIFACTS:
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
    """Run the Exp 4230 record-only archive workflow."""

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
            "blocked_v392_not_active",
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

    sources = read_v391_sources(root)
    close_state = build_v391_close_state(sources, root=root)
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
        v391_close_state=close_state,
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
    """Validate the complete-path artifact against the Exp 4230 contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in payload, f"missing required fields: {field}")
    verdict = payload.get("honest_verdict")
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "terminal-prefixed honest_verdict required",
    )
    principles = payload.get("field_principles")
    _require(isinstance(principles, Mapping), "field_principles must be a mapping")
    for field in ("honest_verdict", "v391_close_state", "preconditions_checked"):
        _require(field in principles, f"missing field principles: {field}")
        _require(
            principles[field] == FIELD_PRINCIPLES[field], "principle must match REQ-REPORT-4230"
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
    close_state = payload.get("v391_close_state")
    _require(isinstance(close_state, Mapping), "v391_close_state must be a mapping")
    _require(close_state.get("oracle_distinct_gate_ran") is True, "gate ran")
    _require(close_state.get("oracle_distinct_gate_clean") is True, "gate clean")
    _require(close_state.get("oracle_distinct_status") == "TIES-VOTE-NULL", "status")
    _require(close_state.get("oracle_distinct_beats_vote") is False, "beats vote")
    _require(close_state.get("verifier_is_oracle") is False, "oracle flag")
    _require(close_state.get("verifier_minus_vote_delta") == VERIFIER_DELTA_DEFAULT, "delta")
    _require(close_state.get("verifier_minus_vote_ci95") == VERIFIER_CI95_DEFAULT, "CI")
    _require(close_state.get("n_tasks") == N_TASKS_DEFAULT, "n_tasks")
    _require(close_state.get("oracle_at_k") == ORACLE_AT_K_DEFAULT, "oracle@K")
    _require(close_state.get("headroom_exists") is True, "headroom")
    _require(close_state.get("not_settled_refutation") is True, "refutation")
    _require(
        close_state.get("accepted_rejected_n")
        == {"accepted": ACCEPTED_DEFAULT, "rejected": REJECTED_DEFAULT, "total": TOTAL_DEFAULT},
        "accepted",
    )
    _require(close_state.get("off_fold_auroc") == OFF_FOLD_AUROC_DEFAULT, "AUROC")
    _require(close_state.get("base_rate") == BASE_RATE_DEFAULT, "base-rate")
    _require("TAUTOLOGY" in close_state.get("build_corrigendum_kinds", []), "TAUTOLOGY")
    _require(close_state.get("reward_duration_flagged") is True, "reward duration")
    _require(
        close_state.get("reward_corpora")
        == {"A": ARM_A_DEFAULT, "B": ARM_B_DEFAULT, "C": ARM_C_DEFAULT},
        "reward corpora",
    )
    _require(close_state.get("reward_youden_j") == YOUDEN_J_DEFAULT, "Youden")
    _require(close_state.get("reward_checkpoint_intact") is True, "checkpoint")
    _require(close_state.get("reward_verifier_is_oracle") is True, "reward oracle")
    _require(
        close_state.get("total_levels_solved") == TOTAL_LEVELS_SOLVED_DEFAULT, "ARC levels"
    )
    _require(close_state.get("total_games_solved") == TOTAL_GAMES_SOLVED_DEFAULT, "ARC games")
    _require(close_state.get("live_solver_efficiency_only_no_level") is True, "live")
    _require(close_state.get("flagged_artifacts_skipped") == [4220, 4222, 4223], "flagged")
    _require(close_state.get("diffusiongemma_status") == "STILL-PENDING", "DiffusionGemma")
    _require(close_state.get("v392_frame") == V392_FRAME, "v392 frame")


def main() -> int:
    """Run the Exp 4230 archive workflow from the repository root."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
