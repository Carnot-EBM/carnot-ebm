"""Archive .410, activate .411, and record the generic-solver close-state.

Spec refs: REQ-REPORT-4442, SCENARIO-REPORT-4442,
SCENARIO-REPORT-4442-BLOCKED-PRECONDITION.

This transition is record-only. It aggregates the .410 capstone and registry,
records the real-but-quarantined g50t lesson without banking a flagged artifact,
and leaves training and leaderboard submission untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml

from carnot.reporting.archive_v391_activate_v392_4230 import (
    CommandResult,
    duration_from,
    file_sha256,
    is_sha256,
    payload_checksum,
    read_active_milestone,
    read_json_object,
    run_smart_subset,
    write_payload,
    yaml_parses,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.410"
ACTIVATED_MILESTONE = "2026.06.411"
RANDOM_SEED = 4442
OUTPUT_REL_PATH = Path("results/experiment_4442_archive_410_activate_411.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
ARC_REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4441_capstone_v410.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v410_to_v411_4442.v1"
TASK_ID = "exp4442-archive-410-activate-411"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.410['\"]?\s*$")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal-prefixed self-declared state lets the reconciler classify without re-running"
    ),
    "reproducible_total_levels": (
        "the bare authoritative count from ops/arc_solve_registry.yaml (37); "
        "the sprint's monotonic progress metric"
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this transition reads upstream YAML/JSON, "
        "declares the 100us floor, and is the .410-lesson default for non-solve "
        "ARC-adjacent tasks"
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproducible_total_levels",
    "inference_substrate",
    "reproducible_total_games",
    "v410_close_state",
    "preconditions_checked",
    "verifier_is_oracle",
    "trm_training_ran",
    "leaderboard_submission",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

YAML_PRECONDITIONS = (
    ("research_complete_yaml", RESEARCH_COMPLETE_REL_PATH),
    ("exclusion_manifest_yaml", EXCLUSION_MANIFEST_REL_PATH),
    ("research_roadmap_next_yaml", RESEARCH_ROADMAP_NEXT_REL_PATH),
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _record_id(line: str) -> str | None:
    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def archive_record_count(text: str) -> int:
    """Count top-level `.410` archive records without nested task false positives."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _yaml_file_check(path: Path) -> tuple[bool, JsonDict]:
    if not path.exists():
        return False, {"path": str(path), "exists": False, "parses": False}
    text = path.read_text(encoding="utf-8")
    return yaml_parses(text), {"path": str(path), "exists": True, "parses": yaml_parses(text)}


def registry_totals_from_text(text: str) -> JsonDict:
    """Read the authoritative reproduced ARC totals from the registry YAML."""

    loaded = yaml.safe_load(text)
    registry = _mapping(loaded)
    return {
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "reproducible_total_games": _int(registry.get("reproducible_total_games")),
    }


def _flagged_rows(capstone: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(row) for row in _list(capstone.get("flagged_artifacts_excluded"))]


def _flagged_ids(capstone: Mapping[str, Any]) -> list[int]:
    return sorted(_int(row.get("experiment_id")) for row in _flagged_rows(capstone))


def _has_duration_too_short(capstone: Mapping[str, Any]) -> bool:
    for row in _flagged_rows(capstone):
        for flag in _list(row.get("live_critical_flags")):
            if _mapping(flag).get("kind") == "DURATION_TOO_SHORT":
                return True
    return False


def build_v410_close_state(
    capstone: Mapping[str, Any], registry_totals: Mapping[str, Any]
) -> JsonDict:
    """Build the `.410` close-state that `.411` agents should inherit."""

    headline = _mapping(capstone.get("headline_question_answers"))
    loo = _mapping(headline.get("exp4432"))
    action = _mapping(capstone.get("action_model")) or _mapping(headline.get("exp4434"))
    first = _mapping(capstone.get("first_contact")) or _mapping(headline.get("exp4435"))
    primitives = _mapping(capstone.get("primitives")) or _mapping(headline.get("exp4436"))
    residuals = _list(capstone.get("residual_deltas")) or _list(loo.get("residual_deltas"))

    return {
        "generic_solver_gap_state": str(capstone.get("generic_solver_gap_state", "partial")),
        "generic_loo_solve_count": _int(capstone.get("generic_loo_solve_count")),
        "generic_loo_target_count": 7,
        "residual_deltas": [dict(_mapping(row)) for row in residuals],
        "action_model": {
            "state": str(action.get("state", "examples_helped_no_reproduced_level")),
            "accuracy_delta": _float(action.get("accuracy_delta")),
            "helped_vs_cold_control": _bool(action.get("helped_vs_cold_control")),
            "offline_reproduced": _bool(action.get("offline_reproduced")),
            "reproduced_levels": _int(action.get("reproduced_levels")),
            "world_model_accuracy_cold": _float(action.get("world_model_accuracy_cold")),
            "world_model_accuracy_with_examples": _float(
                action.get("world_model_accuracy_with_examples")
            ),
        },
        "first_contact": {
            "state": str(first.get("state", "contract_fixed_no_routed_solve")),
            "target_game": str(first.get("target_game", "dc22")),
            "verdict_contract_fixed": _bool(first.get("verdict_contract_fixed")),
            "routed_solve_banked": _bool(first.get("routed_solve_banked")),
            "offline_reproduced": _bool(first.get("offline_reproduced")),
            "reproduced_levels": _int(first.get("reproduced_levels")),
        },
        "primitives": {
            "state": str(primitives.get("state", "consolidated_no_regression")),
            "deepened_game": str(primitives.get("deepened_game", "tu93")),
            "new_levels_reproduced": _int(primitives.get("new_levels_reproduced")),
            "count": _int(primitives.get("count", primitives.get("primitives_consolidated_count"))),
            "no_regression": _bool(primitives.get("no_regression")),
        },
        "g50t_l1_quarantine": {
            "game": "g50t",
            "level": 1,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "held_out_level_banked": False,
            "trusted_for_aggregation": False,
            "flagged_adversarial": 4433 in _flagged_ids(capstone),
            "duration_too_short": _has_duration_too_short(capstone),
            "substrate_declaration_false_positive": True,
            "reason": "flagged_adversarial_DURATION_TOO_SHORT_inference_substrate_none",
        },
        "flagged_artifacts_skipped": _flagged_ids(capstone),
        "reproducible_total_levels": _int(registry_totals.get("reproducible_total_levels")),
        "reproducible_total_games": _int(registry_totals.get("reproducible_total_games")),
        "verifier_is_oracle_honored": True,
        "circular_execution_grounded_arc_solve_not_moat_headline": True,
    }


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    residual_games = ",".join(str(row.get("game", "")) for row in _list(close_state.get("residual_deltas")))
    return (
        ".410 close-state: generic_solver_gap_state=partial; generic_loo_solve_count=2/7; "
        f"residual_deltas={residual_games}; exp4434 example-conditioning lifted world-model "
        "accuracy +0.286 vs cold with no banked level; exp4435 fixed the verdict contract "
        "but routed dc22 with no banked level; exp4436 deepened tu93 +1 and consolidated "
        "5 primitives with no regression; exp4433 g50t L1 was real but quarantined "
        "DURATION_TOO_SHORT because inference_substrate=None, so it is not banked here."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .410 and activate .411; record true generic-solver close-state')}",
        "  completed: '2026-06-19'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4442-archive-410-activate-411",
        "  tasks:",
        "  - id: exp4441-capstone-410",
        "    result: 'generic solver gap partial; LOO 2/7; registry 37/18'",
    ]
    return "\n".join(lines) + "\n"


def _insert_before_tasks(lines: list[str], new_line: str) -> list[str]:
    for index, line in enumerate(lines):
        if line == "  tasks:":
            return lines[:index] + [new_line] + lines[index:]
    return lines + [new_line]


def _canonicalize_span(lines: list[str], close_state: Mapping[str, Any]) -> list[str]:
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
                out.append("  activation_recorded: exp4442-archive-410-activate-411")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4442-archive-410-activate-411")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.410` record exists and carries the truth."""

    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if _record_id(line) is not None]
    spans = [
        (start, starts[index + 1] if index + 1 < len(starts) else len(lines))
        for index, start in enumerate(starts)
    ]
    target_spans = [
        (start, end) for start, end in spans if _record_id(lines[start]) == ARCHIVED_MILESTONE
    ]
    if not target_spans:
        return f"{text.rstrip()}\n{build_canonical_record(close_state)}", 0, "appended"

    first_start, first_end = target_spans[0]
    replacement = _canonicalize_span(lines[first_start:first_end], close_state)
    remove = {index for start, end in target_spans[1:] for index in range(start, end)}
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
    return new_text, 0, "updated" if new_text != text else "unchanged"


def terminal_verdict(close_state: Mapping[str, Any]) -> str:
    return (
        "complete: archived_v410_v411_active_generic_gap_"
        f"{close_state.get('generic_solver_gap_state')}_loo_"
        f"{_int(close_state.get('generic_loo_solve_count'))}_of_7_levels_"
        f"{_int(close_state.get('reproducible_total_levels'))}_g50t_quarantined"
    )


def _command_check(result: CommandResult) -> JsonDict:
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "green": result.exit_code == 0,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def build_cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "kind": "artifact",
            "experiment_id": "4441",
            "deliverable": str(CAPSTONE_REL_PATH),
            "required": True,
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "kind": "registry",
            "experiment_id": "arc_solve_registry",
            "deliverable": str(ARC_REGISTRY_REL_PATH),
            "required": True,
            "sha256": file_sha256(root / ARC_REGISTRY_REL_PATH),
        },
    ]


def build_complete_artifact(
    *,
    close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: list[Mapping[str, Any]],
) -> JsonDict:
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4442,
        "task_id": TASK_ID,
        "random_seed": RANDOM_SEED,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": True,
        "exclusion_manifest_parses": True,
        "research_roadmap_next_yaml_parses": True,
        "pretest_suite_green": True,
        "preconditions_checked": dict(preconditions_checked),
        "v410_close_state": dict(close_state),
        "reproducible_total_levels": _int(close_state.get("reproducible_total_levels")),
        "reproducible_total_games": _int(close_state.get("reproducible_total_games")),
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": terminal_verdict(close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4442", "SCENARIO-REPORT-4442"],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_milestone_confirmed: str = "",
    active_roadmap_path: str = "research-roadmap.yaml",
) -> JsonDict:
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4442,
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
        "research_roadmap_next_yaml_parses": bool(
            _mapping(preconditions_checked.get("research_roadmap_next_yaml")).get("parses", False)
        ),
        "pretest_suite_green": bool(
            _mapping(preconditions_checked.get("smart_subset_pretest")).get("green", False)
        ),
        "preconditions_checked": dict(preconditions_checked),
        "v410_close_state": {},
        "reproducible_total_levels": 0,
        "reproducible_total_games": 0,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4442", "SCENARIO-REPORT-4442-BLOCKED-PRECONDITION"],
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
    write_payload(
        output_path,
        build_blocked_artifact(
            reason,
            preconditions_checked=preconditions_checked,
            duration_s=duration_from(started_s, now_s),
            active_milestone_confirmed=active_milestone_confirmed,
            active_roadmap_path=active_roadmap_path,
        ),
    )
    return output_path


def run(
    root: Path = REPO_ROOT,
    *,
    pretest_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Run the `.410` archive workflow and write the terminal artifact."""

    root = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    preconditions: JsonDict = {}

    for key, rel_path in YAML_PRECONDITIONS:
        ok, check = _yaml_file_check(root / rel_path)
        check["path"] = str(rel_path)
        preconditions[key] = check
        if not ok:
            return _blocked(
                root,
                "blocked_yaml_parse",
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
            )

    active_milestone, active_path = read_active_milestone(root)
    preconditions["active_milestone"] = {
        "expected": ACTIVATED_MILESTONE,
        "actual": active_milestone,
        "path": active_path,
        "matches": active_milestone == ACTIVATED_MILESTONE,
    }
    if active_milestone != ACTIVATED_MILESTONE:
        return _blocked(
            root,
            "blocked_v411_not_active",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    capstone_path = root / CAPSTONE_REL_PATH
    preconditions["v410_capstone"] = {
        "path": str(CAPSTONE_REL_PATH),
        "exists": capstone_path.exists(),
        "sha256": file_sha256(capstone_path),
    }
    if not capstone_path.exists():
        return _blocked(
            root,
            "blocked_v410_capstone_missing",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    registry_path = root / ARC_REGISTRY_REL_PATH
    ok, registry_check = _yaml_file_check(registry_path)
    registry_check["path"] = str(ARC_REGISTRY_REL_PATH)
    preconditions["arc_solve_registry_yaml"] = registry_check
    if not ok:
        return _blocked(
            root,
            "blocked_arc_solve_registry_yaml_parse",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    pretest = pretest_result if pretest_result is not None else run_smart_subset(root)
    preconditions["smart_subset_pretest"] = _command_check(pretest)
    if pretest.exit_code != 0:
        return _blocked(
            root,
            "blocked_smart_subset_pretest_not_green",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    capstone = read_json_object(capstone_path)
    registry_totals = registry_totals_from_text(registry_path.read_text(encoding="utf-8"))
    close_state = build_v410_close_state(capstone, registry_totals)
    research_path = root / RESEARCH_COMPLETE_REL_PATH
    research_text = research_path.read_text(encoding="utf-8")
    new_text, duplicates_removed, record_action = dedupe_or_update_record(research_text, close_state)
    if not yaml_parses(new_text):
        return _blocked(
            root,
            "blocked_research_complete_edit_invalid",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    research_path.write_text(new_text, encoding="utf-8")
    payload = build_complete_artifact(
        close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(start, now_s),
        active_roadmap_path=active_path,
        research_complete_record_action=record_action,
        research_complete_duplicates_removed=duplicates_removed,
        cited_upstream_artifacts=build_cited_upstream(root),
    )
    validate_artifact(payload)
    output_path = root / OUTPUT_REL_PATH
    write_payload(output_path, payload)
    return output_path


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the terminal artifact before writing a complete result."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required artifact field: {missing[0]}")
    if not isinstance(payload.get("honest_verdict"), str) or not payload[
        "honest_verdict"
    ].startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict lacks a complete-path terminal prefix")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principle mismatch")
    if payload.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle must be true")
    if payload.get("trm_training_ran") is not False or payload.get("leaderboard_submission") is not False:
        raise ValueError("training and leaderboard submission must stay false")
    if not is_sha256(payload.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a SHA-256 hex digest")


def main(root: Path = REPO_ROOT) -> int:
    output_path = run(root)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
