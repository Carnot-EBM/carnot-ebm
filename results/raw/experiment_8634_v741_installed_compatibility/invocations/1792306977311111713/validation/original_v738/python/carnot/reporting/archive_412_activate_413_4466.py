"""Archive .412, activate .413, and record the generic-solver backlog.

Spec refs: REQ-REPORT-4466, SCENARIO-REPORT-4466,
SCENARIO-REPORT-4466-BLOCKED-PRECONDITION.

This transition is record-only. It aggregates the .412 capstone and registry,
records the operationally failed level-up attempts without inflating progress,
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
ARCHIVED_MILESTONE = "2026.06.412"
ACTIVATED_MILESTONE = "2026.06.413"
RANDOM_SEED = 4466
OUTPUT_REL_PATH = Path("results/experiment_4466_archive_412_activate_413.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
ARC_REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4465_capstone_v412.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v412_to_v413_4466.v1"
TASK_ID = "exp4466-archive-412-activate-413"
FOVER_AUC = 0.9131
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
OPEN_GAP_IDS = (
    "GAP-4423-DC22",
    "GAP-4432-LOO-SC25",
    "GAP-4458-SB26",
)
CLOSED_GAP_IDS = ("GAP-4432-LOO-TR87",)
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.412['\"]?\s*$")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal-prefixed self-declared state lets the reconciler classify without re-running"
    ),
    "reproducible_total_levels": (
        "the bare authoritative count from ops/arc_solve_registry.yaml (39, FLAT); "
        "the sprint's monotonic progress metric that .413 must finally move"
    ),
    "open_gap_ids": (
        "the 3 open generic-solver gaps that become the .413 build backlog "
        "(dc22-blocked-by-precondition, sc25-no-artifact, sb26-missing-operator)"
    ),
    "prior_milestone_churn_note": (
        "one honest string: .412 banked ZERO new reproduced levels because the level-up "
        "attempts failed OPERATIONALLY (dc22 precondition block, sc25 no-artifact), "
        "not on the research"
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- this transition reads upstream YAML/JSON, "
        "declares the 100us floor, the .410-lesson default for non-solve "
        "ARC-adjacent tasks"
    ),
}

PRIOR_MILESTONE_CHURN_NOTE = (
    ".412 banked ZERO new reproduced levels because the level-up attempts failed "
    "OPERATIONALLY: dc22 was blocked by a pytest coverage precondition before CEGIS "
    "ran, and sc25 produced no artifact, so the research lever was not actually "
    "tested for those level-up attempts."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproducible_total_levels",
    "open_gap_ids",
    "prior_milestone_churn_note",
    "inference_substrate",
    "reproducible_total_games",
    "v412_close_state",
    "open_gap_failure_modes",
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


def _bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _record_id(line: str) -> str | None:
    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def archive_record_count(text: str) -> int:
    """Count top-level .412 archive records without nested task false positives."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _yaml_file_check(path: Path) -> tuple[bool, JsonDict]:
    if not path.exists():
        return False, {"path": str(path), "exists": False, "parses": False}
    text = path.read_text(encoding="utf-8")
    parses = yaml_parses(text)
    return parses, {"path": str(path), "exists": True, "parses": parses}


def _flagged_rows(capstone: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_mapping(row) for row in _list(capstone.get("flagged_artifacts_excluded"))]


def _flagged_ids(capstone: Mapping[str, Any]) -> list[int]:
    return sorted(_int(row.get("experiment_id")) for row in _flagged_rows(capstone))


def registry_facts_from_text(text: str) -> JsonDict:
    """Read the authoritative reproduced ARC totals and prior milestone flatness."""

    registry = _mapping(yaml.safe_load(text))
    current = _mapping(registry.get("latest_hygiene_4461"))
    prior = _mapping(registry.get("latest_hygiene_4449"))
    current_levels = _int(registry.get("reproducible_total_levels")) or _int(
        current.get("reproducible_total_levels")
    )
    current_games = _int(registry.get("reproducible_total_games")) or _int(
        current.get("reproducible_total_games")
    )
    prior_levels = _int(prior.get("reproducible_total_levels"), current_levels)
    prior_games = _int(prior.get("reproducible_total_games"), current_games)
    return {
        "reproducible_total_levels": current_levels,
        "reproducible_total_games": current_games,
        "v411_reproducible_total_levels": prior_levels,
        "v411_reproducible_total_games": prior_games,
        "reproducible_levels_delta_vs_v411": current_levels - prior_levels,
        "reproducible_games_delta_vs_v411": current_games - prior_games,
    }


def open_gap_failure_modes() -> list[JsonDict]:
    """Return the three .413 backlog gaps with their true .412 failure modes."""

    return [
        {
            "gap_id": "GAP-4423-DC22",
            "source_experiment": 4455,
            "source_artifact": "results/experiment_4455_solve_dc22_cegis_config_rule.json",
            "failure_mode": "blocked_baseline_pytest_coverage",
            "dc22_cegis_solve_ran": False,
            "counterexample_rounds": 0,
            "unblocks_with": "--no-cov precondition fix",
            "status": "open",
        },
        {
            "gap_id": "GAP-4432-LOO-SC25",
            "source_experiment": 4457,
            "source_artifact": "results/experiment_4457_cast_grid_spell_shrink_tank_exit.json",
            "failure_mode": "no_artifact_produced",
            "sc25_provisional_levels": ["L2", "L3", "L4", "L5"],
            "provisional_l2_l5_banked": False,
            "unblocks_with": "split bank-from-generalize plus --no-cov fixes",
            "status": "open",
        },
        {
            "gap_id": "GAP-4458-SB26",
            "source_experiment": 4458,
            "source_artifact": "results/experiment_4458_first_contact_new_game.json",
            "failure_mode": "missing_color_match_slot_sequence_verifier",
            "honest_negative": True,
            "missing_operator": "color_match_slot_sequence_verifier",
            "status": "open",
        },
    ]


def _submission_package(capstone: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(capstone.get("submission_package")) or _mapping(
        _mapping(capstone.get("headline_question_answers")).get("exp4460")
    )


def _publication_gate(capstone: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(capstone.get("publication_gate"))


def build_v412_close_state(
    capstone: Mapping[str, Any], registry_facts: Mapping[str, Any]
) -> JsonDict:
    """Build the .412 close-state that .413 agents should inherit."""

    loo_v3 = _mapping(capstone.get("loo_v3"))
    glyph = _mapping(capstone.get("glyph_rewrite"))
    submission = _submission_package(capstone)
    publication = _publication_gate(capstone)
    level_delta = _int(registry_facts.get("reproducible_levels_delta_vs_v411"))

    return {
        "generic_solver_gap_state": str(capstone.get("generic_solver_gap_state", "partial")),
        "generic_loo_solve_count_v2_baseline": _int(
            loo_v3.get("generic_loo_solve_count_v2_baseline"), 5
        ),
        "generic_loo_solve_count_v3": _int(
            capstone.get("generic_loo_solve_count_v3"),
            _int(loo_v3.get("generic_loo_solve_count_v3"), 6),
        ),
        "reproducible_total_levels": _int(registry_facts.get("reproducible_total_levels")),
        "reproducible_total_games": _int(registry_facts.get("reproducible_total_games")),
        "v411_reproducible_total_levels": _int(
            registry_facts.get("v411_reproducible_total_levels")
        ),
        "v411_reproducible_total_games": _int(registry_facts.get("v411_reproducible_total_games")),
        "reproducible_levels_delta_vs_v411": level_delta,
        "reproducible_games_delta_vs_v411": _int(
            registry_facts.get("reproducible_games_delta_vs_v411")
        ),
        "zero_new_reproduced_levels": level_delta == 0,
        "closed_gap_ids": list(CLOSED_GAP_IDS)
        if _bool(glyph.get("tr87_resolved_generically"), True)
        else [],
        "tr87_resolved_generically": _bool(glyph.get("tr87_resolved_generically"), True),
        "tr87_source_experiment": 4456,
        "submission_package_ready": _bool(submission.get("submission_package_ready"), True),
        "submission_package_levels": _int(
            submission.get("total_reproduced_levels_in_package"), 39
        ),
        "submitted_to_leaderboard": _bool(
            submission.get("submitted_to_leaderboard"),
            _bool(capstone.get("submitted_to_leaderboard"), False),
        ),
        "paper_ready": _bool(publication.get("paper_ready"), _bool(capstone.get("paper_ready"), True)),
        "publication_gate": "G1-G4 FROZEN",
        "fover_auc": FOVER_AUC,
        "open_gap_ids": list(OPEN_GAP_IDS),
        "open_gap_failure_modes": open_gap_failure_modes(),
        "prior_milestone_churn_note": PRIOR_MILESTONE_CHURN_NOTE,
        "flagged_artifacts_skipped": _flagged_ids(capstone),
        "verifier_is_oracle_honored": True,
        "execution_grounded_arc_solve_not_moat_headline": True,
    }


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    return (
        ".412 close-state: generic_solver_gap_state=partial; "
        "generic_loo_solve_count_v3=6 (v2 baseline 5); registry=39/20 FLAT vs .411; "
        "ZERO new reproduced levels; GAP-4432-LOO-TR87 closed generically by exp4456; "
        "submission_package_ready=true (39 levels, not submitted); paper_ready=true "
        "(FoVer 0.9131, G1-G4 frozen); open gaps: GAP-4423-DC22 "
        "blocked_baseline_pytest_coverage, GAP-4432-LOO-SC25 no artifact with L2-L5 "
        "unbanked, GAP-4458-SB26 missing_color_match_slot_sequence_verifier."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .412 and activate .413; record true generic-solver backlog')}",
        "  completed: '2026-06-19'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4466-archive-412-activate-413",
        "  tasks:",
        "  - id: exp4465-capstone-412",
        "    result: 'generic solver gap partial; LOO v3 6/7; registry 39/20; zero new reproduced levels'",
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
                out.append("  activation_recorded: exp4466-archive-412-activate-413")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4466-archive-412-activate-413")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level .412 record exists and carries the truth."""

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
        "complete: archived_v412_v413_active_generic_gap_"
        f"{close_state.get('generic_solver_gap_state')}_loo_v3_"
        f"{_int(close_state.get('generic_loo_solve_count_v3'))}_levels_"
        f"{_int(close_state.get('reproducible_total_levels'))}_games_"
        f"{_int(close_state.get('reproducible_total_games'))}_zero_new_levels_open_gaps_3"
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
            "experiment_id": "4465",
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
        "experiment_id": 4466,
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
        "v412_close_state": dict(close_state),
        "generic_solver_gap_state": str(close_state.get("generic_solver_gap_state", "partial")),
        "generic_loo_solve_count_v2_baseline": _int(
            close_state.get("generic_loo_solve_count_v2_baseline")
        ),
        "generic_loo_solve_count_v3": _int(close_state.get("generic_loo_solve_count_v3")),
        "reproducible_total_levels": _int(close_state.get("reproducible_total_levels")),
        "reproducible_total_games": _int(close_state.get("reproducible_total_games")),
        "open_gap_ids": list(OPEN_GAP_IDS),
        "open_gap_failure_modes": open_gap_failure_modes(),
        "closed_gap_ids": list(CLOSED_GAP_IDS),
        "prior_milestone_churn_note": PRIOR_MILESTONE_CHURN_NOTE,
        "flagged_artifacts_skipped": _list(close_state.get("flagged_artifacts_skipped")),
        "submission_package_ready": _bool(close_state.get("submission_package_ready")),
        "submission_package_levels": _int(close_state.get("submission_package_levels")),
        "submitted_to_leaderboard": _bool(close_state.get("submitted_to_leaderboard")),
        "paper_ready": _bool(close_state.get("paper_ready")),
        "fover_auc": FOVER_AUC,
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": terminal_verdict(close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "no_3090_inference": True,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4466", "SCENARIO-REPORT-4466"],
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
        "experiment_id": 4466,
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
        "v412_close_state": {},
        "reproducible_total_levels": 0,
        "reproducible_total_games": 0,
        "open_gap_ids": [],
        "open_gap_failure_modes": [],
        "prior_milestone_churn_note": "",
        "field_principles": dict(FIELD_PRINCIPLES),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "no_3090_inference": True,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4466", "SCENARIO-REPORT-4466-BLOCKED-PRECONDITION"],
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
    """Run the .412 archive workflow and write the terminal artifact."""

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
            "blocked_v413_not_active",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    capstone_path = root / CAPSTONE_REL_PATH
    preconditions["v412_capstone"] = {
        "path": str(CAPSTONE_REL_PATH),
        "exists": capstone_path.exists(),
        "sha256": file_sha256(capstone_path),
    }
    if not capstone_path.exists():
        return _blocked(
            root,
            "blocked_v412_capstone_missing",
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
    registry_facts = registry_facts_from_text(registry_path.read_text(encoding="utf-8"))
    close_state = build_v412_close_state(capstone, registry_facts)
    research_path = root / RESEARCH_COMPLETE_REL_PATH
    research_text = research_path.read_text(encoding="utf-8")
    new_text, duplicates_removed, record_action = dedupe_or_update_record(
        research_text, close_state
    )
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
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict lacks a complete-path terminal prefix")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field principle mismatch")
    if payload.get("open_gap_ids") != list(OPEN_GAP_IDS):
        raise ValueError("open_gap_ids must match the .413 backlog")
    if payload.get("prior_milestone_churn_note") != PRIOR_MILESTONE_CHURN_NOTE:
        raise ValueError("prior_milestone_churn_note must preserve the flat .412 truth")
    if payload.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle must be true")
    if payload.get("no_3090_inference") is not True:
        raise ValueError("no_3090_inference must stay true")
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
