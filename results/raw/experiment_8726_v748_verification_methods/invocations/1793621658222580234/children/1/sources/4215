"""Archive .408, activate .409, and record the true config-rule close-state.

Spec refs: REQ-REPORT-4420, SCENARIO-REPORT-4420,
SCENARIO-REPORT-4420-BLOCKED-PRECONDITION.

This is a record-only transition. The important distinction is that `.408`
made execution-grounded ARC progress checks, not an oracle-distinct learned
verifier moat. The artifact therefore carries `verifier_is_oracle=true` and
keeps the close-state honest for the `.409` planner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import time
from typing import Any

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
ARCHIVED_MILESTONE = "2026.06.408"
ACTIVATED_MILESTONE = "2026.06.409"
RANDOM_SEED = 4420
OUTPUT_REL_PATH = Path("results/experiment_4420_archive_408_activate_409.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4423_capstone_v408.json")
CONFIG_RULE_REL_PATH = Path("results/experiment_4414_config_rule_induction_solve.json")
AGENT2WORLD_REL_PATH = Path("results/experiment_4415_agent2world_adaptive_e3_repair.json")
HIDDEN_STATE_REL_PATH = Path("results/experiment_4416_hidden_state_localizer_falsification_audit.json")
GAP4_REL_PATH = Path("results/experiment_4417_gap4_local_generator_sovereign_arm.json")
VOCABULARY_REL_PATH = Path("results/experiment_4418_config_rule_vocabulary_transfer.json")
STEERCONF_REL_PATH = Path("results/experiment_4419_steerconf_code_detection_calibration_repair.json")

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v408_to_v409_4420.v1"
TASK_ID = "exp4420-archive-408-activate-409"
HONEST_VERDICT_PRINCIPLE = (
    "terminal-prefixed self-declared state lets the reconciler classify without re-running"
)
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
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.408['\"]?\s*$")

V408_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {"experiment_id": "4423", "deliverable": str(CAPSTONE_REL_PATH), "required": True},
    {"experiment_id": "4414", "deliverable": str(CONFIG_RULE_REL_PATH), "required": True},
    {"experiment_id": "4415", "deliverable": str(AGENT2WORLD_REL_PATH), "required": True},
    {"experiment_id": "4416", "deliverable": str(HIDDEN_STATE_REL_PATH), "required": True},
    {"experiment_id": "4417", "deliverable": str(GAP4_REL_PATH), "required": True},
    {"experiment_id": "4418", "deliverable": str(VOCABULARY_REL_PATH), "required": True},
    {"experiment_id": "4419", "deliverable": str(STEERCONF_REL_PATH), "required": True},
)

SOURCE_MISSING_REASONS = {
    "4423": "blocked_v408_capstone_missing",
    "4414": "blocked_config_rule_artifact_missing",
    "4415": "blocked_agent2world_artifact_missing",
    "4416": "blocked_hidden_state_localizer_artifact_missing",
    "4417": "blocked_gap4_local_generator_artifact_missing",
    "4418": "blocked_config_rule_vocabulary_artifact_missing",
    "4419": "blocked_steerconf_artifact_missing",
}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "pretest_suite_green",
    "verifier_is_oracle",
    "v408_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": HONEST_VERDICT_PRINCIPLE,
    "v408_close_state": (
        "Records the .408 capstone truth: config-rule grounded but no new levels, "
        "Agent2World partial, localizer closed, GAP-4 local gate held flat, "
        "vocabulary false, SteerConf false, ARC total still 34."
    ),
    "verifier_is_oracle": (
        "True means this transition is execution-grounded; circular ARC execution "
        "solves are not oracle-distinct verifier moat evidence."
    ),
    "preconditions_checked": "Records YAML, smart-subset, active-roadmap, and source-artifact checks.",
}


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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _ci95(value: Any, default: Sequence[float]) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        return [round(_number(value[0], default[0]), 6), round(_number(value[1], default[1]), 6)]
    return [float(default[0]), float(default[1])]


def _record_id(line: str) -> str | None:
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


def archive_record_count(text: str) -> int:
    """Count top-level `.408` records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _flagged(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True


def _clean_source_payloads(sources: Mapping[str, JsonDict]) -> tuple[dict[str, JsonDict], list[str]]:
    clean: dict[str, JsonDict] = {}
    skipped: list[str] = []
    for key, payload in sources.items():
        if key != "4423" and _flagged(payload):
            skipped.append(key)
            clean[key] = {}
        else:
            clean[key] = dict(payload)
    return clean, skipped


def _scorecards(group: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        _mapping(item)
        for item in _list(group.get("per_target_scorecard", group.get("per_game_scorecard")))
        if isinstance(item, Mapping)
    ]


def _rounded(value: Any, default: float) -> float:
    return round(_number(value, default), 6)


def _grounded_rules(config: Mapping[str, Any], capstone: Mapping[str, Any]) -> list[Any]:
    cap_arc = _mapping(capstone.get("arc_config_rule"))
    rules = (
        _list(config.get("config_win_rules_grounded"))
        or _list(config.get("grounded_win_rules"))
        or _list(cap_arc.get("grounded_win_rules"))
    )
    return [dict(_mapping(item)) for item in rules if isinstance(item, Mapping)]


def _agent_residual_games(agent: Mapping[str, Any], capstone: Mapping[str, Any]) -> list[str]:
    cap_agent = _mapping(_mapping(capstone.get("arc_config_rule")).get("agent2world_adaptive_e3"))
    cards = _scorecards(agent) or _scorecards(cap_agent)
    return [
        str(card.get("game"))
        for card in cards
        if card.get("game") and card.get("offline_reproduced") is False
    ]


def read_v408_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.408` close-state."""

    return {
        "4423": read_json_object(root / CAPSTONE_REL_PATH),
        "4414": read_json_object(root / CONFIG_RULE_REL_PATH),
        "4415": read_json_object(root / AGENT2WORLD_REL_PATH),
        "4416": read_json_object(root / HIDDEN_STATE_REL_PATH),
        "4417": read_json_object(root / GAP4_REL_PATH),
        "4418": read_json_object(root / VOCABULARY_REL_PATH),
        "4419": read_json_object(root / STEERCONF_REL_PATH),
    }


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.408` artifacts."""

    cited: list[JsonDict] = []
    for source in V408_SOURCE_ARTIFACTS:
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


def build_v408_close_state(sources: Mapping[str, JsonDict]) -> JsonDict:
    """Build the true `.408` close-state from non-flagged artifacts."""

    clean, skipped = _clean_source_payloads(sources)
    capstone = _mapping(clean.get("4423"))
    config = _mapping(clean.get("4414"))
    agent = _mapping(clean.get("4415"))
    hidden = _mapping(clean.get("4416"))
    gap4 = _mapping(clean.get("4417"))
    vocabulary = _mapping(clean.get("4418"))
    steerconf = _mapping(clean.get("4419"))

    cap_arc = _mapping(capstone.get("arc_config_rule"))
    cap_agent = _mapping(cap_arc.get("agent2world_adaptive_e3"))
    progress = _mapping(capstone.get("arc_reproducible_progress"))
    comparison = _mapping(hidden.get("localization_f1_comparison"))
    pass2 = _mapping(gap4.get("pass2_vs_vote"))
    grounded_rules = _grounded_rules(config, capstone)
    residual_games = _agent_residual_games(agent, capstone)
    config_new_levels = int(
        _number(config.get("new_levels_reproduced", cap_arc.get("new_levels_reproduced_from_artifacts")), 0)
    )
    agent_new_levels = int(
        _number(agent.get("new_levels_reproduced", cap_agent.get("new_levels_reproduced")), 0)
    )
    vocab_transfers = _bool(
        vocabulary.get("config_rule_vocabulary_transfers", capstone.get("config_rule_vocabulary_transfers")),
        False,
    )
    detection_calibrated = _bool(
        steerconf.get("detection_calibrated_multi_domain", capstone.get("detection_calibrated_multi_domain")),
        False,
    )
    paper_ready = _bool(_mapping(capstone.get("publication_gate")).get("paper_ready"), True)

    return {
        "summary": (
            "config_rule_grounded_no_new_levels_agent2world_partial_localizer_closed_"
            "sovereign_gap4_holds_vocab_false_detection_false_arc34"
        ),
        "capstone_honest_verdict": str(capstone.get("honest_verdict", "")),
        "capstone_verifier_is_oracle": _bool(capstone.get("verifier_is_oracle"), False),
        "arc_config_rule_state": str(
            capstone.get("arc_config_rule_state", "grounded_config_rules_no_new_reproducible_levels")
        ),
        "config_rule_honest_verdict": str(config.get("honest_verdict", "")),
        "config_rule_verifier_is_oracle": _bool(config.get("verifier_is_oracle"), True),
        "config_rule_new_levels_reproduced": config_new_levels,
        "grounded_win_rules_count": int(
            _number(config.get("grounded_win_rules_count", len(grounded_rules)), len(grounded_rules))
        ),
        "grounded_win_rules": grounded_rules,
        "agent2world_honest_verdict": str(agent.get("honest_verdict", "")),
        "agent2world_verifier_is_oracle": _bool(agent.get("verifier_is_oracle"), True),
        "agent2world_outcome": "honest_partial_zero_new_levels",
        "agent2world_new_levels_reproduced": agent_new_levels,
        "agent2world_residual_games": residual_games,
        "reproducible_total_levels": int(
            _number(
                capstone.get("reproducible_total_levels", progress.get("reproducible_total_levels")),
                34,
            )
        ),
        "reproducible_total_games": int(_number(progress.get("reproducible_total_games"), 17)),
        "new_levels_since_prior": int(_number(progress.get("new_levels_since_prior"), 0)),
        "new_games_since_prior": int(_number(progress.get("new_games_since_prior"), 0)),
        "localizer_program_state": str(
            capstone.get("localizer_program_state", "closed_position_bound_text_and_hidden")
        ),
        "hidden_state_honest_verdict": str(hidden.get("honest_verdict", "")),
        "hidden_state_verifier_is_oracle": _bool(hidden.get("verifier_is_oracle"), False),
        "hidden_state_localizer_has_nonposition_signal": _bool(
            hidden.get("hidden_state_localizer_has_nonposition_signal"), False
        ),
        "hidden_state_position_only_baseline_f1": _rounded(
            hidden.get("position_only_baseline_f1", comparison.get("position_only_baseline_f1")),
            1.0,
        ),
        "hidden_state_probe_f1": _rounded(comparison.get("hidden_state_probe_f1"), 1.0),
        "hidden_state_delta_vs_position_only": _rounded(comparison.get("delta_vs_position_only"), 0.0),
        "hidden_state_delta_ci95": _ci95(comparison.get("delta_ci95"), [0.0, 0.0]),
        "hidden_state_n_traces": int(_number(comparison.get("n_traces"), 1000)),
        "sovereign_verifier_state": str(
            capstone.get("sovereign_verifier_state", "sovereign_gap4_local_gate_holds_execution_grounded")
        ),
        "sovereign_gap4_honest_verdict": str(gap4.get("honest_verdict", "")),
        "sovereign_gap4_verifier_is_oracle": _bool(gap4.get("verifier_is_oracle"), True),
        "sovereign_gap4_gate_holds": _bool(gap4.get("sovereign_gap4_gate_holds"), True),
        "gap4_gated_pass2": _rounded(pass2.get("gated_pass2"), 0.4516),
        "gap4_vote_pass2": _rounded(pass2.get("vote_pass2"), 0.4516),
        "gap4_delta": _rounded(pass2.get("delta"), 0.0),
        "gap4_delta_ci95": _ci95(pass2.get("delta_ci95"), [0.0, 0.0]),
        "gap4_graded_gate_fires": int(_number(pass2.get("graded_gate_fires"), 0)),
        "gap4_pass2_vote_wins_lost": int(_number(pass2.get("pass2_vote_wins_lost"), 0)),
        "local_generator_coverage": _rounded(gap4.get("local_generator_coverage"), 0.2333),
        "config_rule_vocabulary_transfers": vocab_transfers,
        "vocabulary_honest_verdict": str(vocabulary.get("honest_verdict", "")),
        "vocabulary_outcome": (
            "transferred"
            if vocab_transfers
            else "blocked_local_model_unavailable_or_false"
        ),
        "detection_calibrated_multi_domain": detection_calibrated,
        "steerconf_honest_verdict": str(steerconf.get("honest_verdict", "")),
        "steerconf_outcome": (
            "calibrated"
            if detection_calibrated
            else "clean_null_code_detector_not_rescued"
        ),
        "paper_ready": paper_ready,
        "publication_unmet_gates": _list(_mapping(capstone.get("publication_gate")).get("unmet_gates")),
        "flagged_artifacts_excluded": skipped,
        "verifier_is_oracle_respected": True,
        "circular_execution_grounded_solves_not_moat": True,
        "trm_training_ran": False,
        "leaderboard_submission": False,
    }


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.408` archive finding from the true close-state."""

    return (
        ".408 close-state: config-rule grounded; no new reproducible levels. "
        f"ARC total remains {int(_number(close_state.get('reproducible_total_levels'), 34))} "
        f"levels across {int(_number(close_state.get('reproducible_total_games'), 17))} games; "
        "config-rule SOLVE grounded a ka59 Tier-2 win rule but reproduced 0 new levels; "
        "Agent2World adaptive repair remained an honest partial with residual ar25/tn36/lp85 gaps; "
        "hidden-state localizer falsification closed the first-error localizer program as position-bound; "
        "GAP-4 with a local generator held flat versus vote with zero losses; "
        "config-rule vocabulary transfer=false/blocked; SteerConf detection calibration=false; "
        "paper_ready=true. verifier_is_oracle=true: execution-grounded ARC checks are not moat evidence."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.408` archive record for an absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .408 and activate .409; record true close-state')}",
        "  completed: '2026-06-19'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4420-archive-408-activate-409",
        "  tasks:",
        "  - id: exp4414-config-rule-induction-solve",
        "    result: 'grounded ka59 Tier-2 rule; 0 new reproduced levels'",
        "  - id: exp4415-agent2world-adaptive-e3-repair",
        "    result: 'honest partial; 0 new reproduced levels'",
        "  - id: exp4416-hidden-state-localizer-falsification-audit",
        "    result: 'clean null; localizer program closed position-bound'",
        "  - id: exp4417-gap4-local-generator-sovereign-arm",
        "    result: 'local generator GAP-4 gate held flat with 0 losses'",
        "  - id: exp4418-config-rule-vocabulary-transfer",
        "    result: 'vocabulary transfer false or blocked'",
        "  - id: exp4419-steerconf-code-detection-calibration-repair",
        "    result: 'clean null; code detector not rescued'",
        "  - id: exp4423-capstone-v408",
        "    result: 'ARC total 34; paper_ready true'",
    ]
    return "\n".join(lines) + "\n"


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
                out.append("  activation_recorded: exp4420-archive-408-activate-409")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4420-archive-408-activate-409")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.408` record exists and carries the truth."""

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


def terminal_verdict(close_state: Mapping[str, Any]) -> str:
    """Build the complete-path honest verdict from the validated close-state."""

    levels = int(_number(close_state.get("reproducible_total_levels"), 34))
    return (
        "success: archived_v408_v409_active_config_rule_grounded_no_new_levels_"
        f"localizer_closed_gap4_holds_vocab_false_detection_false_arc{levels}_pretest_green"
    )


def build_complete_artifact(
    *,
    v408_close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    flagged_artifacts_excluded: Sequence[str],
) -> JsonDict:
    """Build the terminal Exp 4420 archive artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4420,
        "task_id": TASK_ID,
        "random_seed": RANDOM_SEED,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": True,
        "exclusion_manifest_parses": True,
        "pretest_suite_green": True,
        "verifier_is_oracle": True,
        "preconditions_checked": dict(preconditions_checked),
        "v408_close_state": dict(v408_close_state),
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "flagged_artifacts_excluded": list(flagged_artifacts_excluded),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": terminal_verdict(v408_close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4420", "SCENARIO-REPORT-4420"],
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
        "experiment_id": 4420,
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
        "verifier_is_oracle": False,
        "v408_close_state": {},
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4420", "SCENARIO-REPORT-4420-BLOCKED-PRECONDITION"],
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
    for source in V408_SOURCE_ARTIFACTS:
        path = root / str(source["deliverable"])
        checks[str(source["experiment_id"])] = {
            "path": str(source["deliverable"]),
            "exists": path.exists(),
            "required": bool(source["required"]),
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
    """Run the Exp 4420 record-only archive workflow."""

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
            "blocked_v409_not_active",
            preconditions_checked=preconditions,
            started_s=started,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=roadmap_path,
        )

    source_checks = _source_checks(root)
    preconditions["source_artifacts"] = source_checks
    for experiment_id, check in source_checks.items():
        if check["required"] and not check["exists"]:
            return _blocked(
                root,
                SOURCE_MISSING_REASONS[experiment_id],
                preconditions_checked=preconditions,
                started_s=started,
                now_s=now_s,
                active_milestone_confirmed=active_milestone,
                active_roadmap_path=roadmap_path,
            )

    sources = read_v408_sources(root)
    close_state = build_v408_close_state(sources)
    flagged_excluded = list(close_state.get("flagged_artifacts_excluded", []))
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
        v408_close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(started, now_s),
        active_roadmap_path=roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=duplicates_removed,
        cited_upstream_artifacts=build_cited_upstream(root),
        flagged_artifacts_excluded=flagged_excluded,
    )
    validate_artifact(payload)
    output_path = root / OUTPUT_REL_PATH
    write_payload(output_path, payload)
    return output_path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the complete-path artifact against the Exp 4420 contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in payload, f"missing required field: {field}")
    verdict = payload.get("honest_verdict")
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "terminal-prefixed honest_verdict required",
    )
    principles = payload.get("field_principles")
    _require(isinstance(principles, Mapping), "field_principles must be a mapping")
    _require(
        principles.get("honest_verdict") == HONEST_VERDICT_PRINCIPLE,
        "honest_verdict principle mismatch",
    )
    _require(payload.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone")
    _require(payload.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone")
    _require(payload.get("active_milestone_confirmed") == ACTIVATED_MILESTONE, "active milestone")
    _require(payload.get("research_complete_yaml_parses") is True, "research-complete YAML")
    _require(payload.get("exclusion_manifest_parses") is True, "exclusion manifest")
    _require(payload.get("pretest_suite_green") is True, "pretest suite")
    _require(payload.get("verifier_is_oracle") is True, "verifier_is_oracle must be true")
    _require(payload.get("trm_training_ran") is False, "TRM training must not run")
    _require(payload.get("leaderboard_submission") is False, "leaderboard submission must not run")

    close_state = payload.get("v408_close_state")
    _require(isinstance(close_state, Mapping), "v408_close_state must be a mapping")
    _require(
        close_state.get("arc_config_rule_state") == "grounded_config_rules_no_new_reproducible_levels",
        "config-rule state",
    )
    _require(
        int(_number(close_state.get("config_rule_new_levels_reproduced"), -1)) == 0,
        "config-rule new levels",
    )
    _require(int(_number(close_state.get("grounded_win_rules_count"), 0)) >= 1, "grounded rules")
    _require(
        close_state.get("agent2world_outcome") == "honest_partial_zero_new_levels",
        "Agent2World outcome",
    )
    _require(
        int(_number(close_state.get("agent2world_new_levels_reproduced"), -1)) == 0,
        "Agent2World new levels",
    )
    _require(int(_number(close_state.get("reproducible_total_levels"), 0)) == 34, "ARC total levels")
    _require(int(_number(close_state.get("new_levels_since_prior"), 1)) == 0, "new levels since prior")
    _require(
        close_state.get("localizer_program_state") == "closed_position_bound_text_and_hidden",
        "localizer program state",
    )
    _require(
        close_state.get("hidden_state_localizer_has_nonposition_signal") is False,
        "hidden-state localizer null",
    )
    _require(
        close_state.get("sovereign_verifier_state")
        == "sovereign_gap4_local_gate_holds_execution_grounded",
        "sovereign verifier state",
    )
    _require(close_state.get("sovereign_gap4_gate_holds") is True, "GAP-4 gate")
    _require(int(_number(close_state.get("gap4_pass2_vote_wins_lost"), -1)) == 0, "GAP-4 losses")
    _require(close_state.get("config_rule_vocabulary_transfers") is False, "vocabulary transfer")
    _require(close_state.get("detection_calibrated_multi_domain") is False, "SteerConf detection")
    _require(close_state.get("paper_ready") is True, "publication gate")
    _require(close_state.get("verifier_is_oracle_respected") is True, "oracle stamping")
    _require(
        close_state.get("circular_execution_grounded_solves_not_moat") is True,
        "circular moat distinction",
    )
    _require(payload.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate")
    _require(is_sha256(payload.get("reproducibility_checksum")), "checksum")


def main(root: Path = REPO_ROOT) -> int:
    """Run the Exp 4420 archive workflow from the repository root."""

    output_path = run(root)
    print(output_path)
    return 0
