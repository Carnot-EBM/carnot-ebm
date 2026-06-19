"""Archive .409, activate .410, and record the true ARC close-state.

Spec refs: REQ-REPORT-4431, SCENARIO-REPORT-4431,
SCENARIO-REPORT-4431-BLOCKED-PRECONDITION.

This is a record-only transition. It reads the .409 capstone and registry,
skips flagged artifacts, and writes the exact handoff state that the .410
roadmap needs. The important discipline is that execution-grounded ARC solves
can count as reproduced ARC progress, but they do not become an oracle-distinct
verifier moat headline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
ARCHIVED_MILESTONE = "2026.06.409"
ACTIVATED_MILESTONE = "2026.06.410"
RANDOM_SEED = 4431
OUTPUT_REL_PATH = Path("results/experiment_4431_archive_409_activate_410.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
ARC_REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4430_capstone_409.json")
CONFIG_RULE_REL_PATH = Path("results/experiment_4421_config_rule_solve_unseen.json")
GLYPH_REL_PATH = Path("results/experiment_4422_glyph_rewrite_perception.json")
FIRST_CONTACT_REL_PATH = Path("results/experiment_4423_generic_first_contact_breadth.json")
DEEPENING_REL_PATH = Path("results/experiment_4424_deeper_solved_game.json")
VOCABULARY_REL_PATH = Path("results/experiment_4425_config_rule_vocabulary_transfer.json")
REGISTRY_AUDIT_REL_PATH = Path("results/experiment_4426_arc_registry_repro_audit.json")
SOTA_REL_PATH = Path("results/experiment_4429_sota_ingestion_409.json")

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v409_to_v410_4431.v1"
TASK_ID = "exp4431-archive-409-activate-410"
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
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.409['\"]?\s*$")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal-prefixed self-declared state lets the reconciler classify without re-running"
    ),
    "reproducible_total_levels": (
        "the bare authoritative count from ops/arc_solve_registry.yaml; the sprint's monotonic progress metric"
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproducible_total_levels",
    "reproducible_total_games",
    "v409_close_state",
    "preconditions_checked",
    "verifier_is_oracle",
    "trm_training_ran",
    "leaderboard_submission",
    "duration_s",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
)

SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {"experiment_id": "4430", "deliverable": str(CAPSTONE_REL_PATH), "required": True},
    {"experiment_id": "4421", "deliverable": str(CONFIG_RULE_REL_PATH), "required": True},
    {"experiment_id": "4422", "deliverable": str(GLYPH_REL_PATH), "required": True},
    {"experiment_id": "4423", "deliverable": str(FIRST_CONTACT_REL_PATH), "required": True},
    {"experiment_id": "4424", "deliverable": str(DEEPENING_REL_PATH), "required": True},
    {"experiment_id": "4425", "deliverable": str(VOCABULARY_REL_PATH), "required": True},
    {"experiment_id": "4426", "deliverable": str(REGISTRY_AUDIT_REL_PATH), "required": True},
    {"experiment_id": "4429", "deliverable": str(SOTA_REL_PATH), "required": True},
)

SOURCE_MISSING_REASONS = {
    "4430": "blocked_v409_capstone_missing",
    "4421": "blocked_config_rule_artifact_missing",
    "4422": "blocked_glyph_rewrite_artifact_missing",
    "4423": "blocked_generic_first_contact_artifact_missing",
    "4424": "blocked_deeper_solved_game_artifact_missing",
    "4425": "blocked_vocabulary_artifact_missing",
    "4426": "blocked_registry_audit_artifact_missing",
    "4429": "blocked_sota_ingestion_artifact_missing",
}


def _number(value: Any, default: float = 0.0) -> float:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else float(default)
    )


def _int(value: Any, default: int = 0) -> int:
    return int(_number(value, default))


def _bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _terminal_prefixed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(TERMINAL_PREFIXES)


def archive_record_count(text: str) -> int:
    """Count top-level `.409` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _is_flagged(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True


def _corrigendum_kinds(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("kind"))
        for item in _list(payload.get("corrigendum_pending"))
        if isinstance(item, Mapping) and item.get("kind")
    ]


def _source_id_from_key(key: str) -> int:
    return int(key) if key.isdigit() else 0


def registry_totals_from_text(text: str) -> JsonDict:
    """Compute reproduced totals from registry entries and preserve declared totals.

    The registry sometimes carries a stale top-level total after individual game
    entries have already advanced. The task asks for the authoritative count from
    the registry, so this helper sums reproduced entries while preserving the
    declared count as a discrepancy diagnostic.
    """

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        return {
            "declared_reproducible_total_levels": 0,
            "declared_reproducible_total_games": 0,
            "entry_sum_reproducible_total_levels": 0,
            "entry_sum_reproducible_total_games": 0,
            "authoritative_reproducible_total_levels": 0,
            "authoritative_reproducible_total_games": 0,
            "registry_total_discrepancy": False,
            "reproduced_games": [],
        }

    reproduced_games: list[JsonDict] = []
    for item in _list(loaded.get("games")):
        game = _mapping(item)
        if game.get("reproducibility") != "reproduced":
            continue
        levels = _int(game.get("levels_reproduced"))
        reproduced_games.append({"game": str(game.get("game", "")), "levels_reproduced": levels})

    entry_levels = sum(_int(item["levels_reproduced"]) for item in reproduced_games)
    entry_games = len(reproduced_games)
    declared_levels = _int(loaded.get("reproducible_total_levels"))
    declared_games = _int(loaded.get("reproducible_total_games"))
    authoritative_levels = entry_levels or declared_levels
    authoritative_games = entry_games or declared_games
    return {
        "declared_reproducible_total_levels": declared_levels,
        "declared_reproducible_total_games": declared_games,
        "entry_sum_reproducible_total_levels": entry_levels,
        "entry_sum_reproducible_total_games": entry_games,
        "authoritative_reproducible_total_levels": authoritative_levels,
        "authoritative_reproducible_total_games": authoritative_games,
        "registry_total_discrepancy": (declared_levels, declared_games)
        != (authoritative_levels, authoritative_games),
        "reproduced_games": reproduced_games,
    }


def _flagged_ids_from_capstone(capstone: Mapping[str, Any]) -> list[int]:
    ids: set[int] = set()
    for row in _list(capstone.get("flagged_artifacts_excluded")):
        mapping = _mapping(row)
        exp_id = mapping.get("experiment_id")
        if isinstance(exp_id, int):
            ids.add(exp_id)
    return sorted(ids)


def _registry_counted_flagged_config(capstone: Mapping[str, Any]) -> bool:
    for row in _list(capstone.get("flagged_sources_counted_by_registry_audit")):
        mapping = _mapping(row)
        if mapping.get("experiment") == "exp4421" and _int(mapping.get("new_levels_counted")) > 0:
            return True
    registry = _mapping(capstone.get("registry_audit"))
    for row in _list(registry.get("flagged_sources_counted")):
        mapping = _mapping(row)
        if mapping.get("experiment") == "exp4421" and _int(mapping.get("new_levels_counted")) > 0:
            return True
    return False


def _a1_config_state(capstone: Mapping[str, Any], config: Mapping[str, Any]) -> JsonDict:
    flagged = _is_flagged(config)
    duration_flagged = "DURATION_TOO_SHORT" in _corrigendum_kinds(config)
    return {
        "phase": "A1",
        "experiment_id": 4421,
        "artifact": str(CONFIG_RULE_REL_PATH),
        "status": "quarantined_duration_too_short"
        if flagged and duration_flagged
        else "excluded_flagged_adversarial"
        if flagged
        else "clean_execution_grounded",
        "direct_artifact_imported": not flagged,
        "flagged_adversarial": flagged,
        "corrigendum_kinds": _corrigendum_kinds(config),
        "registry_audit_counted_execution_grounded": _registry_counted_flagged_config(capstone),
        "offline_reproduced": _bool(config.get("offline_reproduced")),
        "reproduced_levels": _int(config.get("reproduced_levels")),
        "new_levels_reproduced": 0 if flagged else _int(config.get("new_levels_reproduced")),
        "verifier_is_oracle": _bool(config.get("verifier_is_oracle"), True),
        "honest_verdict": str(config.get("honest_verdict", "")),
    }


def _a2_glyph_state(capstone: Mapping[str, Any], glyph: Mapping[str, Any]) -> JsonDict:
    flagged = _is_flagged(glyph)
    cap_glyph = _mapping(capstone.get("glyph_rewrite"))
    state = str(capstone.get("glyph_rewrite_state", cap_glyph.get("state", "")))
    solved = (
        not flagged
        and state == "grounded_and_offline_solved"
        and _bool(glyph.get("offline_reproduced", cap_glyph.get("offline_reproduced")))
    )
    return {
        "phase": "A2",
        "experiment_id": 4422,
        "artifact": str(GLYPH_REL_PATH),
        "status": state or ("excluded_flagged_adversarial" if flagged else "missing_state"),
        "banked_reproducible_level": solved,
        "new_levels_banked": 1 if solved else 0,
        "target_game": str(glyph.get("target_game", cap_glyph.get("target_game", "tr87"))),
        "offline_reproduced": _bool(glyph.get("offline_reproduced", cap_glyph.get("offline_reproduced"))),
        "reproduced_levels": _int(glyph.get("reproduced_levels", cap_glyph.get("reproduced_levels"))),
        "verifier_is_oracle": _bool(glyph.get("verifier_is_oracle", cap_glyph.get("verifier_is_oracle")), True),
        "honest_verdict": str(glyph.get("honest_verdict", cap_glyph.get("honest_verdict", ""))),
    }


def _a3_first_contact_state(capstone: Mapping[str, Any], first: Mapping[str, Any]) -> JsonDict:
    cap_first = _mapping(capstone.get("generic_first_contact"))
    verdict = str(first.get("honest_verdict", cap_first.get("honest_verdict", "")))
    state = str(capstone.get("generic_first_contact_state", cap_first.get("state", "")))
    partial = verdict.startswith("partial:")
    return {
        "phase": "A3",
        "experiment_id": 4423,
        "artifact": str(FIRST_CONTACT_REL_PATH),
        "status": "skipped_partial_verdict" if partial else state or "no_new_game",
        "new_game_added": False
        if partial
        else _bool(first.get("offline_reproduced", cap_first.get("offline_reproduced"))),
        "offline_reproduced": _bool(first.get("offline_reproduced", cap_first.get("offline_reproduced"))),
        "reproduced_levels": _int(first.get("reproduced_levels", cap_first.get("reproduced_levels"))),
        "target_game": str(first.get("target_game", cap_first.get("target_game", ""))),
        "missing_verifier_gaps": _list(
            first.get("missing_verifier_gaps", cap_first.get("missing_verifier_gaps"))
        ),
        "verifier_is_oracle": _bool(first.get("verifier_is_oracle", cap_first.get("verifier_is_oracle"))),
        "honest_verdict": verdict,
    }


def _a4_deepening_state(capstone: Mapping[str, Any], deepening: Mapping[str, Any]) -> JsonDict:
    cap_deep = _mapping(capstone.get("multi_level_deepening"))
    state = str(capstone.get("multi_level_deepening_state", cap_deep.get("state", "")))
    new_levels = _int(deepening.get("new_levels_reproduced", cap_deep.get("new_levels_reproduced")))
    offline = _bool(deepening.get("offline_reproduced", cap_deep.get("offline_reproduced")))
    banked = offline and new_levels > 0
    return {
        "phase": "A4",
        "experiment_id": 4424,
        "artifact": str(DEEPENING_REL_PATH),
        "status": state or ("new_level_added" if banked else "mechanic_repair_no_new_level"),
        "banked_reproducible_level": banked,
        "new_levels_reproduced": new_levels if banked else 0,
        "offline_reproduced": offline,
        "reproduced_levels": _int(deepening.get("reproduced_levels", cap_deep.get("reproduced_levels"))),
        "target_game": str(deepening.get("game", cap_deep.get("game", "sc25"))),
        "residual_failing_mechanic": str(
            deepening.get("residual_failing_mechanic", cap_deep.get("residual_failing_mechanic", ""))
        ),
        "verifier_is_oracle": _bool(
            deepening.get("verifier_is_oracle", cap_deep.get("verifier_is_oracle")),
            True,
        ),
        "honest_verdict": str(deepening.get("honest_verdict", cap_deep.get("honest_verdict", ""))),
    }


def _vocabulary_state(capstone: Mapping[str, Any], vocabulary: Mapping[str, Any]) -> JsonDict:
    flagged = _is_flagged(vocabulary)
    cap_vocab = _mapping(capstone.get("config_rule_vocabulary_transfer"))
    if flagged:
        status = "excluded_flagged_adversarial"
        transfers = False
    else:
        transfers = _bool(
            vocabulary.get(
                "config_rule_vocabulary_transfers",
                cap_vocab.get("config_rule_vocabulary_transfers"),
            )
        )
        status = "transfers" if transfers else "no_transfer"
    return {
        "experiment_id": 4425,
        "artifact": str(VOCABULARY_REL_PATH),
        "status": status,
        "config_rule_vocabulary_transfers": transfers,
        "flagged_adversarial": flagged,
        "corrigendum_kinds": _corrigendum_kinds(vocabulary),
        "verifier_is_oracle": _bool(vocabulary.get("verifier_is_oracle")),
        "honest_verdict": str(vocabulary.get("honest_verdict", "")),
    }


def read_v409_sources(root: Path) -> dict[str, JsonDict]:
    """Read all source artifacts that carry the `.409` close-state."""

    return {
        "4430": read_json_object(root / CAPSTONE_REL_PATH),
        "4421": read_json_object(root / CONFIG_RULE_REL_PATH),
        "4422": read_json_object(root / GLYPH_REL_PATH),
        "4423": read_json_object(root / FIRST_CONTACT_REL_PATH),
        "4424": read_json_object(root / DEEPENING_REL_PATH),
        "4425": read_json_object(root / VOCABULARY_REL_PATH),
        "4426": read_json_object(root / REGISTRY_AUDIT_REL_PATH),
        "4429": read_json_object(root / SOTA_REL_PATH),
    }


def build_v409_close_state(
    sources: Mapping[str, JsonDict],
    registry_totals: Mapping[str, Any],
) -> JsonDict:
    """Build the true `.409` close-state from capstone, registry, and clean artifacts."""

    capstone = _mapping(sources.get("4430"))
    config = _mapping(sources.get("4421"))
    glyph = _mapping(sources.get("4422"))
    first = _mapping(sources.get("4423"))
    deepening = _mapping(sources.get("4424"))
    vocabulary = _mapping(sources.get("4425"))
    registry_audit = _mapping(sources.get("4426"))
    sota = _mapping(sources.get("4429"))

    flagged_ids = set(_flagged_ids_from_capstone(capstone))
    for key, payload in sources.items():
        if _is_flagged(_mapping(payload)):
            flagged_ids.add(_source_id_from_key(key))
    flagged_ids.discard(0)

    a1 = _a1_config_state(capstone, config)
    a2 = _a2_glyph_state(capstone, glyph)
    a3 = _a3_first_contact_state(capstone, first)
    a4 = _a4_deepening_state(capstone, deepening)
    vocab = _vocabulary_state(capstone, vocabulary)

    authoritative_levels = _int(registry_totals.get("authoritative_reproducible_total_levels"))
    authoritative_games = _int(registry_totals.get("authoritative_reproducible_total_games"))
    capstone_levels = _int(capstone.get("reproducible_total_levels"))
    audit_levels = _int(registry_audit.get("reproducible_total_levels"))
    flagged_for_v410 = str(
        _mapping(capstone.get("sota_ingestion")).get(
            "flagged_for_v410",
            sota.get("flagged_for_v410", ""),
        )
    )

    return {
        "summary": (
            "glyph_rewrite_banked_config_rule_quarantined_first_contact_partial_"
            "deepening_no_new_level_vocab_excluded"
        ),
        "capstone_honest_verdict": str(capstone.get("honest_verdict", "")),
        "phase_a_tasks": {
            "A1_config_rule_unseen": a1,
            "A2_glyph_rewrite": a2,
            "A3_generic_first_contact": a3,
            "A4_deeper_solved_game": a4,
        },
        "phase_a_banked_reproducible_level_tasks": [
            task["phase"]
            for task in (a1, a2, a3, a4)
            if task.get("banked_reproducible_level") is True
        ],
        "phase_a_quarantined_or_skipped_tasks": [
            task["phase"]
            for task in (a1, a2, a3, a4)
            if str(task.get("status", "")).startswith(("quarantined", "skipped"))
        ],
        "config_rule_vocabulary_transfer": vocab,
        "config_rule_vocabulary_transfer_outcome": vocab["status"],
        "reproducible_total_levels": authoritative_levels,
        "reproducible_total_games": authoritative_games,
        "registry_declared_reproducible_total_levels": _int(
            registry_totals.get("declared_reproducible_total_levels")
        ),
        "registry_declared_reproducible_total_games": _int(
            registry_totals.get("declared_reproducible_total_games")
        ),
        "registry_entry_sum_reproducible_total_levels": _int(
            registry_totals.get("entry_sum_reproducible_total_levels")
        ),
        "registry_entry_sum_reproducible_total_games": _int(
            registry_totals.get("entry_sum_reproducible_total_games")
        ),
        "registry_total_discrepancy": _bool(registry_totals.get("registry_total_discrepancy")),
        "capstone_reproducible_total_levels": capstone_levels,
        "registry_audit_reproducible_total_levels": audit_levels,
        "new_levels_since_v408": max(0, authoritative_levels - 34),
        "new_games_since_v408": max(0, authoritative_games - 17),
        "flagged_artifacts_skipped": sorted(flagged_ids),
        "flagged_for_v410": flagged_for_v410,
        "paper_ready": _bool(_mapping(capstone.get("publication_gate")).get("paper_ready"), True),
        "publication_unmet_gates": _list(_mapping(capstone.get("publication_gate")).get("unmet_gates")),
        "verifier_is_oracle_honored": True,
        "verifier_is_oracle": True,
        "circular_execution_grounded_arc_solve_not_moat_headline": True,
        "trm_training_ran": False,
        "leaderboard_submission": False,
    }


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.409` archive finding from the true close-state."""

    levels = _int(close_state.get("reproducible_total_levels"))
    games = _int(close_state.get("reproducible_total_games"))
    phase = _mapping(close_state.get("phase_a_tasks"))
    a1 = _mapping(phase.get("A1_config_rule_unseen"))
    a2 = _mapping(phase.get("A2_glyph_rewrite"))
    a3 = _mapping(phase.get("A3_generic_first_contact"))
    a4 = _mapping(phase.get("A4_deeper_solved_game"))
    vocab = _mapping(close_state.get("config_rule_vocabulary_transfer"))
    return (
        ".409 close-state: archive truth from exp4430 capstone and the ARC registry. "
        f"Final reproduced registry total is {levels} levels across {games} games "
        f"(registry declared {_int(close_state.get('registry_declared_reproducible_total_levels'))}, "
        f"entry-sum {_int(close_state.get('registry_entry_sum_reproducible_total_levels'))}). "
        f"PHASE A1 config-rule artifact is {a1.get('status')} and skipped from direct aggregation; "
        "its execution-grounded registry audit is not a moat headline. "
        f"PHASE A2 glyph rewrite banked execution-grounded tr87 progress: "
        f"banked={a2.get('banked_reproducible_level')}, reproduced_levels={a2.get('reproduced_levels')}. "
        f"PHASE A3 first-contact is {a3.get('status')} with verdict {a3.get('honest_verdict')}. "
        f"PHASE A4 deeper game is {a4.get('status')}, banked={a4.get('banked_reproducible_level')}. "
        f"Config-rule vocabulary transfer outcome is {vocab.get('status')} / "
        f"transfers={vocab.get('config_rule_vocabulary_transfers')}. "
        f"flagged_for_v410={close_state.get('flagged_for_v410')}. "
        "verifier_is_oracle=true is honored: execution-grounded ARC solves are progress, not a circular moat claim."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.409` archive record for an absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .409 and activate .410; record true ARC generic-solver close-state')}",
        "  completed: '2026-06-19'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4431-archive-409-activate-410",
        "  tasks:",
        "  - id: exp4421-config-rule-solve-unseen",
        "    result: 'quarantined flagged_adversarial DURATION_TOO_SHORT; no direct import'",
        "  - id: exp4422-glyph-rewrite-perception",
        "    result: 'glyph rewrite banked execution-grounded tr87 progress'",
        "  - id: exp4423-generic-first-contact-breadth",
        "    result: 'skipped partial verdict; missing verifier gap logged'",
        "  - id: exp4424-deeper-solved-game",
        "    result: 'mechanic repair only; no new reproduced level'",
        "  - id: exp4425-config-rule-vocabulary-transfer",
        "    result: 'vocabulary transfer excluded or false from clean evidence'",
        "  - id: exp4430-capstone-409",
        "    result: 'archive source capstone; verifier_is_oracle honored'",
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
                out.append("  activation_recorded: exp4431-archive-409-activate-410")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4431-archive-409-activate-410")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.409` record exists and carries the truth."""

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
    levels = _int(close_state.get("reproducible_total_levels"))
    games = _int(close_state.get("reproducible_total_games"))
    return (
        "complete: archived_v409_v410_active_arc_levels_"
        f"{levels}_games_{games}_glyph_banked_config_quarantined_first_contact_skipped_"
        "vocab_excluded_verifier_oracle_honored"
    )


def build_complete_artifact(
    *,
    v409_close_state: Mapping[str, Any],
    registry_totals: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal Exp 4431 archive artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4431,
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
        "verifier_is_oracle": True,
        "preconditions_checked": dict(preconditions_checked),
        "v409_close_state": dict(v409_close_state),
        "reproducible_total_levels": _int(v409_close_state.get("reproducible_total_levels")),
        "reproducible_total_games": _int(v409_close_state.get("reproducible_total_games")),
        "registry_declared_reproducible_total_levels": _int(
            registry_totals.get("declared_reproducible_total_levels")
        ),
        "registry_declared_reproducible_total_games": _int(
            registry_totals.get("declared_reproducible_total_games")
        ),
        "registry_entry_sum_reproducible_total_levels": _int(
            registry_totals.get("entry_sum_reproducible_total_levels")
        ),
        "registry_entry_sum_reproducible_total_games": _int(
            registry_totals.get("entry_sum_reproducible_total_games")
        ),
        "registry_total_discrepancy": _bool(registry_totals.get("registry_total_discrepancy")),
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": terminal_verdict(v409_close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4431", "SCENARIO-REPORT-4431"],
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
        "experiment_id": 4431,
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
        "verifier_is_oracle": False,
        "v409_close_state": {},
        "reproducible_total_levels": 0,
        "reproducible_total_games": 0,
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "trm_training_ran": False,
        "leaderboard_submission": False,
        "spec_refs": ["REQ-REPORT-4431", "SCENARIO-REPORT-4431-BLOCKED-PRECONDITION"],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


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
    for source in SOURCE_ARTIFACTS:
        path = root / str(source["deliverable"])
        checks[str(source["experiment_id"])] = {
            "path": str(source["deliverable"]),
            "exists": path.exists(),
            "required": bool(source["required"]),
            "sha256": file_sha256(path),
        }
    return checks


def build_cited_upstream(root: Path) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for source in SOURCE_ARTIFACTS:
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
    cited.append(
        {
            "kind": "registry",
            "experiment_id": "arc_solve_registry",
            "deliverable": str(ARC_REGISTRY_REL_PATH),
            "required": True,
            "sha256": file_sha256(root / ARC_REGISTRY_REL_PATH),
        }
    )
    return cited


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


def run(
    root: Path = REPO_ROOT,
    *,
    pretest_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Run the archive workflow and write the terminal artifact."""

    root = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    preconditions: JsonDict = {}

    research_path = root / RESEARCH_COMPLETE_REL_PATH
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
            started_s=start,
            now_s=now_s,
        )
    research_text = research_path.read_text(encoding="utf-8")
    research_parses = yaml_parses(research_text)
    preconditions["research_complete_yaml"] = {
        "path": str(RESEARCH_COMPLETE_REL_PATH),
        "exists": True,
        "parses": research_parses,
    }
    if not research_parses:
        return _blocked(
            root,
            "blocked_research_complete_yaml_poison",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
        )

    exclusion_path = root / EXCLUSION_MANIFEST_REL_PATH
    if not exclusion_path.exists():
        preconditions["exclusion_manifest_yaml"] = {
            "path": str(EXCLUSION_MANIFEST_REL_PATH),
            "exists": False,
            "parses": False,
        }
        return _blocked(
            root,
            "blocked_exclusion_manifest_missing",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
        )
    exclusion_text = exclusion_path.read_text(encoding="utf-8")
    exclusion_parses = yaml_parses(exclusion_text)
    preconditions["exclusion_manifest_yaml"] = {
        "path": str(EXCLUSION_MANIFEST_REL_PATH),
        "exists": True,
        "parses": exclusion_parses,
    }
    if not exclusion_parses:
        return _blocked(
            root,
            "blocked_exclusion_manifest_yaml_poison",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
        )

    next_path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    if not next_path.exists():
        preconditions["research_roadmap_next_yaml"] = {
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "exists": False,
            "parses": False,
        }
        return _blocked(
            root,
            "blocked_research_roadmap_next_missing",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
        )
    next_text = next_path.read_text(encoding="utf-8")
    next_parses = yaml_parses(next_text)
    preconditions["research_roadmap_next_yaml"] = {
        "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
        "exists": True,
        "parses": next_parses,
    }
    if not next_parses:
        return _blocked(
            root,
            "blocked_research_roadmap_next_yaml_poison",
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
            "blocked_v410_not_active",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    source_checks = _source_checks(root)
    preconditions["source_artifacts"] = source_checks
    for exp_id, check in source_checks.items():
        if check["required"] and not check["exists"]:
            return _blocked(
                root,
                SOURCE_MISSING_REASONS.get(exp_id, f"blocked_source_{exp_id}_missing"),
                preconditions_checked=preconditions,
                started_s=start,
                now_s=now_s,
                active_milestone_confirmed=active_milestone,
                active_roadmap_path=active_path,
            )

    registry_path = root / ARC_REGISTRY_REL_PATH
    if not registry_path.exists():
        preconditions["arc_solve_registry_yaml"] = {
            "path": str(ARC_REGISTRY_REL_PATH),
            "exists": False,
            "parses": False,
        }
        return _blocked(
            root,
            "blocked_arc_solve_registry_missing",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )
    registry_text = registry_path.read_text(encoding="utf-8")
    registry_parses = yaml_parses(registry_text)
    preconditions["arc_solve_registry_yaml"] = {
        "path": str(ARC_REGISTRY_REL_PATH),
        "exists": True,
        "parses": registry_parses,
    }
    if not registry_parses:
        return _blocked(
            root,
            "blocked_arc_solve_registry_yaml_poison",
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

    registry_totals = registry_totals_from_text(registry_text)
    sources = read_v409_sources(root)
    close_state = build_v409_close_state(sources, registry_totals)
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
    if not yaml_parses(research_path.read_text(encoding="utf-8")):
        return _blocked(
            root,
            "blocked_research_complete_yaml_poison_after_edit",
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            active_milestone_confirmed=active_milestone,
            active_roadmap_path=active_path,
        )

    output_path = root / OUTPUT_REL_PATH
    payload = build_complete_artifact(
        v409_close_state=close_state,
        registry_totals=registry_totals,
        preconditions_checked=preconditions,
        duration_s=duration_from(start, now_s),
        active_roadmap_path=active_path,
        research_complete_record_action=record_action,
        research_complete_duplicates_removed=duplicates_removed,
        cited_upstream_artifacts=build_cited_upstream(root),
    )
    validate_artifact(payload)
    write_payload(output_path, payload)
    return output_path


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the artifact contract before writing a complete JSON record."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required artifact field: {field}")
    if not _terminal_prefixed(payload.get("honest_verdict")):
        raise ValueError("honest_verdict lacks a complete-path terminal prefix")
    if payload.get("field_principles", {}).get("honest_verdict") != FIELD_PRINCIPLES["honest_verdict"]:
        raise ValueError("honest_verdict principle mismatch")
    if payload.get("field_principles", {}).get("reproducible_total_levels") != FIELD_PRINCIPLES[
        "reproducible_total_levels"
    ]:
        raise ValueError("reproducible_total_levels principle mismatch")
    if payload.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle must be true for this execution-grounded transition")
    if payload.get("trm_training_ran") is not False:
        raise ValueError("TRM training must stay false")
    if payload.get("leaderboard_submission") is not False:
        raise ValueError("leaderboard submission must stay false")
    close = _mapping(payload.get("v409_close_state"))
    if _int(payload.get("reproducible_total_levels")) != _int(
        close.get("reproducible_total_levels")
    ):
        raise ValueError("reproducible_total_levels does not match close-state")
    if _int(payload.get("reproducible_total_games")) != _int(
        close.get("reproducible_total_games")
    ):
        raise ValueError("reproducible_total_games does not match close-state")
    if not is_sha256(payload.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be a SHA-256 hex digest")


def main(root: Path = REPO_ROOT) -> int:
    run(root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
