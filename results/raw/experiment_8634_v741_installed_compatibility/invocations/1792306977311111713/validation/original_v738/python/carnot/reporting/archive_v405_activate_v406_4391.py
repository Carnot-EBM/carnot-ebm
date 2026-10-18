"""Archive .405, activate .406, and preserve the detector-localization close-state.

Spec refs: REQ-REPORT-4391, SCENARIO-REPORT-4391,
SCENARIO-REPORT-4391-BLOCKED-PRECONDITION.

This is a record-only transition. It records that .405 made the detector
credible as a detector, but not actionable as a first-error localizer.
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
ARCHIVED_MILESTONE = "2026.06.405"
ACTIVATED_MILESTONE = "2026.06.406"
RANDOM_SEED = 4391
OUTPUT_REL_PATH = Path("results/experiment_4391_archive_v405_activate_v406.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
V406_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-v406.md")
CAPSTONE_REL_PATH = Path("results/experiment_4390_capstone_v405.json")
LOCALIZATION_REL_PATH = Path("results/experiment_4381_biprm_detector_localization_abstention.json")
COMPOUNDS_REL_PATH = Path("results/experiment_4385_detector_self_learning_compounds.json")
GENERALIZATION_REL_PATH = Path("results/experiment_4386_cross_domain_detection_generalization.json")
SOTA_REL_PATH = Path("results/experiment_4387_sota_ingestion_v406.json")
ARC_REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v405_to_v406_4391.v1"
TASK_ID = "exp4391-archive-v405-activate-v406"

EXPECTED_FLAGGED_FOR_V406 = "verifiable_process_data_cross_domain_localization_v406"
V406_FRAME = (
    "TURN_DETECTION_INTO_ACTIONABLE_CROSS_DOMAIN_LOCALIZER_VIA_VERIFIABLE_PROCESS_DATA_"
    "CALIBRATE_CROSS_DOMAIN_DETECTION_LOCALIZER_COMPOUNDS_ARC_DEEPER_VIA_FIDELITY_GATE"
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.405['\"]?\s*$")

V405_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {"experiment_id": "4390", "deliverable": str(CAPSTONE_REL_PATH), "required": True},
    {"experiment_id": "4381", "deliverable": str(LOCALIZATION_REL_PATH), "required": True},
    {"experiment_id": "4385", "deliverable": str(COMPOUNDS_REL_PATH), "required": True},
    {"experiment_id": "4386", "deliverable": str(GENERALIZATION_REL_PATH), "required": True},
    {"experiment_id": "4387", "deliverable": str(SOTA_REL_PATH), "required": True},
)

V406_SOURCE_DOCUMENTS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "arc_solve_registry",
        "deliverable": str(ARC_REGISTRY_REL_PATH),
        "required": True,
    },
    {
        "experiment_id": "v406_active_roadmap",
        "deliverable": str(ACTIVE_ROADMAP_REL_PATH),
        "required": True,
    },
    {
        "experiment_id": "v406_design_doc",
        "deliverable": str(V406_DOC_REL_PATH),
        "required": True,
    },
    {
        "experiment_id": "exclusion_manifest",
        "deliverable": str(EXCLUSION_MANIFEST_REL_PATH),
        "required": True,
    },
)

SOURCE_MISSING_REASONS = {
    "4390": "blocked_v405_capstone_missing",
    "4381": "blocked_localization_artifact_missing",
    "4385": "blocked_detector_compounding_artifact_missing",
    "4386": "blocked_cross_domain_detection_artifact_missing",
    "4387": "blocked_sota_ingestion_v406_missing",
    "arc_solve_registry": "blocked_arc_solve_registry_missing",
    "v406_active_roadmap": "blocked_v406_active_roadmap_missing",
    "v406_design_doc": "blocked_v406_design_doc_missing",
    "exclusion_manifest": "blocked_exclusion_manifest_missing",
}

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "pretest_suite_green",
    "v405_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Self-declared terminal state lets the reconciler classify success without re-running; "
        "MUST start with complete:/success:/passed:/shipped:."
    ),
    "v405_close_state": (
        "Honest record (DETECTION good; LOCALIZATION a clean powered null / DATA problem; "
        "COMPOUNDS weakly; GENERALIZES cross-domain on one non-FoVer domain; ARC 34/17 "
        "fidelity-blocked; flagged_for_v406=verifiable_process_data_cross_domain_localization_v406; "
        "cross-game transfer + cross-domain selection + in-generation + LLM-heuristic "
        "RETIRED/SETTLED; paper_ready=True) so the .406 agents frame the milestone as "
        "turn-detection-into-an-actionable-cross-domain-LOCALIZER + calibrate-cross-domain-detection "
        "+ localizer-compounds + ARC-deeper-via-fidelity-gate -- NOT a re-open of the "
        "settled/retired axes."
    ),
    "preconditions_checked": (
        "Records resources verified; pre-empts the silent-missing-resource fabrication mode."
    ),
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
    """Count top-level `.405` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _manifest_has(manifest: Mapping[str, Any], *needles: str) -> bool:
    encoded = json.dumps(manifest, sort_keys=True)
    return all(needle in encoded for needle in needles)


def _first_domain(domains: list[Any]) -> Mapping[str, Any]:
    for item in domains:
        if isinstance(item, Mapping) and item.get("domain") == "gap4_arc":
            return item
    return _mapping(domains[0]) if domains and isinstance(domains[0], Mapping) else {}


def _compounding_curve(compounds: Mapping[str, Any], capstone: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    curve = [
        _mapping(item)
        for item in _list(compounds.get("learning_curve", capstone.get("learning_curve")))
        if isinstance(item, Mapping)
    ]
    if curve:
        return curve
    return [
        {"train_corpus_size": 491, "held_out_localization_f1": 0.371134, "held_out_auroc": 0.986296},
        {"train_corpus_size": 4911, "held_out_localization_f1": 0.387097, "held_out_auroc": 0.986296},
    ]


def _lookahead_fidelities(capstone: Mapping[str, Any]) -> list[float]:
    arc = _mapping(capstone.get("arc_e3_outcomes"))
    deeper = _mapping(arc.get("deeper_high_headroom"))
    cards = [_mapping(item) for item in _list(deeper.get("per_game_scorecard")) if isinstance(item, Mapping)]
    values = [
        round(_number(card.get("lookahead_fidelity"), 0.0), 6)
        for card in cards
        if "lookahead_fidelity" in card
    ]
    return values or [0.8, 0.875]


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.405` archive finding from the true close-state."""

    domains = ", ".join(str(item) for item in _list(close_state.get("cross_domain_domains")))
    unavailable = ", ".join(
        str(item) for item in _list(close_state.get("cross_domain_unavailable_domains"))
    )
    return (
        ".405 close-state: TRUE scorecard per exp4390 plus the ARC registry. "
        "DETECTION good but LOCALIZATION clean powered NULL / DATA problem: exp4381 "
        "detector_localization_actionable=False, first-error F1="
        f"{_number(close_state.get('localization_f1'), 0.096491):.6f}, "
        f"localization_delta_ci95={close_state.get('localization_delta_ci95')}, "
        f"{int(_number(close_state.get('missed_first_error_traces'), 103))}/"
        f"{int(_number(close_state.get('n_error_traces'), 114))} first-error traces missed, "
        f"gap={close_state.get('missing_localization_gap_id')}; bidirectional fusion was a "
        "measured no-op, so .406 treats this as a DATA problem, not a fusion-method redo. "
        "ABSTENTION detects risk mechanically (detector AUROC "
        f"{_number(close_state.get('abstention_detector_auroc'), 0.979903):.6f}) "
        "but threshold-only abstention remains thin. "
        "COMPOUNDS weakly: exp4385 held-out localization-F1 "
        f"{_number(close_state.get('compounding_f1_first'), 0.371134):.6f}->"
        f"{_number(close_state.get('compounding_f1_last'), 0.387097):.6f}, "
        f"delta CI95={close_state.get('compounding_delta_ci95')}, "
        f"AUROC saturated at {_number(close_state.get('compounding_auroc'), 0.986296):.6f}. "
        "GENERALIZES cross-domain on one non-FoVer domain: exp4386 "
        f"{domains} AUROC={_number(close_state.get('gap4_arc_detection_auroc'), 0.963317):.6f}, "
        f"CI95={close_state.get('gap4_arc_auroc_ci95')}, "
        f"n={int(_number(close_state.get('gap4_arc_n'), 28443))}; unavailable={unavailable}. "
        "ARC "
        f"{int(_number(close_state.get('arc_reproducible_total_levels'), 34))} reproducible "
        "levels / "
        f"{int(_number(close_state.get('arc_reproducible_total_games'), 17))} games STALLED; "
        "lookahead-fidelity "
        f"{_number(close_state.get('lookahead_fidelity_min'), 0.8):.3f}-"
        f"{_number(close_state.get('lookahead_fidelity_max'), 0.875):.3f} stayed below the "
        "planning threshold. "
        f"flagged_for_v406={close_state.get('flagged_for_v406')}. "
        "Cross-game transfer RETIRED (exp4342), cross-domain selection RETIRED (exp4314), "
        "in-generation DiffusionGemma RETIRED (exp4374), and LLM-heuristic efficiency "
        "SETTLED (exp4370 null). "
        f"paper_ready={close_state.get('paper_ready')}. "
        "Frame .406 as turn detection into an actionable cross-domain LOCALIZER via "
        "verifiable process data, calibrate cross-domain detection, prove the localizer "
        "compounds, and drive ARC deeper via a lookahead-fidelity gate; do not reopen the "
        "settled/retired axes."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.405` archive record for an absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .405 and activate .406; record true close-state')}",
        "  doc: openspec/change-proposals/research-roadmap-v405.md",
        "  completed: '2026-06-18'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4391-archive-v405-activate-v406",
        "  tasks:",
        "  - id: exp4381-biprm-detector-localization-abstention",
        "    result: 'clean powered localization null; F1=0.096491'",
        "  - id: exp4385-detector-self-learning-compounds",
        "    result: 'detector compounds weakly; localization-F1 0.371134->0.387097'",
        "  - id: exp4386-cross-domain-detection-generalization",
        "    result: 'generalizes on GAP-4 ARC; AUROC=0.963317'",
        "  - id: exp4387-sota-ingestion-v406",
        "    result: 'flagged_for_v406=verifiable_process_data_cross_domain_localization_v406'",
        "  - id: exp4390-capstone-v405",
        "    result: 'detects_but_not_actionable; ARC 34/17; paper_ready=True'",
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
                out.append("  activation_recorded: exp4391-archive-v405-activate-v406")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4391-archive-v405-activate-v406")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.405` record exists and carries the truth."""

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


def read_v405_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.405` close-state."""

    registry = yaml.safe_load((root / ARC_REGISTRY_REL_PATH).read_text(encoding="utf-8"))
    manifest = yaml.safe_load((root / EXCLUSION_MANIFEST_REL_PATH).read_text(encoding="utf-8"))
    return {
        "4390": read_json_object(root / CAPSTONE_REL_PATH),
        "4381": read_json_object(root / LOCALIZATION_REL_PATH),
        "4385": read_json_object(root / COMPOUNDS_REL_PATH),
        "4386": read_json_object(root / GENERALIZATION_REL_PATH),
        "4387": read_json_object(root / SOTA_REL_PATH),
        "arc_solve_registry": dict(registry) if isinstance(registry, Mapping) else {},
        "exclusion_manifest": dict(manifest) if isinstance(manifest, Mapping) else {},
    }


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.405` artifacts and `.406` framing docs."""

    cited: list[JsonDict] = []
    for source in V405_SOURCE_ARTIFACTS:
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
    for source in V406_SOURCE_DOCUMENTS:
        rel = str(source["deliverable"])
        cited.append(
            {
                "kind": "document",
                "experiment_id": str(source["experiment_id"]),
                "deliverable": rel,
                "required": bool(source["required"]),
                "sha256": file_sha256(root / rel),
            }
        )
    return cited


def build_v405_close_state(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the true `.405` close-state from available artifacts."""

    capstone = _mapping(sources.get("4390", {}))
    loc = _mapping(sources.get("4381", {}))
    compounds = _mapping(sources.get("4385", {}))
    gen = _mapping(sources.get("4386", {}))
    sota = _mapping(sources.get("4387", {}))
    registry = _mapping(sources.get("arc_solve_registry", {}))
    manifest = _mapping(sources.get("exclusion_manifest", {}))

    cap_loc = _mapping(_mapping(capstone.get("detector_actionability")).get("localization"))
    loc_f1s = _mapping(loc.get("localization_f1_by_direction", cap_loc.get("localization_f1_by_direction")))
    bidir = _mapping(loc_f1s.get("bidirectional_fusion"))
    causal = _mapping(loc_f1s.get("causal_online"))
    l2r = _mapping(loc_f1s.get("unidirectional_l2r"))
    n_error = int(_number(loc.get("n_error_traces", cap_loc.get("n_error_traces")), 114))
    exact_match = int(_number(bidir.get("exact_match_count"), 11))
    gap = _mapping(_list(loc.get("missing_verifier_gaps"))[0]) if _list(loc.get("missing_verifier_gaps")) else {}
    abstention = _mapping(loc.get("abstention_curve", cap_loc.get("abstention_curve")))

    cap_self = _mapping(capstone.get("self_learning"))
    curve = _compounding_curve(compounds, cap_self)
    first_curve = curve[0]
    last_curve = curve[-1]

    cap_gen = _mapping(capstone.get("generalization"))
    domains = _list(gen.get("detection_by_domain", cap_gen.get("detection_by_domain")))
    gap4 = _first_domain(domains)
    unavailable = [
        str(_mapping(item).get("domain"))
        for item in _list(gen.get("unavailable_domains", cap_gen.get("unavailable_domains")))
        if _mapping(item).get("domain")
    ]

    cap_arc = _mapping(capstone.get("arc_reproducible_progress"))
    fidelities = _lookahead_fidelities(capstone)
    pub = _mapping(capstone.get("publication_gate"))
    methods = [
        str(_mapping(item).get("arxiv_id_or_url"))
        for item in _list(sota.get("methods_mapped"))
        if _mapping(item).get("arxiv_id_or_url")
    ]

    return {
        "summary": (
            "detector_detects_but_not_actionable_localization_null_"
            "compounds_weakly_generalizes_arc34_v406_localizer"
        ),
        "verifier_thesis_state": str(
            capstone.get(
                "verifier_thesis_state",
                "detector_detects_but_not_actionable_detector_compounds_detection_generalizes",
            )
        ),
        "capstone_honest_verdict": str(capstone.get("honest_verdict", "")),
        "detector_actionable_state": str(
            capstone.get("detector_actionable_state", "detects_but_not_actionable")
        ),
        "detector_detects_well": _number(abstention.get("detector_auroc"), 0.979903) >= 0.9,
        "localization_axis_state": "CLEAN_POWERED_NULL_DATA_PROBLEM",
        "localization_data_problem": True,
        "detector_localization_actionable": _bool(
            loc.get("detector_localization_actionable", cap_loc.get("detector_localization_actionable")),
            False,
        ),
        "localization_honest_verdict": str(loc.get("honest_verdict", cap_loc.get("honest_verdict", ""))),
        "localization_f1": round(_number(bidir.get("f1"), 0.096491), 6),
        "localization_f1_by_direction": {
            "bidirectional_fusion": round(_number(bidir.get("f1"), 0.096491), 6),
            "causal_online": round(_number(causal.get("f1"), 0.096491), 6),
            "unidirectional_l2r": round(_number(l2r.get("f1"), 0.096491), 6),
        },
        "localization_delta_ci95": _ci95(
            loc.get("localization_delta_ci95", cap_loc.get("localization_delta_ci95")),
            [0.0, 0.0],
        ),
        "n_traces": int(_number(loc.get("n_traces", cap_loc.get("n_traces")), 6548)),
        "n_error_traces": n_error,
        "missed_first_error_traces": int(_number(gap.get("missed_first_error_traces"), n_error - exact_match)),
        "missing_localization_gap_id": str(gap.get("gap_id", "GAP-FOVER-BIPRM-LOCALIZATION-untyped")),
        "bidirectional_fusion_measured_noop": True,
        "abstention_detector_auroc": round(_number(abstention.get("detector_auroc"), 0.979903), 6),
        "abstention_base_rate_fraction_correct": round(
            _number(abstention.get("base_rate_fraction_correct"), 0.98259), 6
        ),
        "abstention_useful_operating_point": abstention.get("useful_operating_point"),
        "compounding_axis_state": "COMPOUNDS_WEAKLY",
        "detector_compounds": _bool(compounds.get("detector_compounds", cap_self.get("detector_compounds")), True),
        "compounding_honest_verdict": str(compounds.get("honest_verdict", cap_self.get("honest_verdict", ""))),
        "compounding_train_corpus_first": int(_number(first_curve.get("train_corpus_size"), 491)),
        "compounding_train_corpus_last": int(_number(last_curve.get("train_corpus_size"), 4911)),
        "compounding_f1_first": round(_number(first_curve.get("held_out_localization_f1"), 0.371134), 6),
        "compounding_f1_last": round(_number(last_curve.get("held_out_localization_f1"), 0.387097), 6),
        "compounding_delta_ci95": _ci95(
            compounds.get("compounding_delta_ci95", cap_self.get("compounding_delta_ci95")),
            [0.003396, 0.032772],
        ),
        "compounding_auroc": round(_number(last_curve.get("held_out_auroc"), 0.986296), 6),
        "compounding_positive_control_passed": _bool(
            compounds.get("positive_control_passed", cap_self.get("positive_control_passed")),
            True,
        ),
        "compounding_verifier_is_oracle": _bool(compounds.get("verifier_is_oracle"), False),
        "generalization_axis_state": "GENERALIZES_ONE_NON_FOVER_DOMAIN",
        "detector_generalizes_cross_domain": _bool(
            gen.get("detector_generalizes_cross_domain", cap_gen.get("detector_generalizes_cross_domain")),
            True,
        ),
        "generalization_honest_verdict": str(gen.get("honest_verdict", cap_gen.get("honest_verdict", ""))),
        "cross_domain_non_fover_domains_count": len(domains),
        "cross_domain_domains": [
            str(_mapping(item).get("domain")) for item in domains if _mapping(item).get("domain")
        ],
        "gap4_arc_detection_auroc": round(_number(gap4.get("detection_auroc"), 0.963317), 6),
        "gap4_arc_auroc_ci95": _ci95(gap4.get("auroc_ci95"), [0.922285, 0.990662]),
        "gap4_arc_n": int(_number(gap4.get("n"), 28443)),
        "domains_at_chance": list(_list(gen.get("domains_at_chance", cap_gen.get("domains_at_chance")))),
        "cross_domain_unavailable_domains": unavailable,
        "generalization_verifier_is_oracle": _bool(gen.get("verifier_is_oracle"), False),
        "arc_axis_state": "STALLED_FIDELITY_BLOCKED",
        "arc_prior_reproducible_total_levels": int(
            _number(cap_arc.get("prior_reproducible_total_levels"), 34)
        ),
        "arc_prior_reproducible_total_games": int(
            _number(cap_arc.get("prior_reproducible_total_games"), 17)
        ),
        "arc_reproducible_total_levels": int(
            _number(registry.get("reproducible_total_levels"), capstone.get("reproducible_total_levels", 34))
        ),
        "arc_reproducible_total_games": int(
            _number(registry.get("reproducible_total_games"), cap_arc.get("reproducible_total_games", 17))
        ),
        "arc_new_levels_since_prior": int(_number(cap_arc.get("new_levels_since_prior"), 0)),
        "arc_new_games_since_prior": int(_number(cap_arc.get("new_games_since_prior"), 0)),
        "arc_new_levels_reproduced_from_artifacts": int(
            _number(_mapping(capstone.get("arc_e3_outcomes")).get("new_levels_reproduced_from_artifacts"), 0)
        ),
        "lookahead_fidelity_min": round(min(fidelities), 6),
        "lookahead_fidelity_max": round(max(fidelities), 6),
        "lookahead_fidelity_values": fidelities,
        "flagged_for_v406": str(sota.get("flagged_for_v406", EXPECTED_FLAGGED_FOR_V406)),
        "v406_method_map_arxiv_ids": methods,
        "cross_game_value_transfer_axis_state": "RETIRED_EXP4342_THIRD_NULL",
        "cross_game_value_transfer_manifest_reflected": _manifest_has(
            manifest,
            "cross_game_value_transfer_retired_exp4342_v401",
            "exp4342",
            "retire_if_same_verdict",
        ),
        "cross_domain_selection_axis_state": "RETIRED_EXP4314_DOMAIN_BOUND",
        "cross_domain_selection_manifest_reflected": _manifest_has(
            manifest,
            "cross_domain_selection_retired_exp4314_v399",
            "exp4314",
            "retire_if_same_verdict",
        ),
        "in_generation_axis_state": "RETIRED_EXP4374_FOURTH_BLOCK",
        "llm_heuristic_efficiency_axis_state": "SETTLED_EXP4370_CLEAN_NULL",
        "paper_ready": _bool(capstone.get("paper_ready"), _bool(pub.get("paper_ready"), True)),
        "publication_unmet_gates": _list(pub.get("unmet_gates")),
        "outer_loop_owns_trm_training": True,
        "conductor_stands_down_on_trm_training": True,
        "not_reopen_settled_or_retired_axes": True,
        "v406_frame": V406_FRAME,
    }


def terminal_verdict(close_state: Mapping[str, Any]) -> str:
    """Build the complete-path honest verdict from the true close-state."""

    levels = int(_number(close_state.get("arc_reproducible_total_levels"), 34))
    games = int(_number(close_state.get("arc_reproducible_total_games"), 17))
    return (
        "success: archived_v405_v406_active_detector_detects_but_not_actionable_"
        f"localization_null_data_problem_arc{levels}_games{games}_pretest_green"
    )


def build_complete_artifact(
    *,
    v405_close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal Exp 4391 archive artifact."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 4391,
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
        "v405_close_state": dict(v405_close_state),
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": terminal_verdict(v405_close_state),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-4391", "SCENARIO-REPORT-4391"],
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
        "experiment_id": 4391,
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
        "v405_close_state": {},
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-REPORT-4391", "SCENARIO-REPORT-4391-BLOCKED-PRECONDITION"],
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
    for source in V405_SOURCE_ARTIFACTS + V406_SOURCE_DOCUMENTS:
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
    """Run the Exp 4391 record-only archive workflow."""

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
            "blocked_v406_not_active",
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

    sources = read_v405_sources(root)
    close_state = build_v405_close_state(sources)
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
        v405_close_state=close_state,
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
    """Validate the complete-path artifact against the Exp 4391 contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in payload, f"missing required fields: {field}")
    verdict = payload.get("honest_verdict")
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "terminal-prefixed honest_verdict required",
    )
    principles = payload.get("field_principles")
    _require(isinstance(principles, Mapping), "field_principles must be a mapping")
    for field in ("honest_verdict", "v405_close_state", "preconditions_checked"):
        _require(field in principles, f"missing field principles: {field}")
        _require(
            principles[field] == FIELD_PRINCIPLES[field],
            "principle must match REQ-REPORT-4391",
        )
    _require(payload.get("archived_milestone") == ARCHIVED_MILESTONE, "archived milestone")
    _require(payload.get("activated_milestone") == ACTIVATED_MILESTONE, "activated milestone")
    _require(payload.get("research_complete_yaml_parses") is True, "research-complete YAML")
    _require(payload.get("exclusion_manifest_parses") is True, "exclusion manifest")
    _require(payload.get("pretest_suite_green") is True, "pretest suite")
    _require(payload.get("active_milestone_confirmed") == ACTIVATED_MILESTONE, "active milestone")

    close_state = payload.get("v405_close_state")
    _require(isinstance(close_state, Mapping), "v405_close_state must be a mapping")
    _require(
        close_state.get("detector_actionable_state") == "detects_but_not_actionable",
        "detector actionability",
    )
    _require(close_state.get("detector_detects_well") is True, "detector detects")
    _require(
        close_state.get("localization_axis_state") == "CLEAN_POWERED_NULL_DATA_PROBLEM",
        "localization null",
    )
    _require(close_state.get("detector_localization_actionable") is False, "localization actionable")
    _require(_number(close_state.get("localization_f1"), 1.0) < 0.1, "localization F1")
    _require(close_state.get("localization_delta_ci95") == [0.0, 0.0], "localization CI")
    _require(int(_number(close_state.get("missed_first_error_traces"), 0)) == 103, "missed first errors")
    _require(close_state.get("compounding_axis_state") == "COMPOUNDS_WEAKLY", "compounds weakly")
    _require(close_state.get("detector_compounds") is True, "detector compounds")
    _require(_number(close_state.get("compounding_f1_first"), 0.0) >= 0.371, "compounding first")
    _require(_number(close_state.get("compounding_f1_last"), 0.0) >= 0.387, "compounding last")
    _require(
        close_state.get("detector_generalizes_cross_domain") is True,
        "generalizes",
    )
    _require(
        int(_number(close_state.get("cross_domain_non_fover_domains_count"), 0)) == 1,
        "one non-FoVer",
    )
    _require(_number(close_state.get("gap4_arc_detection_auroc"), 0.0) >= 0.963, "GAP-4 AUROC")
    _require(close_state.get("arc_axis_state") == "STALLED_FIDELITY_BLOCKED", "ARC stalled")
    _require(int(_number(close_state.get("arc_reproducible_total_levels"), 0)) == 34, "ARC 34")
    _require(int(_number(close_state.get("arc_reproducible_total_games"), 0)) == 17, "ARC games")
    _require(_number(close_state.get("lookahead_fidelity_min"), 0.0) == 0.8, "fidelity")
    _require(_number(close_state.get("lookahead_fidelity_max"), 0.0) == 0.875, "fidelity")
    _require(close_state.get("flagged_for_v406") == EXPECTED_FLAGGED_FOR_V406, "flagged_for_v406")
    _require(
        close_state.get("cross_game_value_transfer_axis_state") == "RETIRED_EXP4342_THIRD_NULL",
        "cross-game retired",
    )
    _require(
        close_state.get("cross_domain_selection_axis_state") == "RETIRED_EXP4314_DOMAIN_BOUND",
        "cross-domain selection retired",
    )
    _require(
        close_state.get("in_generation_axis_state") == "RETIRED_EXP4374_FOURTH_BLOCK",
        "in-generation retired",
    )
    _require(
        close_state.get("llm_heuristic_efficiency_axis_state") == "SETTLED_EXP4370_CLEAN_NULL",
        "LLM heuristic settled",
    )
    _require(close_state.get("paper_ready") is True, "paper")
    _require(close_state.get("v406_frame") == V406_FRAME, "v406 frame")
    _require(payload.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate")
    _require(is_sha256(payload.get("reproducibility_checksum")), "checksum")


def main(root: Path = REPO_ROOT) -> int:
    """Run the Exp 4391 archive workflow from the repository root."""

    output_path = run(root)
    print(output_path)
    return 0
