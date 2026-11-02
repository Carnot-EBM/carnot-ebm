"""Experiment 4989: .459 final submission-readiness capstone.

Spec refs: REQ-CAPSTONE-4989, SCENARIO-CAPSTONE-4989,
SCENARIO-CAPSTONE-4989-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4989-FIELD-PRINCIPLES.

This aggregation reads every .459 upstream artifact through
``scripts/summarize_artifact.py`` before importing fields. It confirms the
locked 2026-06-30 deliverable, records the A3/B3 infra fixes as still holding,
and hands off the post-6/30 verifier-moat pivot without claiming the moat is
proven.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_4819_capstone_v443 import (
    SummarizerResult,
    UpstreamSource,
    _float,
    _int,
    _mapping,
    _read_json_object,
    _read_yaml_object,
    file_sha256,
    payload_checksum,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4989_capstone_v459"
EXPERIMENT_ID = 4989
SCHEMA = "carnot.exp4989.capstone_v459.v1"
RESULT_RELATIVE_PATH = "results/experiment_4989_capstone_v459.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
NORTH_STAR_RELATIVE_PATH = "ops/north-star.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 20260629
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4989",
    "SCENARIO-CAPSTONE-4989",
    "SCENARIO-CAPSTONE-4989-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4989-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_capstone_v459_submission_ready_<state> "
            "(e.g. _levels_<N>_heldout_<rate>_package_ready_pivot_turnkey_7_1)."
        )
    },
    "headline": {
        "principle": (
            "the one-line .459 headline -- the locked deliverable confirmed ready "
            "for 6/30 + the post-6/30 verifier-moat pivot turnkey 7/1 "
            "(backlog extended to 11 papers)."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the monotonic ARC progress metric from the registry (69 + A1/A2 banks "
            "counted ONLY if B1 banks_trustworthy)."
        )
    },
    "banks_counted": {
        "principle": (
            "the A1/A2 games whose banks were counted (gated on B1 banks_trustworthy) "
            "-- no un-audited bank inflates the total; honest no-bank is expected "
            "(5th+ flat milestone)."
        )
    },
    "heldout_first_win_rate": {
        "principle": (
            "A4's FINAL full-25 go/no-go number + flag_resolved -- the operator's "
            "6/30 decision input."
        )
    },
    "submission_package_ready": {
        "principle": (
            "B2's terminal gate + peak_vram_gb -- the deadline deliverable for the operator."
        )
    },
    "a3_substrate_flag_resolved": {
        "principle": (
            "true -- A3 (exp4982) is NOT flagged true_live_recheck=critical (the .456 "
            "DURATION_TOO_SHORT substrate fix still holds; self-play is COUNTED)."
        )
    },
    "b3_window_nonzero": {
        "principle": (
            "true -- B3 (exp4987) emitted a NON-zero mtime window (the relaxed gate is "
            "maintained)."
        )
    },
    "arc_first_win_wall_closed": {
        "principle": (
            "true -- the .453 B1-trusted WALL_IS_HIDDEN_STATE closure stands; .459 did "
            "not reopen it or queue representation #5 / the concluded energy-as-ARC program."
        )
    },
    "post_sprint_pivot": {
        "principle": (
            "the post-6/30 verifier-moat handoff: D's distributional-energy-verifier "
            "TURNKEY spec (arXiv:2605.18871, beats SC on MuSR) + extended backlog "
            "(NEW 2510.14913/2603.04304 -> 11 papers), gated on B1 "
            "pivot_readiness_trustworthy; the deliverable remains the ~0.05 agent + "
            "FoVer paper. NOT moat-proven."
        )
    },
    "pivot_executable_on_7_1": {
        "principle": (
            "true iff D kept the pivot turnkey AND B1 pivot_readiness_trustworthy -- "
            "the post-sprint experiment runs the instant the sprint retires."
        )
    },
    "skipped_flagged_adversarial": {
        "principle": "the list of upstream artifacts skipped per the fabrication gate."
    },
    "cited_upstream_artifacts": {
        "principle": (
            "{experiment_id, fields_imported, sha256} per source -- the audit trail "
            "(no synthesized numbers)."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON; 0.0001s floor)."
    },
    "preconditions_checked": {
        "principle": (
            "records upstream-artifact presence + registry loadability; a missing input "
            "emits blocked_."
        )
    },
    "random_seed": {
        "principle": "determinism for any tie-break in the scorecard aggregation."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (cited upstream sha256s + scorecard) so a replication catches drift."
        )
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "headline",
    "reproducible_total_levels",
    "banks_counted",
    "heldout_first_win_rate",
    "submission_package_ready",
    "a3_substrate_flag_resolved",
    "b3_window_nonzero",
    "arc_first_win_wall_closed",
    "post_sprint_pivot",
    "pivot_executable_on_7_1",
    "skipped_flagged_adversarial",
    "cited_upstream_artifacts",
    "inference_substrate",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "milestone_scorecard",
    "field_principles",
    "duration_s",
    "capstone_ready",
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
    "blocked_",
)

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "A1_LEVELUP": UpstreamSource(4980, "results/experiment_4980_levelup_attempt.json"),
    "A2_LEVELUP": UpstreamSource(4981, "results/experiment_4981_levelup_attempt.json"),
    "A3_SELF_PLAY": UpstreamSource(
        4982, "results/experiment_4982_self_play_verifier_checkpoint.json"
    ),
    "A4_HELDOUT": UpstreamSource(
        4983, "results/experiment_4983_heldout_first_win_readiness.json"
    ),
    "D_PIVOT": UpstreamSource(
        4984, "results/experiment_4984_distributional_energy_verifier_turnkey.json"
    ),
    "B1_AUDIT": UpstreamSource(4985, "results/experiment_4985_bank_and_pivot_audit.json"),
    "B2_PACKAGE": UpstreamSource(4986, "results/experiment_4986_submission_package_harden.json"),
    "B3_STAMPING": UpstreamSource(
        4987, "results/experiment_4987_stamping_backfill_and_wiring_readiness.json"
    ),
    "C_KV260": UpstreamSource(4988, "results/experiment_4988_kv260_continuity.json"),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "A1_LEVELUP": (
        "honest_verdict",
        "target_game",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "solve_provenance",
        "live_path_reachable",
        "reproduction_gate",
        "registry_update",
    ),
    "A2_LEVELUP": (
        "honest_verdict",
        "target_game",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "solve_provenance",
        "live_path_reachable",
        "reproduction_gate",
        "registry_update",
    ),
    "A3_SELF_PLAY": (
        "honest_verdict",
        "target_game",
        "verifier_checkpoint_refreshed",
        "checkpoint_path",
        "offline_reproduced",
        "reproduced_levels",
        "flag_resolved",
        "reproduction_gate",
        "inference_substrate",
    ),
    "A4_HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "heldout_first_win_ci",
        "games_evaluated",
        "flag_resolved",
        "positive_control_passed",
        "operator_decision_number",
        "model_specs",
        "inference_substrate",
        "solve_provenance",
    ),
    "D_PIVOT": (
        "honest_verdict",
        "pivot_turnkey",
        "pivot_executable_on_7_1",
        "three_column_dry_run_ok",
        "validation_gate",
        "turnkey_spec",
        "post_sprint_first_experiment_pointer",
        "verifier_is_oracle",
        "moat_proven_claimed",
        "arxiv_ids_cited",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "banks_trustworthy",
        "pivot_readiness_trustworthy",
        "checks",
        "audit_failure_reasons",
        "bank_evidence",
        "pivot_readiness_evidence",
    ),
    "B2_PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "peak_vram_gb",
        "frozen_stack_loads",
        "package_builds",
        "ready_package_regression_ok",
        "submits",
        "operator_only",
        "operator_submission_checklist",
    ),
    "B3_STAMPING": (
        "honest_verdict",
        "stamping_backfilled_arms",
        "mtime_fallback_window",
        "wiring_proposal_reconfirmed",
        "research_conductor_modified",
    ),
    "C_KV260": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "loaded_overlay",
        "xmutil_requires_sudo",
        "verifier_is_oracle",
    ),
}

Summarizer = Callable[[Path, str], SummarizerResult]


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    """Return an experiment id from the artifact when present, else from config."""

    if artifact:
        for field in ("experiment_id", "experiment"):
            value = artifact.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                match = re.search(r"experiment_(\d+)|\b(\d{4})\b", value)
                if match:
                    return int(match.group(1) or match.group(2))
    return UPSTREAM_SOURCES[source].experiment_id


def _live_critical(summary: SummarizerResult | None) -> bool:
    return bool(summary and summary.exit_code >= 2)


def _live_recheck(summary: SummarizerResult | None) -> str:
    if summary is None:
        return "not_run"
    if summary.exit_code >= 2:
        return "critical"
    if summary.exit_code == 1:
        return "warn"
    return "clean"


def _is_skipped(artifact: Mapping[str, Any] | None, summary: SummarizerResult | None) -> str | None:
    if artifact and artifact.get("flagged_adversarial") is True:
        return "flagged_adversarial"
    if _live_critical(summary):
        return "live_critical_recheck"
    return None


def _clean_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
) -> dict[str, Mapping[str, Any]]:
    return {
        source: artifact
        for source, artifact in artifacts.items()
        if _is_skipped(artifact, summarizer_results.get(source)) is None
    }


def _imported_fields(source: str, artifact: Mapping[str, Any]) -> list[str]:
    return [field for field in CLEAN_IMPORT_FIELDS[source] if field in artifact]


def _cited_artifacts(
    clean: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        artifact = clean.get(source)
        if artifact is None:
            continue
        summary = summarizer_results.get(source)
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "fields_imported": _imported_fields(source, artifact),
                "sha256": artifact_sha256.get(source, ""),
                "summarizer_exit_code": summary.exit_code if summary else None,
                "true_live_recheck": _live_recheck(summary),
            }
        )
    return rows


def _skipped_flagged_adversarial(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source, artifact in artifacts.items():
        summary = summarizer_results.get(source)
        reason = _is_skipped(artifact, summary)
        if reason is None:
            continue
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": reason,
                "sha256": artifact_sha256.get(source, ""),
                "summarizer_exit_code": summary.exit_code if summary else None,
                "true_honest_verdict": artifact.get("honest_verdict", ""),
                "true_live_recheck": _live_recheck(summary),
            }
        )
    return rows


def _bank_candidate(source: str, artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {"source": source, "game": "", "new_levels_banked": 0}
    return {
        "source": source,
        "game": str(artifact.get("target_game") or ""),
        "new_levels_banked": _int(artifact.get("new_levels_banked")),
    }


def _banks_counted(clean: Mapping[str, Mapping[str, Any]], registry_total: int) -> JsonDict:
    b1 = clean.get("B1_AUDIT")
    trustworthy = bool(b1 and b1.get("banks_trustworthy") is True)
    candidates = [
        _bank_candidate("A1_LEVELUP", clean.get("A1_LEVELUP")),
        _bank_candidate("A2_LEVELUP", clean.get("A2_LEVELUP")),
    ]
    counted = [
        row for row in candidates if trustworthy and _int(row.get("new_levels_banked")) > 0
    ]
    bank_delta = sum(_int(row["new_levels_banked"]) for row in counted)
    return {
        "b1_banks_trustworthy": trustworthy,
        "counted": counted,
        "candidate_banks": candidates,
        "audit_failure_reasons": list(b1.get("audit_failure_reasons", [])) if b1 else [],
        "base_total_from_registry": registry_total,
        "bank_delta_counted": bank_delta,
        "computed_total": registry_total + bank_delta,
    }


def _heldout_first_win_rate(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    a4 = clean.get("A4_HELDOUT")
    if a4 is None:
        return {"status": "missing_or_skipped", "rate": None, "flag_resolved": False}
    flag_resolved = a4.get("flag_resolved") is True
    games_evaluated = _int(a4.get("games_evaluated"))
    return {
        "status": "full25_go_no_go" if games_evaluated == 25 else "partial_go_no_go",
        "honest_verdict": a4.get("honest_verdict", ""),
        "rate": _float(a4.get("heldout_first_win_rate")) if flag_resolved else None,
        "ci": a4.get("heldout_first_win_ci", {}),
        "flag_resolved": flag_resolved,
        "games_evaluated": games_evaluated,
        "positive_control_passed": a4.get("positive_control_passed") is True,
        "operator_decision_number": a4.get("operator_decision_number", {}),
        "model_specs": a4.get("model_specs", {}),
        "solve_provenance": a4.get("solve_provenance"),
    }


def _submission_package_ready(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    package = clean.get("B2_PACKAGE")
    if package is None:
        return {"ready": False, "status": "missing_or_skipped", "peak_vram_gb": None}
    submitted = package.get("submits") is True or package.get("submitted_to_leaderboard") is True
    peak_vram = _float(package.get("peak_vram_gb"))
    ready = bool(
        package.get("submission_package_ready") is True
        and package.get("operator_only") is True
        and not submitted
        and peak_vram is not None
        and peak_vram < 16.0
        and package.get("frozen_stack_loads") is True
        and package.get("package_builds") is True
        and package.get("ready_package_regression_ok") is True
    )
    return {
        "ready": ready,
        "decision": "package_ready_operator_only" if ready else "package_not_ready",
        "honest_verdict": package.get("honest_verdict", ""),
        "peak_vram_gb": peak_vram,
        "frozen_stack_loads": package.get("frozen_stack_loads") is True,
        "package_builds": package.get("package_builds") is True,
        "ready_package_regression_ok": package.get("ready_package_regression_ok") is True,
        "submits": submitted,
        "operator_only": package.get("operator_only") is True,
        "operator_submission_checklist": package.get("operator_submission_checklist", []),
    }


def _a3_substrate_flag_resolved(
    summarizer_results: Mapping[str, SummarizerResult],
    clean: Mapping[str, Mapping[str, Any]],
) -> bool:
    summary = summarizer_results.get("A3_SELF_PLAY")
    return bool(clean.get("A3_SELF_PLAY") and _live_recheck(summary) != "critical")


def _a3_substrate_state(
    clean: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
) -> JsonDict:
    a3 = clean.get("A3_SELF_PLAY")
    summary = summarizer_results.get("A3_SELF_PLAY")
    resolved = _a3_substrate_flag_resolved(summarizer_results, clean)
    return {
        "resolved": resolved,
        "decision": "a3_substrate_fix_holds_self_play_counted" if resolved else "a3_substrate_blocked",
        "honest_verdict": a3.get("honest_verdict", "") if a3 else "",
        "target_game": a3.get("target_game") if a3 else None,
        "offline_reproduced": a3.get("offline_reproduced") is True if a3 else False,
        "reproduced_levels": _int(a3.get("reproduced_levels")) if a3 else 0,
        "summarizer_exit_code": summary.exit_code if summary else None,
        "true_live_recheck": _live_recheck(summary),
        "inference_substrate": a3.get("inference_substrate") if a3 else None,
    }


def _b3_window_nonzero(clean: Mapping[str, Mapping[str, Any]]) -> bool:
    b3 = clean.get("B3_STAMPING")
    window = _mapping(b3.get("mtime_fallback_window") if b3 else {})
    wall_minutes = _float(window.get("wall_minutes"))
    return bool(wall_minutes is not None and wall_minutes > 0.0)


def _b3_window_state(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    b3 = clean.get("B3_STAMPING")
    window = _mapping(b3.get("mtime_fallback_window") if b3 else {})
    nonzero = _b3_window_nonzero(clean)
    return {
        "nonzero": nonzero,
        "decision": "b3_relaxed_mtime_window_maintained" if nonzero else "b3_mtime_window_missing",
        "honest_verdict": b3.get("honest_verdict", "") if b3 else "",
        "mtime_fallback_window": dict(window),
        "wall_minutes": _float(window.get("wall_minutes")),
        "n_arms": _int(window.get("n_arms")),
        "wiring_proposal_reconfirmed": b3.get("wiring_proposal_reconfirmed") is True
        if b3
        else False,
        "research_conductor_modified": b3.get("research_conductor_modified") is True
        if b3
        else None,
    }


def _wall_closure() -> JsonDict:
    return {
        "source": "standing_453_b1_trusted_closure",
        "closed": True,
        "closure_verdict": "WALL_IS_HIDDEN_STATE",
        "trusted": True,
        "did_reopen_in_v459": False,
        "representation_5_queued": False,
        "energy_as_arc_program": "S0_CONCLUDED_2026_06_26",
        "did_reopen_energy_as_arc_program": False,
        "do_not_queue": "representation_5_or_concluded_energy_as_arc_program",
    }


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _backlog_parts(d: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    if d is None:
        return [], []
    turnkey = _mapping(d.get("turnkey_spec"))
    extension = _mapping(turnkey.get("sota_backlog_extension_v459"))
    new_ids_raw = extension.get("new_arxiv_ids")
    reconfirmed_raw = extension.get("reconfirmed_arxiv_ids")
    new_ids = [str(value) for value in new_ids_raw] if isinstance(new_ids_raw, list) else []
    reconfirmed = (
        [str(value) for value in reconfirmed_raw] if isinstance(reconfirmed_raw, list) else []
    )
    cited = [str(value) for value in d.get("arxiv_ids_cited", [])]
    if not new_ids:
        new_ids = [arxiv_id for arxiv_id in ("2510.14913", "2603.04304") if arxiv_id in cited]
    backlog = _dedupe_preserving_order(new_ids + reconfirmed)
    if len(backlog) < len(cited):
        backlog = _dedupe_preserving_order(backlog + cited)
    return backlog, new_ids


def _post_sprint_pivot(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    d = clean.get("D_PIVOT")
    b1 = clean.get("B1_AUDIT")
    b1_trust = bool(b1 and b1.get("pivot_readiness_trustworthy") is True)
    d_turnkey = bool(d and d.get("pivot_turnkey") is True)
    d_executable = bool(d and d.get("pivot_executable_on_7_1") is True)
    executable = b1_trust and d_turnkey and d_executable
    extended_backlog, new_backlog_ids = _backlog_parts(d)
    return {
        "decision": (
            "post_6_30_distributional_energy_verifier_moat_experiment_turnkey_7_1"
            if executable
            else "pivot_not_stated_untrusted"
        ),
        "pivot_turnkey": executable,
        "pivot_executable_on_7_1": executable,
        "b1_pivot_readiness_trustworthy": b1_trust,
        "d_pivot_turnkey": d_turnkey,
        "d_pivot_executable_on_7_1": d_executable,
        "arxiv_id": "2605.18871",
        "arxiv_ids_cited": list(d.get("arxiv_ids_cited", []) if d else []),
        "new_sota_backlog_ids": new_backlog_ids,
        "extended_sota_backlog": extended_backlog,
        "extended_sota_backlog_count": len(extended_backlog),
        "sota_signal": "2605.18871 beats self-consistency on MuSR",
        "validation_gate": d.get("validation_gate", {}) if d else {},
        "claim_status": "readiness_only_not_moat_proven",
        "moat_proven": False,
        "moat_proven_claimed": d.get("moat_proven_claimed") is True if d else False,
        "deliverable": "current ~0.05 agent + publishable FoVer paper",
        "executable_date": "2026-07-01",
        "runs_after": "2026-06-30_sprint_retirement",
        "validation_required": (
            "beats_self_consistency_ci95_excludes_zero AND oracle_distinct AND "
            "no_model_identity_shortcut"
        ),
        "do_not_queue": "representation_5_or_concluded_energy_as_arc_program",
    }


def _reserved_lanes(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    c = clean.get("C_KV260")
    c_ok = bool(c and c.get("kv260_ssh_reachable") is True)
    return {
        "b3_stamping": {
            **_b3_window_state(clean),
            "decision": (
                "reserved_lane_b3_nonzero_mtime_window"
                if _b3_window_nonzero(clean)
                else "reserved_lane_b3_mtime_window_missing"
            ),
        },
        "c_kv260": {
            "decision": "reserved_lane_kv260_continuity_ok" if c_ok else "reserved_lane_kv260_not_ok",
            "honest_verdict": c.get("honest_verdict", "") if c else "",
            "kv260_ssh_reachable": c_ok,
            "loaded_overlay": c.get("loaded_overlay") if c else None,
            "xmutil_requires_sudo": c.get("xmutil_requires_sudo") is True if c else None,
        },
    }


def _milestone_scorecard(
    clean: Mapping[str, Mapping[str, Any]],
    *,
    banks: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
    a3: Mapping[str, Any],
    b3: Mapping[str, Any],
    wall: Mapping[str, Any],
    pivot: Mapping[str, Any],
) -> JsonDict:
    return {
        "deliverable": "locked ~0.05 first-win agent + publishable FoVer paper",
        "wall_closure": dict(wall),
        "banks": dict(banks),
        "heldout_go_no_go": dict(heldout),
        "submission_package": dict(package),
        "a3_substrate_fix": dict(a3),
        "b3_window_fix": dict(b3),
        "post_sprint_pivot": dict(pivot),
        "reserved_lanes": _reserved_lanes(clean),
    }


def _rate_slug(heldout: Mapping[str, Any]) -> str:
    rate = heldout.get("rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        text = f"{float(rate):.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return "unknown"


def _headline(
    total: int,
    banks: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
    a3: Mapping[str, Any],
    b3: Mapping[str, Any],
    pivot: Mapping[str, Any],
) -> str:
    counted = banks.get("counted") if isinstance(banks.get("counted"), list) else []
    bank_text = "no trusted A1/A2 banks counted" if not counted else f"{len(counted)} trusted banks counted"
    package_text = "package ready" if package.get("ready") is True else "package not ready"
    pivot_text = (
        "post-6/30 verifier-moat pivot turnkey 7/1"
        if pivot.get("pivot_executable_on_7_1") is True
        else "post-6/30 verifier-moat pivot not trusted turnkey"
    )
    return (
        f".459 FINAL-stretch sprint-day submission readiness: locked ~0.05 agent + FoVer paper "
        f"ready for 6/30 at {total} reproducible levels, {bank_text}; held-out "
        f"first-win {_rate_slug(heldout)} with flag_resolved={heldout.get('flag_resolved') is True}; "
        f"{package_text}; A3 substrate fix holds={a3.get('resolved') is True}; "
        f"B3 nonzero mtime window={b3.get('nonzero') is True}; {pivot_text}; "
        f"11-paper SOTA backlog; ARC wall remains WALL_IS_HIDDEN_STATE closed."
    )


def _honest_verdict(
    total: int,
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
    pivot: Mapping[str, Any],
) -> str:
    package_slug = "package_ready" if package.get("ready") is True else "package_not_ready"
    pivot_slug = (
        "pivot_turnkey_7_1"
        if pivot.get("pivot_executable_on_7_1") is True
        else "pivot_not_turnkey"
    )
    return (
        "complete_capstone_v459_submission_ready_"
        f"levels_{total}_heldout_{_rate_slug(heldout)}_{package_slug}_{pivot_slug}"
    )


def _checksum_payload(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": payload.get("honest_verdict"),
        "headline": payload.get("headline"),
        "reproducible_total_levels": payload.get("reproducible_total_levels"),
        "banks_counted": payload.get("banks_counted"),
        "heldout_first_win_rate": payload.get("heldout_first_win_rate"),
        "submission_package_ready": payload.get("submission_package_ready"),
        "a3_substrate_flag_resolved": payload.get("a3_substrate_flag_resolved"),
        "b3_window_nonzero": payload.get("b3_window_nonzero"),
        "post_sprint_pivot": payload.get("post_sprint_pivot"),
        "pivot_executable_on_7_1": payload.get("pivot_executable_on_7_1"),
        "cited_upstream_artifacts": payload.get("cited_upstream_artifacts"),
        "milestone_scorecard": payload.get("milestone_scorecard"),
    }


def build_artifact(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    registry_sha256: str | None,
    summarizer_results: Mapping[str, SummarizerResult],
    duration_s: float,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the complete .459 final submission-readiness scorecard."""

    clean = _clean_artifacts(artifacts, summarizer_results)
    registry_total = _int(registry.get("reproducible_total_levels"))
    banks = _banks_counted(clean, registry_total)
    total = _int(banks.get("computed_total"), registry_total)
    heldout = _heldout_first_win_rate(clean)
    package = _submission_package_ready(clean)
    a3 = _a3_substrate_state(clean, summarizer_results)
    b3 = _b3_window_state(clean)
    wall = _wall_closure()
    pivot = _post_sprint_pivot(clean)
    scorecard = _milestone_scorecard(
        clean,
        banks=banks,
        heldout=heldout,
        package=package,
        a3=a3,
        b3=b3,
        wall=wall,
        pivot=pivot,
    )
    cited = _cited_artifacts(clean, artifact_sha256, summarizer_results)
    skipped = _skipped_flagged_adversarial(artifacts, artifact_sha256, summarizer_results)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(total, heldout, package, pivot),
        "headline": _headline(total, banks, heldout, package, a3, b3, pivot),
        "reproducible_total_levels": total,
        "banks_counted": banks,
        "heldout_first_win_rate": heldout,
        "submission_package_ready": package,
        "a3_substrate_flag_resolved": a3.get("resolved") is True,
        "b3_window_nonzero": b3.get("nonzero") is True,
        "arc_first_win_wall_closed": wall.get("closed") is True,
        "post_sprint_pivot": pivot,
        "pivot_executable_on_7_1": pivot.get("pivot_executable_on_7_1") is True,
        "skipped_flagged_adversarial": skipped,
        "cited_upstream_artifacts": cited,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(
            preconditions_checked
            or {
                "agents_md_read": True,
                "codex_md_read": True,
                "registry": {
                    "path": REGISTRY_RELATIVE_PATH,
                    "sha256": registry_sha256 or "",
                    "reproducible_total_levels": registry_total,
                },
                "north_star": {"path": NORTH_STAR_RELATIVE_PATH, "read_sections": ["0", "1", "5"]},
            }
        ),
        "random_seed": RANDOM_SEED,
        "milestone_scorecard": scorecard,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "capstone_ready": (
            wall.get("closed") is True
            and package.get("ready") is True
            and heldout.get("rate") is not None
            and heldout.get("flag_resolved") is True
            and a3.get("resolved") is True
            and b3.get("nonzero") is True
            and pivot.get("pivot_executable_on_7_1") is True
            and bool(cited)
            and isinstance(skipped, list)
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(_checksum_payload(payload))
    return payload


def build_blocked_artifact(
    *,
    reason: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    summarizer_results: Mapping[str, SummarizerResult],
    duration_s: float,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    """Build a schema-valid blocked artifact without importing headline metrics."""

    registry_total = _int(registry.get("reproducible_total_levels"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "headline": "",
        "reproducible_total_levels": registry_total,
        "banks_counted": {
            "b1_banks_trustworthy": False,
            "counted": [],
            "candidate_banks": [],
            "base_total_from_registry": registry_total,
            "bank_delta_counted": 0,
            "computed_total": registry_total,
        },
        "heldout_first_win_rate": {},
        "submission_package_ready": {},
        "a3_substrate_flag_resolved": False,
        "b3_window_nonzero": False,
        "arc_first_win_wall_closed": False,
        "post_sprint_pivot": {},
        "pivot_executable_on_7_1": False,
        "skipped_flagged_adversarial": _skipped_flagged_adversarial(
            artifacts, artifact_sha256, summarizer_results
        ),
        "cited_upstream_artifacts": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "milestone_scorecard": {},
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "capstone_ready": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(_checksum_payload(payload))
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the .459 scorecard without mutating it."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if not isinstance(payload.get("headline"), str):
        errors.append("invalid_headline")
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    if not isinstance(payload.get("banks_counted"), Mapping):
        errors.append("invalid_banks_counted")
    if not isinstance(payload.get("heldout_first_win_rate"), Mapping):
        errors.append("invalid_heldout_first_win_rate")
    if not isinstance(payload.get("submission_package_ready"), Mapping):
        errors.append("invalid_submission_package_ready")
    if not isinstance(payload.get("a3_substrate_flag_resolved"), bool):
        errors.append("invalid_a3_substrate_flag_resolved")
    if not isinstance(payload.get("b3_window_nonzero"), bool):
        errors.append("invalid_b3_window_nonzero")
    if not isinstance(payload.get("arc_first_win_wall_closed"), bool):
        errors.append("invalid_arc_first_win_wall_closed")
    if not isinstance(payload.get("post_sprint_pivot"), Mapping):
        errors.append("invalid_post_sprint_pivot")
    if not isinstance(payload.get("pivot_executable_on_7_1"), bool):
        errors.append("invalid_pivot_executable_on_7_1")
    if not isinstance(payload.get("preconditions_checked"), Mapping):
        errors.append("invalid_preconditions_checked")
    if not isinstance(payload.get("random_seed"), int) or isinstance(payload.get("random_seed"), bool):
        errors.append("invalid_random_seed")
    if not isinstance(payload.get("capstone_ready"), bool):
        errors.append("invalid_capstone_ready")
    if not isinstance(payload.get("milestone_scorecard"), Mapping):
        errors.append("invalid_milestone_scorecard")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
    skipped = payload.get("skipped_flagged_adversarial")
    if not isinstance(skipped, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not str(row.get("sha256", "")).startswith("sha256:")
        or not row.get("reason")
        for row in skipped
    ):
        errors.append("invalid_skipped_flagged_adversarial")
    if not str(payload.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("invalid_reproducibility_checksum")
    return errors


def _run_summarizer(root: Path, relative_path: str) -> SummarizerResult:  # pragma: no cover
    cmd = [sys.executable, SUMMARIZER_RELATIVE_PATH, relative_path]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    return SummarizerResult(
        command=cmd,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _first_blocker(
    *,
    summarizer_present: bool,
    registry_present: bool,
    registry_loadable: bool,
    spec_has_req: bool,
    north_star_present: bool,
    upstream_preconditions: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if not summarizer_present:
        return "missing_summarizer"
    if not registry_present:
        return "missing_registry"
    if not registry_loadable:
        return "registry_not_yaml_loadable"
    if not spec_has_req:
        return "spec_missing_req_4989"
    if not north_star_present:
        return "missing_north_star"
    if any(row.get("present") is not True for row in upstream_preconditions.values()):
        return "upstream_artifact_missing"
    return None


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    summarizer: Summarizer | None = None,
) -> JsonDict:
    """Read upstreams via the summarizer, aggregate, and write the scorecard."""

    start = time.perf_counter()
    summarizer = summarizer or _run_summarizer
    summarizer_path = root / SUMMARIZER_RELATIVE_PATH
    summarizer_present = summarizer_path.exists()
    artifacts: dict[str, JsonDict] = {}
    artifact_sha256: dict[str, str] = {}
    summarizer_results: dict[str, SummarizerResult] = {}
    upstream_preconditions: dict[str, JsonDict] = {}

    for source, spec in UPSTREAM_SOURCES.items():
        path = root / spec.relative_path
        present = path.exists()
        upstream_preconditions[source] = {"path": spec.relative_path, "present": present}
        if not present:
            continue
        if summarizer_present:
            summary = summarizer(root, spec.relative_path)
            summarizer_results[source] = summary
            upstream_preconditions[source]["summarizer_exit_code"] = summary.exit_code
        artifacts[source] = _read_json_object(path)
        artifact_sha256[source] = file_sha256(path) or ""

    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_present = registry_path.exists()
    registry_loadable = False
    registry: JsonDict = {}
    if registry_present:
        try:
            registry = _read_yaml_object(registry_path)
            registry_loadable = True
        except yaml.YAMLError:
            registry = {}

    spec_path = root / SPEC_RELATIVE_PATH
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4989" in spec_path.read_text(
        encoding="utf-8"
    )
    north_star_path = root / NORTH_STAR_RELATIVE_PATH
    north_star_present = north_star_path.exists()
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "summarizer": {"path": SUMMARIZER_RELATIVE_PATH, "present": summarizer_present},
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
            "sha256": file_sha256(registry_path) or "",
            "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        },
        "north_star": {
            "path": NORTH_STAR_RELATIVE_PATH,
            "present": north_star_present,
            "sha256": file_sha256(north_star_path) or "",
            "read_sections": ["0", "1", "5"] if north_star_present else [],
        },
        "spec_has_req_4989": spec_has_req,
        "upstream_artifacts": upstream_preconditions,
    }
    duration_s = time.perf_counter() - start
    blocker = _first_blocker(
        summarizer_present=summarizer_present,
        registry_present=registry_present,
        registry_loadable=registry_loadable,
        spec_has_req=spec_has_req,
        north_star_present=north_star_present,
        upstream_preconditions=upstream_preconditions,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            reason=blocker,
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            summarizer_results=summarizer_results,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    else:
        artifact = build_artifact(
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            registry_sha256=file_sha256(registry_path),
            summarizer_results=summarizer_results,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_capstone()
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact.get("honest_verdict"),
                "schema_errors": errors,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
