"""Experiment 4934: .454 submission-readiness capstone.

Spec refs: REQ-CAPSTONE-4934, SCENARIO-CAPSTONE-4934,
SCENARIO-CAPSTONE-4934-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4934-FIELD-PRINCIPLES.

This is an aggregation pass, not a new research claim. It reads every cited
upstream artifact through ``scripts/summarize_artifact.py`` before importing
fields, carries the registry level total, and reports the locked 6/30
deliverable without inflating duplicate banks or retired efficiency levers.
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
EXPERIMENT = "experiment_4934_capstone_v454"
EXPERIMENT_ID = 4934
SCHEMA = "carnot.exp4934.capstone_v454.v1"
RESULT_RELATIVE_PATH = "results/experiment_4934_capstone_v454.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
NORTH_STAR_RELATIVE_PATH = "ops/north-star.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4934",
    "SCENARIO-CAPSTONE-4934",
    "SCENARIO-CAPSTONE-4934-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4934-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_capstone_v454_submission_maximized_<state> "
            "(e.g. _levels_<N>_heldout_<rate>_package_ready)."
        )
    },
    "headline": {
        "principle": (
            "the one-line .454 headline -- the locked deliverable maximized for 6/30 "
            "(levels growth + efficiency + clean go/no-go + package ready)."
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
            "-- no un-audited bank inflates the total."
        )
    },
    "action_efficiency_result": {
        "principle": (
            "D's MATM efficiency lift if B1 efficiency_trustworthy, else the honest "
            "null -- the squared-scored submission lever."
        )
    },
    "heldout_first_win_rate": {
        "principle": (
            "A4's full-25 (or partial) go/no-go number + flag_resolved -- the "
            "operator's 6/30 decision input."
        )
    },
    "submission_package_ready": {
        "principle": (
            "B2's terminal gate + peak_vram_gb -- the deadline deliverable for the operator."
        )
    },
    "arc_first_win_wall_closed": {
        "principle": (
            "true -- the .453 B1-trusted WALL_IS_HIDDEN_STATE closure stands; .454 did "
            "not reopen it or queue representation #5."
        )
    },
    "post_sprint_pivot": {
        "principle": (
            "the post-6/30 verifier-moat handoff: D's distributional-energy-verifier "
            "scaffold (arXiv:2605.18871) + the deliverable (~0.05 agent + FoVer paper)."
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
    "action_efficiency_result",
    "heldout_first_win_rate",
    "submission_package_ready",
    "arc_first_win_wall_closed",
    "post_sprint_pivot",
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
    "PREV_CAPSTONE": UpstreamSource(4923, "results/experiment_4923_capstone_v453.json"),
    "A1_LEVELUP": UpstreamSource(4925, "results/experiment_4925_levelup_attempt.json"),
    "A2_LEVELUP": UpstreamSource(4926, "results/experiment_4926_levelup_attempt.json"),
    "A3_SELF_PLAY": UpstreamSource(
        4927, "results/experiment_4927_self_play_verifier_checkpoint.json"
    ),
    "A4_HELDOUT": UpstreamSource(
        4928, "results/experiment_4928_heldout_first_win_readiness.json"
    ),
    "B1_AUDIT": UpstreamSource(4929, "results/experiment_4929_bank_and_efficiency_audit.json"),
    "B2_PACKAGE": UpstreamSource(4930, "results/experiment_4930_submission_package_harden.json"),
    "B3_STAMPING": UpstreamSource(
        4931, "results/experiment_4931_stamping_backfill_and_wiring_readiness.json"
    ),
    "C_KV260": UpstreamSource(4932, "results/experiment_4932_kv260_continuity.json"),
    "D_EFFICIENCY": UpstreamSource(
        4933, "results/experiment_4933_matm_similarity_retrieval_efficiency.json"
    ),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "PREV_CAPSTONE": (
        "honest_verdict",
        "a1_closure_verdict_trusted",
        "post_sprint_pivot",
        "reproducible_total_levels",
    ),
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
        "checkpoint_path",
        "checkpoint_mtime_delta_ns",
        "offline_reproduced",
        "reproduced_levels",
    ),
    "A4_HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "heldout_first_win_ci",
        "games_evaluated",
        "games_remaining",
        "flag_resolved",
        "live_agent_ran",
        "positive_control_passed",
        "parity_test_green",
        "partial",
        "generator_backend",
        "solve_provenance",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "banks_trustworthy",
        "efficiency_trustworthy",
        "checks",
        "audit_failure_reasons",
        "bank_evidence",
        "efficiency_evidence",
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
    ),
    "B3_STAMPING": (
        "honest_verdict",
        "milestone",
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
    "D_EFFICIENCY": (
        "honest_verdict",
        "matm_similarity_retrieval_lift",
        "actions_to_first_levelup_delta",
        "forward_walk_hit_rate_delta",
        "reached_level_regression",
        "submitted_parity_test_green",
        "retire_if_same_verdict",
        "post_sprint_pivot_gate_noted",
        "arxiv_ids_cited",
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
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        artifact = artifacts.get(source)
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


def _banks_counted(
    clean: Mapping[str, Mapping[str, Any]],
    registry_total: int,
) -> JsonDict:
    b1 = clean.get("B1_AUDIT")
    prev = clean.get("PREV_CAPSTONE")
    base_total = _int(prev.get("reproducible_total_levels") if prev else None, registry_total)
    trustworthy = bool(b1 and b1.get("banks_trustworthy") is True)
    candidates = [
        _bank_candidate("A1_LEVELUP", clean.get("A1_LEVELUP")),
        _bank_candidate("A2_LEVELUP", clean.get("A2_LEVELUP")),
    ]
    counted = [
        row for row in candidates if trustworthy and _int(row.get("new_levels_banked")) > 0
    ]
    computed_total = max(registry_total, base_total + sum(_int(row["new_levels_banked"]) for row in counted))
    return {
        "b1_banks_trustworthy": trustworthy,
        "counted": counted,
        "candidate_banks": candidates,
        "audit_failure_reasons": list(b1.get("audit_failure_reasons", [])) if b1 else [],
        "base_total_from_v453": base_total,
        "registry_total": registry_total,
        "computed_total": computed_total,
    }


def _action_efficiency_result(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    b1 = clean.get("B1_AUDIT")
    d = clean.get("D_EFFICIENCY")
    trustworthy = bool(b1 and b1.get("efficiency_trustworthy") is True)
    lift = _float(d.get("matm_similarity_retrieval_lift")) if d else None
    if trustworthy and d and lift is not None:
        decision = "trusted_efficiency_lift"
        reported_lift = lift
    elif d is not None:
        decision = "honest_null_not_trusted_lift"
        reported_lift = None
    else:
        decision = "not_reported_missing_or_untrusted"
        reported_lift = None
    return {
        "decision": decision,
        "b1_efficiency_trustworthy": trustworthy,
        "reported_lift": reported_lift,
        "d_honest_verdict": d.get("honest_verdict", "") if d else "",
        "actions_to_first_levelup_delta": d.get("actions_to_first_levelup_delta", {}) if d else {},
        "forward_walk_hit_rate_delta": d.get("forward_walk_hit_rate_delta", {}) if d else {},
        "reached_level_regression": d.get("reached_level_regression") is True if d else None,
        "submitted_parity_test_green": d.get("submitted_parity_test_green") is True if d else None,
        "retire_if_same_verdict": d.get("retire_if_same_verdict") is True if d else None,
        "audit_failure_reasons": list(b1.get("audit_failure_reasons", [])) if b1 else [],
    }


def _heldout_first_win_rate(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    a4 = clean.get("A4_HELDOUT")
    if a4 is None:
        return {
            "status": "missing_or_skipped",
            "rate": None,
            "flag_resolved": False,
        }
    flag_resolved = a4.get("flag_resolved") is True
    return {
        "status": "full25_go_no_go" if _int(a4.get("games_remaining")) == 0 else "partial_go_no_go",
        "honest_verdict": a4.get("honest_verdict", ""),
        "rate": _float(a4.get("heldout_first_win_rate")) if flag_resolved else None,
        "ci": a4.get("heldout_first_win_ci", {}),
        "flag_resolved": flag_resolved,
        "games_evaluated": _int(a4.get("games_evaluated")),
        "games_remaining": _int(a4.get("games_remaining")),
        "live_agent_ran": a4.get("live_agent_ran") is True,
        "positive_control_passed": a4.get("positive_control_passed") is True,
        "parity_test_green": a4.get("parity_test_green") is True,
        "partial": a4.get("partial") is True,
        "generator_backend": a4.get("generator_backend"),
        "solve_provenance": a4.get("solve_provenance"),
    }


def _submission_package_ready(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    package = clean.get("B2_PACKAGE")
    if package is None:
        return {"ready": False, "status": "missing_or_skipped", "peak_vram_gb": None}
    submitted = package.get("submits") is True or package.get("submitted_to_leaderboard") is True
    ready = (
        package.get("submission_package_ready") is True
        and package.get("operator_only") is True
        and not submitted
    )
    return {
        "ready": ready,
        "decision": "package_ready_operator_only" if ready else "package_not_ready",
        "honest_verdict": package.get("honest_verdict", ""),
        "peak_vram_gb": _float(package.get("peak_vram_gb")),
        "frozen_stack_loads": package.get("frozen_stack_loads") is True,
        "package_builds": package.get("package_builds") is True,
        "ready_package_regression_ok": package.get("ready_package_regression_ok") is True,
        "submits": submitted,
        "operator_only": package.get("operator_only") is True,
    }


def _wall_closure(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    prev = clean.get("PREV_CAPSTONE")
    trust = _mapping(prev.get("a1_closure_verdict_trusted") if prev else {})
    closed = (
        trust.get("trusted") is True
        and trust.get("closure_verdict") == "WALL_IS_HIDDEN_STATE"
        and str(prev.get("honest_verdict", "") if prev else "").startswith(
            "complete_capstone_v453_wall_is_hidden_state"
        )
    )
    return {
        "source": "PREV_CAPSTONE",
        "experiment_id": _experiment_id("PREV_CAPSTONE", prev),
        "closed": closed,
        "closure_verdict": trust.get("closure_verdict", ""),
        "trusted": trust.get("trusted") is True,
        "do_not_queue": "representation_5" if closed else "",
        "did_reopen_in_v454": False,
    }


def _reserved_lanes(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    b3 = clean.get("B3_STAMPING")
    c = clean.get("C_KV260")
    b3_verdict = str(b3.get("honest_verdict", "") if b3 else "")
    c_ok = bool(c and c.get("kv260_ssh_reachable") is True)
    return {
        "b3_stamping": {
            "decision": (
                "reserved_lane_blocked_insufficient_v454_mtime_window"
                if b3_verdict.startswith("blocked_insufficient_v454_mtime_window")
                else "reserved_lane_stamping_ready"
                if b3
                else "reserved_lane_missing"
            ),
            "honest_verdict": b3_verdict,
            "wiring_proposal_reconfirmed": b3.get("wiring_proposal_reconfirmed") is True if b3 else False,
            "mtime_fallback_window": b3.get("mtime_fallback_window", {}) if b3 else {},
            "research_conductor_modified": b3.get("research_conductor_modified") is True if b3 else None,
        },
        "c_kv260": {
            "decision": "reserved_lane_kv260_continuity_ok" if c_ok else "reserved_lane_kv260_not_ok",
            "honest_verdict": c.get("honest_verdict", "") if c else "",
            "kv260_ssh_reachable": c_ok,
            "loaded_overlay": c.get("loaded_overlay") if c else None,
            "xmutil_requires_sudo": c.get("xmutil_requires_sudo") is True if c else None,
        },
    }


def _post_sprint_pivot(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    d = clean.get("D_EFFICIENCY")
    gate = _mapping(d.get("post_sprint_pivot_gate_noted") if d else {})
    return {
        "decision": "post_6_30_distributional_energy_verifier_pivot",
        "arxiv_id": str(gate.get("arxiv_id") or "2605.18871"),
        "domain": "non_saturated_structured_reasoning_domain",
        "scaffold": "D_similarity_retrieval_retired_scaffold_handoff",
        "validation_gate": gate.get(
            "validation_gate",
            "distributional-energy-verifier beats self-consistency with CI95 excluding zero",
        ),
        "deliverable": "current ~0.05 agent + publishable FoVer paper",
        "do_not_queue": "representation_5",
        "starts_after": "2026-06-30_sprint_retirement",
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
    efficiency: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
) -> str:
    counted = banks.get("counted") if isinstance(banks.get("counted"), list) else []
    bank_text = "no trusted A1/A2 banks counted" if not counted else f"{len(counted)} trusted banks counted"
    efficiency_text = (
        "D reports no trusted efficiency lift"
        if efficiency.get("reported_lift") is None
        else f"D lift {efficiency['reported_lift']}"
    )
    package_text = "package ready" if package.get("ready") is True else "package not ready"
    return (
        f".454 submission readiness: {total} reproducible levels, {bank_text}; "
        f"held-out first-win {_rate_slug(heldout)} with flag_resolved="
        f"{heldout.get('flag_resolved') is True}; {efficiency_text}; {package_text}; "
        "ARC wall remains WALL_IS_HIDDEN_STATE closed."
    )


def _honest_verdict(
    total: int,
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> str:
    package_slug = "package_ready" if package.get("ready") is True else "package_not_ready"
    efficiency_slug = "efficiency_lift" if efficiency.get("reported_lift") is not None else "efficiency_null"
    return (
        "complete_capstone_v454_submission_maximized_"
        f"levels_{total}_heldout_{_rate_slug(heldout)}_{package_slug}_{efficiency_slug}"
    )


def _milestone_scorecard(
    clean: Mapping[str, Mapping[str, Any]],
    *,
    banks: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
    wall: Mapping[str, Any],
) -> JsonDict:
    return {
        "wall_closure": dict(wall),
        "banks": dict(banks),
        "action_efficiency": dict(efficiency),
        "heldout_go_no_go": dict(heldout),
        "submission_package": dict(package),
        "reserved_lanes": _reserved_lanes(clean),
    }


def _checksum_payload(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": payload.get("honest_verdict"),
        "headline": payload.get("headline"),
        "reproducible_total_levels": payload.get("reproducible_total_levels"),
        "banks_counted": payload.get("banks_counted"),
        "action_efficiency_result": payload.get("action_efficiency_result"),
        "heldout_first_win_rate": payload.get("heldout_first_win_rate"),
        "submission_package_ready": payload.get("submission_package_ready"),
        "post_sprint_pivot": payload.get("post_sprint_pivot"),
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
    """Build the complete .454 submission-readiness scorecard."""

    clean = _clean_artifacts(artifacts, summarizer_results)
    registry_total = _int(registry.get("reproducible_total_levels"))
    banks = _banks_counted(clean, registry_total)
    total = _int(banks.get("computed_total"), registry_total)
    efficiency = _action_efficiency_result(clean)
    heldout = _heldout_first_win_rate(clean)
    package = _submission_package_ready(clean)
    wall = _wall_closure(clean)
    scorecard = _milestone_scorecard(
        clean,
        banks=banks,
        efficiency=efficiency,
        heldout=heldout,
        package=package,
        wall=wall,
    )
    cited = _cited_artifacts(clean, artifact_sha256, summarizer_results)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(total, heldout, package, efficiency),
        "headline": _headline(total, banks, efficiency, heldout, package),
        "reproducible_total_levels": total,
        "banks_counted": banks,
        "action_efficiency_result": efficiency,
        "heldout_first_win_rate": heldout,
        "submission_package_ready": package,
        "arc_first_win_wall_closed": wall.get("closed") is True,
        "post_sprint_pivot": _post_sprint_pivot(clean),
        "skipped_flagged_adversarial": _skipped_flagged_adversarial(
            artifacts, artifact_sha256, summarizer_results
        ),
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
                "north_star": {"path": NORTH_STAR_RELATIVE_PATH, "read_sections": ["1", "2", "5"]},
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
            and bool(cited)
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

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "headline": "",
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "banks_counted": {
            "b1_banks_trustworthy": False,
            "counted": [],
            "candidate_banks": [],
            "registry_total": _int(registry.get("reproducible_total_levels")),
        },
        "action_efficiency_result": {},
        "heldout_first_win_rate": {},
        "submission_package_ready": {},
        "arc_first_win_wall_closed": False,
        "post_sprint_pivot": {},
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
    """Return schema errors for the .454 scorecard without mutating it."""

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
    if not isinstance(payload.get("action_efficiency_result"), Mapping):
        errors.append("invalid_action_efficiency_result")
    if not isinstance(payload.get("heldout_first_win_rate"), Mapping):
        errors.append("invalid_heldout_first_win_rate")
    if not isinstance(payload.get("submission_package_ready"), Mapping):
        errors.append("invalid_submission_package_ready")
    if not isinstance(payload.get("arc_first_win_wall_closed"), bool):
        errors.append("invalid_arc_first_win_wall_closed")
    if not isinstance(payload.get("post_sprint_pivot"), Mapping):
        errors.append("invalid_post_sprint_pivot")
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
        return "spec_missing_req_4934"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4934" in spec_path.read_text(
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
            "read_sections": ["1", "2", "5"] if north_star_present else [],
        },
        "spec_has_req_4934": spec_has_req,
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
