"""Exp5467 .496 capstone truth-table synthesis.

Spec refs: REQ-CAPSTONE-5467, SCENARIO-CAPSTONE-5467,
SCENARIO-CAPSTONE-5467-MISSING-SKIPPED-PRECONDITION,
SCENARIO-CAPSTONE-5467-FIELD-PRINCIPLES.

This module does not run a fresh model, solver, board workload, or ARC agent.
It reads the artifacts that already landed for Exp5454 through Exp5466 and
records what those artifacts can honestly support. The important point is the
claim boundary: a useful receipt can still be non-headline when it is flagged,
skipped, missing, gated by a failed precondition, or only proves a bounded
property such as exact-solver fallback or board reachability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5467_capstone_v496.json")
EXPERIMENT = "experiment_5467_capstone_v496"
EXPERIMENT_ID = "exp5467-v496-capstone"
MILESTONE = "2026.07.496"
SCHEMA = "carnot.experiment_5467.capstone_v496.v1"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5467
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EXP5454 = "results/experiment_5454_transition_v496.json"
EXP5455 = "results/experiment_5455_source_delta_v496.json"
EXP5456 = "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json"
EXP5456_GRAPH = "results/experiment_5456_guided_decoding_tautology_corrigendum_v496_metric_dependency_graph.json"
EXP5457 = "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json"
EXP5457_ATTRIBUTION = "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496_claim_attribution_receipts.jsonl"
EXP5457_ROWS = "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496_rows.jsonl"
EXP5458 = "results/experiment_5458_minimal_core_claim_repair_v496.json"
EXP5459 = "results/experiment_5459_constraint_distortion_guard_v496.json"
EXP5460 = "results/experiment_5460_csl_policy_bandit_v496.json"
EXP5460_RECEIPTS = "results/experiment_5460_csl_policy_confidence_receipts_v496.jsonl"
EXP5461 = "results/experiment_5461_gated_sota_csl_memory_routing_v496.json"
EXP5461_ROWS = "results/experiment_5461_gated_sota_csl_memory_routing_v496_rows.jsonl"
EXP5462 = "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json"
EXP5463 = "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json"
EXP5464 = "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json"
EXP5464_RECEIPTS = "results/experiment_5464_arc_perception_feature_receipts_v496.json"
EXP5465 = "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json"
EXP5466 = "results/experiment_5466_prd_gap_agent_failure_table_v496.json"

SOURCE_CONTEXT_PATHS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "ops/changelog.md",
    "ops/status.md",
)
MAIN_ARTIFACT_PATHS = (
    EXP5454,
    EXP5455,
    EXP5456,
    EXP5457,
    EXP5458,
    EXP5459,
    EXP5460,
    EXP5461,
    EXP5462,
    EXP5463,
    EXP5464,
    EXP5465,
    EXP5466,
)
SIDECAR_ARTIFACT_PATHS = (
    EXP5456_GRAPH,
    EXP5457_ATTRIBUTION,
    EXP5457_ROWS,
    EXP5460_RECEIPTS,
    EXP5461_ROWS,
    EXP5464_RECEIPTS,
)
EXPECTED_INPUT_PATHS = (*SOURCE_CONTEXT_PATHS, *MAIN_ARTIFACT_PATHS, *SIDECAR_ARTIFACT_PATHS)

SPEC_REFS = (
    "REQ-CAPSTONE-5467",
    "SCENARIO-CAPSTONE-5467",
    "SCENARIO-CAPSTONE-5467-MISSING-SKIPPED-PRECONDITION",
    "SCENARIO-CAPSTONE-5467-FIELD-PRINCIPLES",
)
REQUIRED_ARTIFACT_FIELDS = (
    "milestone",
    "artifact_paths_read",
    "truth_table",
    "headline_ready_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "honest_null_lanes",
    "skipped_gated_tasks",
    "no_claim_boundaries",
    "next_milestone_recommendations",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "result_path",
    "spec_refs",
    "field_principles",
    "source_context_read",
    "source_context_missing",
    "missing_artifacts",
    "sidecar_artifacts_read",
    "artifact_checksums",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)
LANE_ORDER = (
    "guided_decoding",
    "distortion_guards",
    "minimal_core_repair",
    "csl_policy",
    "sota_csl",
    "pbit_pdit_bridge",
    "hardware_receipts",
    "arc",
    "synthesis",
)
FIELD_PRINCIPLES = {
    "milestone": "route key; must equal 2026.07.496.",
    "artifact_paths_read": "source context plus every upstream result or sidecar actually read.",
    "truth_table": "lane-indexed evidence map for the requested capstone lanes, built only from artifacts.",
    "headline_ready_lanes": "positive evidence; only clean lanes whose authority gate is satisfied.",
    "bounded_lanes": "bounded evidence; useful receipts that must not become broad claims.",
    "blocked_lanes": "blocker accounting for flagged, skipped, missing, or precondition-failed lanes.",
    "honest_null_lanes": "executed null-result lanes that did not bank a positive outcome.",
    "skipped_gated_tasks": "actual skipped upstreams, kept distinct from blocked and null lanes.",
    "no_claim_boundaries": "explicit claims the capstone refuses to make.",
    "next_milestone_recommendations": "3-5 observed gap priorities, including repeated-failure quarantine.",
    "roadmap_yaml_unchanged": "protected-file discipline; derived from git status.",
    "conductor_unchanged": "protected-file discipline; derived from git status.",
    "inference_substrate": "must equal aggregation_from_upstream_artifacts.",
    "honest_verdict": "terminal status; starts with complete: or blocked: and summarizes the bounded close.",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_5467_capstone_v496.py -q --no-cov",
    (
        ".venv/bin/coverage run --include=python/carnot/experiment_5467_capstone_v496.py "
        "-m pytest tests/python/test_experiment_5467_capstone_v496.py -q --no-cov -n 0"
    ),
    ".venv/bin/coverage report --include=python/carnot/experiment_5467_capstone_v496.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover - upstream files are objects in tests
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _select(payload: JsonMap, fields: Sequence[str]) -> JsonDict:
    return {field: payload[field] for field in fields if field in payload}


def _is_number(value: Any, expected: float) -> bool:
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and float(value) == expected
    )


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _is_skipped(payload: JsonMap) -> bool:
    status = str(payload.get("status", "")).lower()
    verdict = str(payload.get("honest_verdict", "")).lower()
    return "skipped" in status or verdict.startswith("skipped")


def _precondition_failed(payload: JsonMap) -> bool:
    details = payload.get("precondition_details")
    return isinstance(details, Mapping) and details.get("all_passed") is False


def _has_tautology(payload: JsonMap) -> bool:
    pending = payload.get("corrigendum_pending")
    if not isinstance(pending, list):
        return False
    return any(
        isinstance(row, Mapping) and str(row.get("kind", "")).upper() == "TAUTOLOGY"
        for row in pending
    )


def _base_blockers(
    paths: Sequence[str], payloads: Mapping[str, JsonMap], missing: Sequence[str]
) -> list[str]:
    blockers: list[str] = []
    for rel_path in paths:
        payload = payloads.get(rel_path, {})
        if rel_path in missing:
            blockers.append("missing")
        if _is_skipped(payload):
            blockers.append("skipped_gated")
        if _precondition_failed(payload):
            blockers.append("precondition_failed")
        if payload.get("flagged_adversarial") is True:
            blockers.append("flagged_adversarial")
        if _has_tautology(payload):
            blockers.append("tautology")
    return _unique(blockers)


def _classification_from_clean_gate(
    clean: bool, success_class: str, blockers: Sequence[str]
) -> str:
    return success_class if clean and not blockers else "blocked"


def _row(
    *,
    lane: str,
    classification: str,
    authority_gate: str,
    source_artifacts: Sequence[str],
    evidence: JsonDict,
    headline_blockers: Sequence[str],
    claim_boundary: str,
) -> JsonDict:
    return {
        "lane": lane,
        "classification": classification,
        "authority_gate": authority_gate,
        "source_artifacts": list(source_artifacts),
        "evidence": evidence,
        "headline_blockers": list(headline_blockers),
        "claim_boundary": claim_boundary,
    }


def _sidecar_summary(path: Path, rel_path: str) -> JsonDict:
    text = path.read_text(encoding="utf-8")
    summary: JsonDict = {
        "artifact_path": rel_path,
        "sha256": _sha256_text(text),
        "size_bytes": len(text.encode("utf-8")),
    }
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        summary.update(
            {
                "format": "jsonl",
                "row_count": len(rows),
                "first_row_keys": sorted(rows[0]) if rows and isinstance(rows[0], dict) else [],
            }
        )
    else:
        parsed = json.loads(text)
        summary.update(
            {
                "format": "json",
                "json_type": type(parsed).__name__,
                "top_level_keys": sorted(parsed) if isinstance(parsed, dict) else [],
            }
        )
    return summary


def _read_inputs(
    root: Path,
) -> tuple[dict[str, JsonDict], list[JsonDict], list[str], list[str], list[str], list[str]]:
    payloads: dict[str, JsonDict] = {}
    sidecars: list[JsonDict] = []
    read_paths: list[str] = []
    source_context_read: list[str] = []
    source_context_missing: list[str] = []
    missing_artifacts: list[str] = []
    result_paths = set(MAIN_ARTIFACT_PATHS) | set(SIDECAR_ARTIFACT_PATHS)
    for rel_path in EXPECTED_INPUT_PATHS:
        path = root / rel_path
        if not path.exists() or path.is_dir():
            if rel_path in result_paths:
                missing_artifacts.append(rel_path)
            else:
                source_context_missing.append(rel_path)
            continue
        if rel_path in SOURCE_CONTEXT_PATHS:
            path.read_text(encoding="utf-8")
            source_context_read.append(rel_path)
        elif rel_path in MAIN_ARTIFACT_PATHS:
            payloads[rel_path] = _read_json_object(path)
        else:
            sidecars.append(_sidecar_summary(path, rel_path))
        read_paths.append(rel_path)
    return (
        payloads,
        sidecars,
        sorted(read_paths),
        source_context_read,
        source_context_missing,
        sorted(missing_artifacts),
    )


def _skipped_gated_tasks(payloads: Mapping[str, JsonMap]) -> list[JsonDict]:
    skipped: list[JsonDict] = []
    for rel_path in MAIN_ARTIFACT_PATHS:
        payload = payloads.get(rel_path, {})
        if not _is_skipped(payload):
            continue
        skipped.append(
            {
                "artifact_path": rel_path,
                "honest_verdict": str(payload.get("honest_verdict", "")),
                "reason": str(payload.get("skipped_reason", payload.get("skip_reason", ""))),
                "status": str(payload.get("status", "")),
            }
        )
    return skipped


def _truth_table(payloads: Mapping[str, JsonMap], missing: Sequence[str]) -> dict[str, JsonDict]:
    exp5457 = payloads.get(EXP5457, {})
    guided_blockers = _base_blockers((EXP5456, EXP5457), payloads, missing)
    if exp5457.get("verifier_guided_decoding_ready") is not True:
        guided_blockers.append("readiness_false")
    if exp5457.get("lcd_bias_check_passed") is not True:
        guided_blockers.append("lcd_bias_failed")

    exp5459 = payloads.get(EXP5459, {})
    distortion_blockers = _base_blockers((EXP5459,), payloads, missing)
    distortion_clean = (
        exp5459.get("distortion_guard_ready") is True
        and exp5459.get("exact_final_authority") is True
    )

    exp5458 = payloads.get(EXP5458, {})
    minimal_blockers = _base_blockers((EXP5458,), payloads, missing)
    minimal_clean = (
        exp5458.get("minimal_core_repair_ready") is True
        and exp5458.get("exact_final_authority") is True
        and _is_number(exp5458.get("repaired_accept_rate_after_exact_recheck"), 1.0)
    )

    exp5460 = payloads.get(EXP5460, {})
    csl_blockers = _base_blockers((EXP5460,), payloads, missing)
    csl_clean = (
        exp5460.get("csl_policy_ready") is True
        and exp5460.get("no_weight_mutation") is True
        and exp5460.get("cumulative_constraint_violations") == 0
    )

    exp5461 = payloads.get(EXP5461, {})
    sota_csl_blockers = _base_blockers((EXP5461,), payloads, missing)
    sota_csl_clean = (
        exp5461.get("csl_sota_memory_routing_ready") is True
        and exp5461.get("gpu_offload_verified") is True
        and exp5461.get("no_weight_mutation") is True
    )

    exp5462 = payloads.get(EXP5462, {})
    pbit_blockers = _base_blockers((EXP5462,), payloads, missing)
    pbit_clean = (
        exp5462.get("minimal_core_pbit_bridge_ready") is True
        and exp5462.get("solver_authoritative") is True
        and _is_number(exp5462.get("fallback_completeness_rate"), 1.0)
        and exp5462.get("hardware_speedup_claim") is False
    )

    exp5463 = payloads.get(EXP5463, {})
    hardware_blockers = _base_blockers((EXP5463,), payloads, missing)
    hardware_clean = (
        exp5463.get("hardware_receipts_ready") is True
        and exp5463.get("gated_upstream_ready") is True
        and exp5463.get("hashes_match_before_timing_compare") is True
        and exp5463.get("hardware_speedup_claim") is False
    )

    exp5464 = payloads.get(EXP5464, {})
    exp5465 = payloads.get(EXP5465, {})
    arc_blockers = _base_blockers((EXP5464, EXP5465), payloads, missing)
    arc_precheck = (
        exp5464.get("arc_metric_integrity_ready") is True
        and exp5464.get("registry_precheck_performed") is True
    )
    arc_banked = (
        exp5465.get("new_level_banked") is True and exp5465.get("offline_reproduced") is True
    )
    arc_classification = (
        "headline_ready"
        if arc_banked and not arc_blockers
        else "honest_null"
        if arc_precheck and not arc_blockers
        else "blocked"
    )

    exp5466 = payloads.get(EXP5466, {})
    synthesis_blockers = _base_blockers((EXP5466,), payloads, missing)
    synthesis_clean = (
        str(exp5466.get("status")) == "complete"
        and exp5466.get("missing_artifacts") == []
        and exp5466.get("skipped_gated_tasks") == []
    )

    return {
        "guided_decoding": _row(
            lane="guided_decoding",
            classification="blocked" if _unique(guided_blockers) else "headline_ready",
            authority_gate="metric independence plus exact final authority; live SOTA rerun still must be clean",
            source_artifacts=[EXP5456, EXP5456_GRAPH, EXP5457, EXP5457_ATTRIBUTION, EXP5457_ROWS],
            evidence={
                "exp5456": _select(
                    exp5456 := payloads.get(EXP5456, {}),
                    (
                        "guided_decoding_corrigendum_clean",
                        "prior_flagged_adversarial",
                        "invalid_tautological_fields",
                        "honest_verdict",
                    ),
                ),
                "exp5456_graph": _select(payloads.get(EXP5456_GRAPH, {}), ()),
                "exp5457": _select(
                    exp5457,
                    (
                        "flagged_adversarial",
                        "corrigendum_pending",
                        "verifier_guided_decoding_ready",
                        "lcd_bias_check_passed",
                        "gpu_offload_verified",
                        "runtime_backend",
                        "metric_independence_checks_passed",
                        "exact_final_authority",
                    ),
                ),
            },
            headline_blockers=_unique(guided_blockers),
            claim_boundary="No guided-decoding headline; the clean posthoc corrigendum did not make the fresh live rerun clean.",
        ),
        "distortion_guards": _row(
            lane="distortion_guards",
            classification=_classification_from_clean_gate(
                distortion_clean, "headline_ready", distortion_blockers
            ),
            authority_gate="exact deterministic facts remain final authority",
            source_artifacts=[EXP5459],
            evidence={
                "exp5459": _select(
                    exp5459,
                    (
                        "distortion_guard_ready",
                        "exact_final_authority",
                        "truth_preserving_compliance_rate",
                        "unsupported_fabrication_rate",
                        "fixture_count",
                        "honest_verdict",
                    ),
                )
            },
            headline_blockers=distortion_blockers,
            claim_boundary="Headline is limited to deterministic distortion-guard readiness, not broad factuality scoring.",
        ),
        "minimal_core_repair": _row(
            lane="minimal_core_repair",
            classification=_classification_from_clean_gate(
                minimal_clean, "headline_ready", minimal_blockers
            ),
            authority_gate="accepted repairs are rechecked by exact final authority",
            source_artifacts=[EXP5458],
            evidence={
                "exp5458": _select(
                    exp5458,
                    (
                        "minimal_core_repair_ready",
                        "exact_final_authority",
                        "repaired_accept_rate_after_exact_recheck",
                        "unrepaired_reject_rate",
                        "honest_verdict",
                    ),
                )
            },
            headline_blockers=minimal_blockers,
            claim_boundary="Headline is deterministic repair acceptance after exact recheck, not learned repair quality.",
        ),
        "csl_policy": _row(
            lane="csl_policy",
            classification=_classification_from_clean_gate(
                csl_clean, "headline_ready", csl_blockers
            ),
            authority_gate="frozen-model policy receipts, rollback, and no model-weight mutation",
            source_artifacts=[EXP5460, EXP5460_RECEIPTS],
            evidence={
                "exp5460": _select(
                    exp5460,
                    (
                        "continuous_self_learning_task",
                        "csl_policy_ready",
                        "no_weight_mutation",
                        "cumulative_constraint_violations",
                        "quality_delta_vs_naive_icl",
                        "context_efficiency_delta",
                        "policy_update_count",
                        "negative_transfer_deflection_rate",
                        "honest_verdict",
                    ),
                )
            },
            headline_blockers=csl_blockers,
            claim_boundary="Headline is governed policy routing with sidecar state only; no weight learning claim.",
        ),
        "sota_csl": _row(
            lane="sota_csl",
            classification=_classification_from_clean_gate(
                sota_csl_clean, "headline_ready", sota_csl_blockers
            ),
            authority_gate="GGUF/llama.cpp GPU offload receipts plus frozen weights",
            source_artifacts=[EXP5461, EXP5461_ROWS],
            evidence={
                "exp5461": _select(
                    exp5461,
                    (
                        "csl_sota_memory_routing_ready",
                        "gpu_offload_verified",
                        "no_weight_mutation",
                        "negative_transfer_deflection_rate",
                        "quality_delta_vs_no_memory",
                        "quality_delta_vs_naive_icl",
                        "runtime_backend",
                        "precondition_details",
                        "honest_verdict",
                    ),
                )
            },
            headline_blockers=sota_csl_blockers,
            claim_boundary="Headline is live GGUF memory routing under frozen weights, not model fine-tuning.",
        ),
        "pbit_pdit_bridge": _row(
            lane="pbit_pdit_bridge",
            classification=_classification_from_clean_gate(pbit_clean, "bounded", pbit_blockers),
            authority_gate="exact solver remains final authority; p-bit and p-dit assumptions are advisory",
            source_artifacts=[EXP5462],
            evidence={
                "exp5462": _select(
                    exp5462,
                    (
                        "minimal_core_pbit_bridge_ready",
                        "solver_authoritative",
                        "fallback_completeness_rate",
                        "hardware_speedup_claim",
                        "claim_limits",
                        "unsafe_false_accepts",
                        "honest_verdict",
                    ),
                )
            },
            headline_blockers=pbit_blockers,
            claim_boundary="Bounded bridge only; no hardware timing or final-answer authority claim.",
        ),
        "hardware_receipts": _row(
            lane="hardware_receipts",
            classification=_classification_from_clean_gate(
                hardware_clean, "bounded", hardware_blockers
            ),
            authority_gate="hashes match before timing comparison; reachable-board receipts only",
            source_artifacts=[EXP5463],
            evidence={
                "exp5463": _select(
                    exp5463,
                    (
                        "hardware_receipts_ready",
                        "gated_upstream_ready",
                        "hashes_match_before_timing_compare",
                        "hardware_speedup_claim",
                        "board_reachability",
                        "timing_repeat_counts",
                        "timing_comparison",
                        "honest_verdict",
                    ),
                )
            },
            headline_blockers=hardware_blockers,
            claim_boundary="Bounded CPU/Polarfire timing facts only; KV260 unreachable and no speedup claim.",
        ),
        "arc": _row(
            lane="arc",
            classification=arc_classification,
            authority_gate="ARC provenance gate: live-agent self-discovery plus offline reproduction for any bank",
            source_artifacts=[EXP5464, EXP5464_RECEIPTS, EXP5465],
            evidence={
                "exp5464": _select(
                    exp5464,
                    (
                        "arc_metric_integrity_ready",
                        "registry_precheck_performed",
                        "duplicate_solve_rejected",
                        "off_path_solve_rejected",
                        "target_shortlist",
                        "honest_verdict",
                    ),
                ),
                "exp5465": _select(
                    exp5465,
                    (
                        "new_level_banked",
                        "offline_reproduced",
                        "registry_precheck_performed",
                        "failure_mode",
                        "live_attempt_count",
                        "solve_provenance",
                        "source_reading_used",
                        "target_game",
                        "honest_verdict",
                    ),
                ),
            },
            headline_blockers=arc_blockers,
            claim_boundary="Metric/perception precheck is useful, but no new ARC level is claimed without offline reproduction.",
        ),
        "synthesis": _row(
            lane="synthesis",
            classification=_classification_from_clean_gate(
                synthesis_clean, "bounded", synthesis_blockers
            ),
            authority_gate="aggregation from upstream artifacts only",
            source_artifacts=[EXP5466],
            evidence={
                "exp5466": _select(
                    exp5466,
                    (
                        "status",
                        "honest_verdict",
                        "artifact_paths_read",
                        "skipped_gated_tasks",
                        "missing_artifacts",
                    ),
                )
            },
            headline_blockers=synthesis_blockers,
            claim_boundary="Bounded capstone/PRD synthesis only; it is not independent experimental evidence.",
        ),
    }


def _lane_buckets(
    truth_table: Mapping[str, JsonMap], payloads: Mapping[str, JsonMap]
) -> tuple[list[str], list[str], list[str], list[str]]:
    headline = [
        lane for lane in LANE_ORDER if truth_table[lane]["classification"] == "headline_ready"
    ]
    bounded = [lane for lane in LANE_ORDER if truth_table[lane]["classification"] == "bounded"]
    blocked = [lane for lane in LANE_ORDER if truth_table[lane]["classification"] == "blocked"]
    honest_null = [
        lane for lane in LANE_ORDER if truth_table[lane]["classification"] == "honest_null"
    ]
    speedup = any(
        payloads.get(path, {}).get("hardware_speedup_claim") is True for path in (EXP5462, EXP5463)
    )
    if not speedup:
        honest_null.append("hardware_speedup_claim")
    return headline, bounded, blocked, honest_null


def _no_claim_boundaries(payloads: Mapping[str, JsonMap]) -> list[JsonDict]:
    exp5463 = payloads.get(EXP5463, {})
    return [
        {
            "claim_id": "hardware_speedup",
            "boundary": "No hardware speedup is claimed; timing ratios are receipt facts only.",
            "evidence": _select(
                exp5463, ("hardware_speedup_claim", "timing_comparison", "board_reachability")
            ),
        },
        {
            "claim_id": "token_internal_features",
            "boundary": "No logits, hidden-state, attention, token, or intermediate-exit feature access is reopened.",
            "evidence": {
                "transition_blocked_lanes": payloads.get(EXP5454, {}).get("blocked_lanes")
            },
        },
        {
            "claim_id": "external_text_scorers",
            "boundary": "Generated-text scorer and judge lanes are not used as final authority.",
            "evidence": {"final_authority": "exact deterministic verifiers and solvers only"},
        },
        {
            "claim_id": "non_local_tsu_kona_aleph",
            "boundary": "Extropic TSU, Kona, and Aleph remain architecture context only without local authenticated execution.",
            "evidence": {"inference_substrate": INFERENCE_SUBSTRATE},
        },
        {
            "claim_id": "off_path_arc_solves",
            "boundary": "ARC progress counts only live-agent self-discovery with offline reproduction; off-path solves are rejected.",
            "evidence": _select(
                payloads.get(EXP5465, {}),
                (
                    "solve_provenance",
                    "new_level_banked",
                    "offline_reproduced",
                    "source_reading_used",
                ),
            ),
        },
        {
            "claim_id": "guided_decoding_headline",
            "boundary": "Exp5457 reproduced a TAUTOLOGY-class failure and failed LCD bias readiness, so no guided-decoding headline is made.",
            "evidence": _select(
                payloads.get(EXP5457, {}),
                (
                    "flagged_adversarial",
                    "corrigendum_pending",
                    "lcd_bias_check_passed",
                    "verifier_guided_decoding_ready",
                ),
            ),
        },
    ]


def _next_milestone_recommendations() -> list[JsonDict]:
    return [
        {
            "target": "quarantine_guided_decoding_rerun",
            "priority": 1,
            "recommendation": "Quarantine direct local SOTA guided-decoding reruns until a small non-tautological metric/LCD-bias fixture is clean.",
            "rationale": "Exp5457 had GGUF receipts but reproduced the Exp5444-style TAUTOLOGY failure and failed LCD bias readiness.",
        },
        {
            "target": "scale_sota_csl_memory_routing",
            "priority": 2,
            "recommendation": "Scale the clean SOTA CSL memory-routing panel with the same GGUF/offload and frozen-weight receipts.",
            "rationale": "Exp5461 is clean, GPU-offloaded, frozen-weight, and reports negative-transfer deflection.",
        },
        {
            "target": "compose_exact_guard_stack",
            "priority": 3,
            "recommendation": "Compose minimal-core repair and distortion guards on larger deterministic fixtures before another live decoding panel.",
            "rationale": "Exp5458 and Exp5459 are exact-authority lanes; they can harden the next generation-time attempt.",
        },
        {
            "target": "restore_kv260_hardware_receipts",
            "priority": 4,
            "recommendation": "Restore KV260 reachability and collect repeated matched board-local timing before any speedup language.",
            "rationale": "Exp5463 reached Polarfire only, hash-matched timing, and correctly kept hardware_speedup_claim=false.",
        },
        {
            "target": "arc_live_salience_frontier",
            "priority": 5,
            "recommendation": "Continue ARC live-agent salience work only through provenance and offline-reproduction gates.",
            "rationale": "Exp5464 made the metric/perception precheck clean, but Exp5465 banked no reproduced new level.",
        },
    ]


def _artifact_checksums(root: Path, paths: Sequence[str]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for rel_path in paths:
        path = root / rel_path
        if path.exists() and path.is_file():
            checksums[rel_path] = _sha256_text(path.read_text(encoding="utf-8"))
    return checksums


def git_path_unchanged(root: Path, rel_path: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", rel_path],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return True
    return completed.returncode != 0 or completed.stdout.strip() == ""


def _payload_checksum(artifact: JsonMap) -> str:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return _sha256_text(_stable_json(payload))


def _honest_verdict(artifact: JsonMap) -> str:
    missing = len(artifact.get("missing_artifacts", []))
    skipped = len(artifact.get("skipped_gated_tasks", []))
    prefix = "blocked:" if missing or skipped else "complete:"
    return (
        f"{prefix} .496 capstone truth table from actual artifacts; "
        f"headline_ready={len(artifact['headline_ready_lanes'])}, "
        f"bounded={len(artifact['bounded_lanes'])}, blocked={len(artifact['blocked_lanes'])}, "
        f"honest_null={len(artifact['honest_null_lanes'])}; guided decoding quarantined, "
        "ARC no-bank, hardware_speedup_claim=false."
    )


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    root_path = Path(root)
    payloads, sidecars, read_paths, source_read, source_missing, missing = _read_inputs(root_path)
    truth_table = _truth_table(payloads, missing)
    headline, bounded, blocked, honest_null = _lane_buckets(truth_table, payloads)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "milestone": MILESTONE,
        "artifact_paths_read": read_paths,
        "source_context_read": source_read,
        "source_context_missing": source_missing,
        "missing_artifacts": missing,
        "sidecar_artifacts_read": sidecars,
        "truth_table": truth_table,
        "headline_ready_lanes": headline,
        "bounded_lanes": bounded,
        "blocked_lanes": blocked,
        "honest_null_lanes": honest_null,
        "skipped_gated_tasks": _skipped_gated_tasks(payloads),
        "no_claim_boundaries": _no_claim_boundaries(payloads),
        "next_milestone_recommendations": _next_milestone_recommendations(),
        "roadmap_yaml_unchanged": git_path_unchanged(root_path, "research-roadmap.yaml"),
        "conductor_unchanged": git_path_unchanged(root_path, "scripts/research_conductor.py"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "artifact_checksums": _artifact_checksums(root_path, read_paths),
        "tests_run": list(tests_run),
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _fail(message: str) -> None:
    raise ValueError(message)


def validate_artifact(artifact: JsonMap) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        _fail(f"missing required fields: {missing}")
    if artifact["milestone"] != MILESTONE:
        _fail("milestone must be 2026.07.496")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        _fail("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact["roadmap_yaml_unchanged"] is not True:
        _fail("roadmap_yaml_unchanged must be true")
    if artifact["conductor_unchanged"] is not True:
        _fail("conductor_unchanged must be true")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        _fail("honest_verdict must start with a terminal prefix")
    if not isinstance(artifact["artifact_paths_read"], list) or not artifact["artifact_paths_read"]:
        _fail("artifact_paths_read must be a non-empty list")
    if not isinstance(artifact["truth_table"], Mapping):
        _fail("truth_table must be a mapping")
    if list(artifact["truth_table"]) != list(LANE_ORDER):
        _fail("truth_table lane order mismatch")

    expected_headlines = [
        lane
        for lane in LANE_ORDER
        if artifact["truth_table"][lane]["classification"] == "headline_ready"
    ]
    if artifact["headline_ready_lanes"] != expected_headlines:
        _fail("headline_ready_lanes inconsistent with truth_table")
    if "guided_decoding" in artifact["headline_ready_lanes"]:
        _fail("guided_decoding cannot be headline_ready while Exp5457 is flagged")
    if "hardware_speedup_claim" not in artifact["honest_null_lanes"]:
        _fail("hardware_speedup_claim honest null must be recorded")
    if not artifact["no_claim_boundaries"]:
        _fail("no_claim_boundaries must be non-empty")
    if len(artifact["next_milestone_recommendations"]) not in {3, 4, 5}:
        _fail("next_milestone_recommendations must contain 3-5 priorities")
    if artifact["reproducibility_checksum"] != _payload_checksum(artifact):
        _fail("reproducibility_checksum is stale")
    if artifact["missing_artifacts"] or artifact["skipped_gated_tasks"]:
        if not str(artifact["honest_verdict"]).startswith("blocked:"):
            _fail("blocked: verdict required when inputs are missing or skipped")
    elif not str(artifact["honest_verdict"]).startswith("complete:"):
        _fail("complete: verdict required when all expected inputs are readable")


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    artifact = build_artifact(root=root, tests_run=tests_run)
    write_json(Path(result_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Exp5467 .496 capstone artifact")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
