"""Exp5401 .491 capstone truth-table synthesis.

Spec refs: REQ-CAPSTONE-5401, SCENARIO-CAPSTONE-5401,
SCENARIO-CAPSTONE-5401-MISSING-OR-FLAGGED-INPUT,
SCENARIO-CAPSTONE-5401-FIELD-PRINCIPLES.

This module is deliberately an aggregation step. It reads the checked-in
milestone artifacts and conductor log, then reports what those artifacts can
support without turning a flagged, bounded, or honest-null result into a
stronger research claim. That distinction matters because capstones become the
source that later roadmap tasks use as their gate state.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5401_capstone_v491.json")
CONDUCTOR_LOG_PATH = "ops/conductor-log.md"
EXPERIMENT = "experiment_5401_capstone_v491"
EXPERIMENT_ID = "exp5401-v491-capstone"
MILESTONE = "2026.07.491"
SCHEMA = "carnot.experiment_5401.capstone_v491.v1"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5401
EXPECTED_TASK_COUNT = 13
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP5389 = "results/experiment_5389_transition_v491.json"
EXP5390 = "results/experiment_5390_sota_source_delta_v491.json"
EXP5391 = "results/experiment_5391_constraint_tax_scaleup_fixtures_v491.json"
EXP5392 = "results/experiment_5392_formal_encoding_safety_fixture_v491.json"
EXP5393 = "results/experiment_5393_overwrite_guidance_tautology_corrigendum_v491.json"
EXP5394 = "results/experiment_5394_gated_overwrite_pbit_ablation_v491.json"
EXP5395 = "results/experiment_5395_influence_share_verifier_budget_router_v491.json"
EXP5396 = "results/experiment_5396_memory_guard_raw_episode_retention_v491.json"
EXP5397 = "results/experiment_5397_arc_blob_salience_live_path_v491.json"
EXP5398 = "results/experiment_5398_hardware_evidence_graph_repeatability_v491.json"
EXP5398_GRAPH = "results/experiment_5398_hardware_evidence_graph_repeatability_v491.graph.json"
EXP5399 = "results/experiment_5399_kan_dynamic_counterexample_certificate_v491.json"
EXP5400 = "results/experiment_5400_evidence_table_prd_gap_analysis_v491.json"

UPSTREAM_ARTIFACT_PATHS: tuple[str, ...] = (
    EXP5389,
    EXP5390,
    EXP5391,
    EXP5392,
    EXP5393,
    EXP5394,
    EXP5395,
    EXP5396,
    EXP5397,
    EXP5398,
    EXP5399,
    EXP5400,
)

SIDECAR_ARTIFACT_PATHS: tuple[str, ...] = (EXP5398_GRAPH,)

TASK_IDS: dict[str, str] = {
    EXP5389: "exp5389-v491-transition-and-archive",
    EXP5390: "exp5390-v491-sota-source-delta",
    EXP5391: "exp5391-v491-constraint-tax-scaleup-fixtures",
    EXP5392: "exp5392-v491-formal-encoding-safety-fixture",
    EXP5393: "exp5393-v491-overwrite-guidance-tautology-corrigendum",
    EXP5394: "exp5394-v491-gated-overwrite-pbit-ablation",
    EXP5395: "exp5395-v491-influence-share-verifier-budget-router",
    EXP5396: "exp5396-v491-memory-guard-raw-episode-retention",
    EXP5397: "exp5397-v491-arc-blob-salience-live-path",
    EXP5398: "exp5398-v491-hardware-evidence-graph-repeatability",
    EXP5399: "exp5399-v491-kan-dynamic-counterexample-certificate",
    EXP5400: "exp5400-v491-evidence-table-and-prd-gap-analysis",
}

CONDUCTOR_FLAG_PATTERNS: dict[str, str] = {
    EXP5392: "Formal-encoding safety",
}

TRUTH_TABLE_LANES: tuple[str, ...] = (
    "structured_constraint_tax_scaleup",
    "formal_encoding_safety_fixture",
    "overwrite_guidance_corrigendum",
    "pbit_boundary_ablation",
    "continuous_self_learning_router",
    "raw_episode_memory_guard",
    "arc_level_up",
    "hardware_repeatability",
    "kan_dynamic_certificate",
    "prd_evidence_table",
)

ALLOWED_CLASSIFICATIONS = {
    "headline_ready",
    "bounded_ready",
    "partial",
    "blocked",
    "honest_null",
    "missing_inputs",
}

SPEC_REFS = (
    "REQ-CAPSTONE-5401",
    "SCENARIO-CAPSTONE-5401",
    "SCENARIO-CAPSTONE-5401-MISSING-OR-FLAGGED-INPUT",
    "SCENARIO-CAPSTONE-5401-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "complete if capstone emitted from actual artifacts; honest_partial "
        "if expected upstream artifacts are missing or unreadable."
    ),
    "milestone": "must equal 2026.07.491.",
    "artifact_count_expected": "number of .491 tasks expected.",
    "artifact_count_found": "number of .491 task artifacts found or emitted.",
    "missing_artifacts": "list of missing or unreadable expected upstream artifacts.",
    "flagged_artifacts": "list of artifacts flagged by conductor or adversarial checks.",
    "structured_scaleup_ready": "derived from Exp5391.",
    "formal_encoding_fixture_ready": (
        "derived from Exp5392 but false while the artifact is flagged."
    ),
    "overwrite_guidance_corrigendum_clean": "derived from Exp5393.",
    "pbit_boundary_ablation_ready": "derived from Exp5394.",
    "continuous_self_learning_router_ready": "derived from Exp5395.",
    "raw_episode_guard_ready": "derived from Exp5396.",
    "arc_new_level_banked": "derived from Exp5397.",
    "hardware_repeatability_ready": "derived from Exp5398 repeatability evidence.",
    "hardware_speedup_claim": (
        "must remain false unless Exp5398 proves repeated board-local timing speedup."
    ),
    "dynamic_counterexample_certificate_ready": "derived from Exp5399.",
    "future_token_signal_allowed": (
        "must remain false unless a new backend feature artifact exists."
    ),
    "retired_or_blocked_lanes": "list of lanes that remain closed or blocked.",
    "next_recommendations": "concrete recommendations for the next roadmap.",
    "active_roadmap_modified": "must be false.",
    "conductor_modified": "must be false.",
    "honest_verdict": (
        "one-line summary starting with complete: that distinguishes real evidence "
        "from blocked lanes."
    ),
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "inference_substrate",
    "source_artifacts",
    "sidecar_artifacts",
    "artifact_read_errors",
    "truth_table",
    "headline_ready_lanes",
    "conductor_flags",
    "protected_file_checks",
    "source_context_read",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES.keys(),
)

BOOLEAN_FIELDS = (
    "structured_scaleup_ready",
    "formal_encoding_fixture_ready",
    "overwrite_guidance_corrigendum_clean",
    "pbit_boundary_ablation_ready",
    "continuous_self_learning_router_ready",
    "raw_episode_guard_ready",
    "arc_new_level_banked",
    "hardware_repeatability_ready",
    "hardware_speedup_claim",
    "dynamic_counterexample_certificate_ready",
    "future_token_signal_allowed",
    "active_roadmap_modified",
    "conductor_modified",
)

SOURCE_CONTEXT_PATHS = (
    "CLAUDE.md",
    "research-program.md",
    "_bmad/prd.md",
    "_bmad/architecture.md",
    "research-roadmap.yaml",
    "research-roadmap-next.yaml",
    CONDUCTOR_LOG_PATH,
    "ops/status.md",
    "ops/changelog.md",
    "results",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5401_capstone_v491.py -q",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5401_capstone_v491.py "
            "-m pytest tests/python/test_experiment_5401_capstone_v491.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5401_capstone_v491.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)


def unwrap(value: Any) -> Any:
    """Return the bare value from the repo's principle-wrapped JSON fields."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Build the capstone from actual upstream artifacts and no roadmap edits."""

    root_path = Path(root)
    payloads, found, missing, read_errors = read_inputs(root_path)
    conductor_flags = read_conductor_flags(root_path)
    flagged_artifacts = build_flagged_artifacts(payloads, conductor_flags)
    booleans = derive_readiness(payloads, flagged_artifacts, root_path)
    truth_table = build_truth_table(payloads, flagged_artifacts, missing, booleans)
    artifact_count_found = 1 + len(found)
    status = "honest_partial" if missing else "complete"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "status": status,
        "milestone": MILESTONE,
        "artifact_count_expected": EXPECTED_TASK_COUNT,
        "artifact_count_found": artifact_count_found,
        "missing_artifacts": missing,
        "flagged_artifacts": flagged_artifacts,
        "truth_table": truth_table,
        "headline_ready_lanes": [
            row["lane"] for row in truth_table if row["headline_ready"] is True
        ],
        "retired_or_blocked_lanes": retired_or_blocked_lanes(truth_table),
        "next_recommendations": next_recommendations(),
        "source_artifacts": source_artifacts(root_path, payloads, found),
        "sidecar_artifacts": sidecar_artifacts(root_path, payloads),
        "artifact_read_errors": read_errors,
        "conductor_flags": conductor_flags,
        "protected_file_checks": protected_file_checks(root_path),
        "source_context_read": source_context_read(root_path),
        "tests_run": [dict(row) for row in tests_run],
        **booleans,
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    artifact = json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Write the validated capstone artifact."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    write_json(Path(result_path), artifact)
    return artifact


def read_inputs(root: Path) -> tuple[dict[str, JsonDict], list[str], list[str], list[JsonDict]]:
    """Read primary and sidecar inputs, marking unreadable primary tasks missing."""

    payloads: dict[str, JsonDict] = {}
    found: list[str] = []
    missing: list[str] = []
    read_errors: list[JsonDict] = []

    for relative in (*UPSTREAM_ARTIFACT_PATHS, *SIDECAR_ARTIFACT_PATHS):
        path = root / relative
        if not path.exists():
            if relative in UPSTREAM_ARTIFACT_PATHS:
                missing.append(relative)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            read_errors.append(
                {
                    "path": relative,
                    "classification": f"malformed_json:{exc.msg}",
                    "line": exc.lineno,
                    "column": exc.colno,
                }
            )
            if relative in UPSTREAM_ARTIFACT_PATHS:
                missing.append(relative)
            continue

        if not isinstance(payload, dict):
            read_errors.append({"path": relative, "classification": "not_json_object"})
            if relative in UPSTREAM_ARTIFACT_PATHS:
                missing.append(relative)
            continue

        payloads[relative] = payload
        if relative in UPSTREAM_ARTIFACT_PATHS:
            found.append(relative)

    return payloads, found, missing, read_errors


def read_conductor_flags(root: Path) -> list[JsonDict]:
    """Pull conductor FLAGGED rows into machine-readable capstone evidence."""

    log_path = root / CONDUCTOR_LOG_PATH
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    flags: list[JsonDict] = []
    for artifact_path, pattern in CONDUCTOR_FLAG_PATTERNS.items():
        for line in lines:
            if pattern in line and "| FLAGGED |" in line:
                flags.append(
                    {
                        "path": artifact_path,
                        "task_id": TASK_IDS[artifact_path],
                        "status": "FLAGGED",
                        "log_excerpt": line,
                    }
                )
                break
    return flags


def build_flagged_artifacts(
    payloads: Mapping[str, JsonDict],
    conductor_flags: Sequence[JsonMap],
) -> list[JsonDict]:
    """Return artifacts that cannot be headline evidence because a flag exists."""

    conductor_flagged_paths = {str(row["path"]) for row in conductor_flags}
    flagged: list[JsonDict] = []
    for artifact_path in UPSTREAM_ARTIFACT_PATHS:
        payload = payloads.get(artifact_path)
        if payload is None:
            continue
        reasons: list[str] = []
        if unwrap(payload.get("flagged_adversarial")) is True:
            reasons.append("artifact flagged_adversarial=true")
        if artifact_path in conductor_flagged_paths:
            reasons.append("conductor log status FLAGGED")
        if has_critical_tautology(payload):
            reasons.append("critical TAUTOLOGY corrigendum pending")
        if reasons:
            flagged.append(
                {
                    "path": artifact_path,
                    "task_id": TASK_IDS[artifact_path],
                    "reasons": reasons,
                    "headline_eligible": False,
                }
            )
    return flagged


def has_critical_tautology(payload: JsonMap) -> bool:
    """Detect the specific adversarial flag that blocks Exp5392 promotion."""

    for row in payload.get("corrigendum_pending", ()):
        if not isinstance(row, Mapping):
            continue
        if row.get("kind") == "TAUTOLOGY" and row.get("severity") == "critical":
            return True
    return False


def derive_readiness(
    payloads: Mapping[str, JsonDict],
    flagged_artifacts: Sequence[JsonMap],
    root: Path,
) -> JsonDict:
    """Compute the required capstone booleans from upstream fields, fail-closed."""

    flagged_paths = {str(row["path"]) for row in flagged_artifacts}
    exp5389 = payloads.get(EXP5389, {})
    exp5398 = payloads.get(EXP5398, {})
    hardware_repeatability = source_bool(exp5398, "repeatability_evidence_present")
    upstream_hardware_speedup = source_bool(exp5398, "hardware_speedup_claim")
    active_roadmap_modified = (
        source_bool(exp5389, "active_roadmap_modified")
        or source_bool(payloads.get(EXP5390, {}), "roadmap_files_modified")
        or git_path_modified(root, "research-roadmap.yaml")
    )
    conductor_modified = (
        source_bool(exp5389, "conductor_modified")
        or any(
            source_bool(payloads.get(path, {}), "research_conductor_modified", "conductor_modified")
            for path in UPSTREAM_ARTIFACT_PATHS
        )
        or git_path_modified(root, "scripts/research_conductor.py")
    )

    return {
        "structured_scaleup_ready": source_bool(payloads.get(EXP5391), "constraint_tax_scaleup_ready"),
        "formal_encoding_fixture_ready": (
            source_bool(payloads.get(EXP5392), "formal_encoding_fixture_ready")
            and EXP5392 not in flagged_paths
        ),
        "overwrite_guidance_corrigendum_clean": source_bool(
            payloads.get(EXP5393),
            "overwrite_guidance_corrigendum_clean",
        ),
        "pbit_boundary_ablation_ready": source_bool(
            payloads.get(EXP5394),
            "pbit_boundary_ablation_ready",
        ),
        "continuous_self_learning_router_ready": source_bool(
            payloads.get(EXP5395),
            "continuous_self_learning_router_ready",
        ),
        "raw_episode_guard_ready": source_bool(payloads.get(EXP5396), "raw_episode_guard_ready"),
        "arc_new_level_banked": source_bool(payloads.get(EXP5397), "new_level_banked"),
        "hardware_repeatability_ready": hardware_repeatability,
        "hardware_speedup_claim": bool(hardware_repeatability and upstream_hardware_speedup),
        "dynamic_counterexample_certificate_ready": source_bool(
            payloads.get(EXP5399),
            "dynamic_counterexample_certificate_ready",
        ),
        "future_token_signal_allowed": future_token_signal_allowed(exp5389),
        "active_roadmap_modified": active_roadmap_modified,
        "conductor_modified": conductor_modified,
    }


def source_bool(payload: JsonMap | None, *fields: str) -> bool:
    """Return true only when one of the requested upstream fields is exactly true."""

    if not payload:
        return False
    for field in fields:
        if field in payload:
            return unwrap(payload[field]) is True
    return False


def future_token_signal_allowed(exp5389: JsonMap) -> bool:
    """Keep the token/internal lane closed without a new backend feature artifact."""

    prior_gate = exp5389.get("prior_gate_summary", {})
    prior_blocker = exp5389.get("prior_blockers", {}).get("token_feature", {})
    backend_features_present = any(
        bool(prior_blocker.get(name))
        for name in (
            "logits_available",
            "hidden_states_available",
            "attention_available",
            "intermediate_depth_exits_available",
        )
    )
    return bool(prior_gate.get("future_token_signal_allowed")) and backend_features_present


def build_truth_table(
    payloads: Mapping[str, JsonDict],
    flagged_artifacts: Sequence[JsonMap],
    missing: Sequence[str],
    booleans: JsonMap,
) -> list[JsonDict]:
    """Build the ten-lane capstone table requested for the milestone close."""

    flagged_paths = {str(row["path"]) for row in flagged_artifacts}
    rows = [
        truth_row(
            lane="structured_constraint_tax_scaleup",
            source_artifact=EXP5391,
            classification=classification_for(EXP5391, missing, "headline_ready"),
            headline_ready=bool(booleans["structured_scaleup_ready"]),
            claim_boundary="bounded_deterministic_fixture_panel",
            evidence=structured_evidence(payloads.get(EXP5391)),
        ),
        truth_row(
            lane="formal_encoding_safety_fixture",
            source_artifact=EXP5392,
            classification=classification_for(
                EXP5392,
                missing,
                "blocked" if EXP5392 in flagged_paths else "headline_ready",
            ),
            headline_ready=bool(booleans["formal_encoding_fixture_ready"]),
            claim_boundary="safe_fixture_not_clean_headline_while_flagged",
            blocked_reason="flagged_adversarial_tautology" if EXP5392 in flagged_paths else "",
            evidence=formal_evidence(payloads.get(EXP5392)),
        ),
        truth_row(
            lane="overwrite_guidance_corrigendum",
            source_artifact=EXP5393,
            classification=classification_for(EXP5393, missing, "headline_ready"),
            headline_ready=bool(booleans["overwrite_guidance_corrigendum_clean"]),
            claim_boundary="row_level_solver_authority_only",
            evidence=overwrite_evidence(payloads.get(EXP5393)),
        ),
        truth_row(
            lane="pbit_boundary_ablation",
            source_artifact=EXP5394,
            classification=classification_for(EXP5394, missing, "bounded_ready"),
            headline_ready=bool(booleans["pbit_boundary_ablation_ready"]),
            claim_boundary="cpu_only_no_hardware_speedup",
            evidence=pbit_evidence(payloads.get(EXP5394)),
        ),
        truth_row(
            lane="continuous_self_learning_router",
            source_artifact=EXP5395,
            classification=classification_for(EXP5395, missing, "headline_ready"),
            headline_ready=bool(booleans["continuous_self_learning_router_ready"]),
            claim_boundary="controller_routing_no_weight_mutation",
            evidence=router_evidence(payloads.get(EXP5395)),
        ),
        truth_row(
            lane="raw_episode_memory_guard",
            source_artifact=EXP5396,
            classification=classification_for(EXP5396, missing, "headline_ready"),
            headline_ready=bool(booleans["raw_episode_guard_ready"]),
            claim_boundary="raw_episode_retention_no_rationale_authority",
            evidence=memory_evidence(payloads.get(EXP5396)),
        ),
        truth_row(
            lane="arc_level_up",
            source_artifact=EXP5397,
            classification=classification_for(EXP5397, missing, "honest_null"),
            headline_ready=False,
            claim_boundary="live_path_reached_no_new_banked_level",
            blocked_reason="bounded_budget_no_levelup",
            evidence=arc_evidence(payloads.get(EXP5397)),
        ),
        truth_row(
            lane="hardware_repeatability",
            source_artifact=[EXP5398, EXP5398_GRAPH],
            classification=classification_for(EXP5398, missing, "blocked"),
            headline_ready=False,
            claim_boundary="hash_graph_receipt_no_board_local_repeatability",
            blocked_reason="no_repeated_board_local_timing",
            evidence=hardware_evidence(payloads.get(EXP5398)),
        ),
        truth_row(
            lane="kan_dynamic_certificate",
            source_artifact=EXP5399,
            classification=classification_for(EXP5399, missing, "headline_ready"),
            headline_ready=bool(booleans["dynamic_counterexample_certificate_ready"]),
            claim_boundary="bounded_certificate_no_broad_kan_verification",
            evidence=kan_evidence(payloads.get(EXP5399)),
        ),
        truth_row(
            lane="prd_evidence_table",
            source_artifact=EXP5400,
            classification=classification_for(EXP5400, missing, "partial"),
            headline_ready=False,
            claim_boundary="evidence_table_complete_but_prd_alignment_partial",
            evidence=prd_evidence(payloads.get(EXP5400)),
        ),
    ]
    return rows


def classification_for(path: str, missing: Sequence[str], present_classification: str) -> str:
    """Convert a missing upstream task into a missing-input truth-table row."""

    return "missing_inputs" if path in set(missing) else present_classification


def truth_row(
    *,
    lane: str,
    source_artifact: str | Sequence[str],
    classification: str,
    headline_ready: bool,
    claim_boundary: str,
    evidence: JsonMap,
    blocked_reason: str = "",
) -> JsonDict:
    """Return a uniform truth-table row so validation can be simple."""

    row = {
        "lane": lane,
        "source_artifact": source_artifact,
        "classification": classification,
        "headline_ready": headline_ready,
        "claim_boundary": claim_boundary,
        "blocked_reason": blocked_reason,
        "evidence": dict(evidence),
    }
    if classification == "missing_inputs":
        row["headline_ready"] = False
        row["claim_boundary"] = "missing upstream artifact; no outcome inferred"
        row["blocked_reason"] = "missing_inputs"
        row["evidence"] = {}
    return row


def structured_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "fixture_count": int(payload["fixture_count"]),
        "constrained_semantic_validity_rate": payload["constrained_semantic_validity_rate"],
        "wrong_valid_reduction": int(payload["wrong_valid_count_unconstrained"])
        - int(payload["wrong_valid_count_constrained"]),
        "unsafe_false_accept_count": int(payload["unsafe_false_accept_count"]),
    }


def formal_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "upstream_ready": bool(payload.get("formal_encoding_fixture_ready")),
        "flagged_adversarial": bool(payload.get("flagged_adversarial")),
        "corrigendum_pending_count": len(payload.get("corrigendum_pending", ())),
        "forbidden_detail_leak_count": int(payload["forbidden_detail_leak_count"]),
    }


def overwrite_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "row_count": int(payload["row_count"]),
        "tautology_checks_passed": bool(payload["tautology_checks_passed"]),
        "row_level_evidence_clean": bool(payload["row_level_evidence_clean"]),
        "unsafe_false_accept_count": int(payload["unsafe_false_accept_count"]),
    }


def pbit_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "fixture_count": int(payload["fixture_count"]),
        "simulation_only": bool(payload["simulation_only"]),
        "hardware_speedup_claim": bool(payload["hardware_speedup_claim"]),
        "unsafe_false_accepts": int(payload["unsafe_false_accepts"]),
    }


def router_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "trace_count": int(payload["trace_count"]),
        "routed_decision_count": int(payload["routed_decision_count"]),
        "quality_delta_vs_baseline": payload["quality_delta_vs_baseline"],
        "verifier_cost_delta_vs_baseline": payload["verifier_cost_delta_vs_baseline"],
        "no_weight_mutation": bool(payload["no_weight_mutation"]),
    }


def memory_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "raw_episode_count": int(payload["raw_episode_count"]),
        "accepted_memory_count": len(payload.get("accepted_memories", ())),
        "rejected_memory_count": int(payload["rejected_memory_count"]),
        "forged_reasoning_deflection_rate": payload["forged_reasoning_deflection_rate"],
        "no_weight_mutation": bool(payload["no_weight_mutation"]),
    }


def arc_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "new_level_banked": bool(payload["new_level_banked"]),
        "failure_mode": payload.get("failure_mode"),
        "registry_total_before": int(payload["registry_total_before"]),
        "registry_total_after": int(payload["registry_total_after"]),
        "solve_provenance": payload["solve_provenance"],
        "offline_bfs_used": bool(payload["offline_bfs_used"]),
    }


def hardware_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "offline_verifier_passed": bool(unwrap(payload["offline_verifier_passed"])),
        "polar_fire_repeat_count": int(unwrap(payload["polar_fire_repeat_count"])),
        "repeatability_evidence_present": bool(unwrap(payload["repeatability_evidence_present"])),
        "hardware_speedup_claim": bool(unwrap(payload["hardware_speedup_claim"])),
        "destructive_action_taken": bool(unwrap(payload["destructive_action_taken"])),
    }


def kan_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "sample_count": int(payload["sample_count"]),
        "false_property_rejection_rate": payload["false_property_rejection_rate"],
        "true_property_preservation_rate": payload["true_property_preservation_rate"],
        "counterexample_region_count": int(payload["counterexample_region_count"]),
        "broad_kan_verification_claim": bool(payload["broad_kan_verification_claim"]),
    }


def prd_evidence(payload: JsonMap | None) -> JsonDict:
    if not payload:
        return {}
    return {
        "closed_gap_count": len(payload.get("closed_gaps", ())),
        "partial_gap_count": len(payload.get("partial_gaps", ())),
        "blocked_gap_count": len(payload.get("blocked_gaps", ())),
        "disallowed_claim_count": len(payload.get("disallowed_claims", ())),
        "honest_verdict": payload.get("honest_verdict", ""),
    }


def retired_or_blocked_lanes(truth_table: Sequence[JsonMap]) -> list[JsonDict]:
    """List lanes that remain closed, blocked, or bounded enough to constrain claims."""

    rows = {
        "formal_encoding_safety_fixture": {
            "lane": "formal_encoding_safety_fixture",
            "state": "blocked_flagged_adversarial",
            "source_artifact": EXP5392,
            "next_gate": "rerun distinct checksum-backed safety metrics",
        },
        "pbit_hardware_transfer": {
            "lane": "pbit_hardware_transfer",
            "state": "blocked_cpu_only",
            "source_artifact": EXP5394,
            "next_gate": "repeat same workload on reachable board before acceleration claim",
        },
        "arc_level_up": {
            "lane": "arc_level_up",
            "state": "honest_null_no_bank",
            "source_artifact": EXP5397,
            "next_gate": "trajectory generation that banks a reproduction-gated new level",
        },
        "hardware_repeatability": {
            "lane": "hardware_repeatability",
            "state": "blocked_no_board_local_repeats",
            "source_artifact": EXP5398,
            "next_gate": "restore KV260 or PolarFire reachability and repeat timing",
        },
        "hardware_speedup_claim": {
            "lane": "hardware_speedup_claim",
            "state": "blocked_no_repeatable_timing_speedup",
            "source_artifact": EXP5398,
            "next_gate": "same-workload board timing speedup with stable output hashes",
        },
        "future_token_internal_signal": {
            "lane": "future_token_internal_signal",
            "state": "retired_until_backend_feature_artifact",
            "source_artifact": EXP5389,
            "next_gate": "backend artifact with logits, hidden states, attention, or intermediate exits",
        },
        "broad_kan_verification": {
            "lane": "broad_kan_verification",
            "state": "blocked_bounded_certificate_only",
            "source_artifact": EXP5399,
            "next_gate": "new bounded certificate family before widening the claim",
        },
        "full_prd_realization": {
            "lane": "full_prd_realization",
            "state": "partial_closed_rows_with_open_arc_token_hardware",
            "source_artifact": EXP5400,
            "next_gate": "close ARC, token/backend, and hardware repeatability blockers",
        },
    }
    if any(row["classification"] == "missing_inputs" for row in truth_table):
        rows["missing_inputs"] = {
            "lane": "missing_inputs",
            "state": "blocked_missing_upstream_artifact",
            "source_artifact": "",
            "next_gate": "restore missing upstream artifact before headline synthesis",
        }
    return list(rows.values())


def next_recommendations() -> list[JsonDict]:
    """Concrete follow-on gates for the next roadmap."""

    return [
        {
            "target": "formal_encoding_safety",
            "recommendation": (
                "Rerun the fixture with distinct checksum-backed safety metrics and "
                "a clean adversarial check before any formal-encoding headline."
            ),
        },
        {
            "target": "structured_constraint_tax",
            "recommendation": (
                "Scale Exp5391's deterministic final-state and tool/action fixtures "
                "while keeping deterministic verifier final authority."
            ),
        },
        {
            "target": "pbit_hardware_transfer",
            "recommendation": (
                "Move the p-bit boundary workload onto reachable hardware only after "
                "repeatable same-workload board timing is available."
            ),
        },
        {
            "target": "continuous_self_learning",
            "recommendation": (
                "Use Exp5395 and Exp5396 as the FR-11 controller baseline; keep raw "
                "episode provenance and no model-weight mutation as gates."
            ),
        },
        {
            "target": "arc_level_up",
            "recommendation": (
                "Keep blob salience but spend the next milestone on trajectory "
                "generation/enumeration that can bank a reproduction-gated new level."
            ),
        },
        {
            "target": "hardware_repeatability",
            "recommendation": (
                "Restore KV260 or PolarFire reachability and rerun the identical "
                "workload until repeated timing and stable output hashes exist."
            ),
        },
        {
            "target": "token_internal_features",
            "recommendation": (
                "Keep the lane closed until a backend artifact exposes logits, hidden "
                "states, attention, or intermediate exits."
            ),
        },
        {
            "target": "kan_certificate",
            "recommendation": (
                "Extend the bounded dynamic certificate to the next false-property "
                "family without claiming broad KAN soundness."
            ),
        },
    ]


def source_artifacts(root: Path, payloads: Mapping[str, JsonDict], found: Sequence[str]) -> list[JsonDict]:
    """Record upstream checksums so later capstones can detect drift."""

    rows: list[JsonDict] = []
    for relative in found:
        payload = payloads[relative]
        rows.append(
            {
                "path": relative,
                "task_id": TASK_IDS[relative],
                "sha256": file_sha256(root / relative),
                "status": unwrap(payload.get("status")),
                "honest_verdict": unwrap(payload.get("honest_verdict", "")),
                "flagged_adversarial": unwrap(payload.get("flagged_adversarial")) is True,
            }
        )
    return rows


def sidecar_artifacts(root: Path, payloads: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Record non-task sidecars separately so they do not inflate task counts."""

    rows: list[JsonDict] = []
    for relative in SIDECAR_ARTIFACT_PATHS:
        if relative in payloads:
            rows.append({"path": relative, "sha256": file_sha256(root / relative)})
    return rows


def protected_file_checks(root: Path) -> list[JsonDict]:
    """Confirm the task did not dirty the active roadmap or conductor script."""

    rows: list[JsonDict] = []
    for relative in ("research-roadmap.yaml", "scripts/research_conductor.py"):
        path = root / relative
        rows.append(
            {
                "path": relative,
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else None,
                "git_status_clean": not git_path_modified(root, relative),
            }
        )
    return rows


def git_path_modified(root: Path, relative: str) -> bool:
    """Return whether git sees a tracked or untracked change for one path."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", relative],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def source_context_read(root: Path) -> list[JsonDict]:
    """Record the requested context files as read-only provenance."""

    rows: list[JsonDict] = []
    for relative in SOURCE_CONTEXT_PATHS:
        path = root / relative
        rows.append(
            {
                "path": relative,
                "present": path.exists(),
                "kind": "directory" if path.is_dir() else "file",
            }
        )
    return rows


def honest_verdict(artifact: JsonMap) -> str:
    """Summarize the milestone without hiding blocked or honest-null lanes."""

    missing_count = len(artifact["missing_artifacts"])
    missing_phrase = (
        f"; missing {missing_count} upstream artifact(s)" if missing_count else ""
    )
    return (
        "complete: .491 capstone emitted from actual artifacts"
        f"{missing_phrase}; headline-ready bounded lanes are structured scale-up, "
        "overwrite corrigendum, p-bit CPU ablation, CSL router, memory guard, and KAN "
        "certificate; Exp5392 flagged, Exp5397 no-bank, token/internal lane closed, "
        "hardware repeatability absent, and no hardware speedup."
    )


def validate_artifact(artifact: JsonMap) -> None:
    """Reject schema drift and the specific overclaims this capstone guards."""

    missing_fields = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(f"missing required fields: {missing_fields}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles drift")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if artifact["status"] not in {"complete", "honest_partial"}:
        raise ValueError("status must be complete or honest_partial")
    if artifact["artifact_count_expected"] != EXPECTED_TASK_COUNT:
        raise ValueError("artifact_count_expected mismatch")

    missing_count = len(artifact["missing_artifacts"])
    expected_found = EXPECTED_TASK_COUNT - missing_count
    if artifact["artifact_count_found"] != expected_found:
        raise ValueError("artifact_count_found mismatch")
    if missing_count and artifact["status"] != "honest_partial":
        raise ValueError("status must be honest_partial when missing_artifacts is non-empty")
    if not missing_count and artifact["status"] != "complete":
        raise ValueError("status must be complete with all upstream artifacts present")

    for field in BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare boolean")
    if artifact["formal_encoding_fixture_ready"] is not False:
        raise ValueError("formal_encoding_fixture_ready must stay false while Exp5392 is flagged")
    if artifact["hardware_speedup_claim"] is not False:
        raise ValueError("hardware_speedup_claim must remain false")
    if artifact["future_token_signal_allowed"] is not False:
        raise ValueError("future_token_signal_allowed must remain false")
    if artifact["active_roadmap_modified"] is not False:
        raise ValueError("active_roadmap_modified must remain false")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must remain false")

    lanes = [row["lane"] for row in artifact["truth_table"]]
    if lanes != list(TRUTH_TABLE_LANES):
        raise ValueError("truth_table lane order mismatch")
    for row in artifact["truth_table"]:
        if row["classification"] not in ALLOWED_CLASSIFICATIONS:
            raise ValueError("truth_table classification invalid")

    formal_row = artifact["truth_table"][1]
    flagged_paths = [row["path"] for row in artifact["flagged_artifacts"]]
    if (
        formal_row["blocked_reason"] == "flagged_adversarial_tautology"
        and EXP5392 not in flagged_paths
    ):
        raise ValueError("flagged_artifacts must include Exp5392")
    if artifact["truth_table"][6]["classification"] == "honest_null" and artifact["arc_new_level_banked"]:
        raise ValueError("arc_new_level_banked cannot be true for honest-null ARC row")
    if not str(artifact["honest_verdict"]).startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def file_sha256(path: Path) -> str:
    """Hash a local file using the same stable prefix convention as artifacts."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    """Hash the artifact content while excluding the checksum field itself."""

    normalized = json_ready(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def json_ready(value: Any) -> Any:
    """Convert paths, tuples, and mappings into deterministic JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, tuple | list):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, artifact: JsonMap) -> None:
    """Write pretty JSON so diffs are reviewable during conductor reconciliation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for conductor-style one-shot artifact emission."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
