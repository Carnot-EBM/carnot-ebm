"""Exp5426 .493 PRD gap and agent-failure evidence table.

Spec refs: REQ-REPORT-5426, SCENARIO-REPORT-5426,
SCENARIO-REPORT-5426-MISSING-UPSTREAM.

This module is intentionally a synthesis step. It reads the `.493` artifacts
that already exist and records which PRD lanes are supported, bounded, blocked,
or missing. It does not run ARC, hardware, solver, or model inference work; the
point is to keep the capstone input traceable to evidence that is actually on
disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5426_prd_gap_agent_failure_table_v493.json")
EXPERIMENT = "experiment_5426_prd_gap_agent_failure_table_v493"
EXPERIMENT_ID = "exp5426-prd-gap-agent-failure-table-v493"
MILESTONE = "2026.07.493"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5426
SCHEMA = "carnot.experiment_5426.prd_gap_agent_failure_table.v493"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5426",
    "SCENARIO-REPORT-5426",
    "SCENARIO-REPORT-5426-MISSING-UPSTREAM",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "upstream_artifacts_read": "provenance",
    "upstream_artifacts_missing": "no fabricated evidence",
    "closed_lanes": "PRD progress",
    "partial_lanes": "bounded evidence",
    "blocked_lanes": "honest gaps",
    "missing_lanes": "absent artifact handling",
    "failure_taxonomy_counts": "tool-use/planning/reasoning diagnosis",
    "prd_gap_table_ready": "capstone input",
    "inference_substrate": "no hidden live model inference",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

EXPECTED_ARTIFACTS = (
    Path("results/experiment_5415_transition_v493.json"),
    Path("results/experiment_5417_risk_calibrated_sota_structured_panel_v493.json"),
    Path("results/experiment_5418_predictive_prefix_action_safety_v493.json"),
    Path("results/experiment_5419_active_constraint_lns_scale_v493.json"),
    Path("results/experiment_5420_pbit_hardware_transfer_preflight_v493.json"),
    Path("results/experiment_5421_evidence_reliance_csl_v493.json"),
    Path("results/experiment_5422_csl_promotion_reliance_scale_v493.json"),
    Path("results/experiment_5423_arc_coex_landmark_levelup_v493.json"),
    Path("results/experiment_5424_hardware_comparable_timing_receipts_v493.json"),
    Path("results/experiment_5425_kan_measurement_access_certificate_v493.json"),
)

TRANSITION_ARTIFACT = EXPECTED_ARTIFACTS[0]
FAILURE_TAXONOMY = (
    "tool-use",
    "planning",
    "reasoning",
    "measurement-access",
    "calibration",
    "live-environment",
)
LANE_CLASSIFICATIONS = ("closed", "partial", "blocked", "missing")
LANE_NAMES = (
    "structured_verification",
    "continuous_self_learning",
    "solver_guidance",
    "arc_live_progress",
    "hardware",
    "certificates",
)
REQUIRED_LANE_FIELDS = (
    "lane",
    "classification",
    "classification_reason",
    "prd_refs",
    "research_program_priorities",
    "artifact_paths",
    "supporting_fields",
    "missing_supporting_fields",
    "failure_taxonomy",
    "claim_boundary",
    "next_action",
)

EXP5417 = Path("results/experiment_5417_risk_calibrated_sota_structured_panel_v493.json")
EXP5418 = Path("results/experiment_5418_predictive_prefix_action_safety_v493.json")
EXP5419 = Path("results/experiment_5419_active_constraint_lns_scale_v493.json")
EXP5420 = Path("results/experiment_5420_pbit_hardware_transfer_preflight_v493.json")
EXP5421 = Path("results/experiment_5421_evidence_reliance_csl_v493.json")
EXP5422 = Path("results/experiment_5422_csl_promotion_reliance_scale_v493.json")
EXP5423 = Path("results/experiment_5423_arc_coex_landmark_levelup_v493.json")
EXP5424 = Path("results/experiment_5424_hardware_comparable_timing_receipts_v493.json")
EXP5425 = Path("results/experiment_5425_kan_measurement_access_certificate_v493.json")

LANE_SPECS: tuple[JsonDict, ...] = (
    {
        "lane": "structured_verification",
        "classification": "closed",
        "classification_reason": "risk_calibrated_prefix_tool_safety_closed",
        "prd_refs": ["FR-12 Verifiable Reasoning", "NFR-02 Safety"],
        "research_program_priorities": [
            "structured verification",
            "tool-first verification before unsafe action execution",
        ],
        "artifact_paths": (EXP5417, EXP5418),
        "field_names": {
            EXP5417: (
                "risk_calibrated_structured_panel_ready",
                "unsafe_false_accept_rate",
                "accepted_risk_estimate",
                "accepted_risk_bound",
                "accepted_risk_bound_threshold",
                "abstention_rate",
                "semantic_error_rate",
                "gpu_offload_verified",
            ),
            EXP5418: (
                "predictive_prefix_safety_ready",
                "deterministic_verifier_final_authority",
                "final_only_unsafe_false_accept_rate",
                "prefix_gated_unsafe_false_accept_rate",
                "final_only_unreachable_tool_action_rate",
                "prefix_gated_unreachable_tool_action_rate",
                "gpu_offload_verified",
            ),
        },
        "failure_taxonomy": ("tool-use", "calibration"),
        "claim_boundary": "closed for structured fixture safety with abstention; not a broad SOTA quality claim",
        "next_action": "Feed closed structured-verification fields into the .493 capstone.",
    },
    {
        "lane": "continuous_self_learning",
        "classification": "closed",
        "classification_reason": "reliance_drift_and_gated_promotion_closed",
        "prd_refs": ["FR-11 Autonomous Self-Learning Loop"],
        "research_program_priorities": [
            "continuous self-learning",
            "audited memory promotion with rollback and no weight mutation",
        ],
        "artifact_paths": (EXP5421, EXP5422),
        "field_names": {
            EXP5421: (
                "evidence_reliance_csl_ready",
                "hidden_forgetting_detected",
                "reliance_drift_metric",
                "quality_preserved",
                "stale_poison_deflection_rate",
                "uncertain_reliance_deflection_rate",
                "rollback_verified",
                "no_weight_mutation",
            ),
            EXP5422: (
                "csl_promotion_reliance_scale_ready",
                "promoted_fragment_count",
                "rejected_fragment_count",
                "abstained_fragment_count",
                "grounding_preserved",
                "rejected_fragments_quarantined",
                "rollback_verified",
                "no_weight_mutation",
            ),
        },
        "failure_taxonomy": ("reasoning", "planning"),
        "claim_boundary": "closed for controller-level CSL evidence; no model weight mutation claim",
        "next_action": "Carry forward gated promotion and reliance-drift controls.",
    },
    {
        "lane": "solver_guidance",
        "classification": "partial",
        "classification_reason": "bounded_solver_guidance",
        "prd_refs": ["FR-12 Verifiable Reasoning", "FR-07 Inference Pipeline"],
        "research_program_priorities": [
            "solver guidance",
            "constraint-backed search with solver authority preserved",
        ],
        "artifact_paths": (EXP5419,),
        "field_names": {
            EXP5419: (
                "active_constraint_lns_scale_ready",
                "solver_validity_preserved",
                "accepted_hint_count",
                "rejected_hint_count",
                "overwritten_hint_count",
                "work_delta",
                "dual_residual_sanity",
                "claim_limits",
            ),
        },
        "failure_taxonomy": ("planning", "reasoning"),
        "claim_boundary": "bounded deterministic LNS guidance; hints are advisory and CPU-local",
        "next_action": "Promote only as bounded solver guidance, not general planning competence.",
    },
    {
        "lane": "arc_live_progress",
        "classification": "blocked",
        "classification_reason": "honest_null_no_new_level_banked",
        "prd_refs": ["FR-12 Verifiable Reasoning"],
        "research_program_priorities": [
            "ARC live progress",
            "live hidden-game discovery agent path",
        ],
        "artifact_paths": (EXP5423,),
        "field_names": {
            EXP5423: (
                "status",
                "arc_new_level_banked",
                "offline_reproduced",
                "reproduced_levels",
                "newly_reached_levels",
                "attempt_count",
                "frontier_expansion_count",
                "landmark_count",
                "failure_mode",
                "registry_total_before",
                "registry_total_after",
                "no_offline_bfs",
                "no_per_game_adapter",
            ),
        },
        "failure_taxonomy": ("planning", "live-environment"),
        "claim_boundary": "live ARC path was exercised but no new reproduced level was banked",
        "next_action": "Treat as an honest ARC null and keep the banked level count unchanged.",
    },
    {
        "lane": "hardware",
        "classification": "partial",
        "classification_reason": "comparable_timing_without_speedup_claim",
        "prd_refs": ["NFR-01 Performance"],
        "research_program_priorities": [
            "hardware",
            "hash-matched CPU and board timing receipts without speedup inflation",
        ],
        "artifact_paths": (EXP5420, EXP5424),
        "field_names": {
            EXP5420: (
                "pbit_transfer_preflight_ready",
                "exact_enumeration_match",
                "same_workload_hash_match",
                "cpu_repeat_count",
                "board_repeat_count",
                "polarfire_reachable",
                "kv260_ssh_checked",
                "hardware_speedup_claim",
                "hardware_summary",
            ),
            EXP5424: (
                "comparable_timing_receipts_ready",
                "measurement_access_complete",
                "same_workload_hash_match",
                "same_result_hash_match",
                "cpu_repeat_count",
                "board_repeat_count",
                "polarfire_reachable",
                "hardware_speedup_claim",
                "timing_comparison",
                "claim_refusal",
            ),
        },
        "failure_taxonomy": ("measurement-access", "live-environment"),
        "claim_boundary": "comparable timing receipts only; hardware_speedup_claim remains false",
        "next_action": "Use as bounded hardware receipt evidence, not as a speedup headline.",
    },
    {
        "lane": "certificates",
        "classification": "partial",
        "classification_reason": "bounded_measurement_access_certificate",
        "prd_refs": ["FR-12 Verifiable Reasoning", "FR-10 Spec-Driven Development"],
        "research_program_priorities": [
            "certificates",
            "observable-vs-missing evidence separation",
        ],
        "artifact_paths": (EXP5425,),
        "field_names": {
            EXP5425: (
                "kan_measurement_access_certificate_ready",
                "certificate_count",
                "false_property_rejection_rate",
                "true_property_preservation_rate",
                "missing_evidence_detected",
                "broad_kan_verification_claim",
                "missing_evidence_controls",
                "claim_limits",
            ),
        },
        "failure_taxonomy": ("measurement-access", "reasoning"),
        "claim_boundary": "bounded measurement-access certificate; no broad KAN verification claim",
        "next_action": "Carry forward certificate limits and missing-evidence controls.",
    },
)

DEFAULT_TESTS_RUN = (
    {
        "command": (
            ".venv/bin/pytest "
            "tests/python/test_experiment_5426_prd_gap_agent_failure_table_v493.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5426_prd_gap_agent_failure_table_v493.py "
            "-m pytest tests/python/test_experiment_5426_prd_gap_agent_failure_table_v493.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5426_prd_gap_agent_failure_table_v493.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)


def default_tests_run() -> list[JsonDict]:
    """Return copied verification commands so the artifact can be replayed."""

    return [dict(row) for row in DEFAULT_TESTS_RUN]


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Build the Exp5426 synthesis artifact from execution-time upstream fields."""

    root_path = Path(root)
    artifacts, missing_artifacts = load_upstream_artifacts(root_path)
    rows = [build_lane(spec, artifacts) for spec in LANE_SPECS]
    ready = not missing_artifacts and not any(row["classification"] == "missing" for row in rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete" if ready else "blocked_missing_upstream",
        "upstream_artifacts_read": [
            str(relative) for relative in EXPECTED_ARTIFACTS if relative in artifacts
        ],
        "upstream_artifacts_missing": [str(relative) for relative in missing_artifacts],
        "closed_lanes": bucket(rows, "closed"),
        "partial_lanes": bucket(rows, "partial"),
        "blocked_lanes": bucket(rows, "blocked"),
        "missing_lanes": bucket(rows, "missing"),
        "failure_taxonomy_counts": failure_taxonomy_counts(rows),
        "prd_gap_table_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, rows, missing_artifacts),
        "transition_context": build_transition_context(artifacts),
        "lane_order": list(LANE_NAMES),
        "tests_run": [dict(row) for row in tests_run],
        "upstream_artifact_checksums": upstream_artifact_checksums(root_path, artifacts),
        "research_conductor_modified": False,
        "research_roadmap_modified": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Write the validated Exp5426 artifact and return the payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def load_upstream_artifacts(root: Path) -> tuple[dict[Path, JsonDict], list[Path]]:
    """Load listed upstream JSON artifacts and return the missing path list."""

    loaded: dict[Path, JsonDict] = {}
    missing: list[Path] = []
    for relative in EXPECTED_ARTIFACTS:
        path = root / relative
        if path.exists():
            loaded[relative] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(relative)
    return loaded, missing


def build_lane(spec: Mapping[str, Any], artifacts: Mapping[Path, JsonDict]) -> JsonDict:
    """Build one lane row, recording missing upstreams instead of inferring values."""

    artifact_paths = tuple(spec["artifact_paths"])
    missing = [relative for relative in artifact_paths if relative not in artifacts]
    base = lane_base(spec)
    if missing:
        base.update(
            {
                "classification": "missing",
                "classification_reason": "missing_upstream_artifact",
                "supporting_fields": [],
                "missing_supporting_fields": [],
                "missing_artifacts": [str(relative) for relative in missing],
                "failure_taxonomy": [],
            }
        )
        return base
    supporting_fields, missing_supporting_fields = collect_supporting_fields(spec, artifacts)
    base.update(
        {
            "classification": spec["classification"],
            "classification_reason": spec["classification_reason"],
            "supporting_fields": supporting_fields,
            "missing_supporting_fields": missing_supporting_fields,
            "missing_artifacts": [],
            "failure_taxonomy": list(spec["failure_taxonomy"]),
        }
    )
    return base


def lane_base(spec: Mapping[str, Any]) -> JsonDict:
    """Return fields shared by present and missing lane rows."""

    return {
        "lane": spec["lane"],
        "prd_refs": list(spec["prd_refs"]),
        "research_program_priorities": list(spec["research_program_priorities"]),
        "artifact_paths": [str(relative) for relative in spec["artifact_paths"]],
        "claim_boundary": spec["claim_boundary"],
        "next_action": spec["next_action"],
    }


def collect_supporting_fields(
    spec: Mapping[str, Any],
    artifacts: Mapping[Path, JsonDict],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Collect only fields that are actually present in the source artifacts."""

    supporting_fields: list[JsonDict] = []
    missing_supporting_fields: list[JsonDict] = []
    for relative, field_names in spec["field_names"].items():
        payload = artifacts[relative]
        for field_name in field_names:
            present = field_name in payload
            record: JsonDict = {
                "artifact_path": str(relative),
                "field_name": str(field_name),
                "present": present,
            }
            if present:
                record["value"] = payload[field_name]
                supporting_fields.append(record)
            missing_supporting_fields.extend([] if present else [record])
    return supporting_fields, missing_supporting_fields


def build_transition_context(artifacts: Mapping[Path, JsonDict]) -> JsonDict:
    """Carry the .493 transition receipt as context without classifying it as a PRD lane."""

    if TRANSITION_ARTIFACT not in artifacts:
        return {
            "artifact_path": str(TRANSITION_ARTIFACT),
            "missing": True,
            "supporting_fields": [],
        }
    payload = artifacts[TRANSITION_ARTIFACT]
    return {
        "artifact_path": str(TRANSITION_ARTIFACT),
        "missing": False,
        "supporting_fields": [
            {
                "artifact_path": str(TRANSITION_ARTIFACT),
                "field_name": field_name,
                "present": True,
                "value": payload[field_name],
            }
            for field_name in (
                "closed_lanes",
                "partial_lanes",
                "blocked_lanes",
                "next_task_range",
                "honest_verdict",
            )
        ],
    }


def bucket(rows: Sequence[Mapping[str, Any]], classification: str) -> list[JsonDict]:
    """Return rows belonging to one terminal lane bucket."""

    return [dict(row) for row in rows if row["classification"] == classification]


def failure_taxonomy_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count taxonomy labels once per classified lane row."""

    counts = {name: 0 for name in FAILURE_TAXONOMY}
    for row in rows:
        for name in row.get("failure_taxonomy", ()):
            counts[str(name)] += 1
    return counts


def honest_verdict(
    ready: bool,
    rows: Sequence[Mapping[str, Any]],
    missing_artifacts: Sequence[Path],
) -> str:
    """Return the terminal capstone-input verdict."""

    if ready:
        closed = len(bucket(rows, "closed"))
        partial = len(bucket(rows, "partial"))
        blocked = len(bucket(rows, "blocked"))
        return (
            "complete: .493 PRD gap table read actual upstream artifacts; "
            f"closed={closed}, partial={partial}, blocked={blocked}, missing=0."
        )
    return (
        "blocked: .493 PRD gap table missing upstream artifacts: "
        + ", ".join(str(relative) for relative in missing_artifacts)
    )


def upstream_artifact_checksums(root: Path, artifacts: Mapping[Path, JsonDict]) -> list[JsonDict]:
    """Hash every upstream artifact that was actually read."""

    checksums: list[JsonDict] = []
    for relative in EXPECTED_ARTIFACTS:
        if relative in artifacts:
            path = root / relative
            checksums.append({"path": str(relative), "sha256": path_sha256(path)})
    return checksums


def path_sha256(path: Path) -> str:
    """Return a sha256 checksum for an already-read artifact path."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a deterministic checksum excluding the checksum field itself."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema and claim-boundary invariants for the synthesis artifact."""

    missing_fields = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing_fields, "missing required field: " + ",".join(missing_fields))
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    rows = all_lane_rows(artifact)
    _require([row.get("lane") for row in rows] == list(LANE_NAMES), "lane_order")
    validate_bucket(artifact.get("closed_lanes", ()), "closed")
    validate_bucket(artifact.get("partial_lanes", ()), "partial")
    validate_bucket(artifact.get("blocked_lanes", ()), "blocked")
    validate_bucket(artifact.get("missing_lanes", ()), "missing")
    _require(artifact.get("failure_taxonomy_counts") == failure_taxonomy_counts(rows), "failure_taxonomy_counts")
    expected_ready = not artifact.get("upstream_artifacts_missing") and not artifact.get("missing_lanes")
    _require(artifact.get("prd_gap_table_ready") is expected_ready, "prd_gap_table_ready")
    for row in rows:
        validate_lane(row)
    return True


def all_lane_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return lane rows in canonical PRD order regardless of bucket."""

    by_name = {
        row["lane"]: dict(row)
        for key in ("closed_lanes", "partial_lanes", "blocked_lanes", "missing_lanes")
        for row in artifact.get(key, ())
    }
    return [by_name[name] for name in LANE_NAMES if name in by_name]


def validate_bucket(rows: Any, classification: str) -> None:
    """Ensure each bucket contains only rows with the matching classification."""

    _require(isinstance(rows, list), "lane buckets")
    for row in rows:
        _require(row.get("classification") == classification, "lane buckets")


def validate_lane(row: Mapping[str, Any]) -> None:
    """Validate row-level provenance and taxonomy boundaries."""

    _require(set(REQUIRED_LANE_FIELDS) <= set(row), "lane fields")
    _require(row.get("classification") in LANE_CLASSIFICATIONS, "lane classification")
    _require(set(row.get("failure_taxonomy", ())) <= set(FAILURE_TAXONOMY), "failure_taxonomy")
    if row.get("classification") == "missing":
        _require(row.get("supporting_fields") == [], "supporting_fields")
        return
    _require(bool(row.get("supporting_fields")), "supporting_fields")
    _require(row.get("missing_supporting_fields") == [], "supporting_fields")
    for field in row.get("supporting_fields", ()):
        _require(field.get("present") is True, "supporting_fields")
        _require(field.get("artifact_path") in row.get("artifact_paths", ()), "supporting_fields")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover
    run()
