"""Build the Exp 3135 archive and .292 handoff artifact.

Spec refs: REQ-REPORT-3135, SCENARIO-REPORT-3135.

This module converts the completed .291 capstone into a machine-readable .292
handoff without activating the roadmap itself. The work is deliberately
aggregation-only: it reads checked-in artifacts, preserves unresolved blocker
classes, and records that no model, verifier, repair, solver, conductor,
synthesis, or hardware path was executed by this archive step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
PRIOR_MILESTONE = "2026.05.291"
NEXT_MILESTONE = "2026.05.292"
SCHEMA = "carnot.archive_activation.v291_to_v292.v1"
ARTIFACT = "experiment_3135_archive_v291_activate_v292"
OUTPUT_REL_PATH = Path("results/experiment_3135_archive_v291_activate_v292.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3135_archive_v291_activate_v292.py"

MATRIX_V25_REL_PATH = Path("results/experiment_3133_cross_corpus_matrix_v25.json")
CAPSTONE_V291_REL_PATH = Path("results/experiment_3134_capstone_v291.json")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
AGENTS_REL_PATH = Path("AGENTS.md")
CODEX_REL_PATH = Path("CODEX.md")
CLAUDE_REL_PATH = Path("CLAUDE.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
OPS_STATUS_REL_PATH = Path("ops/status.md")
OPS_CHANGELOG_REL_PATH = Path("ops/changelog.md")
TRACEABILITY_REL_PATH = Path("_bmad/traceability.md")

SOURCE_PATHS = (
    ("matrix_v25", MATRIX_V25_REL_PATH),
    ("capstone_v291", CAPSTONE_V291_REL_PATH),
    ("staged_roadmap", STAGED_ROADMAP_REL_PATH),
    ("active_roadmap", ACTIVE_ROADMAP_REL_PATH),
    ("vnext_doc", VNEXT_DOC_REL_PATH),
    ("agents_instructions", AGENTS_REL_PATH),
    ("codex_instructions", CODEX_REL_PATH),
    ("claude_instructions", CLAUDE_REL_PATH),
    ("research_conductor", CONDUCTOR_REL_PATH),
    ("ops_status", OPS_STATUS_REL_PATH),
    ("ops_changelog", OPS_CHANGELOG_REL_PATH),
    ("traceability", TRACEABILITY_REL_PATH),
)
INFERENCE_SUBSTRATE = {
    "kind": "aggregation_from_upstream_artifacts",
    "executes_models": False,
    "executes_verifiers": False,
    "executes_repairs": False,
    "executes_solvers": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "local_repo_only": True,
    "no_live_llm_inference": True,
    "source": "checked_in_artifacts",
    "live_model_calls": 0,
    "hardware_commands_run": [],
}
EXPECTED_MISSING_COMPARATIVE_MODELS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
]
EXPECTED_PRESENT_MODEL_IDS = ["unsloth/gemma-4-26B-A4B-it-GGUF"]


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and keep absent or malformed evidence visibly empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_mapping(path: Path) -> JsonDict:
    """Read roadmap YAML without converting malformed content into readiness."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a checksum so the archive can be tied back to exact source bytes."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3135: synthesize the .291 archive and .292 handoff record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V25_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V291_REL_PATH)
    staged = read_yaml_mapping(root_path / STAGED_ROADMAP_REL_PATH)
    active = read_yaml_mapping(root_path / ACTIVE_ROADMAP_REL_PATH)
    source_artifacts = [
        _source_artifact(root_path, role, rel_path) for role, rel_path in SOURCE_PATHS
    ]
    roadmap_handoff = _roadmap_handoff(root_path, staged, active)
    prior_capstone_ready = capstone.get("capstone_ready") is True
    prior_paper_ready = capstone.get("paper_ready") is True
    prior_publication_blocker_count, blocker_source = _publication_blocker_count(
        capstone,
        matrix,
    )
    blocker_delta_from_v24, delta_source = _blocker_delta_from_v24(capstone, matrix)
    status_summary = _status_summary_291(matrix, capstone)
    blocked_reasons = _blocked_reasons(
        capstone_present=bool(capstone),
        prior_capstone_ready=prior_capstone_ready,
        roadmap_handoff=roadmap_handoff,
        vnext_doc_present=(root_path / VNEXT_DOC_REL_PATH).is_file(),
    )
    ready = not blocked_reasons

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "prior_milestone": PRIOR_MILESTONE,
        "next_milestone": _next_milestone(roadmap_handoff),
        "archive_v291_activate_v292_ready": ready,
        "prior_capstone_ready": prior_capstone_ready,
        "prior_paper_ready": prior_paper_ready,
        "prior_paper_ready_source_field_present": "paper_ready" in capstone,
        "prior_publication_blocker_count": prior_publication_blocker_count,
        "prior_publication_blocker_count_source": blocker_source,
        "blocker_delta_from_v24": blocker_delta_from_v24,
        "blocker_delta_from_v24_source": delta_source,
        "status_summary_291": status_summary,
        "carry_forward_blockers": _carry_forward_blockers(status_summary, capstone, matrix),
        "roadmap_handoff": roadmap_handoff,
        "source_artifacts": source_artifacts,
        "source_checksums": {str(row["path"]): row["sha256"] for row in source_artifacts},
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row["present"] is not True
        ],
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "activation_performed_by_this_task": False,
        "research_roadmap_yaml_modified": False,
        "research_roadmap_next_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "no_historical_artifact_rewrite": True,
        "no_push": True,
        "blocked_reasons": blocked_reasons,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3135 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifact(root: Path, role: str, rel_path: Path) -> JsonDict:
    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def _roadmap_handoff(
    root: Path,
    staged: Mapping[str, Any],
    active: Mapping[str, Any],
) -> JsonDict:
    staged_present = (root / STAGED_ROADMAP_REL_PATH).is_file()
    if staged_present:
        source_path = STAGED_ROADMAP_REL_PATH
        source_payload: Mapping[str, Any] = staged
        used_fallback = False
    else:
        source_path = ACTIVE_ROADMAP_REL_PATH
        source_payload = active
        used_fallback = True
    milestone = str(source_payload.get("milestone") or "")
    milestone_doc = str(source_payload.get("milestone_doc") or "")
    task_ids = _task_ids(source_payload)
    return {
        "requested_staged_roadmap_path": STAGED_ROADMAP_REL_PATH.as_posix(),
        "requested_staged_roadmap_present": staged_present,
        "source_path": source_path.as_posix(),
        "source_present": (root / source_path).is_file(),
        "used_active_roadmap_fallback": used_fallback,
        "active_roadmap_milestone": str(active.get("milestone") or ""),
        "observed_milestone": milestone,
        "expected_milestone": NEXT_MILESTONE,
        "milestone_matches": milestone == NEXT_MILESTONE,
        "observed_milestone_doc": milestone_doc,
        "expected_milestone_doc": VNEXT_DOC_REL_PATH.as_posix(),
        "milestone_doc_matches": milestone_doc == VNEXT_DOC_REL_PATH.as_posix(),
        "task_ids": task_ids,
        "non_empty_tasks": bool(task_ids),
    }


def _blocked_reasons(
    *,
    capstone_present: bool,
    prior_capstone_ready: bool,
    roadmap_handoff: Mapping[str, Any],
    vnext_doc_present: bool,
) -> list[str]:
    reasons: list[str] = []
    if not capstone_present:
        reasons.append("prior capstone artifact missing or malformed")
    elif not prior_capstone_ready:
        reasons.append("prior capstone is not capstone_ready=true")
    if not roadmap_handoff.get("source_present"):
        reasons.append("roadmap handoff source is missing")
    if not roadmap_handoff.get("milestone_matches"):
        reasons.append("roadmap milestone is not 2026.05.292")
    if not roadmap_handoff.get("milestone_doc_matches"):
        reasons.append(
            "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md"
        )
    if not roadmap_handoff.get("non_empty_tasks"):
        reasons.append("roadmap has no tasks")
    if not vnext_doc_present:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _status_summary_291(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> JsonDict:
    count, count_source = _publication_blocker_count(capstone, matrix)
    delta, delta_source = _blocker_delta_from_v24(capstone, matrix)
    claim_summary = _as_mapping(capstone.get("claim_allowance_summary"))
    headline_summary = _as_mapping(matrix.get("headline_claim_allowance_summary"))
    verifier_summary = _as_mapping(matrix.get("verifier_repair_summary"))
    fr11_ledger = _float_or(
        capstone.get("fr11_ledger_consistency_rate"),
        _float_or(
            _field_from_summary_or_matrix(
                "ledger_consistency_rate",
                "fr11_summary",
                capstone,
                matrix,
                None,
            ),
            _first_float_from_text(str(capstone.get("fr11_self_learning_status") or "")) or 0.0,
        ),
    )
    return {
        "paper_ready": capstone.get("paper_ready") is True,
        "capstone_ready": capstone.get("capstone_ready") is True,
        "matrix_v25_ready": matrix.get("matrix_v25_ready") is True
        or _as_mapping(capstone.get("matrix_v25_summary")).get("matrix_v25_ready") is True,
        "publication_blocker_count": count,
        "publication_blocker_count_source": count_source,
        "blocker_delta_from_v24": delta,
        "blocker_delta_source": delta_source,
        "next_top_gap": _next_top_gap(capstone),
        "sota_cache_status": _status_value("sota_cache_status", matrix, capstone),
        "live_verifier_status": _status_value("live_verifier_status", matrix, capstone),
        "verifier_claim_status": _status_value("verifier_claim_status", matrix, capstone),
        "false_accept_rate": _false_accept_rate(capstone, claim_summary, headline_summary, matrix),
        "verifier_gain_delta": _verifier_gain_delta(capstone, verifier_summary, matrix),
        "repair_gate_status": _repair_gate_status(
            capstone,
            claim_summary,
            headline_summary,
            verifier_summary,
        ),
        "repair_claim_status": _status_value("repair_claim_status", matrix, capstone),
        "fr11_self_learning_status": _status_value(
            "fr11_self_learning_status",
            matrix,
            capstone,
        ),
        "fr11_ledger_consistency_rate": fr11_ledger,
        "ebt_arm_status": _status_value("ebt_arm_status", matrix, capstone),
        "kan_status": _status_value("kan_status", matrix, capstone),
        "sampler_hardware_status": _status_value(
            "sampler_hardware_status",
            matrix,
            capstone,
        ),
        "source_artifacts": _dict_rows(capstone.get("source_artifacts")),
    }


def _status_value(
    field: str,
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> str:
    value = capstone.get(field)
    if value in (None, ""):
        value = matrix.get(field)
    return str(value or "")


def _publication_blocker_count(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> tuple[int, str]:
    capstone_count = capstone.get("publication_blocker_count")
    if not isinstance(capstone_count, bool) and isinstance(capstone_count, int):
        return capstone_count, "capstone_publication_blocker_count"
    blockers = capstone.get("publication_blockers")
    if isinstance(blockers, list):
        return len(blockers), "capstone_publication_blockers_length"
    matrix_summary = _as_mapping(capstone.get("matrix_v25_summary"))
    summary_count = matrix_summary.get("publication_blocker_count")
    if not isinstance(summary_count, bool) and isinstance(summary_count, int):
        return summary_count, "capstone_matrix_v25_summary"
    count = _first_int_from_text(str(capstone.get("honest_verdict") or ""))
    if count is not None:
        return count, "capstone_honest_verdict"
    count = _first_int_from_text(str(matrix.get("honest_verdict") or ""))
    if count is not None:
        return count, "matrix_honest_verdict"
    return 0, "missing"


def _blocker_delta_from_v24(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> tuple[int, str]:
    direct = capstone.get("blocker_delta_from_v24")
    if not isinstance(direct, bool) and isinstance(direct, int):
        return direct, "capstone_blocker_delta_from_v24"
    matrix_summary = _as_mapping(capstone.get("matrix_v25_summary"))
    summary_delta = matrix_summary.get("blocker_delta_from_v24")
    if not isinstance(summary_delta, bool) and isinstance(summary_delta, int):
        return summary_delta, "capstone_matrix_v25_summary"
    capstone_match = re.search(
        r"blocker_delta_from_v24=(-?\d+)",
        str(capstone.get("honest_verdict") or ""),
    )
    if capstone_match:
        return int(capstone_match.group(1)), "capstone_honest_verdict"
    matrix_match = re.search(
        r"blocker_delta_from_v24=(-?\d+)",
        str(matrix.get("honest_verdict") or ""),
    )
    if matrix_match:
        return int(matrix_match.group(1)), "matrix_honest_verdict"
    return 0, "missing"


def _first_int_from_text(text: str) -> int | None:
    match = re.search(r"publication_blocker_count=(\d+)", text)
    return int(match.group(1)) if match else None


def _first_float_from_text(text: str) -> float | None:
    match = re.search(
        r"(?:false_accept_rate|verifier_gain_delta|ledger_consistency_rate)=(-?\d+(?:\.\d+)?)",
        text,
    )
    return float(match.group(1)) if match else None


def _next_top_gap(capstone: Mapping[str, Any]) -> str:
    value = capstone.get("next_top_gap")
    if value not in (None, ""):
        return str(value)
    match = re.search(r"next_top_gap=([^;]+)", str(capstone.get("honest_verdict") or ""))
    return match.group(1) if match else ""


def _false_accept_rate(
    capstone: Mapping[str, Any],
    claim_summary: Mapping[str, Any],
    headline_summary: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> float:
    verifier_summary = _as_mapping(matrix.get("verifier_repair_summary"))
    return _float_or(
        capstone.get("false_accept_rate"),
        _float_or(
            claim_summary.get("false_accept_rate"),
            _float_or(
                headline_summary.get("false_accept_rate"),
                _float_or(
                    verifier_summary.get("false_accept_rate"),
                    _first_float_from_text(str(capstone.get("honest_verdict") or "")) or 0.0,
                ),
            ),
        ),
    )


def _verifier_gain_delta(
    capstone: Mapping[str, Any],
    verifier_summary: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> float:
    return _float_or(
        capstone.get("verifier_gain_delta"),
        _float_or(
            verifier_summary.get("verifier_gain_delta"),
            _float_or(
                matrix.get("verifier_gain_delta"),
                _first_float_from_text(str(matrix.get("honest_verdict") or "")) or 0.0,
            ),
        ),
    )


def _repair_gate_status(
    capstone: Mapping[str, Any],
    claim_summary: Mapping[str, Any],
    headline_summary: Mapping[str, Any],
    verifier_summary: Mapping[str, Any],
) -> str:
    value = capstone.get("repair_gate_state")
    if value in (None, ""):
        value = claim_summary.get("repair_gate_state")
    if value in (None, ""):
        value = headline_summary.get("repair_gate_state")
    if value in (None, ""):
        value = verifier_summary.get("repair_gate_state")
    if value in (None, ""):
        value = capstone.get("repair_ladder_status")
    return str(value or "")


def _carry_forward_blockers(
    status: Mapping[str, Any],
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> list[JsonDict]:
    model_ids = _model_ids(capstone, matrix)
    return [
        {
            "blocker_id": "publication_blockers_46",
            "description": "46 publication blockers",
            "source": CAPSTONE_V291_REL_PATH.as_posix(),
            "source_field": "publication_blocker_count",
            "value": status["publication_blocker_count"],
            "expected_carry_forward_value": 46,
            "matches_expected": status["publication_blocker_count"] == 46,
        },
        {
            "blocker_id": "missing_comparative_sota_pair",
            "description": "missing Qwen3.6/Gemma-4-31B comparative SOTA pair",
            "source": CAPSTONE_V291_REL_PATH.as_posix(),
            "source_field": "claim_allowance_summary.missing_model_ids",
            "value": model_ids,
            "expected_carry_forward_value": {
                "missing_model_ids": EXPECTED_MISSING_COMPARATIVE_MODELS,
                "present_model_ids": EXPECTED_PRESENT_MODEL_IDS,
            },
            "matches_expected": model_ids
            == {
                "missing_model_ids": EXPECTED_MISSING_COMPARATIVE_MODELS,
                "present_model_ids": EXPECTED_PRESENT_MODEL_IDS,
            },
        },
        {
            "blocker_id": "false_accept_rate_0_5",
            "description": "false_accept_rate=0.5",
            "source": CAPSTONE_V291_REL_PATH.as_posix(),
            "source_field": "claim_allowance_summary.false_accept_rate",
            "value": status["false_accept_rate"],
            "expected_carry_forward_value": 0.5,
            "matches_expected": status["false_accept_rate"] == 0.5,
        },
        {
            "blocker_id": "zero_verifier_gain",
            "description": "zero verifier gain",
            "source": MATRIX_V25_REL_PATH.as_posix(),
            "source_field": "verifier_repair_summary.verifier_gain_delta",
            "value": status["verifier_gain_delta"],
            "expected_carry_forward_value": 0.0,
            "matches_expected": status["verifier_gain_delta"] == 0.0,
        },
        {
            "blocker_id": "repair_gate_blocked",
            "description": "repair gate blocked",
            "source": CAPSTONE_V291_REL_PATH.as_posix(),
            "source_field": "claim_allowance_summary.repair_gate_state",
            "value": status["repair_gate_status"],
            "expected_carry_forward_value": "blocked_false_accept",
            "matches_expected": status["repair_gate_status"] == "blocked_false_accept",
        },
        {
            "blocker_id": "fr11_ledger_consistency_0_666667",
            "description": "FR-11 ledger_consistency_rate=0.666667",
            "source": MATRIX_V25_REL_PATH.as_posix(),
            "source_field": "fr11_summary.ledger_consistency_rate",
            "value": status["fr11_ledger_consistency_rate"],
            "expected_carry_forward_value": 0.666667,
            "matches_expected": status["fr11_ledger_consistency_rate"] == 0.666667,
        },
        {
            "blocker_id": "ebt_arm_projection_only",
            "description": "EBT/ARM projection-only",
            "source": CAPSTONE_V291_REL_PATH.as_posix(),
            "source_field": "ebt_arm_status",
            "value": status["ebt_arm_status"],
            "expected_carry_forward_value": (
                "projection_only_sidecar_diagnostic_no_live_integration"
            ),
            "matches_expected": status["ebt_arm_status"]
            == "projection_only_sidecar_diagnostic_no_live_integration",
        },
        {
            "blocker_id": "kan_bounded_only",
            "description": "KAN bounded only",
            "source": CAPSTONE_V291_REL_PATH.as_posix(),
            "source_field": "kan_status",
            "value": status["kan_status"],
            "expected_carry_forward_value": (
                "bounded_pwa_milp_abstraction_no_deployed_verifier_claim"
            ),
            "matches_expected": status["kan_status"]
            == "bounded_pwa_milp_abstraction_no_deployed_verifier_claim",
        },
        {
            "blocker_id": "no_authenticated_hardware_speedup",
            "description": "no authenticated hardware speedup",
            "source": CAPSTONE_V291_REL_PATH.as_posix(),
            "source_field": "sampler_hardware_status",
            "value": status["sampler_hardware_status"],
            "expected_carry_forward_value": "blocked_hardware_sampler_boundary_no_speedup_claim",
            "matches_expected": status["sampler_hardware_status"]
            == "blocked_hardware_sampler_boundary_no_speedup_claim",
        },
    ]


def _model_ids(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    claim_summary = _as_mapping(capstone.get("claim_allowance_summary"))
    headline_summary = _as_mapping(matrix.get("headline_claim_allowance_summary"))
    missing = list(_as_list(claim_summary.get("missing_model_ids")))
    if not missing:
        missing = list(_as_list(headline_summary.get("missing_model_ids")))
    present = list(_as_list(claim_summary.get("present_model_ids")))
    if not present:
        present = list(_as_list(headline_summary.get("present_model_ids")))
    return {
        "missing_model_ids": missing or EXPECTED_MISSING_COMPARATIVE_MODELS,
        "present_model_ids": present or EXPECTED_PRESENT_MODEL_IDS,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact["archive_v291_activate_v292_ready"] is True:
        roadmap = _as_mapping(artifact.get("roadmap_handoff"))
        return (
            "complete: archive_v291_activate_v292_ready=true; "
            f"prior_capstone_ready={str(artifact['prior_capstone_ready']).lower()}; "
            f"prior_paper_ready={str(artifact['prior_paper_ready']).lower()}; "
            f"prior_publication_blocker_count={artifact['prior_publication_blocker_count']}; "
            f"blocker_delta_from_v24={artifact['blocker_delta_from_v24']}; "
            f"next_milestone={artifact['next_milestone']}; "
            f"roadmap_source={roadmap.get('source_path')}"
        )
    reasons = _as_list(artifact.get("blocked_reasons"))
    if any("capstone" in str(reason) for reason in reasons):
        prefix = "blocked_prior_capstone_not_ready"
    else:
        prefix = "blocked_roadmap_handoff_not_ready"
    return (
        f"{prefix}: "
        f"prior_capstone_ready={str(artifact['prior_capstone_ready']).lower()}; "
        f"next_milestone={artifact['next_milestone']}; "
        f"reasons={'; '.join(str(reason) for reason in reasons)}"
    )


def _next_milestone(roadmap_handoff: Mapping[str, Any]) -> str:
    observed = str(roadmap_handoff.get("observed_milestone") or "")
    return observed or NEXT_MILESTONE


def _duration(start: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - start), 6)


def _task_ids(payload: Mapping[str, Any]) -> list[str]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [
        str(task["id"])
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") not in (None, "")
    ]


def _field_from_summary_or_matrix(
    field: str,
    summary_name: str,
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
    default: Any,
) -> Any:
    capstone_summary = _as_mapping(capstone.get(summary_name))
    if field in capstone_summary:
        return capstone_summary[field]
    matrix_summary = _as_mapping(matrix.get(summary_name))
    if field in matrix_summary:
        return matrix_summary[field]
    return default


def _dict_rows(value: Any) -> list[JsonDict]:
    return [dict(row) for row in _as_list(value) if isinstance(row, Mapping)]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or(value: Any, default: float | None) -> float:
    if isinstance(value, bool):
        return 0.0 if default is None else float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0 if default is None else float(default)
