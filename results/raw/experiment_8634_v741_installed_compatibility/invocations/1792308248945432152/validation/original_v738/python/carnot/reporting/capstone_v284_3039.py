"""Build the Exp 3039 milestone .284 capstone artifact.

Spec refs: REQ-REPORT-3039, SCENARIO-REPORT-3039.

This module is the milestone closeout ledger for .284. It reads the matrix and
source artifacts that already exist, then carries forward only the claims those
artifacts can support. The distinction matters because the capstone decides
publication readiness, but it is not itself new model, verifier, or hardware
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.284"
NEXT_MILESTONE = "2026.05.285"
SCHEMA = "carnot.milestone_capstone.v284_aggregation.v1"
ARTIFACT = "experiment_3039_capstone_v284"
OUTPUT_REL_PATH = Path("results/experiment_3039_capstone_v284.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3039_capstone_v284.py"

EXP3026_REL_PATH = Path("results/experiment_3026_archive_v283_activate_v284.json")
EXP3027_REL_PATH = Path("results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json")
EXP3028_REL_PATH = Path("results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json")
EXP3029_REL_PATH = Path("results/experiment_3029_repair_promotion_boundary_audit_v2.json")
EXP3030_REL_PATH = Path("results/experiment_3030_validator_frontier_corrigendum_v2.json")
EXP3031_REL_PATH = Path("results/experiment_3031_dccd_structured_repair_panel_v1.json")
EXP3032_REL_PATH = Path("results/experiment_3032_fr11_heldout_dvi_replay_v2.json")
EXP3033_REL_PATH = Path("results/experiment_3033_fr11_nonforgetting_negative_control_stress_v1.json")
EXP3034_REL_PATH = Path("results/experiment_3034_gatemate_output_contract_pinout_decision_v1.json")
EXP3035_REL_PATH = Path("results/experiment_3035_gatemate_output_shim_rtl_ccf_sim_v1.json")
EXP3035_GATE_CHECK_REL_PATH = Path("results/experiment_3035_gatemate_output_shim_rtl_ccf_sim.json")
EXP3036_REL_PATH = Path("results/experiment_3036_gatemate_host_visible_flash_smoke_v4.json")
EXP3037_REL_PATH = Path("results/experiment_3037_ssqa_bounded_rtl_pnr_gate_artifact_v2.json")
MATRIX_V18_REL_PATH = Path("results/experiment_3038_cross_corpus_matrix_v18.json")

STATUSES = (
    "clean",
    "flagged",
    "blocked",
    "gated_skipped",
    "projection_only",
    "pilot_only",
    "missing",
    "retired",
)

HARDWARE_FORBIDDEN_FIELDS = (
    "speedup_claim_made",
    "speedup_claimed",
    "sampler_claim_made",
    "thermodynamic_claim_made",
    "boltzmann_claim_made",
    "annealing_claim_made",
    "energy_claim_made",
    "hardware_performance_claim_made",
    "hardware_execution_claim_made",
)


@dataclass(frozen=True)
class SourceSpec:
    """A checked-in upstream artifact read by the capstone."""

    experiment_id: str
    planned_path: Path
    alternate_path: Path | None = None
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3026", EXP3026_REL_PATH),
    SourceSpec("exp3027", EXP3027_REL_PATH),
    SourceSpec("exp3028", EXP3028_REL_PATH),
    SourceSpec("exp3029", EXP3029_REL_PATH),
    SourceSpec("exp3030", EXP3030_REL_PATH),
    SourceSpec("exp3031", EXP3031_REL_PATH),
    SourceSpec("exp3032", EXP3032_REL_PATH),
    SourceSpec("exp3033", EXP3033_REL_PATH),
    SourceSpec("exp3034", EXP3034_REL_PATH),
    SourceSpec("exp3035", EXP3035_REL_PATH, alternate_path=EXP3035_GATE_CHECK_REL_PATH),
    SourceSpec("exp3036", EXP3036_REL_PATH),
    SourceSpec("exp3037", EXP3037_REL_PATH),
    SourceSpec("exp3038", MATRIX_V18_REL_PATH, required=True),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating absent or malformed files as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a file that is present."""

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
    """REQ-REPORT-3039: synthesize .284 closure from matrix v18 and artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    loaded = _load_sources(root_path)
    payloads = {exp_id: row["payload"] for exp_id, row in loaded.items()}
    source_artifacts = _source_artifacts(root_path, loaded)
    required_errors = _required_source_errors(loaded)
    duration_s = _duration(start, now_s)
    matrix = payloads.get("exp3038", {})
    rows = _matrix_rows(matrix)
    rows_by_exp = _rows_by_exp(rows)
    matrix_summary = _matrix_v18_summary(matrix, rows)
    repair_status = _repair_claim_status(payloads, rows_by_exp)
    fr11_status = _fr11_self_learning_status(payloads, rows_by_exp)
    gatemate_status = _gatemate_status(payloads, rows_by_exp)
    ssqa_status = _ssqa_status(payloads, rows_by_exp)
    capstone_ready = _capstone_ready(matrix, matrix_summary, required_errors)
    paper_checks = _paper_ready_checks(
        capstone_ready,
        matrix_summary,
        repair_status,
        fr11_status,
        gatemate_status,
        ssqa_status,
    )
    paper_ready = all(bool(check["passed"]) for check in paper_checks)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "repair_claim_status": repair_status,
        "fr11_self_learning_status": fr11_status,
        "gatemate_status": gatemate_status,
        "ssqa_status": ssqa_status,
        "matrix_v18_summary": matrix_summary,
        "what_284_proved": _what_284_proved(payloads, rows_by_exp, matrix_summary),
        "paper_ready_checks": paper_checks,
        "blockers_remaining": _blockers_remaining(
            required_errors,
            matrix_summary,
            repair_status,
            fr11_status,
            gatemate_status,
            ssqa_status,
            payloads,
        ),
        "next_milestone_focus": _next_milestone_focus(),
        "recommended_next_actions": _recommended_next_actions(),
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads),
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "missing_artifacts": _missing_artifacts(source_artifacts),
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "publication_action_allowed": False,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
        "honest_verdict": "blocked_required_matrix_v18_missing",
    }
    if required_errors:
        artifact["required_upstream_errors"] = required_errors
        return artifact
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3039 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_sources(root: Path) -> dict[str, JsonDict]:
    loaded: dict[str, JsonDict] = {}
    for spec in SOURCE_SPECS:
        planned = root / spec.planned_path
        alternate = root / spec.alternate_path if spec.alternate_path is not None else None
        actual_path = spec.planned_path
        payload = read_json_object(planned)
        if not payload and alternate is not None and alternate.is_file():
            actual_path = spec.alternate_path or spec.planned_path
            payload = read_json_object(root / actual_path)
        loaded[spec.experiment_id] = {
            "spec": spec,
            "payload": payload,
            "actual_path": actual_path,
            "planned_path_present": planned.is_file(),
            "actual_path_present": (root / actual_path).is_file(),
        }
    return loaded


def _source_artifacts(root: Path, loaded: Mapping[str, JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        loaded_row = loaded[spec.experiment_id]
        actual_path = Path(loaded_row["actual_path"])
        alternate_path = spec.alternate_path
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "planned_path": spec.planned_path.as_posix(),
                "alternate_path": alternate_path.as_posix() if alternate_path else None,
                "actual_path": actual_path.as_posix(),
                "planned_path_present": bool(loaded_row["planned_path_present"]),
                "present": bool(loaded_row["actual_path_present"]),
                "required": spec.required,
                "readable_json_object": bool(loaded_row["payload"]),
                "sha256": sha256_file(root / actual_path),
            }
        )
    return rows


def _required_source_errors(loaded: Mapping[str, JsonDict]) -> list[JsonDict]:
    errors: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        if spec.required and not loaded[spec.experiment_id]["payload"]:
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.planned_path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _matrix_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(row) for row in _as_list(matrix.get("matrix_rows")) if isinstance(row, Mapping)]


def _rows_by_exp(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("experiment_id")): row for row in rows if row.get("experiment_id")}


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        status = str(row.get("status") or "missing")
        counts[status if status in counts else "missing"] += 1
    return counts


def _matrix_v18_summary(matrix: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> JsonDict:
    row_counts = _status_counts(rows)
    top_counts = {status: _int_or(matrix.get(status), 0) for status in STATUSES}
    expected = {f"exp{number}" for number in range(3026, 3040)}
    represented = set(_rows_by_exp(rows))
    return {
        "matrix_v18_ready": bool(matrix.get("matrix_v18_ready")),
        "rows_total": _int_or(matrix.get("rows_total"), len(rows)),
        "row_count_observed": len(rows),
        "clean": top_counts["clean"],
        "flagged": top_counts["flagged"],
        "blocked": top_counts["blocked"],
        "gated_skipped": top_counts["gated_skipped"],
        "projection_only": top_counts["projection_only"],
        "pilot_only": top_counts["pilot_only"],
        "missing": top_counts["missing"],
        "retired": top_counts["retired"],
        "counts_from_rows": row_counts,
        "counts_match_rows": row_counts == top_counts,
        "all_284_tasks_represented": expected <= represented,
        "missing_task_rows": sorted(expected - represented),
        "status_by_experiment": {
            exp_id: str(row.get("status") or "missing")
            for exp_id, row in sorted(_rows_by_exp(rows).items())
        },
        "baseline_v17_summary": dict(_mapping(matrix.get("baseline_v17_summary"))),
    }


def _capstone_ready(
    matrix: Mapping[str, Any],
    matrix_summary: Mapping[str, Any],
    required_errors: list[JsonDict],
) -> bool:
    return (
        not required_errors
        and matrix.get("matrix_v18_ready") is True
        and matrix_summary.get("rows_total") == 14
        and matrix_summary.get("row_count_observed") == 14
        and matrix_summary.get("counts_match_rows") is True
        and matrix_summary.get("all_284_tasks_represented") is True
    )


def _repair_claim_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows_by_exp: Mapping[str, Mapping[str, Any]],
) -> str:
    explicit = str(payloads.get("exp3029", {}).get("repair_claim_status") or "")
    if explicit:
        return explicit
    matrix_value = str(rows_by_exp.get("exp3029", {}).get("repair_claim_status") or "")
    return matrix_value or "unknown"


def _fr11_self_learning_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows_by_exp: Mapping[str, Mapping[str, Any]],
) -> str:
    exp3032 = payloads.get("exp3032", {})
    exp3033 = payloads.get("exp3033", {})
    replay_clean = (
        exp3032.get("fr11_heldout_replay_ready") is True
        and exp3032.get("continuous_self_learning_tested") is True
        and exp3032.get("tautology_risk_cleared") is True
        and exp3032.get("information_asymmetry_enforced") is True
        and not exp3032.get("invariant_violations")
    )
    stress_clean = (
        exp3033.get("fr11_nonforgetting_stress_ready") is True
        and exp3033.get("fr11_self_learning_promotable") is True
        and str(exp3033.get("promotion_decision") or "") == "controller_only_promotable"
        and not exp3033.get("drift_failures")
    )
    matrix_promotable = rows_by_exp.get("exp3033", {}).get("fr11_self_learning_promotable") is True
    if replay_clean and stress_clean and matrix_promotable:
        return "controller_only_promotable"
    if replay_clean:
        return "heldout_replay_clean_not_promoted"
    return "blocked_or_unproven"


def _gatemate_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows_by_exp: Mapping[str, Mapping[str, Any]],
) -> str:
    exp3034 = payloads.get("exp3034", {})
    exp3036 = payloads.get("exp3036", {})
    if exp3034.get("gatemate_output_contract_ready") is True and (
        exp3036.get("host_visible_output_observed") is True
        or rows_by_exp.get("exp3036", {}).get("host_visible_output_observed") is True
    ):
        return "host_visible_output_observed"
    bounded = (
        exp3034.get("gatemate_output_contract_ready") is False
        and bool(_as_list(exp3034.get("exact_operator_action_required")))
        and not _hardware_claim_fields(exp3034)
    )
    if bounded:
        return "blocked_pinout_missing_bounded"
    return "blocked_or_unbounded"


def _ssqa_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows_by_exp: Mapping[str, Mapping[str, Any]],
) -> str:
    exp3037 = payloads.get("exp3037", {})
    gate_status = str(
        exp3037.get("ssqa_gate_status")
        or rows_by_exp.get("exp3037", {}).get("ssqa_gate_status")
        or ""
    )
    performance_allowed = exp3037.get("ssqa_performance_claim_allowed") is True
    if gate_status in {"gate_skipped", "gated_skipped"} and not performance_allowed:
        return "gate_skipped_bounded_no_performance_claim"
    if gate_status == "run" and _as_list(exp3037.get("resource_report_paths")):
        return "bounded_resource_evidence"
    return "blocked_or_unbounded"


def _paper_ready_checks(
    capstone_ready: bool,
    matrix_summary: Mapping[str, Any],
    repair_status: str,
    fr11_status: str,
    gatemate_status: str,
    ssqa_status: str,
) -> list[JsonDict]:
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v18 is present, complete, and count-reconciled",
        },
        {
            "check": "repair_promotable",
            "passed": repair_status == "promotable",
            "reason": f"repair_claim_status={repair_status}",
        },
        {
            "check": "fr11_non_tautological",
            "passed": fr11_status == "controller_only_promotable",
            "reason": f"fr11_self_learning_status={fr11_status}",
        },
        {
            "check": "hardware_resolved_or_bounded",
            "passed": gatemate_status
            in {"host_visible_output_observed", "blocked_pinout_missing_bounded"}
            and ssqa_status
            in {"bounded_resource_evidence", "gate_skipped_bounded_no_performance_claim"},
            "reason": f"gatemate_status={gatemate_status}; ssqa_status={ssqa_status}",
        },
        {
            "check": "matrix_has_no_nonclean_publication_blockers",
            "passed": all(
                _int_or(matrix_summary.get(status), 0) == 0
                for status in ("flagged", "blocked", "gated_skipped", "missing")
            ),
            "reason": (
                f"flagged={matrix_summary.get('flagged')}; "
                f"blocked={matrix_summary.get('blocked')}; "
                f"gated_skipped={matrix_summary.get('gated_skipped')}; "
                f"missing={matrix_summary.get('missing')}"
            ),
        },
    ]


def _what_284_proved(
    payloads: Mapping[str, Mapping[str, Any]],
    rows_by_exp: Mapping[str, Mapping[str, Any]],
    matrix_summary: Mapping[str, Any],
) -> JsonDict:
    exp3028 = payloads.get("exp3028", {})
    exp3029 = payloads.get("exp3029", {})
    exp3030 = payloads.get("exp3030", {})
    exp3032 = payloads.get("exp3032", {})
    exp3033 = payloads.get("exp3033", {})
    exp3034 = payloads.get("exp3034", {})
    exp3037 = payloads.get("exp3037", {})
    return {
        "repair": {
            "clean_repair_rerun_ready": bool(exp3028.get("clean_repair_rerun_ready")),
            "n_tasks": _int_or(exp3028.get("n_tasks"), 0),
            "n_live_transcripts": _int_or(exp3028.get("n_live_transcripts"), 0),
            "pass_at_1_delta": _float_or_none(exp3028.get("pass_at_1_delta")),
            "false_accept_delta": _float_or_none(exp3028.get("false_accept_delta")),
            "repair_claim_status": str(exp3029.get("repair_claim_status") or "unknown"),
            "matrix_status": str(rows_by_exp.get("exp3029", {}).get("status") or "missing"),
        },
        "validator_frontier": {
            "verified_region_count": _int_or(exp3030.get("verified_region_count"), 0),
            "irrelevant_region_count": _int_or(exp3030.get("irrelevant_region_count"), 0),
            "unresolved_region_count": _int_or(exp3030.get("unresolved_region_count"), 0),
            "fallback_only_count": _int_or(exp3030.get("fallback_only_count"), 0),
            "missing_authority_count": _int_or(exp3030.get("missing_authority_count"), 0),
        },
        "fr11_self_learning": {
            "heldout_trace_count": _int_or(exp3032.get("heldout_trace_count"), 0),
            "tautology_risk_cleared": bool(exp3032.get("tautology_risk_cleared")),
            "information_asymmetry_enforced": bool(
                exp3032.get("information_asymmetry_enforced")
            ),
            "promotion_decision": str(exp3033.get("promotion_decision") or ""),
            "model_weight_training": bool(_mapping(exp3033.get("inference_substrate")).get("model_weight_training")),
        },
        "gatemate": {
            "gatemate_output_contract_ready": bool(
                exp3034.get("gatemate_output_contract_ready")
            ),
            "host_visible_output_observed": bool(
                rows_by_exp.get("exp3036", {}).get("host_visible_output_observed")
            ),
            "operator_action_count": len(_as_list(exp3034.get("exact_operator_action_required"))),
        },
        "ssqa": {
            "ssqa_boundary_ready": bool(exp3037.get("ssqa_boundary_ready")),
            "ssqa_gate_status": str(exp3037.get("ssqa_gate_status") or ""),
            "resource_report_count": len(_as_list(exp3037.get("resource_report_paths"))),
            "performance_claim_allowed": bool(exp3037.get("ssqa_performance_claim_allowed")),
        },
        "matrix_accounting": dict(matrix_summary),
    }


def _blockers_remaining(
    required_errors: list[JsonDict],
    matrix_summary: Mapping[str, Any],
    repair_status: str,
    fr11_status: str,
    gatemate_status: str,
    ssqa_status: str,
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    if required_errors:
        blockers.append({"area": "matrix_v18", "status": "blocked", "evidence": required_errors})
    if any(_int_or(matrix_summary.get(status), 0) for status in ("flagged", "blocked", "gated_skipped", "missing")):
        blockers.append(
            {
                "area": "matrix_nonclean",
                "status": "publication_blocking",
                "counts": {
                    status: _int_or(matrix_summary.get(status), 0)
                    for status in ("flagged", "blocked", "gated_skipped", "missing")
                },
                "next_action": "Drive matrix v19 to zero unresolved publication blockers.",
            }
        )
    if repair_status != "promotable":
        blockers.append(
            {
                "area": "repair",
                "status": repair_status,
                "evidence": {
                    "exp3028_flags": _as_list(payloads.get("exp3028", {}).get("corrigendum_pending")),
                    "exp3029_honest_verdict": str(payloads.get("exp3029", {}).get("honest_verdict") or ""),
                },
                "next_action": "Clear or explicitly retire the remaining repair adversarial flags.",
            }
        )
    if fr11_status != "controller_only_promotable":
        blockers.append(
            {
                "area": "fr11_self_learning",
                "status": fr11_status,
                "next_action": "Re-run held-out and negative-control checks before promotion.",
            }
        )
    if gatemate_status != "host_visible_output_observed":
        blockers.append(
            {
                "area": "gatemate",
                "status": gatemate_status,
                "evidence": _as_list(payloads.get("exp3034", {}).get("exact_operator_action_required")),
                "next_action": "Commit a host-visible output contract and reader command.",
            }
        )
    if ssqa_status != "bounded_resource_evidence":
        blockers.append(
            {
                "area": "ssqa",
                "status": ssqa_status,
                "evidence": _as_list(payloads.get("exp3037", {}).get("exact_blocker_or_next_action")),
                "next_action": "Collect bounded RTL/PnR evidence only after host-visible output exists.",
            }
        )
    if not payloads.get("exp3036"):
        blockers.append(
            {
                "area": "exp3036",
                "status": "missing_or_gated_skipped",
                "next_action": "Run GateMate flash smoke after Exp3034/Exp3035 pass their gates.",
            }
        )
    return blockers


def _recommended_next_actions() -> list[str]:
    return [
        "Resolve the GateMate output pinout and commit a CCF binding plus host reader command.",
        "Rerun the GateMate output shim and flash smoke only after the output contract is ready.",
        "Convert SSQA from gate-skipped to bounded RTL/PnR/resource evidence without speedup claims.",
        "Clear or retire the repair adversarial flags before any headline repair wording.",
        "Carry FR-11 forward only as controller-weight verifier-feedback learning until broader training is measured.",
    ]


def _next_milestone_focus() -> str:
    return (
        f"{NEXT_MILESTONE}: GateMate host-visible output contract closure, SSQA bounded "
        "resource evidence, and repair-flag cleanup before paper-promotion work."
    )


def _cited_upstream_artifacts(
    source_artifacts: list[JsonDict],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    citations: list[JsonDict] = []
    for source in source_artifacts:
        exp_id = str(source["experiment_id"])
        payload = payloads.get(exp_id, {})
        citation: JsonDict = {
            "experiment_id": exp_id,
            "planned_path": source["planned_path"],
            "actual_path": source["actual_path"],
            "present": source["present"],
            "planned_path_present": source["planned_path_present"],
            "readable_json_object": source["readable_json_object"],
            "sha256": source["sha256"],
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        }
        model_details = _source_model_details(payload)
        hardware_details = _source_hardware_details(payload)
        gate_details = _source_gate_details(payload)
        if model_details:
            citation["source_model_details"] = model_details
        if hardware_details:
            citation["source_hardware_details"] = hardware_details
        if gate_details:
            citation["source_gate_details"] = gate_details
        citations.append(citation)
    return citations


def _source_model_details(payload: Mapping[str, Any]) -> JsonDict:
    substrate = _mapping(payload.get("inference_substrate"))
    substrate_details = {
        key: substrate.get(key)
        for key in (
            "kind",
            "mode",
            "gguf_cache_paths",
            "gpu_inventory",
            "selected_headline_model",
            "model_checksum_feasibility",
            "loader",
            "model_load_attempted",
        )
        if substrate.get(key) not in (None, [], {})
    }
    fields = {
        "model_specs": payload.get("model_specs"),
        "target_model": payload.get("target_model"),
        "headline_models_used": payload.get("headline_models_used"),
        "headline_models_available": payload.get("headline_models_available"),
        "model_checksums": payload.get("model_checksums"),
        "inference_substrate": substrate_details,
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _source_hardware_details(payload: Mapping[str, Any]) -> JsonDict:
    fields = {
        "gatemate_output_contract_ready": payload.get("gatemate_output_contract_ready"),
        "host_visible_io_plan_ready": payload.get("host_visible_io_plan_ready"),
        "selected_output_path": payload.get("selected_output_path"),
        "host_reader_command": payload.get("host_reader_command"),
        "host_visible_output_observed": payload.get("host_visible_output_observed"),
        "ssqa_gate_status": payload.get("ssqa_gate_status"),
        "ssqa_performance_claim_allowed": payload.get("ssqa_performance_claim_allowed"),
        "resource_report_paths": payload.get("resource_report_paths"),
        "exact_blocker_or_next_action": payload.get("exact_blocker_or_next_action"),
        "exact_operator_action_required": payload.get("exact_operator_action_required"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _source_gate_details(payload: Mapping[str, Any]) -> JsonDict:
    fields = {
        "status": payload.get("status"),
        "blocked_at_layer": payload.get("blocked_at_layer"),
        "gate_check_summary": payload.get("gate_check_summary"),
        "gates_evaluated": payload.get("gates_evaluated"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _source_checksums(source_artifacts: list[JsonDict]) -> dict[str, str | None]:
    return {str(row["actual_path"]): row["sha256"] for row in source_artifacts}


def _missing_artifacts(source_artifacts: list[JsonDict]) -> list[str]:
    missing: list[str] = []
    for row in source_artifacts:
        if row.get("planned_path_present") is not True:
            missing.append(str(row["planned_path"]))
        elif row.get("present") is not True:
            missing.append(str(row["actual_path"]))
    return missing


def _hardware_claim_fields(payload: Mapping[str, Any]) -> list[str]:
    return [field for field in HARDWARE_FORBIDDEN_FIELDS if payload.get(field) is True]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: "
        f"capstone_ready={str(artifact['capstone_ready']).lower()}; "
        f"paper_ready={str(artifact['paper_ready']).lower()}; "
        f"repair_claim_status={artifact['repair_claim_status']}; "
        f"fr11_self_learning_status={artifact['fr11_self_learning_status']}; "
        f"gatemate_status={artifact['gatemate_status']}; "
        f"ssqa_status={artifact['ssqa_status']}; "
        f"next={NEXT_MILESTONE}_gatemate_output_contract_repair_flag_cleanup"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "source": "checked_in_artifacts",
    }


def _duration(start: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - start), 6)


def _safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _mapping(value: Any) -> Mapping[str, Any]:
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
