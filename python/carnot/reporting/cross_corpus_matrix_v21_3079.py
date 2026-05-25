"""Build the Exp 3079 cross-corpus matrix v21 artifact.

Spec refs: REQ-REPORT-3079, SCENARIO-REPORT-3079.

Matrix v21 is an accounting artifact, not a fresh experiment. It reads the
already-written v20/capstone/normalization artifacts plus the checked-in .287
outputs, then records which claims are clean, bounded, blocked, skipped,
projected, missing, flagged, or retired. Keeping this as a pure aggregation
step prevents a paper-readiness table from quietly becoming a rerun of repair,
model inference, solver scoring, or hardware work.
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
MILESTONE = "2026.05.287"
SCHEMA = "carnot.cross_corpus_matrix.v21_287_claim_aggregation.v1"
ARTIFACT = "experiment_3079_cross_corpus_matrix_v21"
OUTPUT_REL_PATH = Path("results/experiment_3079_cross_corpus_matrix_v21.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3079_cross_corpus_matrix_v21.py"

MATRIX_V20_REL_PATH = Path("results/experiment_3065_cross_corpus_matrix_v20.json")
CAPSTONE_V286_REL_PATH = Path("results/experiment_3066_capstone_v286.json")
EXP3059_REQUESTED_REL_PATH = Path(
    "results/experiment_3059_gated_sota_repair_de_tautology_rerun_v1.json"
)
EXP3059_ACTUAL_REL_PATH = Path("results/experiment_3059_gated_sota_repair_de_tautology_rerun.json")
EXP3067_REL_PATH = Path("results/experiment_3067_archive_v286_activate_v287.json")
EXP3068_REL_PATH = Path(
    "results/experiment_3068_matrix_v20_artifact_alias_blocker_normalization_v1.json"
)
EXP3069_REL_PATH = Path("results/experiment_3069_solver_verifier_failure_autopsy_protocol_v1.json")
EXP3070_REL_PATH = Path("results/experiment_3070_first_token_abstention_sota_panel_v1.json")
EXP3071_REL_PATH = Path("results/experiment_3071_verge_mcs_smt_correction_pilot_v1.json")
EXP3072_REL_PATH = Path("results/experiment_3072_gated_local_sota_verifier_calibration_v2.json")
EXP3073_REL_PATH = Path(
    "results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json"
)
EXP3074_REL_PATH = Path("results/experiment_3074_llguidance_aprad_repair_protocol_v1.json")
EXP3075_REL_PATH = Path(
    "results/experiment_3075_gated_grammar_constrained_sota_repair_micro_panel_v1.json"
)
EXP3076_REL_PATH = Path("results/experiment_3076_fr11_online_soundness_completeness_budget_v1.json")
EXP3077_REL_PATH = Path(
    "results/experiment_3077_fr11_soundness_bounded_online_self_learning_pilot_v1.json"
)
EXP3078_REL_PATH = Path("results/experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")

STATUSES = (
    "clean",
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
    "retired",
)
COUNT_FIELDS = {
    "clean": "clean_rows",
    "flagged": "flagged_rows",
    "bounded": "bounded_rows",
    "blocked": "blocked_rows",
    "gated_skipped": "gated_skipped_rows",
    "projection_only": "projection_only_rows",
    "missing": "missing_rows",
    "retired": "retired_rows",
}
V20_CLASS_FIELDS = tuple(COUNT_FIELDS.values())
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}
REQUIRED_ROW_KEYS = {
    "row_id",
    "status",
    "source_artifact",
    "source_field",
    "evidence_class",
    "blocker_class",
    "claim_scope",
    "summary",
}


@dataclass(frozen=True)
class SourceSpec:
    """One upstream source v21 must inspect or cite.

    Required sources are the authority chain for the matrix itself. The .287
    artifacts are expected but not required, because an absent expected result
    is itself a row that the matrix must record instead of treating as success.
    """

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    source_type: str = "json"


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3065", MATRIX_V20_REL_PATH, "matrix_v20_authority", required=True),
    SourceSpec("exp3066", CAPSTONE_V286_REL_PATH, "capstone_v286_authority", required=True),
    SourceSpec("exp3068", EXP3068_REL_PATH, "alias_and_blocker_normalization", required=True),
    SourceSpec("exp3059_actual", EXP3059_ACTUAL_REL_PATH, "exp3059_alias_target"),
    SourceSpec("exp3067", EXP3067_REL_PATH, "archive_v287_activation"),
    SourceSpec("exp3069", EXP3069_REL_PATH, "solver_verifier_failure_autopsy"),
    SourceSpec("exp3070", EXP3070_REL_PATH, "first_token_abstention_panel"),
    SourceSpec("exp3071", EXP3071_REL_PATH, "verge_mcs_feedback_pilot"),
    SourceSpec("exp3072", EXP3072_REL_PATH, "gated_verifier_calibration"),
    SourceSpec("exp3073", EXP3073_REL_PATH, "ebt_arm_adapter_feasibility"),
    SourceSpec("exp3074", EXP3074_REL_PATH, "grammar_repair_protocol"),
    SourceSpec("exp3075", EXP3075_REL_PATH, "gated_grammar_repair_micro_panel"),
    SourceSpec("exp3076", EXP3076_REL_PATH, "fr11_soundness_completeness_budget"),
    SourceSpec("exp3077", EXP3077_REL_PATH, "fr11_soundness_bounded_pilot"),
    SourceSpec("exp3078", EXP3078_REL_PATH, "gatemate_ssqa_refresh"),
    SourceSpec("conductor_log", CONDUCTOR_LOG_REL_PATH, "structured_gate_history", source_type="text"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating absent or malformed evidence as empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read a text evidence source while treating absence as no evidence."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def sha256_file(path: Path) -> str | None:
    """Return a source checksum so rows can be audited without mutating inputs."""

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
    """REQ-REPORT-3079: aggregate matrix v21 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    loaded = {spec.experiment_id: _load_source(root_path, spec) for spec in SOURCE_SPECS}
    payloads = {experiment_id: _as_mapping(row.get("payload")) for experiment_id, row in loaded.items()}
    conductor_log = str(loaded["conductor_log"].get("text") or "")
    aliases = _alias_map(payloads["exp3068"])
    artifact_hygiene_ids = _artifact_hygiene_row_ids(payloads["exp3068"])
    rows = (
        _v20_rows(payloads["exp3065"], aliases, artifact_hygiene_ids)
        + _capstone_v286_rows(payloads["exp3066"])
        + _normalization_rows(payloads["exp3068"])
        + _dot287_rows(payloads, conductor_log)
    )
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    source_artifacts = _source_artifacts(root_path, rows)
    required_errors = _required_source_errors(source_artifacts)
    invariant_violations = _invariant_violations(
        rows,
        status_counts,
        publication_blockers,
        source_artifacts,
        required_errors,
    )
    ready = not invariant_violations

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v21_ready": ready,
        "rows_total": len(rows),
        **{COUNT_FIELDS[status]: status_counts[status] for status in STATUSES},
        "publication_blocker_count": len(publication_blockers),
        "publication_blockers": publication_blockers,
        "rows": rows,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "required_source_errors": required_errors,
        "alias_normalization": _alias_summary(payloads["exp3068"], aliases, artifact_hygiene_ids),
        "status_counts": status_counts,
        "invariant_violations": invariant_violations,
        "paper_ready": ready and not publication_blockers,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "no_historical_artifact_rewrite": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": _duration(start, now_s),
        "honest_verdict": _honest_verdict(ready, len(rows), len(publication_blockers), required_errors),
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3079 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_source(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path) if spec.source_type == "json" else {}
    text = read_text(path) if spec.source_type == "text" else ""
    return {"present": path.is_file(), "payload": payload, "text": text}


def _v20_rows(
    matrix: Mapping[str, Any],
    aliases: Mapping[str, str],
    artifact_hygiene_ids: set[str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for field in V20_CLASS_FIELDS:
        for index, raw_value in enumerate(_as_list(matrix.get(field))):
            raw = _as_mapping(raw_value)
            original_id = str(raw.get("row_id") or f"{field}_{index}")
            original_source = str(raw.get("source_artifact") or MATRIX_V20_REL_PATH.as_posix())
            alias_applied = original_source in aliases
            retired_by_alias = original_id in artifact_hygiene_ids
            status = "retired" if retired_by_alias else normal_status(str(raw.get("status") or "missing"))
            summary = _as_mapping(raw.get("summary"))
            summary.update(
                {
                    "matrix_v20_row_id": original_id,
                    "matrix_v20_status": normal_status(str(raw.get("status") or "missing")),
                    "alias_applied": alias_applied,
                    "retired_by_alias_normalization": retired_by_alias,
                    "status_rationale": "artifact_hygiene_alias_resolved_non_destructively"
                    if retired_by_alias
                    else "carried_forward_from_matrix_v20",
                }
            )
            rows.append(
                _row(
                    row_id=f"v20:{original_id}",
                    status=status,
                    source_artifact=aliases.get(original_source, original_source),
                    source_field=str(raw.get("source_field") or field),
                    evidence_class=str(raw.get("evidence_class") or "matrix_v20_carry_forward"),
                    claim_scope=str(raw.get("claim_scope") or "matrix_v20_carry_forward"),
                    summary=summary,
                    row_origin="matrix_v20",
                )
            )
    return rows


def _capstone_v286_rows(capstone: Mapping[str, Any]) -> list[JsonDict]:
    status = "missing" if not capstone else "blocked" if capstone.get("capstone_ready") is not True else "clean" if capstone.get("paper_ready") is True else "bounded"
    return [
        _row(
            row_id="capstone:v286_paper_readiness",
            status=status,
            source_artifact=CAPSTONE_V286_REL_PATH.as_posix(),
            source_field="paper_ready",
            evidence_class="capstone_v286_authority",
            claim_scope="paper_readiness",
            summary={
                "capstone_ready": capstone.get("capstone_ready") is True,
                "paper_ready": capstone.get("paper_ready") is True,
                "publication_blocker_count": len(_as_list(capstone.get("publication_blockers"))),
            },
            row_origin="capstone_v286",
        )
    ]


def _normalization_rows(ledger: Mapping[str, Any]) -> list[JsonDict]:
    status = "missing" if not ledger else "clean" if ledger.get("matrix_v20_normalization_ready") is True else "blocked"
    return [
        _row(
            row_id="normalization:matrix_v20_alias_blocker_ledger",
            status=status,
            source_artifact=EXP3068_REL_PATH.as_posix(),
            source_field="matrix_v20_normalization_ready",
            evidence_class="artifact_alias_normalization",
            claim_scope="source_artifact_accounting",
            summary={
                "artifact_alias_count": len(_as_list(ledger.get("artifact_aliases"))),
                "normalized_blocker_count_estimate": _int_or_none(
                    ledger.get("normalized_blocker_count_estimate")
                ),
                "no_research_claim_cleaned_by_alias": ledger.get("no_research_claim_cleaned_by_alias")
                is True,
            },
            row_origin="normalization_ledger",
        )
    ]


def _dot287_rows(payloads: Mapping[str, Mapping[str, Any]], conductor_log: str) -> list[JsonDict]:
    return [
        _ready_row(
            "dot287:exp3067_archive_activation",
            payloads["exp3067"],
            EXP3067_REL_PATH,
            "archive_v286_activate_v287_ready",
            "archive_activation",
            "milestone_activation",
        ),
        _ready_row(
            "dot287:exp3069_solver_verifier_autopsy",
            payloads["exp3069"],
            EXP3069_REL_PATH,
            "verifier_failure_autopsy_ready",
            "solver_verifier_failure_autopsy",
            "recovery_protocol",
        ),
        _first_token_row(payloads["exp3070"]),
        _verge_row(payloads["exp3071"]),
        _gate_record_row(
            "dot287:exp3072_verifier_calibration_gate",
            payloads["exp3072"],
            EXP3072_REL_PATH,
            "gate_check_summary",
            "local_sota_verifier_calibration_gate",
            "verifier_gain_recovery_gate",
        ),
        _ebt_adapter_row(payloads["exp3073"]),
        _ready_row(
            "dot287:exp3074_repair_protocol",
            payloads["exp3074"],
            EXP3074_REL_PATH,
            "grammar_constrained_repair_protocol_ready",
            "grammar_constrained_repair_protocol",
            "repair_rerun_protocol",
        ),
        _repair_micro_panel_row(payloads["exp3075"], conductor_log),
        _fr11_budget_row(payloads["exp3076"]),
        _fr11_pilot_row(payloads["exp3077"]),
        *_gatemate_ssqa_rows(payloads["exp3078"]),
    ]


def _ready_row(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
    ready_field: str,
    evidence_class: str,
    claim_scope: str,
) -> JsonDict:
    status = "missing" if not payload else "clean" if payload.get(ready_field) is True else "blocked"
    return _row(
        row_id=row_id,
        status=status,
        source_artifact=source_path.as_posix(),
        source_field=ready_field,
        evidence_class=evidence_class,
        claim_scope=claim_scope,
        summary={"ready": payload.get(ready_field) is True, "honest_verdict": str(payload.get("honest_verdict") or "")},
        row_origin="milestone_287",
    )


def _first_token_row(panel: Mapping[str, Any]) -> JsonDict:
    precision = _float_or_none(panel.get("abstention_precision"))
    gate_failed = precision is not None and precision < 0.7
    flagged = panel.get("flagged_adversarial") is True or bool(_as_list(panel.get("corrigendum_pending"))) or gate_failed
    status = "missing" if not panel else "flagged" if flagged else "clean" if panel.get("first_token_panel_ready") is True else "blocked"
    return _row(
        row_id="dot287:exp3070_first_token_abstention",
        status=status,
        source_artifact=EXP3070_REL_PATH.as_posix(),
        source_field="abstention_precision",
        evidence_class="first_token_abstention_sota_panel",
        claim_scope="local_sota_solution_verifier_gain",
        summary={
            "first_token_panel_ready": panel.get("first_token_panel_ready") is True,
            "abstention_precision": precision,
            "expected_abstention_precision_min": 0.7,
            "verifier_gain_delta_with_abstention": _float_or_none(
                panel.get("verifier_gain_delta_with_abstention")
            ),
            "flagged_adversarial": panel.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(panel.get("corrigendum_pending"))),
        },
        row_origin="milestone_287",
    )


def _verge_row(pilot: Mapping[str, Any]) -> JsonDict:
    guided = _int_or_none(pilot.get("guided_success_count"))
    solver = _int_or_none(pilot.get("solver_only_success_count"))
    no_lift = guided is not None and solver is not None and guided <= solver
    flagged = pilot.get("flagged_adversarial") is True or bool(_as_list(pilot.get("corrigendum_pending"))) or no_lift
    status = "missing" if not pilot else "flagged" if flagged else "clean" if pilot.get("mcs_feedback_ready") is True else "blocked"
    return _row(
        row_id="dot287:exp3071_verge_mcs_feedback",
        status=status,
        source_artifact=EXP3071_REL_PATH.as_posix(),
        source_field="guided_success_count",
        evidence_class="verge_mcs_smt_feedback_pilot",
        claim_scope="solver_grounded_repair_feedback",
        summary={
            "mcs_feedback_ready": pilot.get("mcs_feedback_ready") is True,
            "guided_success_count": guided,
            "solver_only_success_count": solver,
            "guided_lift_positive": bool(guided is not None and solver is not None and guided > solver),
            "flagged_adversarial": pilot.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(pilot.get("corrigendum_pending"))),
        },
        row_origin="milestone_287",
    )


def _gate_record_row(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
    source_field: str,
    evidence_class: str,
    claim_scope: str,
) -> JsonDict:
    failed_count = sum(1 for row in _as_list(payload.get("gates_evaluated")) if _as_mapping(row).get("passed") is not True)
    blocked_gate = str(payload.get("status") or "").lower() == "blocked" or "failed" in str(payload.get("honest_verdict") or "").lower()
    status = "missing" if not payload else "gated_skipped" if blocked_gate or failed_count else "clean"
    return _row(
        row_id=row_id,
        status=status,
        source_artifact=source_path.as_posix(),
        source_field=source_field,
        evidence_class=evidence_class,
        claim_scope=claim_scope,
        summary={
            "status": str(payload.get("status") or ""),
            "gate_check_summary": str(payload.get("gate_check_summary") or ""),
            "failed_gate_count": failed_count,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_287",
    )


def _ebt_adapter_row(audit: Mapping[str, Any]) -> JsonDict:
    implemented = audit.get("adapter_implementation_claimed") is True
    ready = audit.get("ebt_arm_adapter_feasibility_ready") is True
    feasible = audit.get("ebt_arm_adapter_feasible") is True
    status = "missing" if not audit else "blocked" if not ready else "projection_only" if not implemented else "clean" if feasible else "bounded"
    return _row(
        row_id="dot287:exp3073_ebt_arm_adapter_feasibility",
        status=status,
        source_artifact=EXP3073_REL_PATH.as_posix(),
        source_field="adapter_implementation_claimed",
        evidence_class="ebt_arm_adapter_feasibility_audit",
        claim_scope="future_adapter_context",
        summary={
            "feasibility_ready": ready,
            "adapter_feasible": feasible,
            "adapter_implementation_claimed": implemented,
            "blocker_count": len(_as_list(audit.get("blockers"))),
            "status_rationale": "future_context_only_no_adapter_implementation_claimed",
        },
        row_origin="milestone_287",
    )


def _repair_micro_panel_row(panel: Mapping[str, Any], conductor_log: str) -> JsonDict:
    gate_entries = _conductor_gate_entries(conductor_log, "Gated grammar-constrained SOTA repair micro-panel")
    source_artifact = EXP3075_REL_PATH.as_posix() if panel else CONDUCTOR_LOG_REL_PATH.as_posix() if gate_entries else EXP3075_REL_PATH.as_posix()
    status = "gated_skipped" if not panel and gate_entries else "missing" if not panel else "flagged" if panel.get("flagged_adversarial") is True or bool(_as_list(panel.get("corrigendum_pending"))) else "clean" if panel.get("grammar_constrained_repair_ready") is True or panel.get("repair_gate_passed") is True else "blocked"
    return _row(
        row_id="dot287:exp3075_repair_micro_panel",
        status=status,
        source_artifact=source_artifact,
        source_field="conductor_log.exp3075_gate" if not panel and gate_entries else "grammar_constrained_repair_ready",
        evidence_class="gated_grammar_constrained_repair_micro_panel",
        claim_scope="repair_live_rerun",
        summary={
            "artifact_present": bool(panel),
            "conductor_gate_skip_count": len(gate_entries),
            "gate_entries": gate_entries,
            "status_rationale": "structured_gate_skip_from_conductor" if gate_entries and not panel else "artifact_payload_classification",
        },
        row_origin="milestone_287",
    )


def _fr11_budget_row(budget: Mapping[str, Any]) -> JsonDict:
    flagged = budget.get("flagged_adversarial") is True or bool(_as_list(budget.get("corrigendum_pending")))
    status = "missing" if not budget else "flagged" if flagged else "clean" if budget.get("soundness_completeness_budget_ready") is True else "blocked"
    return _row(
        row_id="dot287:exp3076_fr11_budget",
        status=status,
        source_artifact=EXP3076_REL_PATH.as_posix(),
        source_field="soundness_completeness_budget_ready",
        evidence_class="fr11_soundness_completeness_budget",
        claim_scope="controller_only_online_learning_budget",
        summary={
            "soundness_completeness_budget_ready": budget.get("soundness_completeness_budget_ready") is True,
            "flagged_adversarial": budget.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(budget.get("corrigendum_pending"))),
        },
        row_origin="milestone_287",
    )


def _fr11_pilot_row(pilot: Mapping[str, Any]) -> JsonDict:
    budget = _as_mapping(pilot.get("mistake_budget_delta"))
    all_gates_passed = budget.get("all_gates_passed") is True
    mistakes = (_int_or_none(pilot.get("soundness_mistakes")) or 0) + (
        _int_or_none(pilot.get("completeness_mistakes")) or 0
    )
    status = "missing" if not pilot else "flagged" if not all_gates_passed or mistakes else "bounded" if pilot.get("fr11_soundness_bounded_ready") is True else "blocked"
    return _row(
        row_id="dot287:exp3077_fr11_soundness_bounded_pilot",
        status=status,
        source_artifact=EXP3077_REL_PATH.as_posix(),
        source_field="mistake_budget_delta.all_gates_passed",
        evidence_class="fr11_soundness_bounded_online_learning_pilot",
        claim_scope="controller_only_online_learning",
        summary={
            "fr11_soundness_bounded_ready": pilot.get("fr11_soundness_bounded_ready") is True,
            "all_gates_passed": all_gates_passed,
            "soundness_mistakes": _int_or_none(pilot.get("soundness_mistakes")),
            "completeness_mistakes": _int_or_none(pilot.get("completeness_mistakes")),
            "promotion_decision": str(pilot.get("promotion_decision") or ""),
        },
        row_origin="milestone_287",
    )


def _gatemate_ssqa_rows(refresh: Mapping[str, Any]) -> list[JsonDict]:
    refresh_ready = refresh.get("gatemate_ssqa_refresh_ready") is True
    gatemate_allowed = refresh.get("gatemate_rerun_allowed") is True
    ssqa_allowed = refresh.get("ssqa_readback_allowed") is True
    return [
        _row(
            row_id="dot287:exp3078_gatemate_operator_refresh",
            status="missing" if not refresh else "clean" if gatemate_allowed else "blocked",
            source_artifact=EXP3078_REL_PATH.as_posix(),
            source_field="gatemate_rerun_allowed",
            evidence_class="gatemate_no_rerun_operator_refresh",
            claim_scope="hardware_rerun_gate",
            summary={
                "gatemate_ssqa_refresh_ready": refresh_ready,
                "gatemate_rerun_allowed": gatemate_allowed,
                "missing_operator_action_count": len(_as_list(refresh.get("missing_operator_actions"))),
                "hardware_execution_claim_made": refresh.get("hardware_execution_claim_made") is True,
                "speedup_claim_made": refresh.get("speedup_claim_made") is True,
            },
            row_origin="milestone_287",
        ),
        _row(
            row_id="dot287:exp3078_ssqa_readback_refresh",
            status="missing" if not refresh else "clean" if ssqa_allowed else "gated_skipped" if refresh_ready else "blocked",
            source_artifact=EXP3078_REL_PATH.as_posix(),
            source_field="ssqa_readback_allowed",
            evidence_class="ssqa_no_rerun_operator_refresh",
            claim_scope="host_visible_readback_gate",
            summary={
                "gatemate_ssqa_refresh_ready": refresh_ready,
                "ssqa_readback_allowed": ssqa_allowed,
                "hardware_readback_attempted": refresh.get("hardware_readback_attempted") is True,
            },
            row_origin="milestone_287",
        ),
    ]


def _row(
    *,
    row_id: str,
    status: str,
    source_artifact: str,
    source_field: str,
    evidence_class: str,
    claim_scope: str,
    summary: Mapping[str, Any],
    row_origin: str,
) -> JsonDict:
    normalized = normal_status(status)
    return {
        "row_id": row_id,
        "status": normalized,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "evidence_class": evidence_class,
        "blocker_class": blocker_class(normalized),
        "claim_scope": claim_scope,
        "summary": dict(summary),
        "row_origin": row_origin,
    }


def _alias_map(ledger: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row.get("requested_path") or ""): str(row.get("actual_path") or "")
        for row in _as_list(ledger.get("artifact_aliases"))
        if _as_mapping(row).get("actual_present") is True
        for row in [_as_mapping(row)]
        if row.get("requested_path") and row.get("actual_path")
    }


def _artifact_hygiene_row_ids(ledger: Mapping[str, Any]) -> set[str]:
    categories = _as_mapping(ledger.get("blocker_categories"))
    return {
        str(row.get("row_id") or "")
        for row in _as_list(categories.get("artifact_hygiene_blockers"))
        if isinstance(row, Mapping)
    }


def _alias_summary(
    ledger: Mapping[str, Any],
    aliases: Mapping[str, str],
    artifact_hygiene_ids: set[str],
) -> JsonDict:
    return {
        "matrix_v20_normalization_ready": ledger.get("matrix_v20_normalization_ready") is True,
        "aliases_applied": dict(aliases),
        "artifact_hygiene_row_ids_retired": sorted(artifact_hygiene_ids),
        "normalized_blocker_count_estimate": _int_or_none(
            ledger.get("normalized_blocker_count_estimate")
        ),
    }


def _conductor_gate_entries(conductor_log: str, task_title: str) -> list[JsonDict]:
    entries: list[JsonDict] = []
    for line in conductor_log.splitlines():
        if task_title not in line or "GATE_BLOCK" not in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        entries.append(
            {
                "timestamp": parts[0] if len(parts) > 0 else "",
                "task": parts[1] if len(parts) > 1 else task_title,
                "status": parts[2] if len(parts) > 2 else "GATE_BLOCK",
                "details": parts[3] if len(parts) > 3 else "",
                "raw": line,
            }
        )
    return entries


def _source_artifacts(root: Path, rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    spec_by_path = {spec.path.as_posix(): spec for spec in SOURCE_SPECS}
    paths = {spec.path.as_posix() for spec in SOURCE_SPECS}
    paths.update(str(row.get("source_artifact") or "") for row in rows)
    return [_source_record(root, Path(path), spec_by_path.get(path)) for path in sorted(paths) if path]


def _source_record(root: Path, rel_path: Path, spec: SourceSpec | None) -> JsonDict:
    path = root / rel_path
    source_type = spec.source_type if spec is not None else "text" if rel_path.suffix == ".md" else "json"
    payload = read_json_object(path) if source_type == "json" else {}
    text = read_text(path) if source_type == "text" else ""
    return {
        "experiment_id": spec.experiment_id if spec is not None else f"row_source:{rel_path.as_posix()}",
        "path": rel_path.as_posix(),
        "role": spec.role if spec is not None else "row_source_citation",
        "required": spec.required if spec is not None else False,
        "source_type": source_type,
        "present": path.is_file(),
        "readable_json_object": bool(payload) if source_type == "json" else False,
        "readable_text": bool(text) if source_type == "text" else False,
        "sha256": sha256_file(path),
    }


def _required_source_errors(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row.get("experiment_id") or ""),
            "path": str(row.get("path") or ""),
            "reason": "missing_or_malformed_required_source",
        }
        for row in source_artifacts
        if row.get("required") is True
        and row.get("readable_json_object") is not True
        and row.get("readable_text") is not True
    ]


def _invariant_violations(
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blockers: list[Mapping[str, Any]],
    source_artifacts: list[Mapping[str, Any]],
    required_errors: list[Mapping[str, Any]],
) -> list[JsonDict]:
    source_paths = {str(row.get("path") or "") for row in source_artifacts}
    violations = [
        {"kind": "required_source_errors", "count": len(required_errors)}
    ] if required_errors else []
    violations += [
        {"kind": "row_count_mismatch", "rows_total": len(rows), "status_count_sum": sum(status_counts.values())}
    ] if sum(status_counts.values()) != len(rows) else []
    violations += [
        {"kind": "publication_blocker_count_mismatch", "publication_blocker_count": len(publication_blockers)}
    ] if len(publication_blockers) != sum(1 for row in rows if row.get("status") in PUBLICATION_BLOCKING_STATUSES) else []
    violations += [
        {"kind": "malformed_rows", "count": sum(1 for row in rows if not _row_machine_readable(row, source_paths))}
    ] if any(not _row_machine_readable(row, source_paths) for row in rows) else []
    return violations


def _row_machine_readable(row: Mapping[str, Any], source_paths: set[str]) -> bool:
    return (
        REQUIRED_ROW_KEYS <= set(row)
        and normal_status(str(row.get("status") or "missing")) in STATUSES
        and bool(row.get("source_artifact"))
        and str(row.get("source_artifact") or "") in source_paths
        and bool(row.get("source_field"))
    )


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {status: sum(1 for row in rows if normal_status(str(row.get("status") or "missing")) == status) for status in STATUSES}


def _publication_blockers(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_id": str(row.get("row_id") or ""),
            "status": normal_status(str(row.get("status") or "missing")),
            "blocker_class": str(row.get("blocker_class") or ""),
            "source_artifact": str(row.get("source_artifact") or ""),
            "source_field": str(row.get("source_field") or ""),
            "claim_scope": str(row.get("claim_scope") or ""),
        }
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    ]


def _honest_verdict(
    ready: bool,
    rows_total: int,
    publication_blocker_count: int,
    required_errors: list[Mapping[str, Any]],
) -> str:
    if required_errors:
        return (
            "blocked_matrix_v21_preconditions: "
            f"required_source_errors={len(required_errors)}; rows_total={rows_total}"
        )
    if not ready:
        return f"blocked_matrix_v21_preconditions: row_count_or_source_invariant_failed; rows_total={rows_total}"
    return (
        "complete: "
        f"matrix_v21_ready=true; rows_total={rows_total}; "
        f"publication_blocker_count={publication_blocker_count}; paper_ready=false"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts_and_conductor_log",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def normal_status(status: str) -> str:
    """Normalize legacy row labels into the v21 status vocabulary."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map one normalized status to its publication-boundary reason class."""

    return {
        "clean": "none",
        "flagged": "adversarial_or_methodology_flag",
        "bounded": "bounded_claim",
        "blocked": "required_blocker",
        "gated_skipped": "structured_gate_skip",
        "projection_only": "projection_only",
        "missing": "missing_artifact",
        "retired": "retired_claim",
    }[normal_status(status)]


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CAPSTONE_V286_REL_PATH",
    "CONDUCTOR_LOG_REL_PATH",
    "EXP3059_ACTUAL_REL_PATH",
    "EXP3059_REQUESTED_REL_PATH",
    "EXP3067_REL_PATH",
    "EXP3068_REL_PATH",
    "EXP3069_REL_PATH",
    "EXP3070_REL_PATH",
    "EXP3071_REL_PATH",
    "EXP3072_REL_PATH",
    "EXP3073_REL_PATH",
    "EXP3074_REL_PATH",
    "EXP3075_REL_PATH",
    "EXP3076_REL_PATH",
    "EXP3077_REL_PATH",
    "EXP3078_REL_PATH",
    "MATRIX_V20_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "blocker_class",
    "build_artifact",
    "normal_status",
    "read_json_object",
    "read_text",
    "sha256_file",
    "write_artifact",
]
