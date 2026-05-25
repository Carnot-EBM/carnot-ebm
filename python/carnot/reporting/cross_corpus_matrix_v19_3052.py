"""Build the Exp 3052 cross-corpus matrix v19 artifact.

Spec refs: REQ-REPORT-3052, SCENARIO-REPORT-3052.

This module is an accounting step, not a new experiment run. It reads the
already-written .284/.285 JSON artifacts, gives every emitted claim row a
machine-readable status, and keeps bounded, blocked, skipped, missing, and
retired evidence visible for the .285 capstone. That separation matters because
the matrix may cite live or hardware upstream evidence without itself running a
model, solver, synthesis tool, board flash, or smoke test.
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
MILESTONE = "2026.05.285"
SCHEMA = "carnot.cross_corpus_matrix.v19_285_claim_aggregation.v1"
ARTIFACT = "experiment_3052_cross_corpus_matrix_v19"
OUTPUT_REL_PATH = Path("results/experiment_3052_cross_corpus_matrix_v19.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3052_cross_corpus_matrix_v19.py"

MATRIX_V18_REL_PATH = Path("results/experiment_3038_cross_corpus_matrix_v18.json")
CAPSTONE_V284_REL_PATH = Path("results/experiment_3039_capstone_v284.json")
EXP3041_REL_PATH = Path("results/experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.json")
EXP3042_REL_PATH = Path("results/experiment_3042_repair_promotion_reconciliation_v3.json")
EXP3043_REL_PATH = Path("results/experiment_3043_verified_speculation_transcript_fingerprint_v1.json")
EXP3046_REL_PATH = Path("results/experiment_3046_fr11_solver_feedback_self_learning_loop_v1.json")
EXP3047_REL_PATH = Path("results/experiment_3047_kan_locality_nonforgetting_probe_v2.json")
EXP3048_REL_PATH = Path("results/experiment_3048_gatemate_output_contract_operator_package_v1.json")
EXP3050_REL_PATH = Path("results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json")
EXP3051_REL_PATH = Path("results/experiment_3051_ssqa_readback_eligibility_bounded_gate_v3.json")

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
    "clean": "clean_count",
    "flagged": "flagged_count",
    "bounded": "bounded_count",
    "blocked": "blocked_count",
    "gated_skipped": "gated_skipped_count",
    "projection_only": "projection_only_count",
    "missing": "missing_count",
    "retired": "retired_count",
}
REQUIRED_SOURCE_IDS = {"exp3038", "exp3039", "exp3041", "exp3042"}


@dataclass(frozen=True)
class SourceSpec:
    """A checked-in artifact that matrix v19 may consume."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3038", MATRIX_V18_REL_PATH, "matrix_v18_baseline", required=True),
    SourceSpec("exp3039", CAPSTONE_V284_REL_PATH, "capstone_v284_context", required=True),
    SourceSpec("exp3041", EXP3041_REL_PATH, "flag_hygiene_authority", required=True),
    SourceSpec("exp3042", EXP3042_REL_PATH, "repair_reconciliation_authority", required=True),
    SourceSpec("exp3043", EXP3043_REL_PATH, "verified_speculation_fingerprint"),
    SourceSpec("exp3046", EXP3046_REL_PATH, "fr11_solver_feedback"),
    SourceSpec("exp3047", EXP3047_REL_PATH, "fr11_kan_locality"),
    SourceSpec("exp3048", EXP3048_REL_PATH, "gatemate_output_contract"),
    SourceSpec("exp3050", EXP3050_REL_PATH, "gatemate_host_visible_smoke"),
    SourceSpec("exp3051", EXP3051_REL_PATH, "ssqa_readback_gate"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and treat absence, arrays, or malformed JSON as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest for a present artifact."""

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
    """REQ-REPORT-3052: aggregate v19 rows from checked-in artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    loaded = _load_sources(root_path)
    payloads = {exp_id: row["payload"] for exp_id, row in loaded.items()}
    source_artifacts = [_source_artifact(root_path, spec, loaded[spec.experiment_id]) for spec in SOURCE_SPECS]

    rows = _build_rows(payloads, source_artifacts)
    counts = _status_counts(rows)
    duration_s = _duration(start, now_s)
    required_missing = [
        str(row["experiment_id"])
        for row in source_artifacts
        if row["experiment_id"] in REQUIRED_SOURCE_IDS and row["readable_json_object"] is not True
    ]
    all_rows_classified = bool(rows) and all(str(row.get("status")) in STATUSES for row in rows)
    ready = all_rows_classified and not required_missing

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v19_ready": ready,
        "rows_total": len(rows),
        "clean_count": counts["clean"],
        "flagged_count": counts["flagged"],
        "bounded_count": counts["bounded"],
        "blocked_count": counts["blocked"],
        "gated_skipped_count": counts["gated_skipped"],
        "projection_only_count": counts["projection_only"],
        "missing_count": counts["missing"],
        "retired_count": counts["retired"],
        "repair_claim_status": _repair_claim_status(payloads.get("exp3042", {})),
        "fr11_self_learning_status": _fr11_self_learning_status(
            payloads.get("exp3046", {}),
            payloads.get("exp3047", {}),
        ),
        "gatemate_status": _gatemate_status(
            payloads.get("exp3048", {}),
            payloads.get("exp3050", {}),
        ),
        "ssqa_status": _ssqa_status(payloads.get("exp3051", {}), payloads.get("exp3050", {})),
        "rows": rows,
        "source_artifacts": source_artifacts,
        "source_checksums": {str(row["path"]): row.get("sha256") for row in source_artifacts},
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row.get("present") is not True
        ],
        "required_source_errors": [
            {"experiment_id": exp_id, "reason": "missing_or_malformed_required_artifact"}
            for exp_id in required_missing
        ],
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_historical_artifact_rewrite": True,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(ready, counts, len(rows), required_missing),
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3052 deliverable JSON."""

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
        path = root / spec.path
        loaded[spec.experiment_id] = {
            "payload": read_json_object(path),
            "present": path.is_file(),
        }
    return loaded


def _source_artifact(root: Path, spec: SourceSpec, loaded: Mapping[str, Any]) -> JsonDict:
    path = root / spec.path
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "required": spec.required,
        "present": bool(loaded.get("present")),
        "readable_json_object": bool(loaded.get("payload")),
        "sha256": sha256_file(path),
    }


def _build_rows(payloads: Mapping[str, Mapping[str, Any]], source_artifacts: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    rows.extend(_prior_v18_rows(payloads.get("exp3038", {})))
    rows.append(_capstone_row(payloads.get("exp3039", {})))
    rows.append(_flag_hygiene_row(payloads.get("exp3041", {})))
    rows.append(_repair_row(payloads.get("exp3042", {})))
    rows.extend(_retired_repair_rows(payloads.get("exp3042", {})))
    rows.append(_fingerprint_row(payloads.get("exp3043", {})))
    rows.append(_fr11_solver_row(payloads.get("exp3046", {})))
    rows.append(_kan_locality_row(payloads.get("exp3047", {})))
    rows.append(_gatemate_contract_row(payloads.get("exp3048", {})))
    rows.append(_gatemate_smoke_row(payloads.get("exp3050", {}), _source_present(source_artifacts, "exp3050")))
    rows.append(_ssqa_gate_row(payloads.get("exp3051", {}), payloads.get("exp3050", {})))
    return rows


def _prior_v18_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, source_row in enumerate(_as_list(matrix.get("matrix_rows"))):
        row = _as_mapping(source_row)
        exp_id = str(row.get("experiment_id") or f"row_{index}")
        status = _normal_status(str(row.get("status") or "missing"))
        rows.append(
            _row(
                row_id=f"v18:{exp_id}",
                status=status,
                source_artifact=MATRIX_V18_REL_PATH.as_posix(),
                source_field=f"matrix_rows[{exp_id}]",
                evidence_class=str(row.get("task_class") or "prior_v18_matrix_row"),
                blocker_class=_blocker_class(status),
                claim_scope="prior_v18_carry_forward",
                summary={
                    "experiment_id": exp_id,
                    "task_class": str(row.get("task_class") or ""),
                    "source_status": status,
                    "upstream_flags": _as_list(row.get("upstream_flags")),
                    "summary": _as_mapping(row.get("summary")),
                },
            )
        )
    return rows


def _capstone_row(capstone: Mapping[str, Any]) -> JsonDict:
    if not capstone:
        status = "missing"
    elif capstone.get("capstone_ready") is not True:
        status = "blocked"
    elif capstone.get("paper_ready") is True:
        status = "clean"
    else:
        status = "bounded"
    return _row(
        row_id="capstone:v284_paper_readiness",
        status=status,
        source_artifact=CAPSTONE_V284_REL_PATH.as_posix(),
        source_field="paper_ready",
        evidence_class="capstone_synthesis",
        blocker_class="none" if status == "clean" else _blocker_class(status),
        claim_scope="paper_readiness",
        summary={
            "capstone_ready": bool(capstone.get("capstone_ready")),
            "paper_ready": bool(capstone.get("paper_ready")),
            "repair_claim_status": str(capstone.get("repair_claim_status") or ""),
            "fr11_self_learning_status": str(capstone.get("fr11_self_learning_status") or ""),
            "gatemate_status": str(capstone.get("gatemate_status") or ""),
            "ssqa_status": str(capstone.get("ssqa_status") or ""),
        },
    )


def _flag_hygiene_row(hygiene: Mapping[str, Any]) -> JsonDict:
    status = "clean" if hygiene.get("flag_hygiene_ready") is True else "blocked" if hygiene else "missing"
    return _row(
        row_id="methodology:flag_hygiene",
        status=status,
        source_artifact=EXP3041_REL_PATH.as_posix(),
        source_field="flag_hygiene_ready",
        evidence_class="flag_hygiene",
        blocker_class="none" if status == "clean" else _blocker_class(status),
        claim_scope="methodology_boundary",
        summary={
            "rows_reviewed": _int_or_none(hygiene.get("rows_reviewed")),
            "honest_verdict": str(hygiene.get("honest_verdict") or ""),
        },
    )


def _repair_row(repair: Mapping[str, Any]) -> JsonDict:
    blockers = _as_list(repair.get("remaining_blockers"))
    candidate = repair.get("repair_promotion_candidate") is True and not blockers
    if not repair:
        status = "missing"
    elif repair.get("repair_reconciliation_ready") is not True:
        status = "blocked"
    elif candidate:
        status = "clean"
    elif str(repair.get("repair_claim_status") or "") == "retired":
        status = "retired"
    else:
        status = "bounded"
    return _row(
        row_id="repair:headline_status",
        status=status,
        source_artifact=EXP3042_REL_PATH.as_posix(),
        source_field="repair_claim_status",
        evidence_class="repair_reconciliation",
        blocker_class="none" if status == "clean" else _blocker_class(status),
        claim_scope="repair_headline_boundary",
        summary={
            "repair_claim_status": _repair_claim_status(repair),
            "repair_promotion_candidate": candidate,
            "remaining_blocker_count": len(blockers),
            "repair_delta_summary": _as_mapping(repair.get("repair_delta_summary")),
        },
    )


def _retired_repair_rows(repair: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for blocker in _as_list(repair.get("remaining_blockers")):
        row = _as_mapping(blocker)
        if not _is_retired_blocker(row):
            continue
        row_id = str(row.get("row_id") or "repair:retired_claim")
        rows.append(
            _row(
                row_id=row_id,
                status="retired",
                source_artifact=str(row.get("source_artifact") or EXP3042_REL_PATH.as_posix()),
                source_field=str(row.get("source_field") or "remaining_blockers"),
                evidence_class="repair_retirement_boundary",
                blocker_class="retired_claim",
                claim_scope="retired_repair_claim",
                summary={
                    "rationale": str(row.get("rationale") or ""),
                    "evidence": _as_mapping(row.get("evidence")),
                },
            )
        )
    return rows


def _fingerprint_row(fingerprint: Mapping[str, Any]) -> JsonDict:
    clean = (
        fingerprint.get("fingerprint_live_ready") is True
        and fingerprint.get("deterministic_replay_passed") is True
        and fingerprint.get("performance_or_repair_promotion_claim") is not True
        and fingerprint.get("legacy_smoke_only_used") is not True
    )
    status = "clean" if clean else "blocked" if fingerprint else "missing"
    return _row(
        row_id="fingerprint:verified_speculation",
        status=status,
        source_artifact=EXP3043_REL_PATH.as_posix(),
        source_field="fingerprint_live_ready",
        evidence_class="transcript_fingerprint",
        blocker_class="none" if status == "clean" else _blocker_class(status),
        claim_scope="reproducibility_preflight",
        summary={
            "fingerprint_live_ready": bool(fingerprint.get("fingerprint_live_ready")),
            "deterministic_replay_passed": bool(fingerprint.get("deterministic_replay_passed")),
            "n_prompts": _int_or_none(fingerprint.get("n_prompts")),
            "transcript_fingerprint_count": len(_as_list(fingerprint.get("transcript_fingerprints"))),
        },
    )


def _fr11_solver_row(fr11: Mapping[str, Any]) -> JsonDict:
    substrate = _as_mapping(fr11.get("inference_substrate"))
    scope_ok = _fr11_substrate_scope_ok(substrate)
    if not fr11:
        status = "missing"
    elif not scope_ok:
        status = "blocked"
    elif fr11.get("fr11_solver_feedback_ready") is True:
        status = "bounded"
    else:
        status = "blocked"
    return _row(
        row_id="fr11:solver_feedback",
        status=status,
        source_artifact=EXP3046_REL_PATH.as_posix(),
        source_field="fr11_solver_feedback_ready",
        evidence_class="controller_solver_feedback",
        blocker_class=(
            "model_weight_scope_violation"
            if fr11 and not scope_ok
            else "controller_only_scope"
            if status == "bounded"
            else _blocker_class(status)
        ),
        claim_scope="controller_only_self_learning",
        summary={
            "promotion_decision": str(fr11.get("promotion_decision") or ""),
            "edit_targets_used": _as_list(fr11.get("edit_targets_used")),
            "family_holdout_delta": _float_or_none(fr11.get("family_holdout_delta")),
            "prior_retention_delta": _float_or_none(fr11.get("prior_retention_delta")),
            "shuffled_control_delta": _float_or_none(fr11.get("shuffled_control_delta")),
            "model_weight_training": substrate.get("model_weight_training"),
            "model_weight_mutation": substrate.get("model_weight_mutation"),
        },
    )


def _kan_locality_row(kan: Mapping[str, Any]) -> JsonDict:
    substrate = _as_mapping(kan.get("inference_substrate"))
    scope_ok = _fr11_substrate_scope_ok(substrate) and substrate.get("kan_model_weight_training") is not True
    if not kan:
        status = "missing"
    elif not scope_ok:
        status = "blocked"
    elif kan.get("kan_locality_probe_ready") is True:
        status = "bounded"
    else:
        status = "blocked"
    return _row(
        row_id="fr11:kan_locality",
        status=status,
        source_artifact=EXP3047_REL_PATH.as_posix(),
        source_field="kan_locality_probe_ready",
        evidence_class="controller_locality_probe",
        blocker_class=(
            "model_weight_scope_violation"
            if kan and not scope_ok
            else "controller_only_scope"
            if status == "bounded"
            else _blocker_class(status)
        ),
        claim_scope="controller_only_locality",
        summary={
            "promotion_decision": str(kan.get("promotion_decision") or ""),
            "locality_metric": _float_or_none(kan.get("locality_metric")),
            "changed_anchor_count": _int_or_none(kan.get("changed_anchor_count")),
            "anchored_prior_count": _int_or_none(kan.get("anchored_prior_count")),
            "heldout_delta": _float_or_none(kan.get("heldout_delta")),
        },
    )


def _gatemate_contract_row(contract: Mapping[str, Any]) -> JsonDict:
    ready = (
        contract.get("gatemate_output_contract_ready") is True
        and contract.get("host_visible_io_plan_ready") is True
        and bool(contract.get("host_reader_command"))
        and bool(_as_list(contract.get("expected_transcript")))
        and contract.get("hardware_execution_claim_made") is not True
        and contract.get("speedup_claim_made") is not True
    )
    status = "clean" if ready else "blocked" if contract else "missing"
    return _row(
        row_id="gatemate:output_contract",
        status=status,
        source_artifact=EXP3048_REL_PATH.as_posix(),
        source_field="gatemate_output_contract_ready",
        evidence_class="gatemate_output_contract",
        blocker_class="none" if status == "clean" else _blocker_class(status),
        claim_scope="hardware_contract_only",
        summary={
            "gatemate_output_contract_ready": bool(contract.get("gatemate_output_contract_ready")),
            "host_visible_io_plan_ready": bool(contract.get("host_visible_io_plan_ready")),
            "selected_output_signal": str(contract.get("selected_output_signal") or ""),
            "missing_operator_action_count": len(_as_list(contract.get("missing_operator_actions"))),
        },
    )


def _gatemate_smoke_row(smoke: Mapping[str, Any], present: bool) -> JsonDict:
    passed = (
        smoke.get("gatemate_host_visible_smoke_passed") is True
        and bool(_as_list(smoke.get("observed_transcript")))
        and smoke.get("transcript_matched") is True
        and smoke.get("speedup_claim_made") is not True
        and smoke.get("boltzmann_claim_made") is not True
        and smoke.get("sampler_claim_made") is not True
    )
    status = "clean" if passed else "blocked" if present else "missing"
    return _row(
        row_id="gatemate:host_visible_smoke",
        status=status,
        source_artifact=EXP3050_REL_PATH.as_posix(),
        source_field="gatemate_host_visible_smoke_passed",
        evidence_class="host_visible_transcript",
        blocker_class="none" if status == "clean" else _blocker_class(status),
        claim_scope="host_visible_hardware_transcript",
        summary={
            "gatemate_host_visible_smoke_passed": bool(
                smoke.get("gatemate_host_visible_smoke_passed")
            ),
            "observed_transcript_count": len(_as_list(smoke.get("observed_transcript"))),
            "transcript_matched": bool(smoke.get("transcript_matched")),
            "honest_verdict": str(smoke.get("honest_verdict") or ""),
        },
    )


def _ssqa_gate_row(ssqa: Mapping[str, Any], smoke: Mapping[str, Any]) -> JsonDict:
    if not ssqa:
        status = "missing"
    elif ssqa.get("ssqa_gate_artifact_ready") is True and ssqa.get("consumed_gatemate_smoke") is True:
        status = "clean"
    elif _status_from_gate_payload(ssqa) == "gated_skipped" or not smoke:
        status = "gated_skipped"
    else:
        status = "blocked"
    return _row(
        row_id="ssqa:readback_gate",
        status=status,
        source_artifact=EXP3051_REL_PATH.as_posix(),
        source_field="ssqa_status",
        evidence_class="ssqa_gate",
        blocker_class="none" if status == "clean" else _blocker_class(status),
        claim_scope="ssqa_readback_eligibility",
        summary={
            "ssqa_status": str(ssqa.get("ssqa_status") or ssqa.get("status") or ""),
            "consumed_gatemate_smoke": bool(ssqa.get("consumed_gatemate_smoke")),
            "gate_check_summary": str(ssqa.get("gate_check_summary") or ""),
            "gates_evaluated": _as_list(ssqa.get("gates_evaluated")),
        },
    )


def _row(
    *,
    row_id: str,
    status: str,
    source_artifact: str,
    source_field: str,
    evidence_class: str,
    blocker_class: str,
    claim_scope: str,
    summary: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "row_id": row_id,
        "status": _normal_status(status),
        "source_artifact": source_artifact,
        "source_field": source_field,
        "evidence_class": evidence_class,
        "blocker_class": blocker_class,
        "claim_scope": claim_scope,
        "summary": dict(summary or {}),
    }


def _normal_status(status: str) -> str:
    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def _blocker_class(status: str) -> str:
    return {
        "clean": "none",
        "flagged": "adversarial_or_methodology_flag",
        "bounded": "bounded_claim",
        "blocked": "required_blocker",
        "gated_skipped": "structured_gate_skip",
        "projection_only": "projection_only",
        "missing": "missing_artifact",
        "retired": "retired_claim",
    }[_normal_status(status)]


def _repair_claim_status(repair: Mapping[str, Any]) -> str:
    if not repair:
        return "missing"
    blockers = _as_list(repair.get("remaining_blockers"))
    if repair.get("repair_promotion_candidate") is True and not blockers:
        return "clean_candidate"
    return str(repair.get("repair_claim_status") or "blocked")


def _fr11_self_learning_status(fr11: Mapping[str, Any], kan: Mapping[str, Any]) -> str:
    if fr11 and not _fr11_substrate_scope_ok(_as_mapping(fr11.get("inference_substrate"))):
        return "blocked_model_weight_scope_violation"
    if kan and (
        not _fr11_substrate_scope_ok(_as_mapping(kan.get("inference_substrate")))
        or _as_mapping(kan.get("inference_substrate")).get("kan_model_weight_training") is True
    ):
        return "blocked_model_weight_scope_violation"
    fr11_ready = fr11.get("fr11_solver_feedback_ready") is True
    kan_ready = kan.get("kan_locality_probe_ready") is True
    if fr11_ready and kan_ready:
        return "controller_only_solver_feedback_and_locality_ready"
    if fr11_ready:
        return "controller_only_solver_feedback_ready"
    if kan_ready:
        return "controller_only_locality_ready"
    if not fr11 and not kan:
        return "missing"
    return "blocked"


def _gatemate_status(contract: Mapping[str, Any], smoke: Mapping[str, Any]) -> str:
    contract_ready = (
        contract.get("gatemate_output_contract_ready") is True
        and contract.get("host_visible_io_plan_ready") is True
    )
    smoke_ready = (
        smoke.get("gatemate_host_visible_smoke_passed") is True
        and bool(_as_list(smoke.get("observed_transcript")))
    )
    if contract_ready and smoke_ready:
        return "host_visible_transcript_ready"
    if contract and not contract_ready:
        return "blocked_output_contract"
    if not smoke:
        return "missing_host_visible_transcript"
    return "blocked_host_visible_transcript"


def _ssqa_status(ssqa: Mapping[str, Any], smoke: Mapping[str, Any]) -> str:
    if ssqa.get("ssqa_gate_artifact_ready") is True and ssqa.get("consumed_gatemate_smoke") is True:
        return str(ssqa.get("ssqa_status") or "eligible")
    if ssqa and (_status_from_gate_payload(ssqa) == "gated_skipped" or not smoke):
        return "gated_skipped_host_visible_smoke_missing"
    if not ssqa:
        return "missing"
    return "blocked"


def _status_from_gate_payload(payload: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            str(payload.get("status") or ""),
            str(payload.get("blocked_at_layer") or ""),
            str(payload.get("honest_verdict") or ""),
            str(payload.get("gate_check_summary") or ""),
        ]
    ).lower()
    if "gate" in text and ("blocked" in text or "failed" in text):
        return "gated_skipped"
    return "blocked"


def _is_retired_blocker(row: Mapping[str, Any]) -> bool:
    evidence = _as_mapping(row.get("evidence"))
    return (
        str(evidence.get("classification") or "") == "retired"
        or "retired_or_blocked_claims" in str(row.get("source_field") or "")
    )


def _fr11_substrate_scope_ok(substrate: Mapping[str, Any]) -> bool:
    return (
        substrate.get("live_llm_inference") is not True
        and substrate.get("model_weight_training") is not True
        and substrate.get("model_weight_mutation") is not True
    )


def _source_present(source_artifacts: list[Mapping[str, Any]], experiment_id: str) -> bool:
    for row in source_artifacts:
        if row.get("experiment_id") == experiment_id:
            return row.get("present") is True
    return False


def _status_counts(rows: list[JsonDict]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _honest_verdict(
    ready: bool,
    counts: Mapping[str, int],
    rows_total: int,
    required_missing: list[str],
) -> str:
    if required_missing:
        return "blocked_required_source_missing: " + ",".join(required_missing)
    prefix = "complete: matrix_v19_ready=true" if ready else "blocked_matrix_v19_rows_unclassified"
    return (
        f"{prefix}; rows_total={rows_total}; "
        f"clean={counts['clean']}; flagged={counts['flagged']}; "
        f"bounded={counts['bounded']}; blocked={counts['blocked']}; "
        f"gated_skipped={counts['gated_skipped']}; "
        f"projection_only={counts['projection_only']}; missing={counts['missing']}; "
        f"retired={counts['retired']}"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool_field(payload: Mapping[str, Any], key: str) -> bool:
    return payload.get(key) is True


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
    "CAPSTONE_V284_REL_PATH",
    "EXP3041_REL_PATH",
    "EXP3042_REL_PATH",
    "EXP3043_REL_PATH",
    "EXP3046_REL_PATH",
    "EXP3047_REL_PATH",
    "EXP3048_REL_PATH",
    "EXP3050_REL_PATH",
    "EXP3051_REL_PATH",
    "MATRIX_V18_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "build_artifact",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
