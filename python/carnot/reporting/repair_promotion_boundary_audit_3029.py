"""Build the Exp 3029 repair promotion boundary audit artifact.

Spec refs: REQ-REPORT-3029, SCENARIO-REPORT-3029.

This module is a claim-boundary ledger, not a repair rerun. It reads the
methodology corrigendum, the clean-methodology repair reconstruction, and the
older matrix/capstone rows, then decides which repair wording is supported.
Keeping the wording decision separate matters because a clean upstream rerun can
still be too new for the existing matrix/capstone to have promoted safely.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.284"
SCHEMA = "carnot.repair_promotion_boundary_audit.v2"
ARTIFACT = "experiment_3029_repair_promotion_boundary_audit_v2"
INFERENCE_SUBSTRATE_KIND = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3029_repair_promotion_boundary_audit_v2.json")

EXP3027_REL_PATH = Path("results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json")
EXP3028_REL_PATH = Path("results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json")
EXP3016_REL_PATH = Path(
    "results/experiment_3016_sota_repair_rerun_with_acceptance_controller_v1.json"
)
MATRIX_V17_REL_PATH = Path("results/experiment_3024_cross_corpus_matrix_v17.json")
CAPSTONE_V283_REL_PATH = Path("results/experiment_3025_capstone_v283.json")

MIN_REPAIR_EVIDENCE_ROWS = 20


@dataclass(frozen=True)
class SourceSpec:
    """A required upstream JSON artifact for the boundary audit."""

    experiment_id: str
    path: Path
    required: bool = True


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3027", EXP3027_REL_PATH),
    SourceSpec("exp3028", EXP3028_REL_PATH),
    SourceSpec("exp3016", EXP3016_REL_PATH),
    SourceSpec("exp3024", MATRIX_V17_REL_PATH),
    SourceSpec("exp3025", CAPSTONE_V283_REL_PATH),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object while preserving absence as empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA256 digest for an existing file."""

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
) -> dict[str, Any]:
    """REQ-REPORT-3029: classify repair promotion wording from upstream JSON."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    source_artifacts = _source_artifacts(root_path, payloads)
    required_errors = _required_source_errors(payloads)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    if required_errors:
        return {
            "schema": SCHEMA,
            "artifact": ARTIFACT,
            "run_date": RUN_DATE,
            "milestone": MILESTONE,
            "repair_promotion_boundary_ready": False,
            "repair_claim_status": "blocked",
            "promotable_claims": [],
            "bounded_claims": [],
            "retired_or_blocked_claims": [],
            "claim_boundary_table": [],
            "required_source_errors": required_errors,
            "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads),
            "source_artifacts_read": source_artifacts,
            "source_checksums": _source_checksums(source_artifacts),
            "inference_substrate": _inference_substrate(),
            "no_new_llm_call": True,
            "no_new_verifier_run": True,
            "no_new_solver_run": True,
            "no_new_synthesis_run": True,
            "no_new_board_flash": True,
            "no_new_hardware_run": True,
            "status_updates_written": False,
            "ops_docs_reconciliation_left_to_conductor": True,
            "duration_s": duration_s,
            "honest_verdict": "blocked_required_upstream_missing",
        }

    exp3027 = payloads["exp3027"]
    exp3028 = payloads["exp3028"]
    exp3016 = payloads["exp3016"]
    matrix = payloads["exp3024"]
    capstone = payloads["exp3025"]
    claim_table = _claim_boundary_table(exp3027, exp3028, exp3016, matrix, capstone)
    promotable = [row for row in claim_table if row["classification"] == "promotable"]
    bounded = [row for row in claim_table if row["classification"] == "bounded"]
    retired_or_blocked = [
        row for row in claim_table if row["classification"] in {"retired", "blocked"}
    ]
    status = _repair_claim_status(claim_table)

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "repair_promotion_boundary_ready": True,
        "repair_claim_status": status,
        "promotable_claims": promotable,
        "bounded_claims": bounded,
        "retired_or_blocked_claims": retired_or_blocked,
        "claim_boundary_table": claim_table,
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads),
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "inference_substrate": _inference_substrate(),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(status, promotable, bounded, retired_or_blocked),
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3029 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main(root: Path | str = REPO_ROOT) -> int:
    """Write the audit artifact and return process-style success."""

    output = write_artifact(root)
    artifact = read_json_object(output)
    return 0 if artifact.get("repair_promotion_boundary_ready") is True else 1


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    return {spec.experiment_id: read_json_object(root / spec.path) for spec in SOURCE_SPECS}


def _source_artifacts(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "present": path.is_file(),
                "required": spec.required,
                "readable_json_object": bool(payloads.get(spec.experiment_id)),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _source_checksums(source_artifacts: list[dict[str, Any]]) -> dict[str, str | None]:
    return {str(row["path"]): row["sha256"] for row in source_artifacts}


def _required_source_errors(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "experiment_id": spec.experiment_id,
            "path": spec.path.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
        for spec in SOURCE_SPECS
        if spec.required and not payloads.get(spec.experiment_id)
    ]


def _claim_boundary_table(
    exp3027: Mapping[str, Any],
    exp3028: Mapping[str, Any],
    exp3016: Mapping[str, Any],
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> list[dict[str, Any]]:
    clean_blockers = _clean_evidence_blockers(exp3028)
    boundary_blockers = _matrix_capstone_boundary_blockers(matrix, capstone)
    exp3028_classification = _exp3028_claim_classification(clean_blockers, boundary_blockers)
    headline_blockers = [*clean_blockers, *boundary_blockers]
    legacy_blockers = _legacy_exp3016_blockers(exp3027, exp3016)

    return [
        {
            "claim_id": "exp3028_clean_repair_candidate",
            "proposed_repair_claim": (
                "Exp 3028 supplies clean acceptance-controlled SOTA repair evidence."
            ),
            "classification": exp3028_classification,
            "required_support_fields": [
                "exp3028.clean_repair_rerun_ready == true",
                "exp3028.repair_controller_clean == true",
                "exp3028.clean_repair_claim_promotable_candidate == true",
                f"exp3028.n_tasks >= {MIN_REPAIR_EVIDENCE_ROWS}",
                "exp3028.n_live_transcripts >= exp3028.n_tasks",
                "exp3028.pass_at_1_delta > 0",
                "exp3028.pass_at_k_delta >= 0",
                "exp3028.syntax_failure_rate_delta <= 0",
                "exp3028.schema_failure_rate_delta <= 0",
                "exp3028.false_accept_delta <= 0",
                "exp3028.tautology_gate_clean == true",
                "exp3028.intent_drift_count == 0",
                "exp3028.legacy_smoke_only_used == false",
                "exp3028.reproducibility_checksum present",
            ],
            "observed_support_fields": _exp3028_observed_support(exp3028),
            "blockers": clean_blockers if clean_blockers else boundary_blockers,
            "allowed_wording": _exp3028_allowed_wording(exp3028_classification),
        },
        {
            "claim_id": "headline_sota_repair_clean_methodology",
            "proposed_repair_claim": (
                "The repair result is promotable as a headline SOTA repair claim."
            ),
            "classification": "promotable" if not headline_blockers else "retired",
            "required_support_fields": [
                "all Exp 3028 clean-evidence gates pass",
                "matrix_v17 repair row status == clean",
                "capstone repair promotion decision promotable == true",
                "capstone paper-ready blockers contain no repair blocker",
                "no claim-boundary violations are present",
            ],
            "observed_support_fields": _headline_observed_support(exp3028, matrix, capstone),
            "blockers": headline_blockers,
            "allowed_wording": _headline_allowed_wording(bool(headline_blockers)),
        },
        {
            "claim_id": "unsupported_exp3016_headline_repair_promotion",
            "proposed_repair_claim": (
                "The original Exp 3016 headline repair claim is promotable without the Exp 3028 boundary."
            ),
            "classification": "retired" if legacy_blockers else "blocked",
            "required_support_fields": [
                "Exp 3027 does not require repair rerun or reconstruction",
                "Exp 3016 top-level random_seed is present",
                "Exp 3016 top-level transcript hashes are present",
                "Exp 3016 source metadata is sufficient without later repair",
            ],
            "observed_support_fields": _legacy_observed_support(exp3027, exp3016),
            "blockers": legacy_blockers,
            "allowed_wording": (
                "Retire this wording; cite Exp 3028's bounded clean-methodology evidence instead."
                if legacy_blockers
                else "Blocked from promotion because the audit is scoped to Exp 3028 replacement evidence."
            ),
        },
    ]


def _clean_evidence_blockers(exp3028: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    checks = [
        ("clean_repair_rerun_ready is not true", exp3028.get("clean_repair_rerun_ready") is True),
        ("repair_controller_clean is not true", exp3028.get("repair_controller_clean") is True),
        (
            "clean_repair_claim_promotable_candidate is not true",
            exp3028.get("clean_repair_claim_promotable_candidate") is True,
        ),
        (
            f"n_tasks is below {MIN_REPAIR_EVIDENCE_ROWS}",
            _as_float(exp3028.get("n_tasks")) is not None
            and _as_float(exp3028.get("n_tasks")) >= MIN_REPAIR_EVIDENCE_ROWS,
        ),
        (
            "n_live_transcripts is below n_tasks",
            _as_float(exp3028.get("n_live_transcripts")) is not None
            and _as_float(exp3028.get("n_tasks")) is not None
            and _as_float(exp3028.get("n_live_transcripts")) >= _as_float(exp3028.get("n_tasks")),
        ),
        (
            "model_specs are absent",
            bool(exp3028.get("model_specs")) and _count(exp3028.get("model_specs")) > 0,
        ),
        (
            "headline model evidence is absent",
            bool(exp3028.get("headline_models_used"))
            and _count(exp3028.get("headline_models_used")) > 0,
        ),
        ("legacy_smoke_only_used is true", exp3028.get("legacy_smoke_only_used") is False),
        (
            "pass_at_1_delta is not positive",
            _as_float(exp3028.get("pass_at_1_delta")) is not None
            and _as_float(exp3028.get("pass_at_1_delta")) > 0.0,
        ),
        (
            "pass_at_k_delta is negative",
            _as_float(exp3028.get("pass_at_k_delta")) is not None
            and _as_float(exp3028.get("pass_at_k_delta")) >= 0.0,
        ),
        (
            "syntax_failure_rate_delta is positive",
            _as_float(exp3028.get("syntax_failure_rate_delta")) is not None
            and _as_float(exp3028.get("syntax_failure_rate_delta")) <= 0.0,
        ),
        (
            "schema_failure_rate_delta is positive",
            _as_float(exp3028.get("schema_failure_rate_delta")) is not None
            and _as_float(exp3028.get("schema_failure_rate_delta")) <= 0.0,
        ),
        (
            "false_accept_delta is positive",
            _as_float(exp3028.get("false_accept_delta")) is not None
            and _as_float(exp3028.get("false_accept_delta")) <= 0.0,
        ),
        ("tautology_gate_clean is not true", exp3028.get("tautology_gate_clean") is True),
        ("intent_drift_count is not zero", _as_float(exp3028.get("intent_drift_count")) == 0.0),
        ("reproducibility_checksum is absent", bool(exp3028.get("reproducibility_checksum"))),
    ]
    for message, passed in checks:
        if not passed:
            blockers.append(message)
    return blockers


def _matrix_capstone_boundary_blockers(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    repair_row = _matrix_repair_row(matrix)
    repair_status = str(repair_row.get("status") or "missing")
    capstone_decision = _capstone_repair_decision(capstone)
    if matrix.get("matrix_v17_ready") is not True:
        blockers.append("matrix_v17_ready is not true")
    if repair_status != "clean":
        blockers.append(f"matrix repair row is {repair_status}")
    if capstone.get("capstone_ready") is not True:
        blockers.append("capstone_ready is not true")
    if capstone_decision.get("promotable") is not True:
        blockers.append("capstone repair decision is not promotable")
    if _repair_paper_blockers(capstone):
        blockers.append("capstone still lists repair paper-ready blocker")
    if _mapping_list(matrix.get("claim_boundary_violations")):
        blockers.append("matrix claim boundary violations are present")
    return blockers


def _legacy_exp3016_blockers(
    exp3027: Mapping[str, Any],
    exp3016: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if exp3027.get("repair_rerun_required") is True:
        blockers.append("Exp 3027 requires repair rerun or reconstruction")
    if exp3016.get("random_seed") in (None, ""):
        blockers.append("Exp 3016 random_seed missing")
    if exp3016.get("transcript_sha256s") in (None, "", [], {}):
        blockers.append("Exp 3016 transcript_sha256s missing")
    return blockers


def _exp3028_claim_classification(clean_blockers: list[str], boundary_blockers: list[str]) -> str:
    if clean_blockers:
        return "retired"
    if boundary_blockers:
        return "bounded"
    return "promotable"


def _exp3028_allowed_wording(classification: str) -> str:
    if classification == "promotable":
        return (
            "May state that Exp 3028 provides clean repair evidence under the audited methodology."
        )
    if classification == "bounded":
        return (
            "Use bounded wording: Exp 3028 provides clean reconstructed repair evidence, "
            "but headline repair promotion waits for matrix/capstone reconciliation."
        )
    return "Retire clean-repair wording until the rerun evidence satisfies every safety gate."


def _headline_allowed_wording(has_blockers: bool) -> str:
    if has_blockers:
        return (
            "Do not use headline SOTA repair wording; cite the bounded Exp 3028 evidence and "
            "name the unresolved matrix/capstone blockers."
        )
    return "The paper may state the clean-methodology repair claim with Exp 3028 support."


def _exp3028_observed_support(exp3028: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clean_repair_rerun_ready": exp3028.get("clean_repair_rerun_ready") is True,
        "repair_controller_clean": exp3028.get("repair_controller_clean") is True,
        "clean_repair_claim_promotable_candidate": (
            exp3028.get("clean_repair_claim_promotable_candidate") is True
        ),
        "n_tasks": exp3028.get("n_tasks"),
        "n_live_transcripts": exp3028.get("n_live_transcripts"),
        "model_specs_count": _count(exp3028.get("model_specs")),
        "headline_models_used_count": _count(exp3028.get("headline_models_used")),
        "legacy_smoke_only_used": exp3028.get("legacy_smoke_only_used") is True,
        "pass_at_1_delta": exp3028.get("pass_at_1_delta"),
        "pass_at_k_delta": exp3028.get("pass_at_k_delta"),
        "syntax_failure_rate_delta": exp3028.get("syntax_failure_rate_delta"),
        "schema_failure_rate_delta": exp3028.get("schema_failure_rate_delta"),
        "false_accept_delta": exp3028.get("false_accept_delta"),
        "tautology_gate_clean": exp3028.get("tautology_gate_clean") is True,
        "intent_drift_count": exp3028.get("intent_drift_count"),
        "reproducibility_checksum_present": bool(exp3028.get("reproducibility_checksum")),
    }


def _headline_observed_support(
    exp3028: Mapping[str, Any],
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> dict[str, Any]:
    repair_row = _matrix_repair_row(matrix)
    capstone_decision = _capstone_repair_decision(capstone)
    return {
        "exp3028_clean": not _clean_evidence_blockers(exp3028),
        "matrix_v17_ready": matrix.get("matrix_v17_ready") is True,
        "matrix_repair_row_status": str(repair_row.get("status") or "missing"),
        "matrix_repair_row_upstream_flag_count": _count(repair_row.get("upstream_flags")),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "capstone_repair_status": str(capstone_decision.get("status") or "missing"),
        "capstone_repair_promotable": capstone_decision.get("promotable") is True,
        "repair_paper_ready_blocker_count": len(_repair_paper_blockers(capstone)),
        "claim_boundary_violation_count": len(
            _mapping_list(matrix.get("claim_boundary_violations"))
        ),
    }


def _legacy_observed_support(
    exp3027: Mapping[str, Any],
    exp3016: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "exp3027_methodology_corrigendum_ready": (
            exp3027.get("methodology_corrigendum_ready") is True
        ),
        "exp3027_repair_rerun_required": exp3027.get("repair_rerun_required") is True,
        "exp3027_repair_rerun_decision": str(
            _mapping(exp3027.get("repair_rerun_decision")).get("decision") or ""
        ),
        "exp3016_random_seed_present": exp3016.get("random_seed") not in (None, ""),
        "exp3016_transcript_sha256s_present": exp3016.get("transcript_sha256s")
        not in (None, "", [], {}),
        "exp3016_reproducibility_checksum_present": bool(exp3016.get("reproducibility_checksum")),
    }


def _repair_claim_status(claim_table: list[Mapping[str, Any]]) -> str:
    if any(
        row.get("claim_id") == "headline_sota_repair_clean_methodology"
        and row.get("classification") == "promotable"
        for row in claim_table
    ):
        return "clean"
    if any(row.get("classification") == "bounded" for row in claim_table):
        return "bounded"
    if any(row.get("classification") == "retired" for row in claim_table):
        return "retired"
    return "blocked"


def _matrix_repair_row(matrix: Mapping[str, Any]) -> dict[str, Any]:
    claim_rows = matrix.get("claim_rows")
    if isinstance(claim_rows, Mapping) and isinstance(claim_rows.get("exp3016_repair"), Mapping):
        return dict(claim_rows["exp3016_repair"])
    for row in _mapping_list(matrix.get("rows")):
        if row.get("row_id") == "exp3016_repair_acceptance_controller":
            return row
    return {}


def _capstone_repair_decision(capstone: Mapping[str, Any]) -> dict[str, Any]:
    decisions = capstone.get("claim_promotion_decisions")
    if isinstance(decisions, Mapping) and isinstance(decisions.get("repair"), Mapping):
        return dict(decisions["repair"])
    return {}


def _repair_paper_blockers(capstone: Mapping[str, Any]) -> list[str]:
    return [
        blocker
        for blocker in _string_list(capstone.get("paper_ready_blockers"))
        if "repair" in blocker.lower() or "exp3016" in blocker.lower()
    ]


def _cited_upstream_artifacts(
    source_artifacts: list[dict[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for source in source_artifacts:
        exp_id = str(source["experiment_id"])
        payload = payloads.get(exp_id, {})
        citation: dict[str, Any] = {
            "experiment_id": exp_id,
            "path": source["path"],
            "present": source["present"],
            "readable_json_object": source["readable_json_object"],
            "sha256": source["sha256"],
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "inference_substrate": payload.get("inference_substrate"),
            "source_field_summary": _source_field_summary(exp_id, payload),
        }
        model_provenance = _source_model_provenance(payload)
        if model_provenance:
            citation["model_provenance"] = model_provenance
        citations.append(citation)
    return citations


def _source_field_summary(exp_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if exp_id == "exp3028":
        return _exp3028_observed_support(payload)
    if exp_id == "exp3027":
        return {
            "methodology_corrigendum_ready": payload.get("methodology_corrigendum_ready") is True,
            "repair_rerun_required": payload.get("repair_rerun_required") is True,
            "missing_metadata_row_count": _count(payload.get("missing_metadata_rows")),
        }
    if exp_id == "exp3024":
        repair_row = _matrix_repair_row(payload)
        return {
            "matrix_v17_ready": payload.get("matrix_v17_ready") is True,
            "repair_row_status": str(repair_row.get("status") or "missing"),
            "claim_boundary_violation_count": len(
                _mapping_list(payload.get("claim_boundary_violations"))
            ),
        }
    if exp_id == "exp3025":
        repair_decision = _capstone_repair_decision(payload)
        return {
            "capstone_ready": payload.get("capstone_ready") is True,
            "paper_ready": payload.get("paper_ready") is True,
            "repair_decision_status": str(repair_decision.get("status") or "missing"),
            "repair_decision_promotable": repair_decision.get("promotable") is True,
        }
    return {
        "repair_controller_clean": payload.get("repair_controller_clean") is True,
        "headline_result": payload.get("headline_result") is True,
        "random_seed_present": payload.get("random_seed") not in (None, ""),
        "transcript_sha256s_present": payload.get("transcript_sha256s") not in (None, "", [], {}),
    }


def _source_model_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = {
        "model_specs": payload.get("model_specs"),
        "headline_models_used": payload.get("headline_models_used"),
        "model_checksums": payload.get("model_checksums"),
    }
    substrate = payload.get("inference_substrate")
    if isinstance(substrate, Mapping) and substrate.get("gguf_cache_paths"):
        provenance["gguf_cache_paths"] = substrate.get("gguf_cache_paths")
    return {key: value for key, value in provenance.items() if value not in (None, [], {})}


def _inference_substrate() -> dict[str, Any]:
    return {
        "kind": INFERENCE_SUBSTRATE_KIND,
        "no_live_llm_inference": True,
        "no_new_repair_generation": True,
        "no_verifier_scoring_run": True,
        "no_top_level_live_model_metadata": True,
        "source_model_metadata_location": "cited_upstream_artifacts[].model_provenance",
    }


def _honest_verdict(
    status: str,
    promotable: list[Mapping[str, Any]],
    bounded: list[Mapping[str, Any]],
    retired_or_blocked: list[Mapping[str, Any]],
) -> str:
    return (
        f"complete: repair_claim_status={status}; promotable={len(promotable)}; "
        f"bounded={len(bounded)}; retired_or_blocked={len(retired_or_blocked)}"
    )


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: object) -> list[dict[str, Any]]:
    return (
        [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, list)
        else []
    )


def _string_list(value: object) -> list[str]:
    return (
        [str(item) for item in value if item not in (None, "")] if isinstance(value, list) else []
    )


def _count(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
