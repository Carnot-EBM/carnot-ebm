"""Build the Exp 3041 matrix/capstone adversarial flag-hygiene artifact.

Spec refs: REQ-REPORT-3041, SCENARIO-REPORT-3041.

This module is a bookkeeping audit, not a verifier. It reads the .284 repair,
matrix, and capstone JSON artifacts and gives downstream tasks a mechanical
map from each visible flag or bounded row to the reason it must stay blocked,
bounded, or can be ignored as an aggregation-substrate false positive. That
keeps paper-promotion logic from guessing based on prose in earlier artifacts.
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
SCHEMA = "carnot.flag_hygiene.matrix_capstone.v1"
ARTIFACT = "experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1"
OUTPUT_REL_PATH = Path("results/experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.py"

EXP3027_REL_PATH = Path("results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json")
EXP3028_REL_PATH = Path("results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json")
EXP3029_REL_PATH = Path("results/experiment_3029_repair_promotion_boundary_audit_v2.json")
MATRIX_V18_REL_PATH = Path("results/experiment_3038_cross_corpus_matrix_v18.json")
CAPSTONE_V284_REL_PATH = Path("results/experiment_3039_capstone_v284.json")

DOWNSTREAM_CONSUMERS = ("exp3042", "exp3043")
AGGREGATION_FLAG_KINDS = {"DURATION_TOO_SHORT", "METHODOLOGY_MISSING"}
METHODOLOGY_FLAG_KIND = "METHODOLOGY_MISSING"
GATE_SKIPPED_STATUSES = {"gated_skipped", "gated-skipped", "gate_skipped"}
ALLOWED_CLASSIFICATIONS = {
    "true_blocker",
    "aggregation_false_positive",
    "missing_metadata",
    "unresolved_bound",
    "hardware_blocked",
    "gate_skipped",
}


@dataclass(frozen=True)
class SourceSpec:
    """A required checked-in artifact consumed by the flag-hygiene audit."""

    experiment_id: str
    path: Path
    required: bool = True


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3027", EXP3027_REL_PATH),
    SourceSpec("exp3028", EXP3028_REL_PATH),
    SourceSpec("exp3029", EXP3029_REL_PATH),
    SourceSpec("exp3038", MATRIX_V18_REL_PATH),
    SourceSpec("exp3039", CAPSTONE_V284_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and treat absence, malformed JSON, or arrays as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest for an existing file."""

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
    """REQ-REPORT-3041: classify .284 matrix/capstone flags from source JSON only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    loaded = _load_sources(root_path)
    source_artifacts = [_source_artifact(root_path, spec, loaded[spec.experiment_id]) for spec in SOURCE_SPECS]
    required_errors = _required_source_errors(loaded)
    duration_s = _duration(start, now_s)
    base_artifact = _base_artifact(source_artifacts, duration_s)

    if required_errors:
        base_artifact.update(
            {
                "flag_hygiene_ready": False,
                "rows_reviewed": 0,
                "required_source_errors": required_errors,
                "honest_verdict": "blocked_required_source_missing: "
                + ",".join(error["experiment_id"] for error in required_errors),
            }
        )
        return base_artifact

    payloads = {exp_id: row["payload"] for exp_id, row in loaded.items()}
    lists = _classification_lists(payloads)
    base_artifact.update(lists)
    all_rows = classification_rows(base_artifact)
    ready = bool(all_rows) and all(_mechanically_consumable(row) for row in all_rows)
    base_artifact.update(
        {
            "flag_hygiene_ready": ready,
            "rows_reviewed": len(all_rows),
            "classification_summary": _classification_summary(base_artifact),
            "downstream_consumers": list(DOWNSTREAM_CONSUMERS),
            "consumer_contract": {
                "required_row_fields": [
                    "row_id",
                    "classification",
                    "blocking",
                    "source_artifact",
                    "source_field",
                ],
                "remove_only_classification": "aggregation_false_positive",
                "keep_blocking_classifications": [
                    "true_blocker",
                    "missing_metadata",
                    "unresolved_bound",
                    "hardware_blocked",
                    "gate_skipped",
                ],
            },
            "honest_verdict": _honest_verdict(ready, len(all_rows), base_artifact),
        }
    )
    return base_artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3041 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def classification_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return all classification rows in the consumer order used by Exp 3042/3043."""

    rows: list[JsonDict] = []
    for key in (
        "true_blocker_rows",
        "aggregation_false_positive_rows",
        "missing_metadata_rows",
        "unresolved_bound_rows",
        "hardware_blocked_rows",
        "gate_skipped_rows",
    ):
        rows.extend(dict(row) for row in _as_list(artifact.get(key)) if isinstance(row, Mapping))
    return rows


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
        "present": bool(loaded.get("present")),
        "readable_json_object": bool(loaded.get("payload")),
        "required": spec.required,
        "sha256": sha256_file(path),
    }


def _required_source_errors(loaded: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    errors: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        if spec.required and not loaded[spec.experiment_id].get("payload"):
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _base_artifact(source_artifacts: list[JsonDict], duration_s: float) -> JsonDict:
    empty: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "flag_hygiene_ready": False,
        "rows_reviewed": 0,
        "true_blocker_rows": [],
        "aggregation_false_positive_rows": [],
        "missing_metadata_rows": [],
        "unresolved_bound_rows": [],
        "hardware_blocked_rows": [],
        "gate_skipped_rows": [],
        "source_artifacts": source_artifacts,
        "source_checksums": {str(row["path"]): row["sha256"] for row in source_artifacts},
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row.get("present") is not True
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
        "honest_verdict": "blocked_required_source_missing",
    }
    return empty


def _classification_lists(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    true_rows: list[JsonDict] = []
    aggregation_rows: list[JsonDict] = []
    missing_rows: list[JsonDict] = []
    unresolved_rows: list[JsonDict] = []
    hardware_rows: list[JsonDict] = []
    gate_rows: list[JsonDict] = []

    for exp_id, rel_path in (
        ("exp3027", EXP3027_REL_PATH),
        ("exp3028", EXP3028_REL_PATH),
        ("exp3029", EXP3029_REL_PATH),
        ("exp3038", MATRIX_V18_REL_PATH),
        ("exp3039", CAPSTONE_V284_REL_PATH),
    ):
        payload = payloads.get(exp_id, {})
        flags = _as_list(payload.get("corrigendum_pending"))
        if not flags and payload.get("flagged_adversarial") is not True:
            continue
        if _aggregation_flag_false_positive(payload, flags):
            aggregation_rows.append(
                _row(
                    row_id=f"{exp_id}:top_level_aggregation_flags",
                    classification="aggregation_false_positive",
                    source_artifact=rel_path.as_posix(),
                    source_field="corrigendum_pending",
                    evidence={"flagged_adversarial": payload.get("flagged_adversarial"), "flags": flags},
                    rationale=(
                        "Source artifact declares aggregation-only substrate; duration and "
                        "methodology flags came from treating it like a compute-bound live run."
                    ),
                    blocking=False,
                    experiment_id=exp_id,
                    flag_kinds=_flag_kinds(flags),
                )
            )
            continue
        methodology_flags = _flags_with_kind(flags, METHODOLOGY_FLAG_KIND)
        non_methodology_flags = [
            flag
            for flag in flags
            if str(_as_mapping(flag).get("kind") or "") != METHODOLOGY_FLAG_KIND
        ]
        if methodology_flags:
            missing_rows.append(
                _row(
                    row_id=f"{exp_id}:methodology_missing",
                    classification="missing_metadata",
                    source_artifact=rel_path.as_posix(),
                    source_field="corrigendum_pending[METHODOLOGY_MISSING]",
                    evidence=methodology_flags,
                    rationale="Methodology metadata is missing and remains a downstream blocker.",
                    blocking=True,
                    experiment_id=exp_id,
                    flag_kinds=[METHODOLOGY_FLAG_KIND],
                )
            )
        if non_methodology_flags:
            true_rows.append(
                _row(
                    row_id=f"{exp_id}:adversarial_flags",
                    classification="true_blocker",
                    source_artifact=rel_path.as_posix(),
                    source_field="corrigendum_pending",
                    evidence=non_methodology_flags,
                    rationale="Non-aggregation adversarial flags remain blockers until separately cleared.",
                    blocking=True,
                    experiment_id=exp_id,
                    flag_kinds=_flag_kinds(non_methodology_flags),
                )
            )

    true_rows.extend(
        _rows_from_exp3027_list(
            payloads.get("exp3027", {}),
            list_name="true_methodology_blockers",
            classification="true_blocker",
            default_rationale="Exp 3027 classified this source row as a true methodology blocker.",
            blocking=True,
        )
    )
    missing_rows.extend(
        _rows_from_exp3027_list(
            payloads.get("exp3027", {}),
            list_name="missing_metadata_rows",
            classification="missing_metadata",
            default_rationale="Exp 3027 classified this source row as missing provenance metadata.",
            blocking=True,
        )
    )
    unresolved_rows.extend(
        _rows_from_exp3027_list(
            payloads.get("exp3027", {}),
            list_name="unresolved_bound_rows",
            classification="unresolved_bound",
            default_rationale="Exp 3027 classified this source row as an unresolved bound.",
            blocking=True,
        )
    )
    hardware_rows.extend(
        _rows_from_exp3027_list(
            payloads.get("exp3027", {}),
            list_name="hardware_blocked_rows",
            classification="hardware_blocked",
            default_rationale="Exp 3027 classified this source row as hardware-blocked.",
            blocking=True,
        )
    )

    true_rows.extend(_retired_claim_rows(payloads.get("exp3029", {})))
    unresolved_rows.extend(_bounded_claim_rows(payloads.get("exp3029", {})))

    matrix_rows = _matrix_rows(payloads.get("exp3038", {}))
    true_rows.extend(_matrix_flagged_rows(matrix_rows, payloads))
    missing_rows.extend(_matrix_missing_rows(matrix_rows))
    hardware_rows.extend(_matrix_hardware_rows(matrix_rows))
    gate_rows.extend(_matrix_gate_rows(matrix_rows))

    capstone = payloads.get("exp3039", {})
    true_rows.extend(_capstone_true_blockers(capstone))
    unresolved_rows.extend(_capstone_unresolved_rows(capstone))
    hardware_rows.extend(_capstone_hardware_rows(capstone))
    gate_rows.extend(_capstone_gate_rows(capstone))

    return {
        "true_blocker_rows": _unique_rows(true_rows),
        "aggregation_false_positive_rows": _unique_rows(aggregation_rows),
        "missing_metadata_rows": _unique_rows(missing_rows),
        "unresolved_bound_rows": _unique_rows(unresolved_rows),
        "hardware_blocked_rows": _unique_rows(hardware_rows),
        "gate_skipped_rows": _unique_rows(gate_rows),
    }


def _rows_from_exp3027_list(
    payload: Mapping[str, Any],
    *,
    list_name: str,
    classification: str,
    default_rationale: str,
    blocking: bool,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, source_row in enumerate(_as_list(payload.get(list_name))):
        row = _as_mapping(source_row)
        if not row:
            continue
        rows.append(
            _row(
                row_id=str(row.get("row_id") or f"{list_name}:{index}"),
                classification=classification,
                source_artifact=EXP3027_REL_PATH.as_posix(),
                source_field=f"{list_name}[{index}]",
                evidence=row,
                rationale=str(row.get("rationale") or default_rationale),
                blocking=blocking,
                experiment_id=str(row.get("source_experiment_id") or ""),
                matrix_status=str(row.get("matrix_status") or ""),
                nested_source_artifact=str(row.get("source_artifact_path") or ""),
            )
        )
    return rows


def _retired_claim_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, claim in enumerate(_as_list(payload.get("retired_or_blocked_claims"))):
        claim_row = _as_mapping(claim)
        claim_id = str(claim_row.get("claim_id") or f"retired_or_blocked_claim_{index}")
        rows.append(
            _row(
                row_id=f"exp3029:{claim_id}",
                classification="true_blocker",
                source_artifact=EXP3029_REL_PATH.as_posix(),
                source_field=f"retired_or_blocked_claims[{claim_id}]",
                evidence=claim_row,
                rationale="Exp 3029 retired or blocked this claim boundary.",
                blocking=True,
                experiment_id="exp3029",
            )
        )
    return rows


def _bounded_claim_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, claim in enumerate(_as_list(payload.get("bounded_claims"))):
        claim_row = _as_mapping(claim)
        claim_id = str(claim_row.get("claim_id") or f"bounded_claim_{index}")
        rows.append(
            _row(
                row_id=f"exp3029:{claim_id}",
                classification="unresolved_bound",
                source_artifact=EXP3029_REL_PATH.as_posix(),
                source_field=f"bounded_claims[{claim_id}]",
                evidence=claim_row,
                rationale="Bounded repair evidence is not a clean headline-promotion row.",
                blocking=True,
                experiment_id="exp3029",
            )
        )
    return rows


def _matrix_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(row) for row in _as_list(matrix.get("matrix_rows")) if isinstance(row, Mapping)]


def _matrix_flagged_rows(
    rows: list[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    classified_by_source = {"exp3027", "exp3028", "exp3029"}
    out: list[JsonDict] = []
    for row in rows:
        exp_id = str(row.get("experiment_id") or "")
        if str(row.get("status") or "") != "flagged" or exp_id in classified_by_source:
            continue
        out.append(
            _row(
                row_id=f"{exp_id}:matrix_flagged",
                classification="true_blocker",
                source_artifact=MATRIX_V18_REL_PATH.as_posix(),
                source_field=f"matrix_rows[{exp_id}].upstream_flags",
                evidence=row,
                rationale="Matrix v18 marks this row flagged and no aggregation false-positive source clears it.",
                blocking=True,
                experiment_id=exp_id,
                matrix_status="flagged",
                flag_kinds=_flag_kinds_from_strings(_as_list(row.get("upstream_flags"))),
            )
        )
    return out


def _matrix_missing_rows(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for row in rows:
        exp_id = str(row.get("experiment_id") or "")
        if str(row.get("status") or "") != "missing":
            continue
        out.append(
            _row(
                row_id=f"{exp_id}:matrix_missing",
                classification="missing_metadata",
                source_artifact=MATRIX_V18_REL_PATH.as_posix(),
                source_field=f"matrix_rows[{exp_id}].status",
                evidence=row,
                rationale="Matrix v18 reports an absent source row; the gap remains blocking metadata.",
                blocking=True,
                experiment_id=exp_id,
                matrix_status="missing",
            )
        )
    return out


def _matrix_hardware_rows(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for row in rows:
        exp_id = str(row.get("experiment_id") or "")
        status = str(row.get("status") or "")
        task_class = str(row.get("task_class") or "")
        if status != "blocked" or not _looks_hardware_related(row, task_class):
            continue
        out.append(
            _row(
                row_id=f"{exp_id}:hardware_blocked",
                classification="hardware_blocked",
                source_artifact=MATRIX_V18_REL_PATH.as_posix(),
                source_field=f"matrix_rows[{exp_id}].status",
                evidence=row,
                rationale="Hardware row is blocked and must not become a performance claim.",
                blocking=True,
                experiment_id=exp_id,
                matrix_status=status,
            )
        )
    return out


def _matrix_gate_rows(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for row in rows:
        exp_id = str(row.get("experiment_id") or "")
        status = str(row.get("status") or "")
        if status not in GATE_SKIPPED_STATUSES:
            continue
        out.append(
            _row(
                row_id=f"{exp_id}:gate_skipped",
                classification="gate_skipped",
                source_artifact=MATRIX_V18_REL_PATH.as_posix(),
                source_field=f"matrix_rows[{exp_id}].status",
                evidence=row,
                rationale="Matrix v18 reports a structured downstream gate skip.",
                blocking=True,
                experiment_id=exp_id,
                matrix_status=status,
            )
        )
    return out


def _capstone_true_blockers(capstone: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for blocker in _capstone_blockers(capstone, "matrix_nonclean"):
        rows.append(
            _row(
                row_id="capstone:matrix_nonclean",
                classification="true_blocker",
                source_artifact=CAPSTONE_V284_REL_PATH.as_posix(),
                source_field="blockers_remaining[matrix_nonclean]",
                evidence=blocker,
                rationale="Capstone keeps non-clean matrix rows publication-blocking.",
                blocking=True,
                experiment_id="exp3039",
            )
        )
    return rows


def _capstone_unresolved_rows(capstone: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for blocker in _capstone_blockers(capstone, "repair"):
        rows.append(
            _row(
                row_id="capstone:repair_bounded",
                classification="unresolved_bound",
                source_artifact=CAPSTONE_V284_REL_PATH.as_posix(),
                source_field="blockers_remaining[repair]",
                evidence=blocker,
                rationale="Capstone repair status remains bounded rather than promotable.",
                blocking=True,
                experiment_id="exp3039",
            )
        )
    for check in _as_list(capstone.get("paper_ready_checks")):
        check_row = _as_mapping(check)
        if check_row.get("check") == "repair_promotable" and check_row.get("passed") is False:
            rows.append(
                _row(
                    row_id="capstone:repair_promotable_check_failed",
                    classification="unresolved_bound",
                    source_artifact=CAPSTONE_V284_REL_PATH.as_posix(),
                    source_field="paper_ready_checks[repair_promotable]",
                    evidence=check_row,
                    rationale="Capstone paper-readiness check explicitly rejects repair promotion.",
                    blocking=True,
                    experiment_id="exp3039",
                )
            )
    return rows


def _capstone_hardware_rows(capstone: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for blocker in _capstone_blockers(capstone, "gatemate"):
        rows.append(
            _row(
                row_id="capstone:gatemate_blocked",
                classification="hardware_blocked",
                source_artifact=CAPSTONE_V284_REL_PATH.as_posix(),
                source_field="blockers_remaining[gatemate]",
                evidence=blocker,
                rationale="Capstone keeps GateMate output blocked on missing output contract.",
                blocking=True,
                experiment_id="exp3039",
            )
        )
    return rows


def _capstone_gate_rows(capstone: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for area in ("ssqa", "exp3036"):
        for blocker in _capstone_blockers(capstone, area):
            rows.append(
                _row(
                    row_id=f"capstone:{area}_gate_skipped",
                    classification="gate_skipped",
                    source_artifact=CAPSTONE_V284_REL_PATH.as_posix(),
                    source_field=f"blockers_remaining[{area}]",
                    evidence=blocker,
                    rationale="Capstone keeps this downstream task bounded by a skipped gate.",
                    blocking=True,
                    experiment_id="exp3039",
                )
            )
    return rows


def _capstone_blockers(capstone: Mapping[str, Any], area: str) -> list[JsonDict]:
    return [
        dict(row)
        for row in _as_list(capstone.get("blockers_remaining"))
        if isinstance(row, Mapping) and str(row.get("area") or "") == area
    ]


def _row(
    *,
    row_id: str,
    classification: str,
    source_artifact: str,
    source_field: str,
    evidence: Any,
    rationale: str,
    blocking: bool,
    experiment_id: str = "",
    matrix_status: str = "",
    flag_kinds: list[str] | None = None,
    nested_source_artifact: str = "",
) -> JsonDict:
    row: JsonDict = {
        "row_id": row_id,
        "classification": classification,
        "blocking": blocking,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "evidence": evidence,
        "rationale": rationale,
    }
    if experiment_id:
        row["experiment_id"] = experiment_id
    if matrix_status:
        row["matrix_status"] = matrix_status
    if flag_kinds:
        row["flag_kinds"] = flag_kinds
    if nested_source_artifact:
        row["nested_source_artifact"] = nested_source_artifact
    return row


def _unique_rows(rows: list[JsonDict]) -> list[JsonDict]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[JsonDict] = []
    for row in rows:
        key = (
            str(row.get("row_id") or ""),
            str(row.get("classification") or ""),
            str(row.get("source_artifact") or ""),
            str(row.get("source_field") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _aggregation_flag_false_positive(payload: Mapping[str, Any], flags: list[Any]) -> bool:
    kinds = set(_flag_kinds(flags))
    return (
        bool(kinds)
        and kinds <= AGGREGATION_FLAG_KINDS
        and payload.get("flagged_adversarial") is True
        and _is_aggregation_only(payload)
    )


def _is_aggregation_only(payload: Mapping[str, Any]) -> bool:
    substrate = payload.get("inference_substrate")
    if isinstance(substrate, str):
        return substrate == "aggregation_from_upstream_artifacts"
    substrate_map = _as_mapping(substrate)
    return str(substrate_map.get("kind") or "") == "aggregation_from_upstream_artifacts"


def _looks_hardware_related(row: Mapping[str, Any], task_class: str) -> bool:
    if any(token in task_class for token in ("gatemate", "ssqa", "hardware", "flash", "rtl")):
        return True
    return any(
        row.get(field) is not None
        for field in (
            "gatemate_output_contract_ready",
            "host_visible_output_observed",
            "ssqa_gate_status",
        )
    )


def _flags_with_kind(flags: list[Any], kind: str) -> list[JsonDict]:
    return [
        dict(flag)
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("kind") or "") == kind
    ]


def _flag_kinds(flags: list[Any]) -> list[str]:
    kinds: list[str] = []
    for flag in flags:
        mapping = _as_mapping(flag)
        kind = str(mapping.get("kind") or "")
        if kind:
            kinds.append(kind)
    return kinds


def _flag_kinds_from_strings(flags: list[Any]) -> list[str]:
    kinds: list[str] = []
    for flag in flags:
        text = str(flag)
        if ":" in text:
            text = text.split(":", 1)[0]
        if "=" in text:
            text = text.split("=", 1)[0]
        if text:
            kinds.append(text)
    return kinds


def _classification_summary(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "true_blocker_rows": len(_as_list(artifact.get("true_blocker_rows"))),
        "aggregation_false_positive_rows": len(
            _as_list(artifact.get("aggregation_false_positive_rows"))
        ),
        "missing_metadata_rows": len(_as_list(artifact.get("missing_metadata_rows"))),
        "unresolved_bound_rows": len(_as_list(artifact.get("unresolved_bound_rows"))),
        "hardware_blocked_rows": len(_as_list(artifact.get("hardware_blocked_rows"))),
        "gate_skipped_rows": len(_as_list(artifact.get("gate_skipped_rows"))),
    }


def _mechanically_consumable(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("row_id"))
        and row.get("classification") in ALLOWED_CLASSIFICATIONS
        and isinstance(row.get("blocking"), bool)
        and bool(row.get("source_artifact"))
        and bool(row.get("source_field"))
    )


def _honest_verdict(ready: bool, rows_reviewed: int, artifact: Mapping[str, Any]) -> str:
    summary = _classification_summary(artifact)
    if not ready:
        return f"blocked_flag_hygiene_incomplete: rows_reviewed={rows_reviewed}"
    return (
        "complete: flag_hygiene_ready=true; "
        f"rows_reviewed={rows_reviewed}; "
        f"true_blockers={summary['true_blocker_rows']}; "
        f"aggregation_false_positives={summary['aggregation_false_positive_rows']}; "
        f"missing_metadata={summary['missing_metadata_rows']}; "
        f"unresolved_bounds={summary['unresolved_bound_rows']}; "
        f"hardware_blocked={summary['hardware_blocked_rows']}; "
        f"gate_skipped={summary['gate_skipped_rows']}"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "source": "checked_in_artifacts",
    }


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


__all__ = [
    "CAPSTONE_V284_REL_PATH",
    "EXP3027_REL_PATH",
    "EXP3028_REL_PATH",
    "EXP3029_REL_PATH",
    "MATRIX_V18_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "build_artifact",
    "classification_rows",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
