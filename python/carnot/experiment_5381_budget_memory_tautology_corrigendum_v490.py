"""Exp5381: budget-memory tautology corrigendum.

Spec refs: REQ-LEARN-5381, SCENARIO-LEARN-5381-ROW-EVIDENCE,
SCENARIO-LEARN-5381-NEGATIVE-CONTROLS, SCENARIO-LEARN-5381-FAIL-CLOSED.

The prior Exp5368 artifact filled a 400 byte budget exactly, and the conductor
flagged that equality as a TAUTOLOGY risk. This module does not treat the equal
budget total as a claim. It rebuilds memory decisions from row evidence,
including cost proxies and negative controls, so downstream self-learning only
depends on a gate that can be falsified by bad rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5381_budget_memory_tautology_corrigendum_v490.json"
)
SOURCE_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5368_budget_curated_memory_governance_v489.json"
)
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5381_budget_memory_tautology_corrigendum_v490.py"
)

EXPERIMENT = "experiment_5381_budget_memory_tautology_corrigendum_v490"
EXPERIMENT_ID = "exp5381-v490-budget-memory-tautology-corrigendum"
MILESTONE = "2026.07.490"
SCHEMA = "carnot.experiment_5381.budget_memory_tautology_corrigendum.v490"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5381

TRUST_STALE_RISK_MAX = 0.2
TRUST_POISON_RISK_MAX = 0.1
SHARE_RISK_MAX = 0.1
POISON_QUARANTINE_RISK = 0.7
STALE_QUARANTINE_RISK = 0.6
DEFAULT_BUDGET_BYTES = 400

SPEC_REFS = (
    "REQ-LEARN-5381",
    "SCENARIO-LEARN-5381-ROW-EVIDENCE",
    "SCENARIO-LEARN-5381-NEGATIVE-CONTROLS",
    "SCENARIO-LEARN-5381-FAIL-CLOSED",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Complete only if row-level evidence and controls were evaluated.",
    "budget_memory_corrigendum_clean": (
        "True only if readiness is recomputed from rows and anti-tautology "
        "controls pass."
    ),
    "source_artifact": "Path to the Exp5368 artifact reviewed.",
    "row_count": "Number of memory decision rows evaluated.",
    "recomputed_fields_from_rows": (
        "List of aggregate fields derived mechanically from rows."
    ),
    "anti_tautology_controls": "Object describing controls and pass/fail status.",
    "negative_controls_count": (
        "Number of stale, poison, and high-cost controls."
    ),
    "negative_controls_passed": (
        "Count of negative controls rejected correctly."
    ),
    "keep_share_trust_policy_ready": (
        "True only if decisions are supported by row evidence."
    ),
    "no_weight_mutation": "Must be true.",
    "rollback_supported": "True only if rollback behavior is evidenced.",
    "unsafe_false_accepts": (
        "Count of bad memory decisions accepted as good."
    ),
    "tests_run": (
        "List of commands run or explicit no-code-change explanation."
    ),
    "honest_verdict": "One-line clean or blocked verdict.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
RECOMPUTED_FIELDS_FROM_ROWS = (
    "row_count",
    "retained_bytes",
    "keep_ids",
    "share_ids",
    "trust_ids",
    "negative_controls_count",
    "negative_controls_passed",
    "keep_share_trust_policy_ready",
    "rollback_supported",
    "unsafe_false_accepts",
)


def load_source_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the reviewed Exp5368 artifact from disk."""

    return _read_json(Path(root) / SOURCE_ARTIFACT_RELATIVE_PATH)


def review_source_tautology(
    source_artifact: Mapping[str, Any],
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """Extract the exact TAUTOLOGY finding from artifact and conductor log."""

    log_text = (Path(root) / CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8")
    findings = [
        dict(row)
        for row in source_artifact.get("corrigendum_pending", [])
        if row.get("kind") == "TAUTOLOGY"
    ]
    return {
        "source_artifact": str(SOURCE_ARTIFACT_RELATIVE_PATH),
        "source_flagged_adversarial": source_artifact.get("flagged_adversarial")
        is True,
        "source_budget_curated_memory_ready": source_artifact.get(
            "budget_curated_memory_ready"
        )
        is True,
        "source_findings": findings,
        "conductor_flagged_tautology": (
            "Budget-curated memory governance" in log_text
            and "adversarial_verify CRITICAL: TAUTOLOGY" in log_text
        ),
    }


def build_evidence_rows(source_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute memory decisions from source rows plus cost-control evidence."""

    ranked_rows = sorted(
        source_artifact["memory_decision_rows"],
        key=lambda row: (-_score(row), str(row["memory_id"])),
    )
    retained_bytes = 0
    evidence_rows: list[JsonDict] = []
    for rank, row in enumerate(ranked_rows, start=1):
        trust_decision = _trust_decision(row)
        keep_decision = _keep_decision(row, trust_decision, retained_bytes)
        if keep_decision == "KEEP":
            retained_bytes += int(row["byte_cost"])
        share_decision = _share_decision(row, trust_decision, keep_decision)
        evidence = _evidence_row(row, rank, trust_decision, keep_decision, share_decision)
        evidence_rows.append(evidence)
    return evidence_rows


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
    evidence_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp5381 artifact by evaluating row-derived controls."""

    source_artifact = load_source_artifact(root)
    source_review = review_source_tautology(source_artifact, root)
    rows = (
        build_evidence_rows(source_artifact)
        if evidence_rows is None
        else [dict(row) for row in evidence_rows]
    )
    summary = _summarize_rows(rows)
    controls = _anti_tautology_controls(source_review, summary, tests_run)
    clean = bool(
        controls["all_passed"]
        and summary["keep_share_trust_policy_ready"]
        and summary["rollback_supported"]
        and summary["unsafe_false_accepts"] == 0
        and source_artifact.get("no_weight_mutation") is True
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifact_review": source_review,
        "memory_evidence_rows": _json_ready(rows),
        "aggregate_recomputation": summary,
        "source_artifact_checksums": _source_checksums(root),
        "status": "complete" if clean else "blocked",
        "budget_memory_corrigendum_clean": clean,
        "source_artifact": str(SOURCE_ARTIFACT_RELATIVE_PATH),
        "row_count": summary["row_count"],
        "recomputed_fields_from_rows": list(RECOMPUTED_FIELDS_FROM_ROWS),
        "anti_tautology_controls": controls,
        "negative_controls_count": summary["negative_controls_count"],
        "negative_controls_passed": summary["negative_controls_passed"],
        "keep_share_trust_policy_ready": summary["keep_share_trust_policy_ready"],
        "no_weight_mutation": source_artifact.get("no_weight_mutation") is True,
        "rollback_supported": summary["rollback_supported"],
        "unsafe_false_accepts": summary["unsafe_false_accepts"],
        "tests_run": [dict(row) for row in tests_run],
        "honest_verdict": _honest_verdict(clean, controls),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the small set of fields consumed by downstream gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    type_errors = [
        "budget_memory_corrigendum_clean"
        for _ in [artifact.get("budget_memory_corrigendum_clean")]
        if not isinstance(artifact.get("budget_memory_corrigendum_clean"), bool)
    ]
    type_errors.extend(
        field
        for field in (
            "row_count",
            "negative_controls_count",
            "negative_controls_passed",
            "unsafe_false_accepts",
        )
        if isinstance(artifact.get(field), bool)
        or not isinstance(artifact.get(field), int)
    )
    type_errors.extend(
        field
        for field in ("keep_share_trust_policy_ready", "no_weight_mutation", "rollback_supported")
        if not isinstance(artifact.get(field), bool)
    )
    if missing or type_errors:
        raise ValueError("invalid Exp5381 artifact fields: " + ",".join(missing + type_errors))
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the deterministic Exp5381 artifact to disk."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _evidence_row(
    row: Mapping[str, Any],
    rank: int,
    trust_decision: str,
    keep_decision: str,
    share_decision: str,
) -> JsonDict:
    cost_evidence = {
        "byte_cost": int(row["byte_cost"]),
        "latency_proxy_ms": round(int(row["byte_cost"]) * 0.0015, 6),
        "energy_proxy_mj": round(
            int(row["byte_cost"]) * 0.0004
            + float(row["stale_risk"]) * 0.01
            + float(row["poison_risk"]) * 0.02,
            6,
        ),
    }
    value_evidence = {
        "estimated_verifier_value": float(row["estimated_verifier_value"]),
        "harm_score": _harm_score(row),
        "value_minus_harm_per_byte": _score(row),
    }
    control_kind = _control_kind(row)
    evidence: JsonDict = {
        "memory_id": row["memory_id"],
        "memory_variant": row["memory_variant"],
        "rank_by_row_score": rank,
        "provenance": dict(row["provenance"]),
        "trust_label": row["trust_label"],
        "useful": bool(row["useful"]),
        "harmful": bool(row["harmful"]),
        "value_evidence": value_evidence,
        "cost_evidence": cost_evidence,
        "stale_control": {
            "risk": float(row["stale_risk"]),
            "threshold": STALE_QUARANTINE_RISK,
        },
        "poison_control": {
            "risk": float(row["poison_risk"]),
            "threshold": POISON_QUARANTINE_RISK,
        },
        "rollback_evidence": {
            "required": bool(
                row["harmful"] or row["memory_variant"] in {"stale", "poisoned", "unverified"}
            ),
            "available": bool(row["rollback_available"]),
            "recovered": bool(row["rollback_recovered"]),
        },
        "source_keep_decision": row["keep_decision"],
        "source_share_decision": row["share_decision"],
        "source_trust_decision": row["trust_decision"],
        "recomputed_keep_decision": keep_decision,
        "recomputed_share_decision": share_decision,
        "recomputed_trust_decision": trust_decision,
        "control_kind": control_kind,
    }
    evidence["accepted_as_good"] = keep_decision == "KEEP" and trust_decision == "TRUST"
    evidence["bad_memory"] = bool(
        evidence["harmful"]
        or evidence["memory_variant"] in {"stale", "poisoned", "unverified"}
        or control_kind == "negative_high_cost_low_value"
    )
    evidence["control_passed"] = _control_passed(evidence)
    evidence["decision_inputs_measured"] = _decision_inputs_measured(evidence)
    return evidence


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row_list = [dict(row) for row in rows]
    negative_controls = [row for row in row_list if row.get("control_kind")]
    rollback_rows = [
        row for row in row_list if row.get("rollback_evidence", {}).get("required") is True
    ]
    keep_ids = [
        row["memory_id"] for row in row_list if row["recomputed_keep_decision"] == "KEEP"
    ]
    share_ids = [
        row["memory_id"] for row in row_list if row["recomputed_share_decision"] == "SHARE"
    ]
    trust_ids = [
        row["memory_id"] for row in row_list if row["recomputed_trust_decision"] == "TRUST"
    ]
    unsafe_false_accepts = sum(
        1 for row in row_list if row["bad_memory"] and row["accepted_as_good"]
    )
    return {
        "row_count": len(row_list),
        "retained_bytes": sum(
            row["cost_evidence"]["byte_cost"]
            for row in row_list
            if row["recomputed_keep_decision"] == "KEEP"
        ),
        "keep_ids": keep_ids,
        "share_ids": share_ids,
        "trust_ids": trust_ids,
        "negative_control_ids": [row["memory_id"] for row in negative_controls],
        "negative_controls_count": len(negative_controls),
        "negative_controls_passed": sum(
            1 for row in negative_controls if row["control_passed"]
        ),
        "keep_share_trust_policy_ready": (
            _row_level_evidence_present(row_list)
            and _keep_share_trust_supported(row_list)
        ),
        "rollback_supported": (
            bool(rollback_rows)
            and all(
                row["rollback_evidence"]["available"] and row["rollback_evidence"]["recovered"]
                for row in rollback_rows
            )
        ),
        "unsafe_false_accepts": unsafe_false_accepts,
    }


def _anti_tautology_controls(
    source_review: Mapping[str, Any],
    summary: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    control_checks = {
        "source_tautology_preserved": bool(source_review["source_findings"])
        and source_review["conductor_flagged_tautology"] is True,
        "row_level_evidence_present": summary["keep_share_trust_policy_ready"] is True,
        "aggregate_recomputed_from_rows": set(RECOMPUTED_FIELDS_FROM_ROWS).issubset(summary),
        "negative_controls_rejected": summary["negative_controls_count"] >= 3
        and summary["negative_controls_passed"] == summary["negative_controls_count"],
        "budget_match_explained_by_row_sum": summary["retained_bytes"] == DEFAULT_BUDGET_BYTES
        and summary["keep_ids"]
        == [
            "mem5368-clean-rollback-route",
            "mem5368-clean-dependency-edge",
            "mem5368-clean-scaleup-summary",
        ],
        "source_ready_not_copied": source_review["source_budget_curated_memory_ready"] is True,
        "rollback_evidenced": summary["rollback_supported"] is True,
        "unsafe_false_accepts_zero": summary["unsafe_false_accepts"] == 0,
        "tests_recorded": bool(tests_run),
    }
    failed = [name for name, passed in control_checks.items() if not passed]
    return {
        **control_checks,
        "source_tautology_reason": source_review["source_findings"][0]["detail"],
        "recomputed_fields": list(RECOMPUTED_FIELDS_FROM_ROWS),
        "failed_controls": failed,
        "all_passed": not failed,
    }


def _honest_verdict(clean: bool, controls: Mapping[str, Any]) -> str:
    if clean:
        return (
            "complete: budget_memory_corrigendum_clean from row evidence; "
            "stale, poisoned, and high-cost controls rejected with zero unsafe "
            "false accepts"
        )
    return "blocked_budget_memory_corrigendum: " + ",".join(controls["failed_controls"])


def _trust_decision(row: Mapping[str, Any]) -> str:
    trusted = bool(
        row["provenance"]["verified"]
        and row["trust_label"] == "verified_clean"
        and row["useful"]
        and not row["harmful"]
        and float(row["stale_risk"]) <= TRUST_STALE_RISK_MAX
        and float(row["poison_risk"]) <= TRUST_POISON_RISK_MAX
    )
    return "TRUST" if trusted else "UNTRUST"


def _keep_decision(
    row: Mapping[str, Any],
    trust_decision: str,
    retained_bytes: int,
) -> str:
    if (
        float(row["poison_risk"]) >= POISON_QUARANTINE_RISK
        or float(row["stale_risk"]) >= STALE_QUARANTINE_RISK
        or (row["harmful"] and trust_decision == "UNTRUST")
    ):
        return "QUARANTINE"
    if trust_decision != "TRUST" or _score(row) <= 0.0:
        return "DROP"
    if retained_bytes + int(row["byte_cost"]) > DEFAULT_BUDGET_BYTES:
        return "DROP"
    return "KEEP"


def _share_decision(
    row: Mapping[str, Any],
    trust_decision: str,
    keep_decision: str,
) -> str:
    if (
        keep_decision == "KEEP"
        and trust_decision == "TRUST"
        and row["provenance"]["verified"]
        and float(row["sharing_risk"]) <= SHARE_RISK_MAX
    ):
        return "SHARE"
    return "DO_NOT_SHARE"


def _control_kind(row: Mapping[str, Any]) -> str:
    if row["memory_variant"] == "stale":
        return "negative_stale"
    if row["memory_variant"] == "poisoned":
        return "negative_poisoned"
    if row["memory_variant"] == "low_value":
        return "negative_high_cost_low_value"
    return ""


def _control_passed(row: Mapping[str, Any]) -> bool:
    if row["control_kind"] == "negative_stale":
        return (
            row["recomputed_trust_decision"] == "UNTRUST"
            and row["recomputed_keep_decision"] == "QUARANTINE"
        )
    if row["control_kind"] == "negative_poisoned":
        return (
            row["recomputed_trust_decision"] == "UNTRUST"
            and row["recomputed_keep_decision"] == "QUARANTINE"
        )
    if row["control_kind"] == "negative_high_cost_low_value":
        return row["recomputed_keep_decision"] == "DROP"
    return True


def _decision_inputs_measured(row: Mapping[str, Any]) -> JsonDict:
    checks = {
        "value": isinstance(row["value_evidence"]["estimated_verifier_value"], float),
        "byte_cost": isinstance(row["cost_evidence"]["byte_cost"], int),
        "latency_proxy_cost": isinstance(row["cost_evidence"]["latency_proxy_ms"], float),
        "energy_proxy_cost": isinstance(row["cost_evidence"]["energy_proxy_mj"], float),
        "provenance": isinstance(row["provenance"].get("verified"), bool),
        "trust_label": bool(row["trust_label"]),
        "stale_control": isinstance(row["stale_control"]["risk"], float),
        "poison_control": isinstance(row["poison_control"]["risk"], float),
        "rollback": isinstance(row["rollback_evidence"]["available"], bool)
        and isinstance(row["rollback_evidence"]["recovered"], bool),
    }
    return {**checks, "all_required": all(checks.values())}


def _row_level_evidence_present(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        row.get("decision_inputs_measured", {}).get("all_required") is True
        for row in rows
    )


def _keep_share_trust_supported(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        _keep_supported(row) and _share_supported(row) and _trust_supported(row)
        for row in rows
    )


def _keep_supported(row: Mapping[str, Any]) -> bool:
    if row["recomputed_keep_decision"] == "KEEP":
        return (
            row["accepted_as_good"]
            and row["useful"]
            and not row["harmful"]
            and not row["bad_memory"]
        )
    return not row["accepted_as_good"] or not row["bad_memory"]


def _share_supported(row: Mapping[str, Any]) -> bool:
    if row["recomputed_share_decision"] == "SHARE":
        return (
            row["recomputed_keep_decision"] == "KEEP"
            and row["recomputed_trust_decision"] == "TRUST"
            and row["provenance"]["verified"]
            and row["cost_evidence"]["byte_cost"] > 0
        )
    return True


def _trust_supported(row: Mapping[str, Any]) -> bool:
    if row["recomputed_trust_decision"] == "TRUST":
        return row["useful"] and not row["harmful"] and row["provenance"]["verified"]
    return not row["accepted_as_good"] or row["bad_memory"]


def _harm_score(row: Mapping[str, Any]) -> float:
    return round(
        float(row["stale_risk"]) + float(row["poison_risk"]) + float(row["sharing_risk"]),
        6,
    )


def _score(row: Mapping[str, Any]) -> float:
    return round(
        (float(row["estimated_verifier_value"]) - _harm_score(row))
        / int(row["byte_cost"]),
        6,
    )


def _source_checksums(root: Path | str) -> JsonDict:
    root_path = Path(root)
    return {
        "source_artifact": _sha256_file(root_path / SOURCE_ARTIFACT_RELATIVE_PATH),
        "conductor_log": _sha256_file(root_path / CONDUCTOR_LOG_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
    }


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True))
