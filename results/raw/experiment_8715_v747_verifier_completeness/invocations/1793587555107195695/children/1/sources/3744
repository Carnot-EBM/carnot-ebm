"""Residual-drift context induction for Exp 1609.

Spec: REQ-VERIFY-1609, SCENARIO-VERIFY-1609.

The residual-drift ledger tells us where a prior answer forgot a still
satisfiable commitment.  The repair policy tells us whether the local repair
replayed cleanly or remained unresolved.  This module mines those failure
records and turns repeated local contexts into candidate constraints.  The
result is intentionally not a global text rule: each candidate is anchored to a
source domain, a localized commitment span, a repair kind, and the deterministic
validator that must remain the final authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

JsonDict = dict[str, Any]

RUN_DATE = "20260509"
EXPERIMENT_ID = "1609"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1609_context_induction.json")
DEFAULT_LEDGER_PATH = Path("results/residual_drift_commitment_ledger_1538.jsonl")
DEFAULT_REPAIR_MANIFEST_PATH = Path("results/residual_drift_repair_policy_1552.jsonl")
DEFAULT_CONDUCTOR_LOG_PATH = Path("logs/conductor.log")
MODULE_PATH = "python/carnot/pipeline/context_induction.py"

CLASS_SATISFIABLE_DRIFT = "satisfiable_drift"
CLASS_TRUE_CONTRADICTION = "true_contradiction"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "context_induction_ready",
    "failure_logs_mined",
    "source_paths",
    "candidate_constraints_generated",
    "selected_candidate",
    "constraint_candidates",
    "true_contradiction_exclusions",
    "blockers",
    "focused_tests_passed",
    "honest_verdict",
)


@dataclass(frozen=True)
class FailureEvidence:
    """One mined failure or exclusion row used by the induction loop."""

    case_id: str
    source_path: str
    source_kind: str
    source_domain: str
    failure_classification: str
    localized_span: str
    repair_kind: str
    validator: str
    contract_family: str | None
    signal: str
    is_positive: bool
    is_exclusion: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Return the JSON artifact form of this evidence record."""

        return {
            "case_id": self.case_id,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_domain": self.source_domain,
            "failure_classification": self.failure_classification,
            "localized_span": self.localized_span,
            "repair_kind": self.repair_kind,
            "validator": self.validator,
            "contract_family": self.contract_family,
            "signal": self.signal,
            "is_positive": self.is_positive,
            "is_exclusion": self.is_exclusion,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MiningResult:
    """Collected evidence plus source accounting from the mining pass."""

    evidence: list[FailureEvidence]
    blockers: list[str]
    source_paths: list[str]
    source_counts: JsonDict


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write a durable bootstrap artifact before source-log loading."""

    mining = MiningResult(
        evidence=[],
        blockers=["experiment_1609_context_induction_in_progress"],
        source_paths=[],
        source_counts=_empty_source_counts(),
    )
    payload = _terminal_artifact(
        status="in_progress",
        run_date=run_date,
        mining=mining,
        candidates=[],
        focused_tests_passed=False,
        blockers=mining.blockers,
    )
    _write_json(Path(output_path), payload)
    return payload


def mine_failure_logs(
    *,
    project_root: Path | str | None = None,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    repair_manifest_path: Path | str = DEFAULT_REPAIR_MANIFEST_PATH,
    conductor_log_path: Path | str = DEFAULT_CONDUCTOR_LOG_PATH,
    recent_log_lines: int = 240,
) -> MiningResult:
    """Mine recent residual-drift failures from ledger, repair, and conductor logs."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    ledger = _resolve_under_root(root, Path(ledger_path))
    repairs = _resolve_under_root(root, Path(repair_manifest_path))
    conductor = _resolve_under_root(root, Path(conductor_log_path))
    evidence: list[FailureEvidence] = []
    blockers: list[str] = []
    source_paths: list[str] = []
    counts = _empty_source_counts()

    if ledger.exists():
        source_paths.append(_display_path(ledger, root))
        ledger_rows = _read_jsonl(ledger)
        counts["ledger_rows"] = len(ledger_rows)
        evidence.extend(
            item
            for row in ledger_rows
            if (item := _evidence_from_ledger_row(row, ledger, root)) is not None
        )
    else:
        blockers.append(f"missing_residual_drift_ledger:{_display_path(ledger, root)}")

    if repairs.exists():
        source_paths.append(_display_path(repairs, root))
        repair_rows = _read_jsonl(repairs)
        counts["repair_rows"] = len(repair_rows)
        evidence.extend(
            item
            for row in repair_rows
            if (item := _evidence_from_repair_row(row, repairs, root)) is not None
        )
    else:
        blockers.append(f"missing_residual_drift_repair_manifest:{_display_path(repairs, root)}")

    if conductor.exists():
        source_paths.append(_display_path(conductor, root))
        log_evidence = _evidence_from_conductor_log(conductor, root, recent_log_lines)
        counts["conductor_failure_lines"] = len(log_evidence)
        evidence.extend(log_evidence)
    else:
        blockers.append(f"missing_conductor_log:{_display_path(conductor, root)}")

    counts["positive_evidence_rows"] = sum(1 for item in evidence if item.is_positive)
    counts["true_contradiction_exclusions"] = sum(1 for item in evidence if item.is_exclusion)
    return MiningResult(
        evidence=evidence,
        blockers=blockers,
        source_paths=source_paths,
        source_counts=counts,
    )


def generate_constraint_candidates(
    evidence: Sequence[FailureEvidence],
    *,
    min_support: int = 1,
) -> list[JsonDict]:
    """Induce context-sensitive constraint candidates from mined drift evidence."""

    exclusions = sorted({item.case_id for item in evidence if item.is_exclusion})
    grouped: dict[tuple[str, str, str, str, str | None], list[FailureEvidence]] = {}
    for item in evidence:
        if not item.is_positive or item.failure_classification == CLASS_TRUE_CONTRADICTION:
            continue
        key = (
            item.source_domain,
            item.localized_span,
            item.repair_kind,
            item.validator,
            item.contract_family,
        )
        grouped.setdefault(key, []).append(item)

    candidates = [
        _candidate_from_group(key, rows, exclusions)
        for key, rows in grouped.items()
        if len(rows) >= min_support
    ]
    candidates.sort(
        key=lambda item: (
            -int(item["support_count"]),
            -float(item["confidence"]),
            str(item["constraint_id"]),
        )
    )
    return candidates


def run_experiment(
    *,
    project_root: Path | str | None = None,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    repair_manifest_path: Path | str = DEFAULT_REPAIR_MANIFEST_PATH,
    conductor_log_path: Path | str = DEFAULT_CONDUCTOR_LOG_PATH,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    focused_tests_passed: bool = False,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run the Exp 1609 induction loop and persist the terminal JSON artifact."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    write_in_progress_artifact(output, run_date=run_date)
    mining = mine_failure_logs(
        project_root=root,
        ledger_path=ledger_path,
        repair_manifest_path=repair_manifest_path,
        conductor_log_path=conductor_log_path,
    )
    candidates = generate_constraint_candidates(mining.evidence)
    blockers = list(mining.blockers)
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")
    if not any(item.is_positive for item in mining.evidence):
        blockers.append("no_qualifying_residual_drift_failures")
    if not candidates:
        blockers.append("no_context_constraint_candidates")

    status = "complete" if focused_tests_passed and candidates else "blocked"
    artifact = _terminal_artifact(
        status=status,
        run_date=run_date,
        mining=mining,
        candidates=candidates,
        focused_tests_passed=focused_tests_passed,
        blockers=list(dict.fromkeys(blockers)),
    )
    _write_json(output, artifact)
    return artifact


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    mining: MiningResult,
    candidates: Sequence[Mapping[str, Any]],
    focused_tests_passed: bool,
    blockers: Sequence[str],
) -> JsonDict:
    positive_count = sum(1 for item in mining.evidence if item.is_positive)
    contradictions = sorted({item.case_id for item in mining.evidence if item.is_exclusion})
    positives_are_clean = all(
        item.failure_classification != CLASS_TRUE_CONTRADICTION
        for item in mining.evidence
        if item.is_positive
    )
    ready = (
        status == "complete"
        and positive_count > 0
        and bool(candidates)
        and positives_are_clean
        and focused_tests_passed
    )
    return {
        "status": status,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "schema_version": 1,
        "context_induction_ready": bool(ready),
        "failure_logs_mined": len(mining.evidence),
        "source_paths": list(mining.source_paths),
        "source_counts": dict(mining.source_counts),
        "candidate_constraints_generated": len(candidates),
        "selected_candidate": dict(candidates[0]) if candidates else None,
        "constraint_candidates": [dict(item) for item in candidates],
        "true_contradiction_exclusions": contradictions,
        "blockers": list(dict.fromkeys(blockers)),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": (
            "complete: context_induction_ready"
            if ready
            else "blocked: context_induction_not_ready"
        ),
        "module_path": MODULE_PATH,
        "mined_evidence": [item.to_dict() for item in mining.evidence],
        "claim_scope": "bounded induction over checked-in residual-drift failure logs only",
    }


def _candidate_from_group(
    key: tuple[str, str, str, str, str | None],
    rows: Sequence[FailureEvidence],
    exclusions: Sequence[str],
) -> JsonDict:
    source_domain, localized_span, repair_kind, validator, contract_family = key
    trigger_context = {
        "source_domain": source_domain,
        "localized_span": localized_span,
        "repair_kind": repair_kind,
        "validator": validator,
    }
    if contract_family:
        trigger_context["contract_family"] = contract_family
    material = json.dumps(
        {
            "trigger_context": trigger_context,
            "positive_case_ids": sorted(item.case_id for item in rows),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    support_count = len(rows)
    confidence = round(support_count / (support_count + (0.5 * len(exclusions))), 6)
    if not exclusions:
        confidence = 1.0
    return {
        "constraint_id": f"ctx1609_{_slug(source_domain)}_{digest}",
        "constraint_type": "context_sensitive_residual_drift_guard",
        "trigger_context": trigger_context,
        "predicate": (
            f"When {source_domain} reaches {localized_span}, require the final "
            f"decision to replay against {validator} before acceptance."
        ),
        "positive_evidence_case_ids": sorted(item.case_id for item in rows),
        "negative_evidence_case_ids": list(exclusions),
        "support_count": support_count,
        "exclusion_count": len(exclusions),
        "confidence": confidence,
        "guardrails": [
            "exclude failure_classification=true_contradiction from positive evidence",
            "keep deterministic validator replay as final acceptance authority",
            "apply only when every trigger_context field matches",
        ],
        "evidence_sources": sorted({item.source_path for item in rows}),
        "induced_from_signals": sorted({item.signal for item in rows}),
    }


def _evidence_from_ledger_row(
    row: Mapping[str, Any],
    source_path: Path,
    root: Path,
) -> FailureEvidence | None:
    if row.get("row_type") != "residual_drift_case":
        return None
    failure_classification = str(row.get("failure_classification") or "")
    if failure_classification not in {CLASS_SATISFIABLE_DRIFT, CLASS_TRUE_CONTRADICTION}:
        return None
    source_domain = str(row.get("source_domain") or "unknown")
    is_exclusion = failure_classification == CLASS_TRUE_CONTRADICTION
    return FailureEvidence(
        case_id=str(row.get("source_case_id") or row.get("case_id") or ""),
        source_path=_display_path(source_path, root),
        source_kind="ledger",
        source_domain=source_domain,
        failure_classification=failure_classification,
        localized_span=_infer_localized_span(row),
        repair_kind=_infer_repair_kind(source_domain),
        validator=_infer_validator(source_domain),
        contract_family=_infer_contract_family(row),
        signal=(
            "true_contradiction_exclusion"
            if is_exclusion
            else "ledger_satisfiable_drift"
        ),
        is_positive=not is_exclusion,
        is_exclusion=is_exclusion,
        details={"live_sota_model_inference_used": bool(row.get("live_sota_model_inference_used"))},
    )


def _evidence_from_repair_row(
    row: Mapping[str, Any],
    source_path: Path,
    root: Path,
) -> FailureEvidence | None:
    if row.get("row_type") != "residual_drift_repair_case":
        return None
    failure_classification = str(row.get("failure_classification") or "")
    if failure_classification == CLASS_TRUE_CONTRADICTION:
        return FailureEvidence(
            case_id=str(row.get("case_id") or ""),
            source_path=_display_path(source_path, root),
            source_kind="repair_manifest",
            source_domain=str(row.get("source_domain") or "unknown"),
            failure_classification=failure_classification,
            localized_span="true_contradiction",
            repair_kind="no_repair",
            validator=str(_mapping(row.get("replay")).get("validator") or "deterministic_replay"),
            contract_family=None,
            signal="repair_true_contradiction_exclusion",
            is_positive=False,
            is_exclusion=True,
            details={"rejection_reason": row.get("rejection_reason")},
        )
    if failure_classification != CLASS_SATISFIABLE_DRIFT or row.get("accepted") is True:
        return None
    localization = _mapping(row.get("localization"))
    replay = _mapping(row.get("replay"))
    source_domain = str(row.get("source_domain") or "unknown")
    return FailureEvidence(
        case_id=str(row.get("case_id") or ""),
        source_path=_display_path(source_path, root),
        source_kind="repair_manifest",
        source_domain=source_domain,
        failure_classification=failure_classification,
        localized_span=str(localization.get("localized_span") or _infer_span_from_domain(source_domain)),
        repair_kind=str(localization.get("repair_kind") or _infer_repair_kind(source_domain)),
        validator=str(replay.get("validator") or _infer_validator(source_domain)),
        contract_family=(
            str(localization.get("contract_family"))
            if localization.get("contract_family")
            else None
        ),
        signal="repair_residual_drift_unrepaired",
        is_positive=True,
        details={"rejection_reason": row.get("rejection_reason") or replay.get("reason")},
    )


def _evidence_from_conductor_log(
    path: Path,
    root: Path,
    recent_log_lines: int,
) -> list[FailureEvidence]:
    lines = path.read_text(encoding="utf-8").splitlines()[-recent_log_lines:]
    evidence: list[FailureEvidence] = []
    for index, line in enumerate(lines, start=1):
        lower = line.lower()
        if "residual_drift" not in lower and "residual drift" not in lower:
            continue
        fields = dict(re.findall(r"([A-Za-z_]+)=([^ ]+)", line))
        source_domain = fields.get("source_domain", "unknown_log")
        evidence.append(
            FailureEvidence(
                case_id=fields.get("case_id", f"log-line-{index}"),
                source_path=_display_path(path, root),
                source_kind="conductor_log",
                source_domain=source_domain,
                failure_classification=CLASS_SATISFIABLE_DRIFT,
                localized_span=fields.get("localized_span", _infer_span_from_domain(source_domain)),
                repair_kind=fields.get("repair_kind", _infer_repair_kind(source_domain)),
                validator=fields.get("validator", _infer_validator(source_domain)),
                contract_family=fields.get("contract_family"),
                signal="conductor_residual_drift_failure",
                is_positive=True,
                details={"log_excerpt": line[:240]},
            )
        )
    return evidence


def _infer_localized_span(row: Mapping[str, Any]) -> str:
    source_domain = str(row.get("source_domain") or "")
    if source_domain == "satquest":
        classification = str(_mapping(row.get("deterministic_validator")).get("classification") or "")
        if classification == "invalid_assignment":
            return "commitments[1].evidence.assignment"
        return "commitments[1].evidence.answer"
    return _infer_span_from_domain(source_domain)


def _infer_span_from_domain(source_domain: str) -> str:
    if source_domain == "product_line":
        return "commitments[1].evidence.selected_features"
    if source_domain == "runtime_contract":
        return "commitments[1].evidence.root_cause_category"
    return "commitments[1].evidence.answer"


def _infer_repair_kind(source_domain: str) -> str:
    if source_domain == "product_line":
        return "product_line_feature_selection_patch"
    if source_domain == "runtime_contract":
        return "runtime_contract_root_cause_patch"
    return "sat_answer_or_assignment_patch"


def _infer_validator(source_domain: str) -> str:
    if source_domain == "product_line":
        return "product_line_oracle"
    if source_domain == "runtime_contract":
        return "runtime_contract"
    return "sat_oracle"


def _infer_contract_family(row: Mapping[str, Any]) -> str | None:
    validation = _commitments_by_name(row).get("deterministic_contract_validation", {})
    structural = _mapping(validation.get("structural_contract_result"))
    family = structural.get("contract_family")
    return str(family) if family else None


def _commitments_by_name(row: Mapping[str, Any]) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    commitments = row.get("commitments")
    if not isinstance(commitments, Sequence) or isinstance(commitments, (str, bytes)):
        return result
    for item in commitments:
        if isinstance(item, Mapping):
            name = item.get("name")
            if isinstance(name, str):
                result[name] = _mapping(item.get("evidence"))
    return result


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _empty_source_counts() -> JsonDict:
    return {
        "ledger_rows": 0,
        "repair_rows": 0,
        "conductor_failure_lines": 0,
        "positive_evidence_rows": 0,
        "true_contradiction_exclusions": 0,
    }


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str, root: Path | None = None) -> str:
    as_path = Path(path)
    base = root or Path.cwd()
    try:
        return str(as_path.resolve().relative_to(base.resolve()))
    except ValueError:  # pragma: no cover - only for out-of-tree custom paths.
        return str(as_path)


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-tests-passed", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment(focused_tests_passed=args.focused_tests_passed)
    print(
        "[exp1609] "
        f"ready={artifact['context_induction_ready']} "
        f"candidates={artifact['candidate_constraints_generated']} "
        f"evidence={artifact['failure_logs_mined']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "CLASS_SATISFIABLE_DRIFT",
    "CLASS_TRUE_CONTRADICTION",
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_CONDUCTOR_LOG_PATH",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_REPAIR_MANIFEST_PATH",
    "EXPERIMENT_ID",
    "FailureEvidence",
    "MiningResult",
    "REQUIRED_ARTIFACT_FIELDS",
    "generate_constraint_candidates",
    "mine_failure_logs",
    "run_experiment",
    "write_in_progress_artifact",
]
