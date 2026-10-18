"""Exp 5325: deterministic Theoria rewrite-state verifier fixture.

Spec refs: REQ-VERIFY-5325, SCENARIO-VERIFY-5325.

This module treats a rewrite as a typed transition between two small structured
states. The free-form sentence is kept for human auditability, but the verifier
decides from typed facts, citations, premise validity, and explicit change
obligations. That makes the fixture useful as a deterministic gate before Exp
5326 spends SOTA model runtime on rewrite quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_5325_theoria_rewrite_state_fixture_v486"
EXPERIMENT_NUMBER = 5325
MILESTONE = "2026.07.486"
RUN_DATE = "20260706"
SCHEMA = "carnot.experiment_5325.theoria_rewrite_state_fixture.v486"
FIXTURE_RELATIVE_PATH = Path("data/rewrite_state_fixture_v486.json")
RESULT_RELATIVE_PATH = Path("results/experiment_5325_theoria_rewrite_state_fixture_v486.json")
SPEC_REFS = ("REQ-VERIFY-5325", "SCENARIO-VERIFY-5325")
INFERENCE_SUBSTRATE = "deterministic_rewrite_state_fixture"
TERMINAL_PREFIXES = ("complete:", "blocked_")
SEMANTIC_LABELS = ("supported", "contradictory", "premise-invalid", "unsupported")
REQUIRED_CASE_TYPES = (
    "safe_paraphrase",
    "contradiction_introduction",
    "missing_required_change",
    "fabricated_premise_citation",
    "invalid_premise_preserved",
    "overbroad_rewrite",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5325 deterministic Theoria rewrite-state fixture.",
    "milestone": "Milestone accountability for the V486 rewrite-state fixture gate.",
    "status": "Machine-readable terminal state for downstream rewrite-quality gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether Exp5326 "
        "can consume the deterministic rewrite-state fixture."
    ),
    "inference_substrate": (
        "Declares deterministic_rewrite_state_fixture so the artifact is read as typed "
        "offline rewrite-state checks, not live model quality."
    ),
    "fixture_path": (
        "Points downstream gates to the exact deterministic rewrite-state fixture used to "
        "compute the reported metrics."
    ),
    "tests_run": (
        "Commands run to validate the rewrite-state module, artifact schema, new-code "
        "coverage, and repository tests."
    ),
}

WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "fixture_path",
    "tests_run",
)
REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "rewrite_case_count",
    "rewrite_acceptability_rate",
    "complete_change_coverage_rate",
    "unsafe_rewrite_rejection_rate",
    "false_accept_count",
    "rewrite_state_fixture_ready",
    "tests_run",
)


@dataclass(frozen=True)
class RewriteState:
    """One side of a rewrite transition, represented as typed verifier state."""

    text: str
    premise_valid: bool
    facts: dict[str, str]
    attributes: dict[str, str]
    citations: tuple[str, ...]
    expected_label: str


@dataclass(frozen=True)
class ChangeObligation:
    """A required typed transition that the target state must complete."""

    obligation_id: str
    change_type: str
    field: str
    source_value: Any
    target_value: Any


@dataclass(frozen=True)
class RewriteCase:
    """A source-to-target rewrite with evidence and expected gate outcomes."""

    case_id: str
    case_type: str
    description: str
    evidence_facts: dict[str, str]
    allowed_citations: tuple[str, ...]
    source: RewriteState
    target: RewriteState
    change_obligations: tuple[ChangeObligation, ...]
    expected_label_preservation: bool
    expected_complete_change_coverage: bool
    expected_accept: bool
    expected_unsafe_rejected: bool
    label_source: str


@dataclass(frozen=True)
class StateScore:
    """Deterministic semantic label plus structured evidence failure reasons."""

    label: str
    conflict_keys: tuple[str, ...]
    fabricated_fact_keys: tuple[str, ...]
    fabricated_citations: tuple[str, ...]


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_map(payload: JsonDict) -> dict[str, str]:
    return {str(key): str(value) for key, value in payload.items()}


def _state_from_payload(payload: JsonDict) -> RewriteState:
    return RewriteState(
        text=str(payload["text"]),
        premise_valid=bool(payload["premise_valid"]),
        facts=_string_map(payload["facts"]),
        attributes=_string_map(payload["attributes"]),
        citations=tuple(str(citation) for citation in payload["citations"]),
        expected_label=str(payload["expected_label"]),
    )


def _obligation_from_payload(payload: JsonDict) -> ChangeObligation:
    return ChangeObligation(
        obligation_id=str(payload["obligation_id"]),
        change_type=str(payload["change_type"]),
        field=str(payload["field"]),
        source_value=payload["source_value"],
        target_value=payload["target_value"],
    )


def _case_from_payload(payload: JsonDict, label_source: str) -> RewriteCase:
    return RewriteCase(
        case_id=str(payload["case_id"]),
        case_type=str(payload["case_type"]),
        description=str(payload["description"]),
        evidence_facts=_string_map(payload["evidence_facts"]),
        allowed_citations=tuple(str(citation) for citation in payload["allowed_citations"]),
        source=_state_from_payload(payload["source"]),
        target=_state_from_payload(payload["target"]),
        change_obligations=tuple(
            _obligation_from_payload(item) for item in payload["change_obligations"]
        ),
        expected_label_preservation=bool(payload["expected_label_preservation"]),
        expected_complete_change_coverage=bool(payload["expected_complete_change_coverage"]),
        expected_accept=bool(payload["expected_accept"]),
        expected_unsafe_rejected=bool(payload["expected_unsafe_rejected"]),
        label_source=label_source,
    )


def load_fixture(path: Path | None = None) -> tuple[RewriteCase, ...]:
    """Load the checked-in no-LLM rewrite-state fixture."""

    fixture_path = REPO_ROOT / FIXTURE_RELATIVE_PATH if path is None else path
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    label_source = str(payload["label_source"])
    return tuple(_case_from_payload(case, label_source) for case in payload["cases"])


def case_by_id(cases: tuple[RewriteCase, ...], case_id: str) -> RewriteCase:
    """Return one rewrite case by its stable fixture ID."""

    return next(case for case in cases if case.case_id == case_id)


def rewrite_case_type_counts(cases: tuple[RewriteCase, ...]) -> dict[str, int]:
    """Count each required rewrite case type, including absent types as zero."""

    counts = Counter(case.case_type for case in cases)
    return {case_type: counts.get(case_type, 0) for case_type in REQUIRED_CASE_TYPES}


def score_state(
    state: RewriteState,
    *,
    evidence_facts: dict[str, str],
    allowed_citations: tuple[str, ...],
) -> StateScore:
    """Score one typed state without consulting the free-form rewrite text."""

    if not state.premise_valid:
        return StateScore(
            label="premise-invalid",
            conflict_keys=(),
            fabricated_fact_keys=(),
            fabricated_citations=(),
        )

    conflict_keys = tuple(
        sorted(
            key
            for key, value in state.facts.items()
            if key in evidence_facts and evidence_facts[key] != value
        )
    )
    fabricated_fact_keys = tuple(sorted(key for key in state.facts if key not in evidence_facts))
    fabricated_citations = tuple(
        sorted(citation for citation in state.citations if citation not in allowed_citations)
    )
    label = (
        "contradictory"
        if conflict_keys
        else "unsupported"
        if fabricated_fact_keys or fabricated_citations
        else "supported"
    )
    return StateScore(
        label=label,
        conflict_keys=conflict_keys,
        fabricated_fact_keys=fabricated_fact_keys,
        fabricated_citations=fabricated_citations,
    )


def _state_value(state: RewriteState, field: str) -> Any:
    if field == "premise_valid":
        return state.premise_valid
    namespace, _, key = field.partition(".")
    return {"facts": state.facts, "attributes": state.attributes}.get(namespace, {}).get(key)


def _obligation_coverage(case: RewriteCase) -> tuple[bool, tuple[str, ...]]:
    missing = []
    for obligation in case.change_obligations:
        source_matches = _state_value(case.source, obligation.field) == obligation.source_value
        target_matches = _state_value(case.target, obligation.field) == obligation.target_value
        if not (source_matches and target_matches):
            missing.append(obligation.obligation_id)
    return not missing, tuple(missing)


def _changed_fact_keys(source: RewriteState, target: RewriteState) -> tuple[str, ...]:
    fact_keys = set(source.facts) | set(target.facts)
    return tuple(sorted(key for key in fact_keys if source.facts.get(key) != target.facts.get(key)))


def _required_fact_change_keys(case: RewriteCase) -> set[str]:
    return {
        obligation.field.partition(".")[2]
        for obligation in case.change_obligations
        if obligation.field.startswith("facts.")
        and obligation.source_value != obligation.target_value
    }


def _case_result(case: RewriteCase) -> JsonDict:
    source_score = score_state(
        case.source,
        evidence_facts=case.evidence_facts,
        allowed_citations=case.allowed_citations,
    )
    target_score = score_state(
        case.target,
        evidence_facts=case.evidence_facts,
        allowed_citations=case.allowed_citations,
    )
    complete_change_coverage, missing_obligations = _obligation_coverage(case)
    label_preserved = source_score.label == target_score.label
    label_preservation_matches_expected = (
        label_preserved == case.expected_label_preservation
    )
    source_label_matches_expected = source_score.label == case.source.expected_label
    target_label_matches_expected = target_score.label == case.target.expected_label
    changed_fact_keys = _changed_fact_keys(case.source, case.target)
    overbroad_fact_keys = tuple(
        key for key in changed_fact_keys if key not in _required_fact_change_keys(case)
    )

    rejection_reasons = []
    if not complete_change_coverage:
        rejection_reasons.append("missing_required_change")
    if case.expected_label_preservation and not label_preserved:
        rejection_reasons.append("label_preservation_failed")
    if not case.expected_label_preservation and label_preserved:
        rejection_reasons.append("expected_label_change_missing")
    if source_score.label != "contradictory" and target_score.label == "contradictory":
        rejection_reasons.append("contradiction_introduced")
    if target_score.fabricated_fact_keys:
        rejection_reasons.append("fabricated_fact")
    if target_score.fabricated_citations:
        rejection_reasons.append("fabricated_citation")
    if not case.source.premise_valid and not case.target.premise_valid:
        rejection_reasons.append("invalid_premise_preserved")
    if overbroad_fact_keys:
        rejection_reasons.append("overbroad_fact_change")

    accepted = (
        not rejection_reasons
        and source_label_matches_expected
        and target_label_matches_expected
    )
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "source_label": source_score.label,
        "target_label": target_score.label,
        "source_label_matches_expected": source_label_matches_expected,
        "target_label_matches_expected": target_label_matches_expected,
        "expected_label_preservation": case.expected_label_preservation,
        "label_preserved": label_preserved,
        "label_preservation_matches_expected": label_preservation_matches_expected,
        "complete_change_coverage": complete_change_coverage,
        "expected_complete_change_coverage": case.expected_complete_change_coverage,
        "complete_change_coverage_matches_expected": (
            complete_change_coverage == case.expected_complete_change_coverage
        ),
        "expected_accept": case.expected_accept,
        "accepted": accepted,
        "acceptability_matches_expected": accepted == case.expected_accept,
        "expected_unsafe_rejected": case.expected_unsafe_rejected,
        "unsafe_rewrite_rejected": (
            case.expected_unsafe_rejected and not accepted and bool(rejection_reasons)
        ),
        "missing_obligations": list(missing_obligations),
        "rejection_reasons": rejection_reasons,
        "conflict_keys": list(target_score.conflict_keys),
        "fabricated_fact_keys": list(target_score.fabricated_fact_keys),
        "fabricated_citations": list(target_score.fabricated_citations),
        "changed_fact_keys": list(changed_fact_keys),
        "overbroad_fact_keys": list(overbroad_fact_keys),
    }


def _rate(passed: int, total: int) -> float:
    return 1.0 if total == 0 else passed / total


def evaluate_fixture(cases: tuple[RewriteCase, ...]) -> JsonDict:
    """Evaluate acceptability, obligation coverage, and unsafe rejection gates."""

    case_results = [_case_result(case) for case in cases]
    case_type_counts = rewrite_case_type_counts(cases)
    missing_case_types = [
        case_type for case_type, count in case_type_counts.items() if count == 0
    ]
    unsafe_rows = [row for row in case_results if row["expected_unsafe_rejected"] is True]
    false_accept_ids = [
        row["case_id"]
        for row in case_results
        if row["expected_accept"] is False and row["accepted"] is True
    ]
    label_mismatch_ids = [
        row["case_id"]
        for row in case_results
        if not row["source_label_matches_expected"] or not row["target_label_matches_expected"]
    ]
    rewrite_acceptability_rate = _rate(
        sum(1 for row in case_results if row["acceptability_matches_expected"]),
        len(case_results),
    )
    complete_change_coverage_rate = _rate(
        sum(1 for row in case_results if row["complete_change_coverage_matches_expected"]),
        len(case_results),
    )
    unsafe_rewrite_rejection_rate = _rate(
        sum(1 for row in unsafe_rows if row["unsafe_rewrite_rejected"]),
        len(unsafe_rows),
    )
    ready = (
        len(case_results) == len(REQUIRED_CASE_TYPES)
        and not missing_case_types
        and not false_accept_ids
        and not label_mismatch_ids
        and rewrite_acceptability_rate == 1.0
        and complete_change_coverage_rate == 1.0
        and unsafe_rewrite_rejection_rate == 1.0
    )
    return {
        "ready": ready,
        "case_results": case_results,
        "case_type_counts": case_type_counts,
        "missing_case_types": missing_case_types,
        "false_accept_ids": false_accept_ids,
        "label_mismatch_ids": label_mismatch_ids,
        "rewrite_case_count": len(case_results),
        "rewrite_acceptability_rate": rewrite_acceptability_rate,
        "complete_change_coverage_rate": complete_change_coverage_rate,
        "unsafe_rewrite_rejection_rate": unsafe_rewrite_rejection_rate,
        "false_accept_count": len(false_accept_ids),
    }


def _consumer_contract() -> JsonDict:
    return {
        "next_experiment": "Exp5326",
        "fixture_path": str(FIXTURE_RELATIVE_PATH),
        "ready_field": "rewrite_state_fixture_ready",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "required_metrics": [
            "rewrite_case_count",
            "rewrite_acceptability_rate",
            "complete_change_coverage_rate",
            "unsafe_rewrite_rejection_rate",
            "false_accept_count",
        ],
    }


def _readiness_blockers(evaluation: JsonDict) -> list[str]:
    blockers = []
    if evaluation["missing_case_types"]:
        blockers.append("missing case types: " + ", ".join(evaluation["missing_case_types"]))
    if evaluation["false_accept_ids"]:
        blockers.append("false accepts: " + ", ".join(evaluation["false_accept_ids"]))
    if evaluation["label_mismatch_ids"]:
        blockers.append("label mismatches: " + ", ".join(evaluation["label_mismatch_ids"]))
    if evaluation["rewrite_acceptability_rate"] != 1.0:
        blockers.append("rewrite acceptability mismatch")
    if evaluation["complete_change_coverage_rate"] != 1.0:
        blockers.append("complete-change coverage mismatch")
    if evaluation["unsafe_rewrite_rejection_rate"] != 1.0:
        blockers.append("unsafe rewrite rejection mismatch")
    return blockers


def build_artifact(
    cases: tuple[RewriteCase, ...],
    *,
    tests_run: list[JsonDict],
) -> JsonDict:
    """Build the Exp 5325 result artifact from deterministic rewrite states."""

    evaluation = evaluate_fixture(cases)
    consumer_contract = _consumer_contract()
    ready = bool(evaluation["ready"] and consumer_contract["next_experiment"] == "Exp5326")
    status = "complete" if ready else "blocked"
    verdict = (
        "complete: deterministic rewrite-state fixture usable by Exp5326"
        if ready
        else "blocked_rewrite_state_fixture_not_ready"
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NUMBER,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_NAME),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "fixture_path": _wrap("fixture_path", str(FIXTURE_RELATIVE_PATH)),
        "rewrite_case_count": evaluation["rewrite_case_count"],
        "rewrite_acceptability_rate": evaluation["rewrite_acceptability_rate"],
        "complete_change_coverage_rate": evaluation["complete_change_coverage_rate"],
        "unsafe_rewrite_rejection_rate": evaluation["unsafe_rewrite_rejection_rate"],
        "false_accept_count": evaluation["false_accept_count"],
        "rewrite_state_fixture_ready": ready,
        "readiness_blockers": _readiness_blockers(evaluation),
        "case_type_counts": evaluation["case_type_counts"],
        "case_results": evaluation["case_results"],
        "consumer_contract": consumer_contract,
        "fixture_checksum": _stable_json(evaluation["case_results"]),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "tests_run": _wrap("tests_run", tests_run),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Assert the schema fields that Exp 5326 and conductor gates depend on."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    for field in WRAPPED_FIELDS:
        assert artifact[field]["principle"] == FIELD_PRINCIPLES[field]
        assert "value" in artifact[field]
    assert artifact["honest_verdict"]["value"].startswith(TERMINAL_PREFIXES)
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE
    assert type(artifact["rewrite_case_count"]) is int
    assert isinstance(artifact["rewrite_acceptability_rate"], int | float)
    assert isinstance(artifact["complete_change_coverage_rate"], int | float)
    assert isinstance(artifact["unsafe_rewrite_rejection_rate"], int | float)
    assert 0.0 <= artifact["rewrite_acceptability_rate"] <= 1.0
    assert 0.0 <= artifact["complete_change_coverage_rate"] <= 1.0
    assert 0.0 <= artifact["unsafe_rewrite_rejection_rate"] <= 1.0
    assert type(artifact["false_accept_count"]) is int
    assert type(artifact["rewrite_state_fixture_ready"]) is bool
    assert artifact["fixture_path"]["value"] == str(FIXTURE_RELATIVE_PATH)
    assert artifact["consumer_contract"]["next_experiment"] == "Exp5326"
    assert artifact["consumer_contract"]["inference_substrate"] == INFERENCE_SUBSTRATE
    assert set(REQUIRED_CASE_TYPES) <= set(artifact["case_type_counts"])
    assert isinstance(artifact["tests_run"]["value"], list)
    if artifact["rewrite_state_fixture_ready"]:
        assert artifact["status"]["value"] == "complete"
        assert artifact["false_accept_count"] == 0
        assert artifact["rewrite_acceptability_rate"] == 1.0
        assert artifact["complete_change_coverage_rate"] == 1.0
        assert artifact["unsafe_rewrite_rejection_rate"] == 1.0
    else:
        assert artifact["status"]["value"] == "blocked"


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    cases: tuple[RewriteCase, ...] | list[RewriteCase] | None = None,
    tests_run: list[JsonDict] | None = None,
) -> JsonDict:
    """Run the offline rewrite-state evaluation and write the result artifact."""

    fixture_cases = load_fixture() if cases is None else tuple(cases)
    artifact = build_artifact(fixture_cases, tests_run=[] if tests_run is None else tests_run)
    write_json(result_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
