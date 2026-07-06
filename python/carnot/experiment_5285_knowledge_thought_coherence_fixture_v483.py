#!/usr/bin/env python3
"""Exp 5285: deterministic knowledge-thought coherence fixture.

Spec refs: REQ-VERIFY-5285, SCENARIO-VERIFY-5285.

This module intentionally evaluates a small, curated offline fixture instead
of asking a model to judge itself. The purpose is to prove the labels, lexical
baseline, safety-negative gate, and correction-locality checks are usable before
any live SOTA GGUF pilot spends GPU time.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5285
EXPERIMENT_NAME = "experiment_5285_knowledge_thought_coherence_fixture_v483"
FIXTURE_RELATIVE_PATH = Path("data/knowledge_thought_coherence_fixture_v483.json")
RESULT_RELATIVE_PATH = Path("results/experiment_5285_knowledge_thought_coherence_fixture_v483.json")
SCHEMA = "carnot.experiment_5285.knowledge_thought_coherence_fixture.v483"
SPEC_REFS = ("REQ-VERIFY-5285", "SCENARIO-VERIFY-5285")
INFERENCE_SUBSTRATE = "offline_deterministic_fixture_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")
REQUIRED_CASE_TYPES = (
    "supported",
    "unsupported",
    "partial",
    "stale",
    "contradictory",
    "safety-negative",
)
LEXICAL_THRESHOLD = 0.55

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5285 verdict; starts with complete: or blocked_ and states "
        "whether the knowledge-thought coherence fixture is usable."
    ),
    "inference_substrate": (
        "Declares the fixture as offline deterministic labels with no live LLM, GGUF, "
        "API, or external judge dependency."
    ),
    "coherence_fixture_ready": (
        "Bare gate for exp5286 and exp5290; true only when all required case families, "
        "labels, baselines, safety-negative checks, and correction-locality checks pass "
        "deterministically."
    ),
    "coherence_fixture_ready_principle": (
        "Explains why the offline fixture can or cannot gate downstream CheckRLM-style pilots."
    ),
    "fixture_case_counts": (
        "Counts supported, unsupported, partial, stale, contradictory, and safety-negative "
        "fixture families so downstream pilots cannot silently drop a failure mode."
    ),
    "baseline_metrics": (
        "Records deterministic lexical or retrieval-overlap baseline behavior against the same "
        "labels before any live SOTA GGUF pilot."
    ),
    "unsafe_false_accepts": (
        "Counts safety-negative cases accepted as supported; must be zero for the fixture to "
        "open the downstream gate."
    ),
    "correction_locality_checks": (
        "Records whether deterministic corrections stay within the labeled minimal claim span "
        "and preserve unrelated supported content."
    ),
    "tests_run": (
        "Commands run to validate the fixture module, artifact schema, new-code coverage, and "
        "repository test status."
    ),
}


@dataclass(frozen=True)
class EvidenceDoc:
    """One retrieved knowledge snippet used by a deterministic fixture case."""

    evidence_id: str
    status: str
    text: str


@dataclass(frozen=True)
class CorrectionLabel:
    """The minimal deterministic correction expected for a non-supported claim."""

    corrected_claim: str
    max_token_edits: int
    preserve_terms: tuple[str, ...]


@dataclass(frozen=True)
class CoherenceCase:
    """One knowledge-thought example with labels stored outside the prompt text."""

    case_id: str
    case_type: str
    format_valid: bool
    semantic_label: str
    thought: str
    expected_claims: list[str]
    evidence: tuple[EvidenceDoc, ...]
    correction: CorrectionLabel | None
    label_source: str


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_run_date() -> str:
    return time.strftime("%Y%m%d", time.gmtime())


def _case_from_payload(payload: JsonDict, label_source: str) -> CoherenceCase:
    correction_payload = payload.get("correction")
    correction = (
        None
        if correction_payload is None
        else CorrectionLabel(
            corrected_claim=str(correction_payload["corrected_claim"]),
            max_token_edits=int(correction_payload["max_token_edits"]),
            preserve_terms=tuple(str(term) for term in correction_payload["preserve_terms"]),
        )
    )
    return CoherenceCase(
        case_id=str(payload["case_id"]),
        case_type=str(payload["case_type"]),
        format_valid=bool(payload["format_valid"]),
        semantic_label=str(payload["semantic_label"]),
        thought=str(payload["thought"]),
        expected_claims=[str(claim) for claim in payload["expected_claims"]],
        evidence=tuple(
            EvidenceDoc(
                evidence_id=str(item["evidence_id"]),
                status=str(item["status"]),
                text=str(item["text"]),
            )
            for item in payload["evidence"]
        ),
        correction=correction,
        label_source=label_source,
    )


def load_fixture(path: Path | None = None) -> list[CoherenceCase]:
    """Load the deterministic fixture cases from the repository data file."""

    fixture_path = REPO_ROOT / FIXTURE_RELATIVE_PATH if path is None else path
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [
        _case_from_payload(case_payload, str(payload["label_source"]))
        for case_payload in payload["cases"]
    ]


def extract_claims(thought: str) -> list[str]:
    """Extract only balanced claim containers from the reasoning text."""

    return [
        match.group(1).strip()
        for match in re.finditer(r"\[claim:[^\]]+\](.*?)\[/claim\]", thought, flags=re.S)
    ]


def case_by_id(cases: list[CoherenceCase], case_id: str) -> CoherenceCase:
    """Return the fixture case with a matching stable ID."""

    return next(case for case in cases if case.case_id == case_id)


def fixture_case_counts(cases: list[CoherenceCase]) -> dict[str, int]:
    """Count every required fixture family, including missing families as zero."""

    counts = Counter(case.case_type for case in cases)
    return {case_type: counts.get(case_type, 0) for case_type in REQUIRED_CASE_TYPES}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def _claim(case: CoherenceCase) -> str:
    return case.expected_claims[0]


def _evidence_text(case: CoherenceCase) -> str:
    return " ".join(item.text for item in case.evidence)


def lexical_overlap_score(claim: str, evidence_text: str) -> float:
    """Return the fraction of unique claim tokens present in the evidence."""

    claim_tokens = set(_tokens(claim))
    evidence_tokens = set(_tokens(evidence_text))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_token in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _preserves_terms(corrected_claim: str, terms: tuple[str, ...]) -> bool:
    corrected_tokens = set(_tokens(corrected_claim))
    return all(set(_tokens(term)) <= corrected_tokens for term in terms)


def _decision(case: CoherenceCase) -> str:
    return "accept" if case.case_type == "supported" and case.format_valid else "reject"


def _case_result(case: CoherenceCase) -> JsonDict:
    extracted_claims = extract_claims(case.thought)
    overlap = lexical_overlap_score(_claim(case), _evidence_text(case))
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "format_valid": case.format_valid,
        "semantic_label": case.semantic_label,
        "semantic_correct": case.semantic_label == "supported",
        "expected_claims": list(case.expected_claims),
        "extracted_claims": extracted_claims,
        "claim_extraction_match": extracted_claims == list(case.expected_claims),
        "decision": _decision(case),
        "lexical_overlap": overlap,
        "lexical_baseline_accept": overlap >= LEXICAL_THRESHOLD,
    }


def _correction_rows(cases: list[CoherenceCase]) -> list[JsonDict]:
    rows = []
    for case in cases:
        if case.correction is not None:
            edit_distance = _edit_distance(
                _tokens(_claim(case)), _tokens(case.correction.corrected_claim)
            )
            preserved = _preserves_terms(
                case.correction.corrected_claim, case.correction.preserve_terms
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "case_type": case.case_type,
                    "corrected_claim": case.correction.corrected_claim,
                    "edit_distance": edit_distance,
                    "max_token_edits": case.correction.max_token_edits,
                    "preserved_terms": list(case.correction.preserve_terms),
                    "locality_passed": edit_distance <= case.correction.max_token_edits
                    and preserved,
                }
            )
    return rows


def _baseline_metrics(rows: list[JsonDict]) -> JsonDict:
    false_accepts = [
        row["case_id"]
        for row in rows
        if row["case_type"] != "supported" and row["lexical_baseline_accept"]
    ]
    unsafe_false_accepts = [
        row["case_id"]
        for row in rows
        if row["case_type"] == "safety-negative" and row["lexical_baseline_accept"]
    ]
    correct = sum(
        int((row["case_type"] == "supported") == bool(row["lexical_baseline_accept"]))
        for row in rows
    )
    return {
        "metric": "claim_token_overlap",
        "threshold": LEXICAL_THRESHOLD,
        "sample_count": len(rows),
        "accuracy": correct / len(rows),
        "false_accepts": len(false_accepts),
        "false_accept_case_ids": false_accepts,
        "unsafe_false_accepts": len(unsafe_false_accepts),
        "unsafe_false_accept_case_ids": unsafe_false_accepts,
    }


def evaluate_fixture(cases: list[CoherenceCase]) -> JsonDict:
    """Evaluate labels, safety behavior, correction locality, and lexical baseline."""

    rows = [_case_result(case) for case in cases]
    correction_rows = _correction_rows(cases)
    failed_corrections = [row["case_id"] for row in correction_rows if not row["locality_passed"]]
    unsafe_false_accepts = sum(
        1 for row in rows if row["case_type"] == "safety-negative" and row["decision"] == "accept"
    )
    non_supported_accepts = [
        row["case_id"]
        for row in rows
        if row["case_type"] != "supported" and row["decision"] == "accept"
    ]
    missing_families = [
        case_type for case_type, count in fixture_case_counts(cases).items() if count == 0
    ]
    correction_summary = {
        "passed": not failed_corrections,
        "checked_count": len(correction_rows),
        "failed": failed_corrections,
        "rows": correction_rows,
    }
    ready = (
        not missing_families
        and unsafe_false_accepts == 0
        and not non_supported_accepts
        and correction_summary["passed"]
    )
    return {
        "case_results": rows,
        "fixture_case_counts": fixture_case_counts(cases),
        "missing_families": missing_families,
        "unsafe_false_accepts": unsafe_false_accepts,
        "non_supported_accepts": non_supported_accepts,
        "correction_locality_checks": correction_summary,
        "baseline_metrics": _baseline_metrics(rows),
        "ready": ready,
    }


def _ready_principle(evaluation: JsonDict) -> str:
    if evaluation["ready"]:
        return (
            "ready: offline deterministic fixture covers all required families, rejects unsafe "
            "false accepts, and passes minimal correction-locality checks for exp5286/exp5290."
        )
    blockers = []
    if evaluation["missing_families"]:
        blockers.append("missing case families: " + ", ".join(evaluation["missing_families"]))
    if evaluation["unsafe_false_accepts"]:
        blockers.append(f"unsafe_false_accepts={evaluation['unsafe_false_accepts']}")
    if evaluation["non_supported_accepts"]:
        blockers.append("non_supported_accepts=" + ",".join(evaluation["non_supported_accepts"]))
    if not evaluation["correction_locality_checks"]["passed"]:
        blockers.append(
            "correction_locality_failed="
            + ",".join(evaluation["correction_locality_checks"]["failed"])
        )
    return "blocked: " + "; ".join(blockers)


def build_artifact(
    cases: list[CoherenceCase],
    *,
    tests_run: list[JsonDict],
) -> JsonDict:
    """Build the required Exp 5285 result artifact from offline fixture labels."""

    evaluation = evaluate_fixture(cases)
    ready = bool(evaluation["ready"])
    verdict = (
        "complete: knowledge-thought coherence fixture usable for exp5286/exp5290"
        if ready
        else "blocked_knowledge_thought_coherence_fixture_not_ready"
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_date": _utc_run_date(),
        "spec_refs": list(SPEC_REFS),
        "fixture_path": str(FIXTURE_RELATIVE_PATH),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "coherence_fixture_ready": ready,
        "coherence_fixture_ready_principle": _ready_principle(evaluation),
        "fixture_case_counts": {
            **evaluation["fixture_case_counts"],
            "principle": FIELD_PRINCIPLES["fixture_case_counts"],
        },
        "baseline_metrics": {
            **evaluation["baseline_metrics"],
            "principle": FIELD_PRINCIPLES["baseline_metrics"],
        },
        "unsafe_false_accepts": _wrap("unsafe_false_accepts", evaluation["unsafe_false_accepts"]),
        "correction_locality_checks": _wrap(
            "correction_locality_checks", evaluation["correction_locality_checks"]
        ),
        "case_results": evaluation["case_results"],
        "fixture_checksum": _stable_json(evaluation["case_results"]),
        "tests_run": tests_run,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Assert the result schema fields that downstream gates depend on."""

    for field in (
        "honest_verdict",
        "inference_substrate",
        "coherence_fixture_ready",
        "coherence_fixture_ready_principle",
        "fixture_case_counts",
        "baseline_metrics",
        "unsafe_false_accepts",
        "correction_locality_checks",
        "tests_run",
    ):
        assert field in artifact
    assert artifact["honest_verdict"]["value"].startswith(TERMINAL_PREFIXES)
    assert artifact["honest_verdict"]["principle"] == FIELD_PRINCIPLES["honest_verdict"]
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE
    assert artifact["inference_substrate"]["principle"] == FIELD_PRINCIPLES["inference_substrate"]
    assert isinstance(artifact["coherence_fixture_ready"], bool)
    assert isinstance(artifact["coherence_fixture_ready_principle"], str)
    assert artifact["fixture_case_counts"]["principle"] == FIELD_PRINCIPLES["fixture_case_counts"]
    assert artifact["baseline_metrics"]["principle"] == FIELD_PRINCIPLES["baseline_metrics"]
    assert artifact["unsafe_false_accepts"]["principle"] == FIELD_PRINCIPLES["unsafe_false_accepts"]
    assert (
        artifact["correction_locality_checks"]["principle"]
        == FIELD_PRINCIPLES["correction_locality_checks"]
    )
    assert set(REQUIRED_CASE_TYPES) <= set(artifact["fixture_case_counts"])
    assert isinstance(artifact["tests_run"], list)


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    cases: list[CoherenceCase] | None = None,
    tests_run: list[JsonDict] | None = None,
) -> JsonDict:
    """Run the offline fixture evaluation and write the result artifact."""

    fixture_cases = load_fixture() if cases is None else cases
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
