"""Exp 5310: deterministic claim paraphrase-consistency fixture.

Spec refs: REQ-VERIFY-5310, SCENARIO-VERIFY-5310.

This module intentionally scores a tiny structured fixture instead of asking a
language model whether two sentences mean the same thing. The free-form text is
kept for human readability, but the verifier reads only curated fact fields,
premise-validity labels, and expected preservation labels. That makes the gate
cheap and reproducible before Exp 5311 spends local SOTA GGUF runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_5310_paraphrase_consistency_fixture_v485"
EXPERIMENT_NUMBER = 5310
MILESTONE = "2026.07.485"
RUN_DATE = "20260706"
SCHEMA = "carnot.experiment_5310.paraphrase_consistency_fixture.v485"
FIXTURE_RELATIVE_PATH = Path("data/claim_paraphrase_consistency_fixture_v485.json")
RESULT_RELATIVE_PATH = Path("results/experiment_5310_paraphrase_consistency_fixture_v485.json")
SPEC_REFS = ("REQ-VERIFY-5310", "SCENARIO-VERIFY-5310")
INFERENCE_SUBSTRATE = "deterministic_claim_paraphrase_fixture_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")
REQUIRED_FAMILIES = (
    "equivalent",
    "contradiction-preserving",
    "premise-invalid",
    "surface-only",
)
SEMANTIC_LABELS = ("supported", "contradictory", "premise-invalid", "unsupported")

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5310 deterministic paraphrase-consistency fixture.",
    "milestone": "Milestone accountability for the V485 paraphrase fixture gate.",
    "status": "Machine-readable terminal state for downstream gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether the "
        "deterministic paraphrase fixture is usable by Exp5311."
    ),
    "inference_substrate": (
        "Declares deterministic_claim_paraphrase_fixture_no_llm so the artifact is read as "
        "a fixture/scoring helper, not a live model claim."
    ),
    "fixture_path": (
        "Points downstream gates to the exact deterministic fixture file used to compute the "
        "reported metrics."
    ),
    "tests_run": (
        "Commands run to validate the fixture module, artifact schema, new-code coverage, and "
        "repository tests."
    ),
}


@dataclass(frozen=True)
class ParaphraseClaim:
    """One claim row whose semantic label is scored from curated structured facts."""

    claim_id: str
    text: str
    premise_valid: bool
    facts: dict[str, str]
    expected_label: str
    expected_label_preservation: bool
    expected_violation_type: str | None


@dataclass(frozen=True)
class ParaphraseGroup:
    """One anchor claim plus variants checked against the same evidence facts."""

    group_id: str
    family: str
    evidence_facts: dict[str, str]
    anchor: ParaphraseClaim
    variants: tuple[ParaphraseClaim, ...]
    label_source: str


@dataclass(frozen=True)
class ClaimScore:
    """Deterministic label plus the evidence keys that caused any contradiction."""

    label: str
    conflict_keys: tuple[str, ...]


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fact_map(payload: JsonDict) -> dict[str, str]:
    return {str(key): str(value) for key, value in payload.items()}


def _claim_from_payload(payload: JsonDict) -> ParaphraseClaim:
    return ParaphraseClaim(
        claim_id=str(payload["claim_id"]),
        text=str(payload["text"]),
        premise_valid=bool(payload["premise_valid"]),
        facts=_fact_map(payload["facts"]),
        expected_label=str(payload["expected_label"]),
        expected_label_preservation=bool(payload["expected_label_preservation"]),
        expected_violation_type=(
            None
            if payload["expected_violation_type"] is None
            else str(payload["expected_violation_type"])
        ),
    )


def _group_from_payload(payload: JsonDict, label_source: str) -> ParaphraseGroup:
    return ParaphraseGroup(
        group_id=str(payload["group_id"]),
        family=str(payload["family"]),
        evidence_facts=_fact_map(payload["evidence_facts"]),
        anchor=_claim_from_payload(payload["anchor"]),
        variants=tuple(_claim_from_payload(item) for item in payload["variants"]),
        label_source=label_source,
    )


def load_fixture(path: Path | None = None) -> tuple[ParaphraseGroup, ...]:
    """Load the checked-in no-LLM paraphrase fixture."""

    fixture_path = REPO_ROOT / FIXTURE_RELATIVE_PATH if path is None else path
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    label_source = str(payload["label_source"])
    return tuple(_group_from_payload(group, label_source) for group in payload["groups"])


def group_by_id(groups: tuple[ParaphraseGroup, ...], group_id: str) -> ParaphraseGroup:
    """Return a fixture group by its stable ID."""

    return next(group for group in groups if group.group_id == group_id)


def fixture_family_counts(groups: tuple[ParaphraseGroup, ...]) -> dict[str, int]:
    """Count each required paraphrase family, including absent families as zero."""

    counts = Counter(group.family for group in groups)
    return {family: counts.get(family, 0) for family in REQUIRED_FAMILIES}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()))


def token_overlap(left: str, right: str) -> float:
    """Return unique-token overlap against the first string."""

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    return len(left_tokens & right_tokens) / len(left_tokens)


def score_claim(claim: ParaphraseClaim, group: ParaphraseGroup) -> ClaimScore:
    """Score one claim deterministically from premise validity and fact conflicts."""

    if not claim.premise_valid:
        return ClaimScore(label="premise-invalid", conflict_keys=())
    if claim.facts == group.evidence_facts:
        return ClaimScore(label="supported", conflict_keys=())

    conflict_keys = tuple(
        sorted(
            key
            for key, value in claim.facts.items()
            if key in group.evidence_facts and group.evidence_facts[key] != value
        )
    )
    if conflict_keys:
        return ClaimScore(label="contradictory", conflict_keys=conflict_keys)
    return ClaimScore(label="unsupported", conflict_keys=())


def _claim_result(
    group: ParaphraseGroup,
    claim: ParaphraseClaim,
    anchor_score: ClaimScore,
) -> JsonDict:
    score = score_claim(claim, group)
    label_preserved = score.label == anchor_score.label
    label_matches_expected = score.label == claim.expected_label
    preservation_matches_expected = label_preserved == claim.expected_label_preservation
    caught_expected_violation = (
        claim.expected_violation_type is not None and not label_preserved and label_matches_expected
    )
    return {
        "group_id": group.group_id,
        "family": group.family,
        "claim_id": claim.claim_id,
        "computed_label": score.label,
        "expected_label": claim.expected_label,
        "label_matches_expected": label_matches_expected,
        "anchor_label": anchor_score.label,
        "expected_label_preservation": claim.expected_label_preservation,
        "label_preserved": label_preserved,
        "preservation_matches_expected": preservation_matches_expected,
        "expected_violation_type": claim.expected_violation_type,
        "caught_expected_violation": caught_expected_violation,
        "conflict_keys": list(score.conflict_keys),
        "premise_valid": claim.premise_valid,
        "token_overlap_with_anchor": token_overlap(group.anchor.text, claim.text),
    }


def _rate(passed: int, total: int) -> float:
    return 1.0 if total == 0 else passed / total


def evaluate_fixture(groups: tuple[ParaphraseGroup, ...]) -> JsonDict:
    """Evaluate label preservation, expected violations, and readiness blockers."""

    claim_results = []
    for group in groups:
        anchor_score = score_claim(group.anchor, group)
        claim_results.append(_claim_result(group, group.anchor, anchor_score))
        for claim in group.variants:
            claim_results.append(_claim_result(group, claim, anchor_score))

    label_preservation_rows = [
        row for row in claim_results if row["expected_label_preservation"] is True
    ]
    label_preservation_passes = [
        row
        for row in label_preservation_rows
        if row["label_preserved"] is True and row["label_matches_expected"] is True
    ]
    contradiction_violation_rows = [
        row for row in claim_results if row["expected_violation_type"] == "contradiction_erased"
    ]
    contradiction_violation_caught = [
        row for row in contradiction_violation_rows if row["caught_expected_violation"] is True
    ]
    invalid_premise_rows = [row for row in claim_results if row["family"] == "premise-invalid"]
    surface_violation_rows = [
        row
        for row in claim_results
        if row["expected_violation_type"] == "surface_overlap_label_flip"
    ]
    label_mismatches = [
        row["claim_id"] for row in claim_results if row["label_matches_expected"] is False
    ]
    preservation_mismatches = [
        row["claim_id"] for row in claim_results if row["preservation_matches_expected"] is False
    ]
    uncaught_violations = [
        row["claim_id"]
        for row in claim_results
        if row["expected_violation_type"] is not None and row["caught_expected_violation"] is False
    ]
    family_counts = fixture_family_counts(groups)
    missing_families = [family for family, count in family_counts.items() if count == 0]
    invalid_premise_handled = bool(invalid_premise_rows) and all(
        row["computed_label"] == "premise-invalid" for row in invalid_premise_rows
    )
    surface_false_positive_resisted = bool(surface_violation_rows) and all(
        row["caught_expected_violation"] is True for row in surface_violation_rows
    )
    label_preservation_pass_rate = _rate(
        len(label_preservation_passes), len(label_preservation_rows)
    )
    contradiction_violation_caught_rate = _rate(
        len(contradiction_violation_caught), len(contradiction_violation_rows)
    )
    ready = (
        not missing_families
        and not label_mismatches
        and not preservation_mismatches
        and not uncaught_violations
        and invalid_premise_handled
        and surface_false_positive_resisted
        and label_preservation_pass_rate == 1.0
        and contradiction_violation_caught_rate == 1.0
    )
    return {
        "ready": ready,
        "claim_results": claim_results,
        "family_counts": family_counts,
        "missing_families": missing_families,
        "label_mismatches": label_mismatches,
        "preservation_mismatches": preservation_mismatches,
        "uncaught_violations": uncaught_violations,
        "surface_false_positive_resisted": surface_false_positive_resisted,
        "paraphrase_group_count": len(groups),
        "label_preservation_pass_rate": label_preservation_pass_rate,
        "contradiction_violation_caught_rate": contradiction_violation_caught_rate,
        "invalid_premise_handled": invalid_premise_handled,
    }


def _readiness_blockers(evaluation: JsonDict) -> list[str]:
    blockers = []
    if evaluation["missing_families"]:
        blockers.append("missing families: " + ", ".join(evaluation["missing_families"]))
    if evaluation["label_mismatches"]:
        blockers.append("label mismatches: " + ", ".join(evaluation["label_mismatches"]))
    if evaluation["preservation_mismatches"]:
        blockers.append(
            "preservation mismatches: " + ", ".join(evaluation["preservation_mismatches"])
        )
    if evaluation["uncaught_violations"]:
        blockers.append("uncaught violations: " + ", ".join(evaluation["uncaught_violations"]))
    if not evaluation["invalid_premise_handled"]:
        blockers.append("invalid premise not handled")
    if not evaluation["surface_false_positive_resisted"]:
        blockers.append("surface false positive not resisted")
    return blockers


def build_artifact(
    groups: tuple[ParaphraseGroup, ...],
    *,
    tests_run: list[JsonDict],
) -> JsonDict:
    """Build the Exp 5310 result artifact from deterministic fixture labels."""

    evaluation = evaluate_fixture(groups)
    ready = bool(evaluation["ready"])
    status = "complete" if ready else "blocked"
    verdict = (
        "complete: deterministic paraphrase-consistency fixture usable by Exp5311"
        if ready
        else "blocked_paraphrase_consistency_fixture_not_ready"
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
        "paraphrase_fixture_ready": ready,
        "paraphrase_group_count": evaluation["paraphrase_group_count"],
        "label_preservation_pass_rate": evaluation["label_preservation_pass_rate"],
        "contradiction_violation_caught_rate": evaluation["contradiction_violation_caught_rate"],
        "invalid_premise_handled": evaluation["invalid_premise_handled"],
        "readiness_blockers": _readiness_blockers(evaluation),
        "family_counts": evaluation["family_counts"],
        "claim_results": evaluation["claim_results"],
        "fixture_checksum": _stable_json(evaluation["claim_results"]),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "tests_run": _wrap("tests_run", tests_run),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Assert the schema fields that Exp 5311 and conductor gates depend on."""

    for field, principle in FIELD_PRINCIPLES.items():
        assert artifact[field]["principle"] == principle
        assert "value" in artifact[field]
    assert artifact["honest_verdict"]["value"].startswith(TERMINAL_PREFIXES)
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE
    assert type(artifact["paraphrase_fixture_ready"]) is bool
    assert type(artifact["paraphrase_group_count"]) is int
    assert isinstance(artifact["label_preservation_pass_rate"], int | float)
    assert isinstance(artifact["contradiction_violation_caught_rate"], int | float)
    assert type(artifact["invalid_premise_handled"]) is bool
    assert artifact["fixture_path"]["value"] == str(FIXTURE_RELATIVE_PATH)
    assert set(REQUIRED_FAMILIES) <= set(artifact["family_counts"])
    assert isinstance(artifact["tests_run"]["value"], list)


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    groups: tuple[ParaphraseGroup, ...] | list[ParaphraseGroup] | None = None,
    tests_run: list[JsonDict] | None = None,
) -> JsonDict:
    """Run the offline fixture evaluation and write the result artifact."""

    fixture_groups = load_fixture() if groups is None else tuple(groups)
    artifact = build_artifact(fixture_groups, tests_run=[] if tests_run is None else tests_run)
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
