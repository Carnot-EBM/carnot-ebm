"""Exp5343: deterministic qualitative temporal/spatial constraint fixture.

Spec refs: REQ-VERIFY-5343, SCENARIO-VERIFY-5343.

This module keeps qualitative spatio-temporal reasoning grounded in exact typed
objects. Interval and rectangle coordinates are the authority; relation names
are checked against those objects before any downstream model can use the cases
as prompts. That makes failures local and deterministic instead of depending on
how plausible a natural-language explanation sounds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_5343_qstr_temporal_spatial_constraint_fixture_v487"
EXPERIMENT_NUMBER = 5343
MILESTONE = "2026.07.487"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5343.qstr_temporal_spatial_constraint_fixture.v487"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5343_qstr_temporal_spatial_constraint_fixture_v487.json"
)
INFERENCE_SUBSTRATE = "deterministic_qstr_constraint_fixture"
SPEC_REFS = ("REQ-VERIFY-5343", "SCENARIO-VERIFY-5343")
TERMINAL_PREFIXES = ("complete:", "blocked_")
TEMPORAL = "temporal"
SPATIAL = "spatial"
TEMPORAL_COMPOSITION_SOURCE = "finite_exact_interval_enumeration"
SPATIAL_COMPOSITION_SOURCE = "exact_rectangle_subset_rules"

TEMPORAL_RELATION_ORDER = (
    "before",
    "meets",
    "overlaps",
    "finished_by",
    "contains",
    "starts",
    "equals",
    "started_by",
    "during",
    "finishes",
    "overlapped_by",
    "met_by",
    "after",
)
SPATIAL_RELATION_ORDER = (
    "contains",
    "inside",
    "overlap",
    "disconnected",
    "east_of",
    "west_of",
)
REQUIRED_TEMPORAL_CASE_TYPES = (
    "before",
    "overlaps",
    "during",
    "meets",
    "contradiction",
    "ambiguous",
)
REQUIRED_SPATIAL_CASE_TYPES = (
    "disconnected",
    "overlap",
    "containment",
    "cardinal_direction",
    "contradiction",
)

TEMPORAL_CONVERSE = {
    "before": "after",
    "after": "before",
    "meets": "met_by",
    "met_by": "meets",
    "overlaps": "overlapped_by",
    "overlapped_by": "overlaps",
    "during": "contains",
    "contains": "during",
    "starts": "started_by",
    "started_by": "starts",
    "finishes": "finished_by",
    "finished_by": "finishes",
    "equals": "equals",
}
SPATIAL_CONVERSE = {
    "contains": "inside",
    "inside": "contains",
    "overlap": "overlap",
    "disconnected": "disconnected",
    "east_of": "west_of",
    "west_of": "east_of",
}
SPATIAL_COMPOSITION = {
    ("contains", "contains"): ("contains",),
    ("east_of", "east_of"): ("east_of",),
}

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Traceability for the Exp5343 deterministic QSTR temporal/spatial "
        "constraint fixture."
    ),
    "milestone": (
        "Milestone accountability for the V487 qualitative temporal/spatial "
        "fixture gate."
    ),
    "status": "Machine-readable terminal state for downstream QSTR constraint gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether "
        "the deterministic QSTR fixture can gate downstream model reasoning."
    ),
    "inference_substrate": (
        "Declares deterministic_qstr_constraint_fixture so the artifact is read as "
        "exact typed temporal/spatial constraint checking, not live model quality."
    ),
    "tests_run": (
        "Commands run to validate the QSTR module, artifact schema, new-code "
        "coverage, and repository tests."
    ),
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "calculus_count",
    "composition_case_count",
    "contradiction_case_count",
    "solver_authoritative",
    "false_accept_count",
    "failure_localization_rate",
    "qstr_fixture_ready",
    "tests_run",
)


@dataclass(frozen=True)
class Interval:
    """A closed-open temporal interval used by the Allen-style checker."""

    entity_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle used for the small RCC/cardinal subset."""

    entity_id: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class RelationCase:
    """One claimed qualitative relation and its expected satisfiability label."""

    case_id: str
    calculus: str
    case_type: str
    subject: str
    object: str
    allowed_relations: tuple[str, ...]
    expected_satisfiable: bool
    expected_failure_ids: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class CompositionCase:
    """One A-B, B-C relation pair with a deterministic A-C outcome check."""

    case_id: str
    calculus: str
    subject: str
    via: str
    object: str
    relation_ab: str
    relation_bc: str
    expected_relations_ac: tuple[str, ...]


@dataclass(frozen=True)
class QSTRFixture:
    """The full typed fixture consumed by the deterministic checker."""

    intervals: dict[str, Interval]
    boxes: dict[str, Box]
    relation_cases: tuple[RelationCase, ...]
    composition_cases: tuple[CompositionCase, ...]


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def temporal_relation(left: Interval, right: Interval) -> str:
    """Return the exact Allen relation between two valid intervals."""

    if left.end < right.start:
        return "before"
    if left.end == right.start:
        return "meets"
    if left.start < right.start < left.end < right.end:
        return "overlaps"
    if left.start < right.start and left.end == right.end:
        return "finished_by"
    if left.start < right.start and left.end > right.end:
        return "contains"
    if left.start == right.start and left.end < right.end:
        return "starts"
    if left.start == right.start and left.end == right.end:
        return "equals"
    if left.start == right.start and left.end > right.end:
        return "started_by"
    if left.start > right.start and left.end < right.end:
        return "during"
    if left.start > right.start and left.end == right.end:
        return "finishes"
    if right.start < left.start < right.end < left.end:
        return "overlapped_by"
    if left.start == right.end:
        return "met_by"
    return "after"


def _interval_universe(limit: int = 7) -> tuple[Interval, ...]:
    return tuple(
        Interval(f"i-{start}-{end}", start, end)
        for start in range(limit)
        for end in range(start + 1, limit + 1)
    )


@lru_cache(maxsize=None)
def _compose_temporal(first: str, second: str) -> tuple[str, ...]:
    outcomes: set[str] = set()
    intervals = _interval_universe()
    for left in intervals:
        for middle in intervals:
            if temporal_relation(left, middle) != first:
                continue
            for right in intervals:
                if temporal_relation(middle, right) == second:
                    outcomes.add(temporal_relation(left, right))
    return tuple(relation for relation in TEMPORAL_RELATION_ORDER if relation in outcomes)


def spatial_relations(left: Box, right: Box) -> tuple[str, ...]:
    """Return exact rectangle relations from the selected RCC/cardinal subset."""

    relations: set[str] = set()
    separated = (
        left.x_max <= right.x_min
        or right.x_max <= left.x_min
        or left.y_max <= right.y_min
        or right.y_max <= left.y_min
    )
    if separated:
        relations.add("disconnected")
    else:
        left_contains = (
            left.x_min < right.x_min
            and left.y_min < right.y_min
            and left.x_max > right.x_max
            and left.y_max > right.y_max
        )
        right_contains = (
            right.x_min < left.x_min
            and right.y_min < left.y_min
            and right.x_max > left.x_max
            and right.y_max > left.y_max
        )
        if left_contains:
            relations.add("contains")
        elif right_contains:
            relations.add("inside")
        else:
            relations.add("overlap")
    if left.x_min > right.x_max:
        relations.add("east_of")
    if left.x_max < right.x_min:
        relations.add("west_of")
    return tuple(relation for relation in SPATIAL_RELATION_ORDER if relation in relations)


def compose_relations(calculus: str, first: str, second: str) -> tuple[str, ...]:
    """Return deterministic composition outcomes for the selected calculus."""

    if calculus == TEMPORAL:
        return _compose_temporal(first, second)
    if calculus == SPATIAL:
        return SPATIAL_COMPOSITION[(first, second)]
    raise ValueError(f"unknown calculus: {calculus}")  # pragma: no cover


def relation_converse(calculus: str, relation: str) -> str:
    """Return the converse relation used to verify B-A from an A-B claim."""

    if calculus == TEMPORAL:
        return TEMPORAL_CONVERSE[relation]
    if calculus == SPATIAL:
        return SPATIAL_CONVERSE[relation]
    raise ValueError(f"unknown calculus: {calculus}")  # pragma: no cover


def build_fixture() -> QSTRFixture:
    """Assemble the tiny exact QSTR fixture for Exp5343."""

    intervals = {
        "tb-a": Interval("tb-a", 0, 1),
        "tb-b": Interval("tb-b", 3, 4),
        "to-a": Interval("to-a", 0, 3),
        "to-b": Interval("to-b", 2, 5),
        "td-a": Interval("td-a", 2, 4),
        "td-b": Interval("td-b", 1, 5),
        "tm-a": Interval("tm-a", 0, 2),
        "tm-b": Interval("tm-b", 2, 4),
        "ta-a": Interval("ta-a", 0, 2),
        "ta-b": Interval("ta-b", 2, 5),
        "tcm-a": Interval("tcm-a", 0, 1),
        "tcm-b": Interval("tcm-b", 3, 4),
        "tcm-c": Interval("tcm-c", 4, 6),
        "tmo-a": Interval("tmo-a", 0, 2),
        "tmo-b": Interval("tmo-b", 2, 5),
        "tmo-c": Interval("tmo-c", 4, 7),
        "tod-a": Interval("tod-a", 0, 4),
        "tod-b": Interval("tod-b", 2, 5),
        "tod-c": Interval("tod-c", 1, 8),
        "tdb-a": Interval("tdb-a", 2, 4),
        "tdb-b": Interval("tdb-b", 1, 5),
        "tdb-c": Interval("tdb-c", 8, 9),
    }
    boxes = {
        "sd-a": Box("sd-a", 0, 0, 1, 1),
        "sd-b": Box("sd-b", 3, 0, 4, 1),
        "so-a": Box("so-a", 0, 0, 3, 3),
        "so-b": Box("so-b", 2, 2, 5, 5),
        "sc-a": Box("sc-a", 0, 0, 6, 6),
        "sc-b": Box("sc-b", 1, 1, 2, 2),
        "se-a": Box("se-a", 6, 0, 7, 1),
        "se-b": Box("se-b", 3, 0, 4, 1),
        "sf-a": Box("sf-a", 0, 0, 1, 1),
        "sf-b": Box("sf-b", 3, 0, 4, 1),
        "scc-a": Box("scc-a", 0, 0, 10, 10),
        "scc-b": Box("scc-b", 2, 2, 8, 8),
        "scc-c": Box("scc-c", 3, 3, 4, 4),
        "see-a": Box("see-a", 6, 0, 7, 1),
        "see-b": Box("see-b", 3, 0, 4, 1),
        "see-c": Box("see-c", 0, 0, 1, 1),
    }
    relation_cases = (
        RelationCase("t-before", TEMPORAL, "before", "tb-a", "tb-b", ("before",), True),
        RelationCase("t-overlaps", TEMPORAL, "overlaps", "to-a", "to-b", ("overlaps",), True),
        RelationCase("t-during", TEMPORAL, "during", "td-a", "td-b", ("during",), True),
        RelationCase("t-meets", TEMPORAL, "meets", "tm-a", "tm-b", ("meets",), True),
        RelationCase(
            "t-contradiction-before-vs-meets",
            TEMPORAL,
            "contradiction",
            "tm-a",
            "tm-b",
            ("before",),
            False,
            ("t-contradiction-before-vs-meets:claim-before",),
        ),
        RelationCase(
            "t-ambiguous-before-or-meets",
            TEMPORAL,
            "ambiguous",
            "ta-a",
            "ta-b",
            ("before", "meets"),
            True,
        ),
        RelationCase(
            "s-disconnected",
            SPATIAL,
            "disconnected",
            "sd-a",
            "sd-b",
            ("disconnected",),
            True,
        ),
        RelationCase("s-overlap", SPATIAL, "overlap", "so-a", "so-b", ("overlap",), True),
        RelationCase(
            "s-containment",
            SPATIAL,
            "containment",
            "sc-a",
            "sc-b",
            ("contains",),
            True,
        ),
        RelationCase(
            "s-east-of",
            SPATIAL,
            "cardinal_direction",
            "se-a",
            "se-b",
            ("east_of",),
            True,
        ),
        RelationCase(
            "s-contradiction-contains-vs-disconnected",
            SPATIAL,
            "contradiction",
            "sf-a",
            "sf-b",
            ("contains",),
            False,
            ("s-contradiction-contains-vs-disconnected:claim-contains",),
        ),
    )
    composition_cases = (
        CompositionCase(
            "tc-before-meets",
            TEMPORAL,
            "tcm-a",
            "tcm-b",
            "tcm-c",
            "before",
            "meets",
            ("before",),
        ),
        CompositionCase(
            "tc-meets-overlaps",
            TEMPORAL,
            "tmo-a",
            "tmo-b",
            "tmo-c",
            "meets",
            "overlaps",
            ("before",),
        ),
        CompositionCase(
            "tc-overlaps-during",
            TEMPORAL,
            "tod-a",
            "tod-b",
            "tod-c",
            "overlaps",
            "during",
            ("overlaps",),
        ),
        CompositionCase(
            "tc-during-before",
            TEMPORAL,
            "tdb-a",
            "tdb-b",
            "tdb-c",
            "during",
            "before",
            ("before",),
        ),
        CompositionCase(
            "sc-contains-contains",
            SPATIAL,
            "scc-a",
            "scc-b",
            "scc-c",
            "contains",
            "contains",
            ("contains",),
        ),
        CompositionCase(
            "sc-east-east",
            SPATIAL,
            "see-a",
            "see-b",
            "see-c",
            "east_of",
            "east_of",
            ("east_of",),
        ),
    )
    return QSTRFixture(intervals, boxes, relation_cases, composition_cases)


def _relation_set(fixture: QSTRFixture, calculus: str, subject: str, object: str) -> tuple[str, ...]:
    if calculus == TEMPORAL:
        return (temporal_relation(fixture.intervals[subject], fixture.intervals[object]),)
    if calculus == SPATIAL:
        return spatial_relations(fixture.boxes[subject], fixture.boxes[object])
    raise ValueError(f"unknown calculus: {calculus}")  # pragma: no cover


def _ordered_relations(calculus: str, relations: set[str] | tuple[str, ...]) -> tuple[str, ...]:
    order = TEMPORAL_RELATION_ORDER if calculus == TEMPORAL else SPATIAL_RELATION_ORDER
    return tuple(relation for relation in order if relation in relations)


def _select_actual_relation(
    calculus: str,
    actual_relations: tuple[str, ...],
    allowed_relations: tuple[str, ...],
) -> str:
    accepted = [relation for relation in allowed_relations if relation in actual_relations]
    if accepted:
        return accepted[0]
    return _ordered_relations(calculus, actual_relations)[0]


def _case_type_counts(cases: tuple[RelationCase, ...], calculus: str) -> dict[str, int]:
    required = REQUIRED_TEMPORAL_CASE_TYPES if calculus == TEMPORAL else REQUIRED_SPATIAL_CASE_TYPES
    counts = Counter(case.case_type for case in cases if case.calculus == calculus)
    return {case_type: counts.get(case_type, 0) for case_type in required}


def _evaluate_relation_case(fixture: QSTRFixture, case: RelationCase) -> JsonDict:
    actual_relations = _relation_set(fixture, case.calculus, case.subject, case.object)
    accepted = bool(set(case.allowed_relations).intersection(actual_relations))
    actual_relation = _select_actual_relation(case.calculus, actual_relations, case.allowed_relations)
    converse_relation = relation_converse(case.calculus, actual_relation)
    converse_relations = _relation_set(fixture, case.calculus, case.object, case.subject)
    claimed_relation = (
        case.allowed_relations[0]
        if len(case.allowed_relations) == 1
        else "one_of:" + ",".join(case.allowed_relations)
    )
    violation_ids = (
        []
        if accepted
        else [f"{case.case_id}:claim-{case.allowed_relations[0]}"]
    )
    actual_label = "satisfiable" if accepted else "unsatisfiable"
    expected_label = "satisfiable" if case.expected_satisfiable else "unsatisfiable"
    localized_failure = accepted or tuple(violation_ids) == case.expected_failure_ids
    return {
        "case_id": case.case_id,
        "calculus": case.calculus,
        "case_type": case.case_type,
        "subject": case.subject,
        "object": case.object,
        "claimed_relation": claimed_relation,
        "allowed_relations": list(case.allowed_relations),
        "ambiguous": len(case.allowed_relations) > 1,
        "actual_relation": actual_relation,
        "actual_relations": list(actual_relations),
        "expected_satisfiable": case.expected_satisfiable,
        "accepted": accepted,
        "expected_label": expected_label,
        "actual_label": actual_label,
        "label_matches_expected": actual_label == expected_label,
        "converse_relation": converse_relation,
        "actual_converse_relations": list(converse_relations),
        "converse_valid": converse_relation in converse_relations,
        "expected_failure_ids": list(case.expected_failure_ids),
        "violation_ids": violation_ids,
        "localized_failure": localized_failure,
    }


def _evaluate_composition_case(fixture: QSTRFixture, case: CompositionCase) -> JsonDict:
    actual_ab = _relation_set(fixture, case.calculus, case.subject, case.via)
    actual_bc = _relation_set(fixture, case.calculus, case.via, case.object)
    actual_ac = _relation_set(fixture, case.calculus, case.subject, case.object)
    possible = compose_relations(case.calculus, case.relation_ab, case.relation_bc)
    selected_ac = _select_actual_relation(case.calculus, actual_ac, case.expected_relations_ac)
    accepted = (
        case.relation_ab in actual_ab
        and case.relation_bc in actual_bc
        and selected_ac in possible
        and selected_ac in case.expected_relations_ac
    )
    source = (
        TEMPORAL_COMPOSITION_SOURCE
        if case.calculus == TEMPORAL
        else SPATIAL_COMPOSITION_SOURCE
    )
    return {
        "case_id": case.case_id,
        "calculus": case.calculus,
        "relation_ab": case.relation_ab,
        "relation_bc": case.relation_bc,
        "actual_relations_ab": list(actual_ab),
        "actual_relations_bc": list(actual_bc),
        "actual_relation_ac": selected_ac,
        "actual_relations_ac": list(actual_ac),
        "possible_composed_relations": list(possible),
        "expected_relations_ac": list(case.expected_relations_ac),
        "composition_source": source,
        "accepted": accepted,
        "violation_ids": [] if accepted else [f"{case.case_id}:composition"],
    }


def _rate(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _all_case_types_present(counts: dict[str, int]) -> bool:
    return all(count == 1 for count in counts.values())


def evaluate_fixture(fixture: QSTRFixture) -> JsonDict:
    """Evaluate relation, composition, converse, and localization checks."""

    relation_results = [_evaluate_relation_case(fixture, case) for case in fixture.relation_cases]
    composition_results = [
        _evaluate_composition_case(fixture, case) for case in fixture.composition_cases
    ]
    temporal_counts = _case_type_counts(fixture.relation_cases, TEMPORAL)
    spatial_counts = _case_type_counts(fixture.relation_cases, SPATIAL)
    contradiction_case_count = sum(
        1 for row in relation_results if row["expected_satisfiable"] is False
    )
    false_accept_count = sum(
        1
        for row in relation_results
        if row["expected_satisfiable"] is False and row["accepted"] is True
    )
    invalid_rows = [
        row for row in relation_results if row["expected_satisfiable"] is False
    ]
    localized_count = sum(1 for row in invalid_rows if row["localized_failure"] is True)
    failure_localization_rate = _rate(localized_count, len(invalid_rows))
    relation_checks_passed = all(
        row["label_matches_expected"] and row["converse_valid"] and row["localized_failure"]
        for row in relation_results
    )
    composition_checks_passed = all(row["accepted"] for row in composition_results)
    case_families_present = _all_case_types_present(temporal_counts) and _all_case_types_present(
        spatial_counts
    )
    deterministic_checks_passed = (
        case_families_present
        and relation_checks_passed
        and composition_checks_passed
        and false_accept_count == 0
        and failure_localization_rate == 1.0
    )
    return {
        "relation_results": relation_results,
        "composition_results": composition_results,
        "temporal_case_type_counts": temporal_counts,
        "spatial_case_type_counts": spatial_counts,
        "calculus_count": len({TEMPORAL, SPATIAL}),
        "relation_case_count": len(relation_results),
        "composition_case_count": len(composition_results),
        "contradiction_case_count": contradiction_case_count,
        "solver_authoritative": True,
        "false_accept_count": false_accept_count,
        "failure_localization_rate": failure_localization_rate,
        "deterministic_checks_passed": deterministic_checks_passed,
    }


def _readiness_blockers(evaluation: JsonDict, tests_run: list[JsonDict]) -> list[str]:
    return [
        blocker
        for failed, blocker in (
            (not _all_case_types_present(evaluation["temporal_case_type_counts"]), "temporal_case_family_missing"),
            (not _all_case_types_present(evaluation["spatial_case_type_counts"]), "spatial_case_family_missing"),
            (not all(row["label_matches_expected"] for row in evaluation["relation_results"]), "relation_label_mismatch"),
            (not all(row["converse_valid"] for row in evaluation["relation_results"]), "converse_check_failed"),
            (not all(row["accepted"] for row in evaluation["composition_results"]), "composition_check_failed"),
            (evaluation["false_accept_count"] != 0, "false_accept_count_nonzero"),
            (evaluation["failure_localization_rate"] != 1.0, "failure_localization_incomplete"),
            (not tests_run, "tests_not_recorded"),
        )
        if failed
    ]


def build_artifact(fixture: QSTRFixture, *, tests_run: list[JsonDict]) -> JsonDict:
    """Build the result artifact from the deterministic QSTR fixture."""

    evaluation = evaluate_fixture(fixture)
    blockers = _readiness_blockers(evaluation, tests_run)
    ready = bool(evaluation["deterministic_checks_passed"] and tests_run and not blockers)
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NUMBER,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_NAME),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap(
            "status",
            "qstr_fixture_ready" if ready else "blocked_qstr_fixture_not_ready",
        ),
        "honest_verdict": _wrap(
            "honest_verdict",
            "complete: deterministic QSTR fixture ready for downstream model reasoning"
            if ready
            else "blocked_qstr_fixture_not_ready",
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "calculus_count": evaluation["calculus_count"],
        "composition_case_count": evaluation["composition_case_count"],
        "contradiction_case_count": evaluation["contradiction_case_count"],
        "solver_authoritative": evaluation["solver_authoritative"],
        "false_accept_count": evaluation["false_accept_count"],
        "failure_localization_rate": evaluation["failure_localization_rate"],
        "qstr_fixture_ready": ready,
        "readiness_blockers": blockers,
        "relation_case_count": evaluation["relation_case_count"],
        "temporal_case_type_counts": evaluation["temporal_case_type_counts"],
        "spatial_case_type_counts": evaluation["spatial_case_type_counts"],
        "relation_results": evaluation["relation_results"],
        "composition_results": evaluation["composition_results"],
        "fixture_checksum": _stable_json(
            {
                "relation_results": evaluation["relation_results"],
                "composition_results": evaluation["composition_results"],
            }
        ),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "tests_run": _wrap("tests_run", tests_run),
    }
    validate_artifact(artifact)
    return artifact


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def _is_bare_bool(value: Any) -> bool:
    return type(value) is bool


def _is_bare_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the schema fields that downstream QSTR gates depend on."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        _require(isinstance(artifact[field], dict), f"{field} must be principle wrapped")
        _require(artifact[field].get("principle") == FIELD_PRINCIPLES[field], field)
        _require("value" in artifact[field], f"{field} missing value")
    _require(artifact["honest_verdict"]["value"].startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, "inference_substrate")
    for field in ("calculus_count", "composition_case_count", "contradiction_case_count", "false_accept_count"):
        _require(_is_bare_int(artifact[field]), field)
    _require(artifact["solver_authoritative"] is True, "solver_authoritative")
    _require(_is_bare_numeric(artifact["failure_localization_rate"]), "failure_localization_rate")
    _require(_is_bare_bool(artifact["qstr_fixture_ready"]), "qstr_fixture_ready")
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run")
    if artifact["qstr_fixture_ready"]:
        _require(artifact["status"]["value"] == "qstr_fixture_ready", "status")
        _require(artifact["false_accept_count"] == 0, "false_accept_count")
        _require(artifact["failure_localization_rate"] == 1.0, "failure_localization_rate")
        _require(bool(artifact["tests_run"]["value"]), "tests_run")


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: list[JsonDict] | None = None,
) -> JsonDict:
    """Run the offline QSTR fixture evaluation and write the result artifact."""

    artifact = build_artifact(build_fixture(), tests_run=[] if tests_run is None else tests_run)
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
