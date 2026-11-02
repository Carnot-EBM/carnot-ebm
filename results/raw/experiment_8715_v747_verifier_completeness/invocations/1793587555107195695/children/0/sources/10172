#!/usr/bin/env python3
"""Generate the reusable LTLZinc temporal non-forgetting benchmark.

The file written by this script is a deterministic JSON dataset, not an
experiment artifact.  It packages the finite-trace temporal cases from the
LTLZinc adapter together with source-template provenance and retention metadata
so later evaluation code can replay anchor cases after update cases.

Spec: REQ-LEARN-1630-6, REQ-LEARN-1630-7.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carnot.reporting import ltlzinc_temporal_continual_learning_adapter as temporal
from scripts import experiment_1630_ltlzinc as retention


JsonDict = dict[str, Any]

DEFAULT_TEMPLATE_PATH = REPO_ROOT / "data" / "constraint_templates.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "ltlzinc_benchmark.json"

SCHEMA = "carnot.ltlzinc_temporal_nonforgetting_benchmark.v1"
BENCHMARK_ID = "ltlzinc_temporal_nonforgetting_v1"
GENERATOR = "scripts/generate_ltlzinc.py"
RUN_DATE = "20260509"
VERIFIER_PATH = (
    "carnot.reporting.ltlzinc_temporal_continual_learning_adapter.verify_temporal_case"
)
SUPPORTED_OPERATORS = temporal.SUPPORTED_OPERATORS
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema",
    "benchmark_id",
    "generator",
    "run_date",
    "spec",
    "source",
    "case_count",
    "anchor_case_count",
    "update_case_count",
    "sat_case_count",
    "repair_hint_case_count",
    "supported_operators",
    "cases",
)
REQUIRED_CASE_FIELDS = (
    "case_id",
    "source_template",
    "nonforgetting_phase",
    "constraint_family",
    "temporal_operator",
    "signal",
    "guard_signal",
    "ltl_formula",
    "minizinc_constraint",
    "trace",
    "expected_satisfied",
    "label",
    "certificate_state",
    "dvi_label",
    "fr11_memory_hint",
    "evaluation",
    "retention",
    "tags",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_constraint_templates(path: Path | str = DEFAULT_TEMPLATE_PATH) -> list[JsonDict]:
    """REQ-LEARN-1630-6: load source template rows used as benchmark provenance."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    templates = payload.get("templates", [])
    _require(isinstance(templates, list) and len(templates) > 0, "templates must be non-empty")
    return [dict(template) for template in templates]


def _template_source(template: Mapping[str, Any]) -> JsonDict:
    return {
        "template_id": str(template["template_id"]),
        "violation_type": str(template.get("violation_type", "")),
        "source_experiment": int(template.get("source_experiment", 0)),
        "violation_pattern_excerpt": str(template.get("violation_pattern", "")),
    }


def _case_source_template(templates: Sequence[Mapping[str, Any]], index: int) -> JsonDict:
    return _template_source(templates[index % len(templates)])


def _retention_metadata(case_id: str, phase: str) -> JsonDict:
    return {
        "phase": phase,
        "anchor_case_id": case_id,
        "must_retrieve_after_updates": phase == "anchor",
        "nonforgetting_check": (
            "retrieve_same_case_after_update_rows" if phase == "anchor" else "post_anchor_update"
        ),
    }


def _adapt_case(
    case: Mapping[str, Any],
    *,
    source_template: Mapping[str, Any],
    phase: str,
) -> JsonDict:
    expected_satisfied = bool(case["expected_satisfied"])
    adapted = dict(case)
    adapted["source_template"] = dict(source_template)
    adapted["nonforgetting_phase"] = phase
    adapted["evaluation"] = {
        "verifier_path": VERIFIER_PATH,
        "expected_verifier_result": expected_satisfied,
    }
    adapted["retention"] = _retention_metadata(str(case["case_id"]), phase)
    adapted["tags"] = [
        "ltlzinc",
        "temporal",
        "nonforgetting",
        f"operator:{case['temporal_operator']}",
        f"phase:{phase}",
    ]
    return adapted


def _benchmark_cases(templates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    anchor_cases = temporal.generate_temporal_cases()
    update_cases = retention.generate_update_cases()
    cases: list[JsonDict] = []
    for index, case in enumerate(anchor_cases):
        cases.append(
            _adapt_case(
                case,
                source_template=_case_source_template(templates, index),
                phase="anchor",
            )
        )
    for offset, case in enumerate(update_cases, start=len(anchor_cases)):
        cases.append(
            _adapt_case(
                case,
                source_template=_case_source_template(templates, offset),
                phase="update",
            )
        )
    return cases


def validate_case_schema(case: Mapping[str, Any]) -> None:
    """REQ-LEARN-1630-7: enforce one reusable benchmark case contract."""

    missing = sorted(set(REQUIRED_CASE_FIELDS).difference(case))
    _require(not missing, f"missing benchmark case fields: {missing}")
    temporal.validate_case_schema(case)
    _require(case["nonforgetting_phase"] in {"anchor", "update"}, "unsupported phase")
    _require(isinstance(case["source_template"], Mapping), "source_template must be a mapping")
    _require(isinstance(case["evaluation"], Mapping), "evaluation must be a mapping")
    _require(isinstance(case["retention"], Mapping), "retention must be a mapping")
    _require(case["evaluation"].get("verifier_path") == VERIFIER_PATH, "unsupported verifier")
    _require(
        bool(case["evaluation"].get("expected_verifier_result"))
        is bool(case["expected_satisfied"]),
        "expected verifier result must match expected_satisfied",
    )
    _require(
        temporal.verify_temporal_case(case) is bool(case["expected_satisfied"]),
        "temporal verifier disagrees with expected label",
    )


def _count_cases(cases: Sequence[Mapping[str, Any]], key: str, value: Any) -> int:
    return sum(1 for case in cases if case.get(key) == value)


def _source_summary(
    *,
    templates: Sequence[Mapping[str, Any]],
    template_path: Path,
) -> JsonDict:
    template_ids = [str(template["template_id"]) for template in templates]
    source_experiments = sorted(
        {int(template.get("source_experiment", 0)) for template in templates}
    )
    return {
        "constraint_templates_path": _repo_relative(template_path),
        "template_count": len(templates),
        "template_ids": template_ids,
        "source_experiments": source_experiments,
    }


def build_benchmark(template_path: Path | str = DEFAULT_TEMPLATE_PATH) -> JsonDict:
    """REQ-LEARN-1630-6: build the full deterministic benchmark payload."""

    source_path = Path(template_path)
    templates = load_constraint_templates(source_path)
    cases = _benchmark_cases(templates)
    certificate_counts = Counter(str(case["certificate_state"]) for case in cases)
    payload: JsonDict = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "generator": GENERATOR,
        "run_date": RUN_DATE,
        "spec": ["REQ-LEARN-1630-6", "REQ-LEARN-1630-7", "SCENARIO-LEARN-1630"],
        "source": _source_summary(templates=templates, template_path=source_path),
        "case_count": len(cases),
        "anchor_case_count": _count_cases(cases, "nonforgetting_phase", "anchor"),
        "update_case_count": _count_cases(cases, "nonforgetting_phase", "update"),
        "sat_case_count": int(certificate_counts.get("SAT", 0)),
        "repair_hint_case_count": int(certificate_counts.get("REPAIR_HINT", 0)),
        "supported_operators": list(SUPPORTED_OPERATORS),
        "cases": cases,
    }
    validate_benchmark(payload)
    return payload


def validate_benchmark(payload: Mapping[str, Any]) -> None:
    """REQ-LEARN-1630-6: enforce top-level count and schema invariants."""

    missing = sorted(set(REQUIRED_TOP_LEVEL_FIELDS).difference(payload))
    _require(not missing, f"missing benchmark fields: {missing}")
    _require(payload["schema"] == SCHEMA, "unsupported schema")
    cases = payload["cases"]
    _require(isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)), "cases invalid")
    for case in cases:
        validate_case_schema(case)
    case_ids = [str(case["case_id"]) for case in cases]
    _require(len(set(case_ids)) == len(case_ids), "case_id values must be unique")
    _require(payload["case_count"] == len(cases), "case_count must match cases")
    _require(
        payload["anchor_case_count"] == _count_cases(cases, "nonforgetting_phase", "anchor"),
        "anchor_case_count must match cases",
    )
    _require(
        payload["update_case_count"] == _count_cases(cases, "nonforgetting_phase", "update"),
        "update_case_count must match cases",
    )
    _require(
        payload["sat_case_count"] == _count_cases(cases, "certificate_state", "SAT"),
        "sat_case_count must match cases",
    )
    _require(
        payload["repair_hint_case_count"]
        == _count_cases(cases, "certificate_state", "REPAIR_HINT"),
        "repair_hint_case_count must match cases",
    )
    _require(
        set(payload["supported_operators"]) == set(SUPPORTED_OPERATORS),
        "supported_operators mismatch",
    )


def write_benchmark(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    template_path: Path | str = DEFAULT_TEMPLATE_PATH,
) -> JsonDict:
    """REQ-LEARN-1630-6: write `data/ltlzinc_benchmark.json` deterministically."""

    benchmark = build_benchmark(template_path=template_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(benchmark, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return benchmark


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = list(sys.argv[1:] if argv is None else argv)
    output_path = Path(args[0]) if args else DEFAULT_OUTPUT_PATH
    benchmark = write_benchmark(output_path=output_path)
    print(json.dumps(benchmark, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
