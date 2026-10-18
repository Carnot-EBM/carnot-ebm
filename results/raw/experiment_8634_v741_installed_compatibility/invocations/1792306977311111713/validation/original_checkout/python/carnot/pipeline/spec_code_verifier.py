"""Spec-aware verification for generated Python code.

Spec: REQ-CODE-025, REQ-CODE-026, REQ-CODE-027,
SCENARIO-CODE-022, SCENARIO-CODE-023, SCENARIO-CODE-024,
SCENARIO-CODE-025
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from carnot.pipeline.code_learning import PropertyScore, RepairStrategy, TraceAnalyzer
from carnot.pipeline.code_spec_corpus import CORPUS_PATH
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.humaneval_live_benchmark import HarnessResult, execute_humaneval
from carnot.pipeline.pbt_code_verifier import PBTCodeVerificationResult, PBTCodeVerifier
from carnot.pipeline.property_code_verifier import (
    PropertyCodeVerifier,
    _build_sample_inputs,
    _extract_signature,
    extract_official_test_examples,
    extract_prompt_examples,
)
from carnot.verify.python_types import safe_exec_function

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEARNING_ARTIFACT_PATHS = (
    REPO_ROOT / "results" / "experiment_225_results.json",
    REPO_ROOT / "results" / "experiment_226_results.json",
    REPO_ROOT / "results" / "experiment_227_results.json",
)
_PROPERTY_BY_KIND = {
    "sorted_output": "sorted_output",
    "reverse_output": "reverse_output",
    "input_immutability": "input_immutability",
    "deterministic": "deterministic",
    "no_exception": "no_exception",
    "typed_output": "annotated_return_type",
    "example_consistency": "example_consistency",
}
_PROPERTY_FROM_VERIFIER = {
    "example_regression": "example_consistency",
    "annotated_return_type": "annotated_return_type",
    "deterministic": "deterministic",
    "input_immutability": "input_immutability",
    "sorted_output": "sorted_output",
    "reverse_output": "reverse_output",
}
_CLAUSE_PROMPTS = {
    "sorted_output": "Return the input sorted in ascending order.",
    "reverse_output": "Return the input in reverse order.",
    "input_immutability": "Do not mutate caller-owned inputs.",
}


@dataclass(frozen=True)
class ExplicitSpecClause:
    """One explicit clause from the checked-in code-spec corpus."""

    family: str
    kind: str
    text: str
    sources: tuple[str, ...]
    trace_refs: tuple[str, ...]


SpecClause = ExplicitSpecClause


@dataclass(frozen=True)
class CodeSpecRow:
    """One explicit code-spec corpus row."""

    task_id: str
    case_id: str
    row_id: str
    run_date: str
    schema_version: str
    entry_point: str
    signature: str
    preconditions: tuple[ExplicitSpecClause, ...]
    postconditions: tuple[ExplicitSpecClause, ...]
    invariants: tuple[ExplicitSpecClause, ...]
    mutation_constraints: tuple[ExplicitSpecClause, ...]
    oracle_hints: tuple[ExplicitSpecClause, ...]
    source_traces: tuple[dict[str, Any], ...]
    trace_summary: dict[str, Any]

    @property
    def clause_families(self) -> tuple[tuple[ExplicitSpecClause, ...], ...]:
        return (
            self.preconditions,
            self.postconditions,
            self.invariants,
            self.mutation_constraints,
            self.oracle_hints,
        )

    def clause_for_kind(self, kind: str) -> ExplicitSpecClause | None:
        for family in self.clause_families:
            for clause in family:
                if clause.kind == kind:
                    return clause
        return None


@dataclass(frozen=True)
class SpecClauseResult:
    """Verification status for one explicit spec clause."""

    family: str
    kind: str
    text: str
    status: str
    checked_by: str
    detail: str
    sources: tuple[str, ...]
    trace_refs: tuple[str, ...]
    matched_properties: tuple[str, ...] = ()

    def to_constraint_result(self) -> ConstraintResult | None:
        if self.status != "violated":
            return None
        return ConstraintResult(
            constraint_type="spec_code",
            description=f"{self.kind} ({self.family}) failed: {self.detail}",
            metadata={
                "family": self.family,
                "kind": self.kind,
                "text": self.text,
                "status": self.status,
                "checked_by": self.checked_by,
                "detail": self.detail,
                "sources": self.sources,
                "trace_refs": self.trace_refs,
                "matched_properties": self.matched_properties,
                "satisfied": False,
            },
        )


ExplicitSpecFailure = SpecClauseResult


@dataclass(frozen=True)
class RepairHint:
    """One ranked repair recommendation."""

    strategy_name: str
    error_family: str
    score: float
    success_rate: float
    attempts: int
    partial_recoveries: int
    supporting_properties: tuple[str, ...]
    support_case_ids: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "error_family": self.error_family,
            "score": self.score,
            "success_rate": self.success_rate,
            "attempts": self.attempts,
            "partial_recoveries": self.partial_recoveries,
            "supporting_properties": list(self.supporting_properties),
            "support_case_ids": list(self.support_case_ids),
            "rationale": self.rationale,
        }


RankedRepairHint = RepairHint


def _default_harness() -> HarnessResult:
    return HarnessResult(passed=True, error_type="disabled", error_message="", stdout="")


def _default_pbt() -> PBTCodeVerificationResult:
    return PBTCodeVerificationResult()


@dataclass
class SpecCodeVerificationResult:
    """Structured result for the spec-aware code verifier."""

    harness: HarnessResult
    pbt: PBTCodeVerificationResult = field(default_factory=_default_pbt)
    spec: CodeSpecRow | None = None
    spec_clause_results: tuple[SpecClauseResult, ...] = ()
    repair_hints: tuple[RepairHint, ...] = ()

    @property
    def verified(self) -> bool:
        return (
            self.harness.passed
            and self.pbt.verified
            and all(result.status != "violated" for result in self.spec_clause_results)
        )

    def to_constraint_results(self) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []
        if not self.harness.passed:
            results.append(
                ConstraintResult(
                    constraint_type="official_tests",
                    description=f"Official tests failed: {self.harness.error_message}",
                    metadata={
                        "passed": False,
                        "error_type": self.harness.error_type,
                        "error_message": self.harness.error_message,
                        "stdout": self.harness.stdout,
                        "satisfied": False,
                    },
                )
            )
        results.extend(self.pbt.to_constraint_results())
        for clause_result in self.spec_clause_results:
            constraint = clause_result.to_constraint_result()
            if constraint is not None:
                results.append(constraint)
        return results

    def to_certificate(self) -> dict[str, Any]:
        return {
            "official_test_summary": {
                "passed": self.harness.passed,
                "error_type": self.harness.error_type,
                "error_message": self.harness.error_message,
                "stdout": self.harness.stdout,
            },
            "pbt_summary": {
                "enabled": bool(
                    self.pbt.max_examples or self.pbt.derived_properties or self.pbt.failures
                ),
                "verified": self.pbt.verified,
                "n_properties": len(self.pbt.derived_properties),
                "n_failures": len(self.pbt.failures),
                "property_names": [prop.name for prop in self.pbt.derived_properties],
                "wall_clock_seconds": self.pbt.wall_clock_seconds,
            },
            "spec_summary": {
                "task_id": self.spec.task_id if self.spec is not None else "",
                "case_id": self.spec.case_id if self.spec is not None else "",
                "entry_point": self.spec.entry_point if self.spec is not None else "",
                "corpus_run_date": self.spec.run_date if self.spec is not None else "",
                "n_clause_results": len(self.spec_clause_results),
                "n_violations": sum(
                    1 for result in self.spec_clause_results if result.status == "violated"
                ),
            },
            "repair_ranking": {"hints": [hint.to_dict() for hint in self.repair_hints]},
        }


ExplicitSpecVerificationResult = SpecCodeVerificationResult
ExplicitCodeSpec = CodeSpecRow


def _parse_family(row: dict[str, Any], family: str) -> tuple[ExplicitSpecClause, ...]:
    raw = row.get(family)
    if not isinstance(raw, list):
        return ()
    clauses: list[ExplicitSpecClause] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clauses.append(
            ExplicitSpecClause(
                family=family,
                kind=str(item.get("kind") or ""),
                text=str(item.get("text") or ""),
                sources=tuple(str(source) for source in item.get("sources", []) if source),
                trace_refs=tuple(str(ref) for ref in item.get("trace_refs", []) if ref),
            )
        )
    return tuple(clauses)


def _row_from_json(row: dict[str, Any]) -> CodeSpecRow:
    return CodeSpecRow(
        task_id=str(row.get("task_id") or ""),
        case_id=str(row.get("case_id") or ""),
        row_id=str(row.get("row_id") or ""),
        run_date=str(row.get("run_date") or ""),
        schema_version=str(row.get("schema_version") or ""),
        entry_point=str(row.get("entry_point") or ""),
        signature=str(row.get("signature") or ""),
        preconditions=_parse_family(row, "preconditions"),
        postconditions=_parse_family(row, "postconditions"),
        invariants=_parse_family(row, "invariants"),
        mutation_constraints=_parse_family(row, "mutation_constraints"),
        oracle_hints=_parse_family(row, "oracle_hints"),
        source_traces=tuple(
            item for item in row.get("source_traces", []) if isinstance(item, dict)
        ),
        trace_summary=dict(row.get("trace_summary") or {}),
    )


def _load_rows(corpus_path: Path) -> tuple[CodeSpecRow, ...]:
    if not corpus_path.exists():
        return ()
    rows: list[CodeSpecRow] = []
    for raw_line in corpus_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(_row_from_json(payload))
    return tuple(rows)


def load_code_spec_row(
    *,
    task_id: str | None = None,
    case_id: str | None = None,
    entry_point: str | None = None,
    corpus_path: str | Path = CORPUS_PATH,
) -> CodeSpecRow | None:
    rows = _load_rows(Path(corpus_path))
    if task_id:
        for row in rows:
            if row.task_id == task_id:
                return row
    if case_id:
        for row in rows:
            if row.case_id == case_id:
                return row
    if entry_point:
        for row in rows:
            if row.entry_point == entry_point:
                return row
    return None


def _parse_example_hint(text: str, entry_point: str) -> tuple[tuple[Any, ...], Any] | None:
    call_text, separator, expected_text = text.partition(" -> ")
    if not separator:
        return None
    try:
        parsed = ast.parse(call_text, mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == entry_point
    ):
        return None
    try:
        args = tuple(ast.literal_eval(arg) for arg in call.args)
        expected = ast.literal_eval(expected_text)
    except (SyntaxError, ValueError):
        return None
    return (args, expected)


def _example_clause_result(
    clause: ExplicitSpecClause,
    *,
    code: str,
    entry_point: str,
) -> SpecClauseResult:
    parsed = _parse_example_hint(clause.text, entry_point)
    if parsed is None:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="not_checked",
            checked_by="example",
            detail="example hint not parseable",
            sources=clause.sources,
            trace_refs=clause.trace_refs,
            matched_properties=("example_consistency",),
        )
    args, expected = parsed
    actual_args = tuple(copy.deepcopy(arg) for arg in args)
    actual, error = safe_exec_function(code, entry_point, actual_args)
    if error is not None:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="violated",
            checked_by="example",
            detail=str(error),
            sources=clause.sources,
            trace_refs=clause.trace_refs,
            matched_properties=("example_consistency",),
        )
    if actual != expected:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="violated",
            checked_by="example",
            detail=f"expected {expected!r}; got {actual!r}",
            sources=clause.sources,
            trace_refs=clause.trace_refs,
            matched_properties=("example_consistency",),
        )
    return SpecClauseResult(
        family=clause.family,
        kind=clause.kind,
        text=clause.text,
        status="satisfied",
        checked_by="example",
        detail="",
        sources=clause.sources,
        trace_refs=clause.trace_refs,
        matched_properties=("example_consistency",),
    )


def _example_consistency_result(
    clause: ExplicitSpecClause,
    *,
    example_results: Sequence[SpecClauseResult],
) -> SpecClauseResult:
    if not example_results:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="not_checked",
            checked_by="example",
            detail="no example clauses matched",
            sources=clause.sources,
            trace_refs=clause.trace_refs,
            matched_properties=("example_consistency",),
        )
    first_violated = next((item for item in example_results if item.status == "violated"), None)
    if first_violated is not None:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="violated",
            checked_by="example",
            detail=first_violated.detail,
            sources=clause.sources,
            trace_refs=clause.trace_refs,
            matched_properties=("example_consistency",),
        )
    return SpecClauseResult(
        family=clause.family,
        kind=clause.kind,
        text=clause.text,
        status="satisfied",
        checked_by="example",
        detail="",
        sources=clause.sources,
        trace_refs=clause.trace_refs,
        matched_properties=("example_consistency",),
    )


def _property_clause_result(
    clause: ExplicitSpecClause,
    *,
    checked_properties: set[str],
    failed_properties: Sequence[str],
) -> SpecClauseResult:
    property_name = _PROPERTY_BY_KIND.get(clause.kind)
    if property_name is None:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="not_checked",
            checked_by="property",
            detail="no verifier property maps to this clause",
            sources=clause.sources,
            trace_refs=clause.trace_refs,
        )
    if property_name not in checked_properties:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="not_checked",
            checked_by="property",
            detail="property not checked for this candidate",
            sources=clause.sources,
            trace_refs=clause.trace_refs,
            matched_properties=(property_name,),
        )
    if property_name in failed_properties:
        return SpecClauseResult(
            family=clause.family,
            kind=clause.kind,
            text=clause.text,
            status="violated",
            checked_by="property",
            detail=f"{property_name} failed",
            sources=clause.sources,
            trace_refs=clause.trace_refs,
            matched_properties=(property_name,),
        )
    return SpecClauseResult(
        family=clause.family,
        kind=clause.kind,
        text=clause.text,
        status="satisfied",
        checked_by="property",
        detail="",
        sources=clause.sources,
        trace_refs=clause.trace_refs,
        matched_properties=(property_name,),
    )


def _unique_examples(
    examples: Iterable[tuple[tuple[Any, ...], Any]],
) -> list[tuple[tuple[Any, ...], Any]]:
    seen: set[str] = set()
    ordered: list[tuple[tuple[Any, ...], Any]] = []
    for args, expected in examples:
        key = f"{repr(args)}->{repr(expected)}"
        if key in seen:
            continue
        seen.add(key)
        ordered.append((args, expected))
    return ordered


def _build_spec_examples(spec: CodeSpecRow) -> list[tuple[tuple[Any, ...], Any]]:
    return _unique_examples(
        parsed
        for clause in spec.oracle_hints
        for parsed in [_parse_example_hint(clause.text, spec.entry_point)]
        if parsed is not None
    )


def _build_spec_prompt(spec: CodeSpecRow, examples: Sequence[tuple[tuple[Any, ...], Any]]) -> str:
    doc_lines: list[str] = []
    for clause in spec.postconditions + spec.invariants + spec.mutation_constraints:
        prompt_line = _CLAUSE_PROMPTS.get(clause.kind)
        if prompt_line and prompt_line not in doc_lines:
            doc_lines.append(prompt_line)
    if examples:
        doc_lines.append("")
        for args, expected in examples:
            call = f"{spec.entry_point}({', '.join(repr(arg) for arg in args)})"
            doc_lines.append(f">>> {call}")
            doc_lines.append(repr(expected))
    if doc_lines:
        rendered = "\n".join(f"    {line}" if line else "" for line in doc_lines)
        return f'def {spec.signature}:\n    """\n{rendered}\n    """\n'
    return f"def {spec.signature}:\n    pass\n"


def _build_spec_harness(spec: CodeSpecRow, examples: Sequence[tuple[tuple[Any, ...], Any]]) -> str:
    lines = ["def check(candidate):"]
    if not examples:
        lines.append("    return None")
        return "\n".join(lines) + "\n"
    for args, expected in examples:
        rendered_args = ", ".join(repr(arg) for arg in args)
        lines.append(f"    assert candidate({rendered_args}) == {repr(expected)}")
    return "\n".join(lines) + "\n"


def _sample_inputs(
    *,
    code: str,
    prompt: str,
    entry_point: str,
    spec_prompt: str,
    spec_harness: str,
) -> list[tuple[Any, ...]]:
    signature = _extract_signature(spec_prompt, entry_point) or _extract_signature(
        prompt, entry_point
    )
    if signature is None:
        signature = _extract_signature(code, entry_point)
    prompt_examples = extract_prompt_examples(spec_prompt, entry_point)
    official_examples = extract_official_test_examples(spec_harness)
    return _build_sample_inputs(signature, prompt_examples, official_examples, spec_prompt)


def _checked_properties_from_property_verifier(result: Any) -> set[str]:
    checked: set[str] = set()
    for prop in result.derived_properties:
        mapped = _PROPERTY_FROM_VERIFIER.get(prop.name)
        if mapped:
            checked.add(mapped)
    return checked


def _failed_properties_from_property_verifier(result: Any) -> set[str]:
    failed: set[str] = set()
    for failure in result.failures:
        mapped = _PROPERTY_FROM_VERIFIER.get(failure.property_name)
        if mapped:
            failed.add(mapped)
    return failed


@lru_cache(maxsize=8)
def _load_learning(paths: tuple[str, ...]) -> tuple[dict[str, PropertyScore], RepairStrategy]:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return ({}, RepairStrategy().fit(()))
    analyzer = TraceAnalyzer.from_paths(existing)
    analysis = analyzer.analyze()
    property_scores = {item.property_name: item for item in analysis.property_rankings}
    return (property_scores, RepairStrategy().fit(analyzer.cases))


class SpecCodeVerifier:
    """Combine official tests, PBT, and explicit checked-in specs."""

    def __init__(
        self,
        *,
        spec_corpus_path: str | Path | None = None,
        learning_artifact_paths: Sequence[str | Path] | None = None,
        include_official_tests: bool = True,
        include_pbt: bool = True,
    ) -> None:
        self._spec_corpus_path = Path(spec_corpus_path or CORPUS_PATH)
        self._learning_artifact_paths = tuple(
            Path(path) for path in (learning_artifact_paths or DEFAULT_LEARNING_ARTIFACT_PATHS)
        )
        self._include_official_tests = include_official_tests
        self._include_pbt = include_pbt
        self._pbt_verifier = PBTCodeVerifier()

    def verify(
        self,
        code: str,
        prompt: str,
        entry_point: str,
        official_tests: str,
        *,
        task_id: str | None = None,
        case_id: str | None = None,
    ) -> SpecCodeVerificationResult:
        harness = (
            execute_humaneval(
                code,
                {"prompt": prompt, "test": official_tests, "entry_point": entry_point},
                timeout=1.0,
            )
            if self._include_official_tests
            else _default_harness()
        )
        pbt = (
            self._pbt_verifier.verify(code, prompt, entry_point, official_tests)
            if self._include_pbt
            else _default_pbt()
        )
        spec = load_code_spec_row(
            task_id=task_id,
            case_id=case_id,
            entry_point=entry_point,
            corpus_path=self._spec_corpus_path,
        )
        clause_results = self._build_clause_results(
            code=code,
            prompt=prompt,
            entry_point=entry_point,
            spec=spec,
        )
        repair_hints = self._rank_repair_hints(
            harness=harness,
            pbt=pbt,
            clause_results=clause_results,
        )
        return SpecCodeVerificationResult(
            harness=harness,
            pbt=pbt,
            spec=spec,
            spec_clause_results=clause_results,
            repair_hints=repair_hints,
        )

    def _build_clause_results(
        self,
        *,
        code: str,
        prompt: str,
        entry_point: str,
        spec: CodeSpecRow | None,
    ) -> tuple[SpecClauseResult, ...]:
        if spec is None:
            return ()

        examples = _build_spec_examples(spec)
        spec_prompt = _build_spec_prompt(spec, examples)
        spec_harness = _build_spec_harness(spec, examples)
        property_result = PropertyCodeVerifier().verify(
            code,
            spec_prompt,
            entry_point,
            spec_harness,
        )
        checked_properties = _checked_properties_from_property_verifier(property_result)
        failed_properties = _failed_properties_from_property_verifier(property_result)

        if spec.clause_for_kind("no_exception") is not None:
            sample_inputs = _sample_inputs(
                code=code,
                prompt=prompt,
                entry_point=entry_point,
                spec_prompt=spec_prompt,
                spec_harness=spec_harness,
            )
            if sample_inputs:
                checked_properties.add("no_exception")
            for args in sample_inputs:
                _, error = safe_exec_function(
                    code,
                    entry_point,
                    tuple(copy.deepcopy(arg) for arg in args),
                )
                if error is not None:
                    failed_properties.add("no_exception")
                    break

        example_clause_results = [
            _example_clause_result(clause, code=code, entry_point=entry_point)
            for clause in spec.oracle_hints
            if clause.kind in {"prompt_example", "official_test_example"}
        ]

        results: list[SpecClauseResult] = []
        example_index = 0
        for family in spec.clause_families:
            for clause in family:
                if clause.kind in {"prompt_example", "official_test_example"}:
                    results.append(example_clause_results[example_index])
                    example_index += 1
                elif clause.kind == "official_test_miss_trace":
                    results.append(
                        SpecClauseResult(
                            family=clause.family,
                            kind=clause.kind,
                            text=clause.text,
                            status="not_checked",
                            checked_by="trace",
                            detail="trace-only provenance clause",
                            sources=clause.sources,
                            trace_refs=clause.trace_refs,
                        )
                    )
                elif clause.kind == "example_consistency":
                    results.append(
                        _example_consistency_result(
                            clause,
                            example_results=example_clause_results,
                        )
                    )
                else:
                    results.append(
                        _property_clause_result(
                            clause,
                            checked_properties=checked_properties,
                            failed_properties=tuple(sorted(failed_properties)),
                        )
                    )
        return tuple(results)

    def _rank_repair_hints(
        self,
        *,
        harness: HarnessResult,
        pbt: PBTCodeVerificationResult,
        clause_results: Sequence[SpecClauseResult],
    ) -> tuple[RepairHint, ...]:
        property_scores, strategy_learner = _load_learning(
            tuple(str(path) for path in self._learning_artifact_paths)
        )
        supporting_properties = tuple(
            sorted(
                {failure.property_name for failure in pbt.failures}
                | {
                    prop
                    for result in clause_results
                    if result.status == "violated"
                    for prop in result.matched_properties
                }
            )
        )
        official_test_miss = harness.passed and bool(
            pbt.failures or any(result.status == "violated" for result in clause_results)
        )
        recommendations = strategy_learner.recommend(
            harness_error_message="" if harness.passed else harness.error_message,
            property_names=supporting_properties,
            official_test_miss=official_test_miss,
            limit=3,
        )
        property_bonus = 0.0
        for name in supporting_properties:
            if name in property_scores:
                property_bonus += property_scores[name].score
        if not recommendations:
            return (
                RepairHint(
                    strategy_name="generic_repair",
                    error_family="unknown",
                    score=property_bonus,
                    success_rate=0.0,
                    attempts=0,
                    partial_recoveries=0,
                    supporting_properties=supporting_properties,
                    support_case_ids=(),
                    rationale="fallback",
                ),
            )
        return tuple(
            RepairHint(
                strategy_name=item.strategy_name,
                error_family=item.error_family,
                score=item.score + property_bonus,
                success_rate=item.success_rate,
                attempts=item.attempts,
                partial_recoveries=item.partial_recoveries,
                supporting_properties=supporting_properties,
                support_case_ids=item.support_case_ids,
                rationale=(
                    f"Prefer {item.strategy_name} for "
                    f"{', '.join(supporting_properties) or 'unknown'}"
                ),
            )
            for item in recommendations
        )
