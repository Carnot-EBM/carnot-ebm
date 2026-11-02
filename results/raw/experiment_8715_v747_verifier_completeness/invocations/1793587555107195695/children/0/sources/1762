"""Exp5445 deterministic AST/KB witness constraints for code/API hallucinations.

This module is deliberately execution-free. It parses fixture source into a
Python AST, resolves simple import aliases, checks visible calls against a
small whitelisted API knowledge base, and writes row-level witnesses that a
structured verifier can inspect without trusting a model self-report.

Spec refs: REQ-CODE-5445, SCENARIO-CODE-5445.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


JsonDict = dict[str, Any]
Importer = Callable[[str], ModuleType]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5445_static_ast_kb_witness_constraints_v495.json")
EXPERIMENT_ID = "experiment_5445_static_ast_kb_witness_constraints_v495"
TASK_ID = "exp5445-v495-static-ast-kb-witness-constraints"
MILESTONE = "2026.07.495"
RUN_DATE = "2026-07-08"
SCHEMA = "carnot.experiment_5445.static_ast_kb_witness_constraints.v495"
SPEC_REFS = ("REQ-CODE-5445", "SCENARIO-CODE-5445")
INFERENCE_SUBSTRATE = "deterministic_ast_kb_verifier_no_llm"
RANDOM_SEED = 5445
DEFAULT_API_MODULES = ("json", "math", "statistics", "builtins")

FALLBACK_API_KB: dict[str, tuple[str, ...]] = {
    "json": ("dump", "dumps", "load", "loads"),
    "math": ("ceil", "floor", "sqrt", "sin"),
    "statistics": ("mean", "median"),
    "builtins": ("len", "print", "range", "sorted"),
}

WITNESS_FIELD_NAMES = (
    "row_id",
    "fixture_family",
    "api_family",
    "source_sha256",
    "ast_parse_ok",
    "ast_error",
    "alias_map",
    "imported_symbol_checks",
    "fully_qualified_call_sites",
    "kb_lookup_results",
    "semantic_intent",
    "expected_outcome",
    "outcome",
    "accepted",
    "reject_reasons",
    "witness_checksum",
)

FIELD_PRINCIPLES = {
    "fixture_count": "bounded coverage.",
    "api_family_counts": "fixture diversity.",
    "ast_parse_success_rate": "structural basis.",
    "kb_source_paths": "KB provenance.",
    "witness_field_names": "inspectable evidence.",
    "nonexistent_call_reject_rate": "hallucination catch.",
    "valid_call_accept_rate": "false rejection guard.",
    "unsafe_false_accepts": "safety boundary.",
    "row_provenance_checksum": "row reproducibility.",
    "ast_kb_witness_ready": "downstream readiness.",
    "inference_substrate": "no hidden live model inference.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class AstKbFixture:
    """One bounded code/API row with an explicit expected deterministic outcome."""

    row_id: str
    fixture_family: str
    api_family: str
    source: str
    expected_outcome: str
    intent: str
    expected_call_fqns: tuple[str, ...]
    metric_tags: tuple[str, ...]


FIXTURES: tuple[AstKbFixture, ...] = (
    AstKbFixture(
        row_id="fixture.valid_json_alias",
        fixture_family="valid_api_call",
        api_family="json",
        source="import json as js\nresult = js.loads(payload)\n",
        expected_outcome="accept",
        intent="parse_json_to_object",
        expected_call_fqns=("json.loads",),
        metric_tags=("valid_call",),
    ),
    AstKbFixture(
        row_id="fixture.valid_math_from_import",
        fixture_family="valid_api_call",
        api_family="math",
        source="from math import sqrt\nresult = sqrt(x)\n",
        expected_outcome="accept",
        intent="compute_square_root",
        expected_call_fqns=("math.sqrt",),
        metric_tags=("valid_call",),
    ),
    AstKbFixture(
        row_id="fixture.nonexistent_json_method",
        fixture_family="nonexistent_method",
        api_family="json",
        source="import json\nresult = json.parse(payload)\n",
        expected_outcome="reject",
        intent="parse_json_to_object",
        expected_call_fqns=("json.loads",),
        metric_tags=("nonexistent_call", "invalid_row"),
    ),
    AstKbFixture(
        row_id="fixture.nonexistent_math_alias_method",
        fixture_family="nonexistent_method",
        api_family="math",
        source="import math as m\nresult = m.relu(x)\n",
        expected_outcome="reject",
        intent="compute_relu",
        expected_call_fqns=("math.relu",),
        metric_tags=("nonexistent_call", "invalid_row"),
    ),
    AstKbFixture(
        row_id="fixture.wrong_module_alias",
        fixture_family="wrong_module_alias",
        api_family="statistics",
        source="import statistics as json\nresult = json.loads(payload)\n",
        expected_outcome="reject",
        intent="parse_json_to_object",
        expected_call_fqns=("json.loads",),
        metric_tags=("nonexistent_call", "invalid_row"),
    ),
    AstKbFixture(
        row_id="fixture.missing_bare_import",
        fixture_family="bare_call_missing_import",
        api_family="json",
        source="result = loads(payload)\n",
        expected_outcome="reject",
        intent="parse_json_to_object",
        expected_call_fqns=("json.loads",),
        metric_tags=("invalid_row",),
    ),
    AstKbFixture(
        row_id="fixture.imported_symbol_missing",
        fixture_family="imported_symbol_missing",
        api_family="json",
        source="from json import parse\nresult = parse(payload)\n",
        expected_outcome="reject",
        intent="parse_json_to_object",
        expected_call_fqns=("json.loads",),
        metric_tags=("invalid_row",),
    ),
    AstKbFixture(
        row_id="fixture.argument_intent_mismatch",
        fixture_family="argument_intent_mismatch",
        api_family="json",
        source="import json\ntext = json.dumps(payload)\n",
        expected_outcome="reject",
        intent="parse_json_to_object",
        expected_call_fqns=("json.loads",),
        metric_tags=("invalid_row",),
    ),
    AstKbFixture(
        row_id="fixture.benign_builtin_control",
        fixture_family="benign_control",
        api_family="builtins",
        source="count = len(items)\n",
        expected_outcome="accept",
        intent="count_items",
        expected_call_fqns=("builtins.len",),
        metric_tags=("valid_call",),
    ),
    AstKbFixture(
        row_id="fixture.benign_no_call_control",
        fixture_family="benign_control",
        api_family="no_api",
        source="total = 1 + 2\n",
        expected_outcome="accept",
        intent="plain_arithmetic",
        expected_call_fqns=(),
        metric_tags=("valid_call",),
    ),
)


@dataclass(frozen=True)
class ApiKnowledgeBase:
    """Whitelisted API member map with provenance for every module entry."""

    attrs: Mapping[str, frozenset[str]]
    source_by_module: Mapping[str, str]
    source_paths: list[str]

    @classmethod
    def build(
        cls,
        module_names: Sequence[str] = DEFAULT_API_MODULES,
        *,
        importer: Importer = importlib.import_module,
    ) -> ApiKnowledgeBase:
        attrs: dict[str, frozenset[str]] = {}
        source_by_module: dict[str, str] = {}
        source_paths: list[str] = []
        for module_name in module_names:
            try:
                module = importer(module_name)
                names = set(dir(module)) | set(FALLBACK_API_KB.get(module_name, ()))
                attrs[module_name] = frozenset(names)
                source_by_module[module_name] = "local_introspection"
                source_paths.append(_module_origin(module_name, module))
            except Exception:
                attrs[module_name] = frozenset(FALLBACK_API_KB[module_name])
                source_by_module[module_name] = "fallback_metadata"
                source_paths.append(f"fallback_metadata:{module_name}")
        return cls(attrs=attrs, source_by_module=source_by_module, source_paths=source_paths)

    @classmethod
    def from_fallback_metadata(
        cls,
        module_names: Sequence[str] = DEFAULT_API_MODULES,
    ) -> ApiKnowledgeBase:
        attrs = {name: frozenset(FALLBACK_API_KB[name]) for name in module_names}
        sources = {name: "fallback_metadata" for name in module_names}
        paths = [f"fallback_metadata:{name}" for name in module_names]
        return cls(attrs=attrs, source_by_module=sources, source_paths=paths)

    def has_module(self, module_name: str) -> bool:
        return module_name in self.attrs

    def lookup(self, fqn: str) -> JsonDict:
        module_name, _, symbol = fqn.rpartition(".")
        if not module_name or module_name == "<unresolved>":
            return {
                "fully_qualified_name": fqn,
                "exists": False,
                "status": "unresolved_bare_call",
                "kb_source": "none",
            }
        known = self.attrs.get(module_name)
        if known is None:
            return {
                "fully_qualified_name": fqn,
                "exists": False,
                "status": "unknown_module",
                "kb_source": "none",
            }
        return {
            "fully_qualified_name": fqn,
            "exists": symbol in known,
            "status": "known_module",
            "kb_source": self.source_by_module[module_name],
        }


def fixture_by_id(row_id: str) -> AstKbFixture:
    """Return a checked-in fixture by stable identifier."""

    return next(fixture for fixture in FIXTURES if fixture.row_id == row_id)


def evaluate_fixture_rows(
    fixtures: Sequence[AstKbFixture] = FIXTURES,
    *,
    kb: ApiKnowledgeBase | None = None,
) -> list[JsonDict]:
    """Evaluate every fixture into a row-level AST/KB witness."""

    knowledge_base = kb or ApiKnowledgeBase.build()
    return [evaluate_fixture(fixture, kb=knowledge_base) for fixture in fixtures]


def evaluate_fixture(fixture: AstKbFixture, *, kb: ApiKnowledgeBase) -> JsonDict:
    """Parse one fixture, resolve aliases, validate calls, and attach evidence."""

    source_sha256 = _sha256(fixture.source)
    reject_reasons: list[str] = []
    try:
        tree = ast.parse(fixture.source)
        ast_parse_ok = True
        ast_error = ""
    except SyntaxError as exc:
        tree = ast.Module(body=[], type_ignores=[])
        ast_parse_ok = False
        ast_error = exc.msg
        reject_reasons.append(f"ast_parse_error:{exc.msg}")

    alias_map, imported_symbol_checks = _build_alias_map(tree, kb)
    call_sites = _collect_call_sites(tree, alias_map, kb, fixture.source)
    kb_results = [kb.lookup(call["fqn"]) for call in call_sites]
    for check in imported_symbol_checks:
        if check["exists"] is False:
            reject_reasons.append(f"imported_symbol_missing:{check['fully_qualified_name']}")
    for call, result in zip(call_sites, kb_results, strict=True):
        if result["status"] == "unresolved_bare_call":
            reject_reasons.append(f"missing_import_for_bare_call:{call['call_name']}")
        elif result["exists"] is False:
            reject_reasons.append(f"kb_missing_call:{call['fqn']}")

    semantic_intent = _semantic_intent(fixture, call_sites)
    if semantic_intent["matched"] is False:
        reject_reasons.append(f"intent_mismatch:{fixture.intent}")

    accepted = ast_parse_ok and not reject_reasons
    row: JsonDict = {
        "row_id": fixture.row_id,
        "fixture_family": fixture.fixture_family,
        "api_family": fixture.api_family,
        "source_sha256": source_sha256,
        "source": fixture.source,
        "ast_parse_ok": ast_parse_ok,
        "ast_error": ast_error,
        "alias_map": dict(sorted(alias_map.items())),
        "imported_symbol_checks": imported_symbol_checks,
        "fully_qualified_call_sites": call_sites,
        "kb_lookup_results": kb_results,
        "semantic_intent": semantic_intent,
        "expected_outcome": fixture.expected_outcome,
        "outcome": "accept" if accepted else "reject",
        "accepted": accepted,
        "reject_reasons": sorted(set(reject_reasons)),
        "metric_tags": list(fixture.metric_tags),
    }
    row["witness_checksum"] = row_witness_checksum(row)
    return row


def derive_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive terminal rates only from row witnesses and expected tags."""

    row_list = [dict(row) for row in rows]
    nonexistent = [row for row in row_list if "nonexistent_call" in row.get("metric_tags", [])]
    valid = [row for row in row_list if "valid_call" in row.get("metric_tags", [])]
    invalid = [row for row in row_list if row.get("expected_outcome") == "reject"]
    return {
        "fixture_count": len(row_list),
        "api_family_counts": dict(sorted(Counter(row["api_family"] for row in row_list).items())),
        "ast_parse_success_rate": _rate(
            sum(1 for row in row_list if row.get("ast_parse_ok") is True),
            len(row_list),
        ),
        "nonexistent_call_reject_rate": _rate(
            sum(1 for row in nonexistent if row.get("accepted") is False),
            len(nonexistent),
        ),
        "valid_call_accept_rate": _rate(
            sum(1 for row in valid if row.get("accepted") is True),
            len(valid),
        ),
        "unsafe_false_accepts": sum(1 for row in invalid if row.get("accepted") is True),
    }


def build_artifact(
    *,
    kb: ApiKnowledgeBase | None = None,
    tests_run: Sequence[Mapping[str, Any] | str] = (),
) -> JsonDict:
    """Build the terminal Exp5445 artifact from deterministic fixture witnesses."""

    knowledge_base = kb or ApiKnowledgeBase.build()
    rows = evaluate_fixture_rows(kb=knowledge_base)
    metrics = derive_metrics(rows)
    row_checksum = row_provenance_checksum(rows)
    ready = bool(
        metrics["fixture_count"] == len(FIXTURES)
        and metrics["ast_parse_success_rate"] == 1.0
        and metrics["nonexistent_call_reject_rate"] == 1.0
        and metrics["valid_call_accept_rate"] == 1.0
        and metrics["unsafe_false_accepts"] == 0
        and all(row["witness_checksum"] == row_witness_checksum(row) for row in rows)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "fixture_count": metrics["fixture_count"],
        "api_family_counts": metrics["api_family_counts"],
        "ast_parse_success_rate": metrics["ast_parse_success_rate"],
        "kb_source_paths": list(knowledge_base.source_paths),
        "witness_field_names": list(WITNESS_FIELD_NAMES),
        "nonexistent_call_reject_rate": metrics["nonexistent_call_reject_rate"],
        "valid_call_accept_rate": metrics["valid_call_accept_rate"],
        "unsafe_false_accepts": metrics["unsafe_false_accepts"],
        "row_provenance_checksum": row_checksum,
        "ast_kb_witness_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            "complete: deterministic AST/KB witness constraints ready"
            if ready
            else "blocked: ast_kb_witness_checks_failed"
        ),
        "witness_rows": rows,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": [_test_run_json(row) for row in tests_run],
        "research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any] | str] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally persist the Exp5445 result artifact."""

    artifact = build_artifact(tests_run=tests_run)
    if write:
        output_path = Path(result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact no longer supports the Exp5445 witness claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and provenance errors for the Exp5445 artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(_error_if(bool(missing), f"missing required fields: {missing}"))
    errors.extend(
        _error_if(
            artifact.get("field_principles") != FIELD_PRINCIPLES,
            "field_principles mismatch",
        )
    )
    rows = artifact.get("witness_rows")
    rows_are_list = isinstance(rows, list)
    errors.extend(
        _error_if(
            not rows_are_list or len(rows) != artifact.get("fixture_count"),
            "fixture_count mismatch",
        )
    )
    errors.extend(
        _error_if(
            rows_are_list
            and any(row.get("witness_checksum") != row_witness_checksum(row) for row in rows),
            "row witness_checksum mismatch",
        )
    )
    errors.extend(
        _error_if(
            rows_are_list
            and artifact.get("row_provenance_checksum") != row_provenance_checksum(rows),
            "row_provenance_checksum mismatch",
        )
    )
    errors.extend(
        _error_if(
            artifact.get("witness_field_names") != list(WITNESS_FIELD_NAMES),
            "witness_field_names mismatch",
        )
    )
    errors.extend(
        _error_if(
            artifact.get("inference_substrate") != INFERENCE_SUBSTRATE,
            "inference_substrate mismatch",
        )
    )
    verdict = artifact.get("honest_verdict")
    errors.extend(
        _error_if(
            not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")),
            "honest_verdict terminal prefix missing",
        )
    )
    for field in ("ast_parse_success_rate", "nonexistent_call_reject_rate", "valid_call_accept_rate"):
        value = artifact.get(field)
        errors.extend(
            _error_if(
                not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0,
                f"{field} outside [0, 1]",
            )
        )
    errors.extend(
        _error_if(
            artifact.get("ast_kb_witness_ready") is True
            and artifact.get("unsafe_false_accepts") != 0,
            "ready artifact has unsafe false accepts",
        )
    )
    errors.extend(
        _error_if(
            artifact.get("research_conductor_modified") is not False,
            "scripts/research_conductor.py must not be modified",
        )
    )
    return errors


def row_witness_checksum(row: Mapping[str, Any]) -> str:
    """Hash one witness row without its self-referential checksum field."""

    payload = {key: value for key, value in row.items() if key != "witness_checksum"}
    return _sha256(_stable_json(payload))


def row_provenance_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash reproducibility-bearing witness identities and row checksums."""

    payload = [
        {
            "row_id": row.get("row_id"),
            "source_sha256": row.get("source_sha256"),
            "witness_checksum": row_witness_checksum(row),
        }
        for row in rows
    ]
    return _sha256(_stable_json(payload))


def _build_alias_map(tree: ast.AST, kb: ApiKnowledgeBase) -> tuple[dict[str, str], list[JsonDict]]:
    alias_map: dict[str, str] = {}
    imported_symbol_checks: list[JsonDict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".", maxsplit=1)[0]
                alias_map[alias.asname or module_name] = module_name
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module.split(".", maxsplit=1)[0]
            for alias in node.names:
                local_name = alias.asname or alias.name
                fqn = f"{module_name}.{alias.name}"
                lookup = kb.lookup(fqn)
                alias_map[local_name] = fqn
                imported_symbol_checks.append(
                    {
                        "module": module_name,
                        "symbol": alias.name,
                        "local_name": local_name,
                        "fully_qualified_name": fqn,
                        "exists": lookup["exists"],
                        "kb_source": lookup["kb_source"],
                    }
                )
    return alias_map, imported_symbol_checks


def _collect_call_sites(
    tree: ast.AST,
    alias_map: Mapping[str, str],
    kb: ApiKnowledgeBase,
    source: str,
) -> list[JsonDict]:
    call_sites: list[JsonDict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_sites.append(_call_site(node, alias_map, kb, source))
    return call_sites


def _call_site(
    node: ast.Call,
    alias_map: Mapping[str, str],
    kb: ApiKnowledgeBase,
    source: str,
) -> JsonDict:
    if isinstance(node.func, ast.Name):
        return _name_call_site(node, alias_map, kb, source)
    if isinstance(node.func, ast.Attribute):
        return _attribute_call_site(node, alias_map, source)
    return {
        "call_name": type(node.func).__name__,
        "call_kind": "unresolved",
        "fqn": f"<unresolved>.{type(node.func).__name__}",
        "resolved": False,
        "lineno": getattr(node, "lineno", None),
        "col_offset": getattr(node, "col_offset", None),
        "source_segment": ast.get_source_segment(source, node) or "",
    }


def _name_call_site(
    node: ast.Call,
    alias_map: Mapping[str, str],
    kb: ApiKnowledgeBase,
    source: str,
) -> JsonDict:
    call_name = node.func.id
    if call_name in alias_map and "." in alias_map[call_name]:
        fqn = alias_map[call_name]
        call_kind = "imported_symbol"
        resolved = True
    elif call_name in kb.attrs.get("builtins", frozenset()):
        fqn = f"builtins.{call_name}"
        call_kind = "builtin"
        resolved = True
    else:
        fqn = f"<unresolved>.{call_name}"
        call_kind = "bare"
        resolved = False
    return {
        "call_name": call_name,
        "call_kind": call_kind,
        "fqn": fqn,
        "resolved": resolved,
        "lineno": node.lineno,
        "col_offset": node.col_offset,
        "source_segment": ast.get_source_segment(source, node) or "",
    }


def _attribute_call_site(
    node: ast.Call,
    alias_map: Mapping[str, str],
    source: str,
) -> JsonDict:
    receiver = node.func.value.id if isinstance(node.func.value, ast.Name) else "<unresolved>"
    module_name = alias_map.get(receiver, receiver)
    resolved = module_name != "<unresolved>"
    fqn = f"{module_name}.{node.func.attr}" if resolved else f"<unresolved>.{node.func.attr}"
    return {
        "call_name": node.func.attr,
        "call_kind": "attribute",
        "fqn": fqn,
        "resolved": resolved,
        "lineno": node.lineno,
        "col_offset": node.col_offset,
        "source_segment": ast.get_source_segment(source, node) or "",
    }


def _semantic_intent(fixture: AstKbFixture, call_sites: Sequence[Mapping[str, Any]]) -> JsonDict:
    actual = [str(call["fqn"]) for call in call_sites]
    expected = list(fixture.expected_call_fqns)
    matched = True if not expected else any(call in expected for call in actual)
    return {
        "intent": fixture.intent,
        "expected_call_fqns": expected,
        "actual_call_fqns": actual,
        "matched": matched,
    }


def _test_run_json(row: Mapping[str, Any] | str) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    return {"command": str(row), "outcome": "recorded"}


def _module_origin(module_name: str, module: ModuleType) -> str:
    return str(getattr(module, "__file__", None) or f"builtin:{module_name}")


def _error_if(condition: bool, message: str) -> list[str]:
    return [message] if condition else []


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
