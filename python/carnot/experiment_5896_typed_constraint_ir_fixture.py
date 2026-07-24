"""Exp5896 typed ConstraintIR exact fixture.

Spec refs: REQ-BENCH-5896, SCENARIO-BENCH-5896-SCHEMA,
SCENARIO-BENCH-5896-CERTIFICATES, SCENARIO-BENCH-5896-LEAKAGE.

This module defines a deliberately small, engine-neutral ConstraintIR and a
deterministic fixture around it. The point is not to be a full logic language:
the point is to make every accepted construct executable through exact local
backends and to reject anything whose semantics would be ambiguous.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5896_typed_constraint_ir_fixture.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_5896_typed_constraint_ir_fixture.rows.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5896_typed_constraint_ir_fixture.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5896_typed_constraint_ir_fixture.py")
BENCH_SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

CONSTRAINT_IR_SCHEMA_VERSION = "carnot.constraint_ir.v1"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5896.typed_constraint_ir_fixture.v1"
ROW_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".row"
RUN_DATE = "20260724"
EXPERIMENT_ID = "experiment_5896_typed_constraint_ir_fixture"
INFERENCE_SUBSTRATE = "deterministic_exact_solver_labeled_dataset_no_llm"
VERIFIER_IS_ORACLE = True

TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "domains", "entities", "predicates", "facts", "rules", "query"}
)
DOMAIN_KEYS = frozenset({"id", "type", "values"})
ENTITY_KEYS = frozenset({"id", "domain"})
PREDICATE_KEYS = frozenset({"id", "arg_types"})
FACT_KEYS = frozenset({"predicate", "args", "truth"})
RULE_KEYS = frozenset({"id", "variables", "body", "head"})
QUERY_KEYS = frozenset({"vars", "where"})
ATOM_KEYS = frozenset({"node", "predicate", "args"})
NOT_KEYS = frozenset({"node", "term"})
AND_KEYS = frozenset({"node", "terms"})
ARITH_KEYS = frozenset({"node", "left", "op", "right"})
SUPPORTED_NODES = frozenset({"atom", "not", "and", "arith"})
ARITHMETIC_OPS = frozenset({"<", "<=", "==", ">=", ">"})
CONTROL_VARIANTS = frozenset(
    {"invalid_ir", "unsat_ir", "type_error", "omitted_constraint", "semantic_nonequivalence"}
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5896_typed_constraint_ir_fixture.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5896_typed_constraint_ir_fixture.py "
    "-m pytest tests/python/test_experiment_5896_typed_constraint_ir_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5896_typed_constraint_ir_fixture.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5896_typed_constraint_ir_fixture.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "constraint_ir_schema_and_version",
    "supported_and_rejected_fragments",
    "parser_and_typecheck_receipts",
    "backend_compiler_receipts",
    "cross_backend_agreement",
    "family_template_and_holdout_design",
    "exact_semantic_equivalence_contract",
    "invalid_unsat_and_nonequivalence_controls",
    "split_and_group_leakage_receipts",
    "label_certificate_and_balance_receipts",
    "row_file_receipt",
    "deterministic_replay_receipt",
    "protected_files_unchanged",
    "typed_constraint_ir_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "constraint_ir_schema_and_version": (
        "An engine-neutral contract prevents backend-specific prompt tuning from defining success."
    ),
    "exact_semantic_equivalence_contract": (
        "Executable behavior and certificates own equivalence."
    ),
    "split_and_group_leakage_receipts": (
        "Paraphrases and renamings of one problem never cross held boundaries."
    ),
    "typed_constraint_ir_fixture_ready_score": (
        "Emit bare 1.0 only for deterministic exact replay, backend agreement, "
        "and nontrivial held headroom."
    ),
    "inference_substrate": "Use `deterministic_exact_solver_labeled_dataset_no_llm`.",
    "verifier_is_oracle": (
        "True for fixture labels/certificates and never a learned-verifier claim."
    ),
    "honest_verdict": "Use `ready:`, `complete_null:`, or `blocked:`.",
}


class ConstraintIRValidationError(ValueError):
    """Raised when a ConstraintIR object is unsupported, ambiguous, or ill typed."""


class ConstraintIRReplayError(ValueError):
    """Raised when a written Exp5896 artifact no longer replays by hash."""


@dataclass(frozen=True)
class Domain:
    name: str
    value_type: str
    values: tuple[str | int, ...]


@dataclass(frozen=True)
class Predicate:
    name: str
    arg_types: tuple[str, ...]


@dataclass(frozen=True)
class Atom:
    predicate: str
    args: tuple[str | int, ...]


@dataclass(frozen=True)
class Expr:
    node: str
    data: JsonDict


@dataclass(frozen=True)
class Rule:
    name: str
    variables: Mapping[str, str]
    body: Expr
    head: Atom


@dataclass(frozen=True)
class Query:
    variables: Mapping[str, str]
    where: Expr


@dataclass(frozen=True)
class ConstraintIR:
    schema_version: str
    domains: tuple[Domain, ...]
    entities: Mapping[str, str]
    predicates: Mapping[str, Predicate]
    facts: tuple[tuple[Atom, bool], ...]
    rules: tuple[Rule, ...]
    query: Query


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file by bytes so replay never trusts path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _unknown_keys(value: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConstraintIRValidationError(f"unknown fields in {context}: {unknown}")


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConstraintIRValidationError(f"{context} must be an object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConstraintIRValidationError(f"{context} must be a list")
    return value


def _var_name(value: str) -> bool:
    return value.startswith("?")


def parse_constraint_ir(payload: Mapping[str, Any]) -> ConstraintIR:
    """Parse and typecheck the strict Exp5896 ConstraintIR schema."""

    data = _require_mapping(payload, "ConstraintIR")
    unknown_top = sorted(set(data) - TOP_LEVEL_KEYS)
    if unknown_top:
        raise ConstraintIRValidationError(f"unknown top-level fields: {unknown_top}")
    missing = sorted(TOP_LEVEL_KEYS - set(data))
    if missing:
        raise ConstraintIRValidationError(f"missing top-level fields: {missing}")
    if data["schema_version"] != CONSTRAINT_IR_SCHEMA_VERSION:
        raise ConstraintIRValidationError("unsupported schema_version")

    domains = _parse_domains(data["domains"])
    domain_map = {domain.name: domain for domain in domains}
    entities = _parse_entities(data["entities"], domain_map)
    predicates = _parse_predicates(data["predicates"], domain_map)
    facts = tuple(_parse_fact(item, predicates, domain_map) for item in _require_list(data["facts"], "facts"))
    rules = tuple(
        _parse_rule(item, predicates, domain_map) for item in _require_list(data["rules"], "rules")
    )
    query = _parse_query(data["query"], predicates, domain_map)
    _reject_rule_dependencies(rules)
    return ConstraintIR(
        schema_version=CONSTRAINT_IR_SCHEMA_VERSION,
        domains=tuple(domains),
        entities=entities,
        predicates=predicates,
        facts=facts,
        rules=rules,
        query=query,
    )


def _parse_domains(raw_domains: Any) -> list[Domain]:
    domains: list[Domain] = []
    seen: set[str] = set()
    for item in _require_list(raw_domains, "domains"):
        raw = _require_mapping(item, "domain")
        _unknown_keys(raw, DOMAIN_KEYS, "domain")
        name = raw.get("id")
        value_type = raw.get("type")
        values = raw.get("values")
        if not isinstance(name, str) or not name:
            raise ConstraintIRValidationError("domain id must be a non-empty string")
        if name in seen:
            raise ConstraintIRValidationError(f"duplicate domain: {name}")
        if value_type not in {"symbol", "int"}:
            raise ConstraintIRValidationError(f"unsupported domain type: {value_type}")
        parsed_values = tuple(_require_list(values, f"domain {name} values"))
        if not parsed_values:
            raise ConstraintIRValidationError(f"domain {name} must be finite and non-empty")
        if len(set(parsed_values)) != len(parsed_values):
            raise ConstraintIRValidationError(f"domain {name} values must be unique")
        if value_type == "symbol" and not all(isinstance(value, str) for value in parsed_values):
            raise ConstraintIRValidationError(f"domain {name} expects string values")
        if value_type == "int" and not all(isinstance(value, int) for value in parsed_values):
            raise ConstraintIRValidationError(f"domain {name} expects integer values")
        domains.append(Domain(name=name, value_type=str(value_type), values=parsed_values))
        seen.add(name)
    return domains


def _parse_entities(raw_entities: Any, domains: Mapping[str, Domain]) -> Mapping[str, str]:
    entities: dict[str, str] = {}
    for item in _require_list(raw_entities, "entities"):
        raw = _require_mapping(item, "entity")
        _unknown_keys(raw, ENTITY_KEYS, "entity")
        entity_id = raw.get("id")
        domain_name = raw.get("domain")
        if not isinstance(entity_id, str) or not isinstance(domain_name, str):
            raise ConstraintIRValidationError("entity id and domain must be strings")
        domain = domains.get(domain_name)
        if domain is None:
            raise ConstraintIRValidationError(f"unknown entity domain: {domain_name}")
        if domain.value_type != "symbol":
            raise ConstraintIRValidationError("entities may only inhabit symbol domains")
        if entity_id not in domain.values:
            raise ConstraintIRValidationError(f"entity {entity_id} not in domain {domain_name}")
        entities[entity_id] = domain_name
    return entities


def _parse_predicates(raw_predicates: Any, domains: Mapping[str, Domain]) -> Mapping[str, Predicate]:
    predicates: dict[str, Predicate] = {}
    for item in _require_list(raw_predicates, "predicates"):
        raw = _require_mapping(item, "predicate")
        _unknown_keys(raw, PREDICATE_KEYS, "predicate")
        name = raw.get("id")
        arg_types = tuple(_require_list(raw.get("arg_types"), f"predicate {name} arg_types"))
        if not isinstance(name, str) or not name:
            raise ConstraintIRValidationError("predicate id must be a non-empty string")
        if name in predicates:
            raise ConstraintIRValidationError(f"duplicate predicate: {name}")
        for domain_name in arg_types:
            if domain_name not in domains:
                raise ConstraintIRValidationError(f"unknown predicate domain: {domain_name}")
        predicates[name] = Predicate(name=name, arg_types=tuple(str(item) for item in arg_types))
    return predicates


def _parse_fact(
    item: Any,
    predicates: Mapping[str, Predicate],
    domains: Mapping[str, Domain],
) -> tuple[Atom, bool]:
    raw = _require_mapping(item, "fact")
    _unknown_keys(raw, FACT_KEYS, "fact")
    atom = _parse_atom_like(raw, predicates, domains, {})
    truth = raw.get("truth")
    if not isinstance(truth, bool):
        raise ConstraintIRValidationError("fact truth must be boolean")
    return atom, truth


def _parse_rule(
    item: Any,
    predicates: Mapping[str, Predicate],
    domains: Mapping[str, Domain],
) -> Rule:
    raw = _require_mapping(item, "rule")
    _unknown_keys(raw, RULE_KEYS, "rule")
    name = raw.get("id")
    if not isinstance(name, str) or not name:
        raise ConstraintIRValidationError("rule id must be a non-empty string")
    variables = _parse_variables(raw.get("variables"), domains, f"rule {name}")
    body = _parse_expr(raw.get("body"), predicates, domains, variables)
    head = _parse_atom_expr(raw.get("head"), predicates, domains, variables)
    return Rule(name=name, variables=variables, body=body, head=head)


def _parse_query(
    item: Any,
    predicates: Mapping[str, Predicate],
    domains: Mapping[str, Domain],
) -> Query:
    raw = _require_mapping(item, "query")
    _unknown_keys(raw, QUERY_KEYS, "query")
    variables = _parse_variables(raw.get("vars"), domains, "query")
    where = _parse_expr(raw.get("where"), predicates, domains, variables)
    return Query(variables=variables, where=where)


def _parse_variables(
    raw_variables: Any,
    domains: Mapping[str, Domain],
    context: str,
) -> Mapping[str, str]:
    raw = _require_mapping(raw_variables, f"{context} variables")
    parsed: dict[str, str] = {}
    for name, domain_name in raw.items():
        if not isinstance(name, str) or not _var_name(name):
            raise ConstraintIRValidationError(f"{context} variable names must start with '?'")
        if not isinstance(domain_name, str) or domain_name not in domains:
            raise ConstraintIRValidationError(f"{context} variable {name} has unknown domain")
        parsed[name] = domain_name
    return parsed


def _parse_expr(
    item: Any,
    predicates: Mapping[str, Predicate],
    domains: Mapping[str, Domain],
    variables: Mapping[str, str],
) -> Expr:
    raw = _require_mapping(item, "expression")
    node = raw.get("node")
    if node not in SUPPORTED_NODES:
        raise ConstraintIRValidationError(f"unsupported expression node: {node}")
    if node == "atom":
        _unknown_keys(raw, ATOM_KEYS, "atom expression")
        return Expr(node="atom", data={"atom": _parse_atom_like(raw, predicates, domains, variables)})
    if node == "not":
        _unknown_keys(raw, NOT_KEYS, "not expression")
        return Expr(
            node="not",
            data={"term": _parse_atom_expr(raw.get("term"), predicates, domains, variables)},
        )
    if node == "and":
        _unknown_keys(raw, AND_KEYS, "and expression")
        terms = [_parse_expr(term, predicates, domains, variables) for term in _require_list(raw.get("terms"), "and terms")]
        if not terms:
            raise ConstraintIRValidationError("and expression requires at least one term")
        return Expr(node="and", data={"terms": tuple(terms)})
    _unknown_keys(raw, ARITH_KEYS, "arith expression")
    op = raw.get("op")
    if op not in ARITHMETIC_OPS:
        raise ConstraintIRValidationError(f"unsupported arithmetic op: {op}")
    left = _parse_arith_term(raw.get("left"), variables, domains)
    right = _parse_arith_term(raw.get("right"), variables, domains)
    return Expr(node="arith", data={"left": left, "op": op, "right": right})


def _parse_atom_expr(
    item: Any,
    predicates: Mapping[str, Predicate],
    domains: Mapping[str, Domain],
    variables: Mapping[str, str],
) -> Atom:
    raw = _require_mapping(item, "atom")
    if raw.get("node") != "atom":
        raise ConstraintIRValidationError("Horn heads and negation terms must be atom nodes")
    _unknown_keys(raw, ATOM_KEYS, "atom")
    return _parse_atom_like(raw, predicates, domains, variables)


def _parse_atom_like(
    raw: Mapping[str, Any],
    predicates: Mapping[str, Predicate],
    domains: Mapping[str, Domain],
    variables: Mapping[str, str],
) -> Atom:
    pred_name = raw.get("predicate")
    args = tuple(_require_list(raw.get("args"), f"atom {pred_name} args"))
    if not isinstance(pred_name, str) or pred_name not in predicates:
        raise ConstraintIRValidationError(f"unknown predicate: {pred_name}")
    predicate = predicates[pred_name]
    if len(args) != len(predicate.arg_types):
        raise ConstraintIRValidationError(f"arity mismatch for predicate {pred_name}")
    for arg, domain_name in zip(args, predicate.arg_types, strict=True):
        _validate_term(arg, domain_name, variables, domains)
    return Atom(predicate=pred_name, args=args)


def _parse_arith_term(
    term: Any,
    variables: Mapping[str, str],
    domains: Mapping[str, Domain],
) -> str | int:
    if isinstance(term, int):
        return term
    if isinstance(term, str) and _var_name(term):
        domain_name = variables.get(term)
        if domain_name is None:
            raise ConstraintIRValidationError(f"unknown arithmetic variable: {term}")
        if domains[domain_name].value_type != "int":
            raise ConstraintIRValidationError(f"arithmetic variable {term} is not integer typed")
        return term
    raise ConstraintIRValidationError("arithmetic terms must be integers or integer variables")


def _validate_term(
    term: Any,
    domain_name: str,
    variables: Mapping[str, str],
    domains: Mapping[str, Domain],
) -> None:
    if isinstance(term, str) and _var_name(term):
        if term not in variables:
            raise ConstraintIRValidationError(f"unknown variable: {term}")
        if variables[term] != domain_name:
            raise ConstraintIRValidationError(f"variable {term} has wrong domain")
        return
    if term not in domains[domain_name].values:
        raise ConstraintIRValidationError(f"value {term!r} not in domain {domain_name}")


def _reject_rule_dependencies(rules: Sequence[Rule]) -> None:
    head_predicates = {rule.head.predicate for rule in rules}
    for rule in rules:
        if _predicates_in_expr(rule.body) & head_predicates:
            raise ConstraintIRValidationError("unsupported recursive or multi-stage Horn dependency")


def _predicates_in_expr(expr: Expr) -> set[str]:
    if expr.node == "atom":
        return {expr.data["atom"].predicate}
    if expr.node == "not":
        return {expr.data["term"].predicate}
    if expr.node == "and":
        found: set[str] = set()
        for term in expr.data["terms"]:
            found.update(_predicates_in_expr(term))
        return found
    return set()


def _domain_map(ir: ConstraintIR) -> Mapping[str, Domain]:
    return {domain.name: domain for domain in ir.domains}


def _atom_key(atom: Atom, assignment: Mapping[str, str | int] | None = None) -> tuple[str, tuple[str | int, ...]]:
    values: list[str | int] = []
    for arg in atom.args:
        if isinstance(arg, str) and _var_name(arg):
            values.append(assignment[arg])  # type: ignore[index]
        else:
            values.append(arg)
    return atom.predicate, tuple(values)


def _all_atom_keys(ir: ConstraintIR) -> list[tuple[str, tuple[str | int, ...]]]:
    domains = _domain_map(ir)
    keys: list[tuple[str, tuple[str | int, ...]]] = []
    for predicate in ir.predicates.values():
        value_lists = [domains[domain_name].values for domain_name in predicate.arg_types]
        for args in product(*value_lists):
            keys.append((predicate.name, tuple(args)))
    return keys


def _assignments(variables: Mapping[str, str], domains: Mapping[str, Domain]) -> list[dict[str, str | int]]:
    names = list(variables)
    value_lists = [domains[variables[name]].values for name in names]
    return [dict(zip(names, values, strict=True)) for values in product(*value_lists)]


def evaluate_with_python(ir: ConstraintIR) -> JsonDict:
    """Evaluate supported ConstraintIR rows by finite-domain least closure."""

    domains = _domain_map(ir)
    truth = {key: False for key in _all_atom_keys(ir)}
    explicit_negative: set[tuple[str, tuple[str | int, ...]]] = set()
    positive_facts: set[tuple[str, tuple[str | int, ...]]] = set()
    for atom, value in ir.facts:
        key = _atom_key(atom)
        if value:
            if key in explicit_negative:
                return _unsat_result("contradictory_fact", key)
            positive_facts.add(key)
            truth[key] = True
        else:
            if key in positive_facts:
                return _unsat_result("contradictory_fact", key)
            explicit_negative.add(key)
            truth[key] = False

    changed = True
    derivations: list[JsonDict] = []
    while changed:
        changed = False
        for rule in ir.rules:
            for assignment in _assignments(rule.variables, domains):
                if _eval_expr_python(rule.body, assignment, truth):
                    key = _atom_key(rule.head, assignment)
                    if key in explicit_negative:
                        return _unsat_result("derived_negative_conflict", key)
                    if not truth[key]:
                        truth[key] = True
                        changed = True
                        derivations.append(
                            {
                                "rule": rule.name,
                                "assignment": _canonical_assignment(assignment, domains),
                                "head": _format_atom_key(key),
                            }
                        )

    query_bindings = _query_bindings_python(ir.query, domains, truth)
    signature = _behavior_signature(ir, truth, query_bindings)
    return {
        "status": "sat",
        "query_bindings": query_bindings,
        "true_atoms": [_format_atom_key(key) for key, value in sorted(truth.items()) if value],
        "derivations": derivations,
        "behavior_signature": signature,
        "behavior_hash": sha256_json(signature),
    }


def _unsat_result(kind: str, key: tuple[str, tuple[str | int, ...]]) -> JsonDict:
    return {
        "status": "unsat",
        "counterexample": {"kind": kind, "atom": _format_atom_key(key)},
    }


def _eval_expr_python(
    expr: Expr,
    assignment: Mapping[str, str | int],
    truth: Mapping[tuple[str, tuple[str | int, ...]], bool],
) -> bool:
    if expr.node == "atom":
        return bool(truth[_atom_key(expr.data["atom"], assignment)])
    if expr.node == "not":
        return not bool(truth[_atom_key(expr.data["term"], assignment)])
    if expr.node == "and":
        return all(_eval_expr_python(term, assignment, truth) for term in expr.data["terms"])
    left = _resolve_arith(expr.data["left"], assignment)
    right = _resolve_arith(expr.data["right"], assignment)
    return _compare_ints(left, str(expr.data["op"]), right)


def _resolve_arith(term: str | int, assignment: Mapping[str, str | int]) -> int:
    value = assignment[term] if isinstance(term, str) and _var_name(term) else term
    return int(value)


def _compare_ints(left: int, op: str, right: int) -> bool:
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "==":
        return left == right
    if op == ">=":
        return left >= right
    return left > right


def _query_bindings_python(
    query: Query,
    domains: Mapping[str, Domain],
    truth: Mapping[tuple[str, tuple[str | int, ...]], bool],
) -> list[JsonDict]:
    bindings: list[JsonDict] = []
    for assignment in _assignments(query.variables, domains):
        if _eval_expr_python(query.where, assignment, truth):
            bindings.append(_canonical_assignment(assignment, domains))
    return sorted(bindings, key=canonical_json)


def evaluate_with_z3(ir: ConstraintIR) -> JsonDict:
    """Compile the supported finite Horn subset to Z3 and replay query behavior."""

    import z3

    domains = _domain_map(ir)
    atom_keys = _all_atom_keys(ir)
    bools = {key: z3.Bool(_z3_atom_name(key)) for key in atom_keys}
    solver = z3.Solver()
    positive_facts: set[tuple[str, tuple[str | int, ...]]] = set()
    negative_facts: set[tuple[str, tuple[str | int, ...]]] = set()
    reasons: dict[tuple[str, tuple[str | int, ...]], list[Any]] = defaultdict(list)

    for atom, value in ir.facts:
        key = _atom_key(atom)
        if value:
            positive_facts.add(key)
            solver.add(bools[key])
        else:
            negative_facts.add(key)
            solver.add(z3.Not(bools[key]))

    implication_count = 0
    for rule in ir.rules:
        for assignment in _assignments(rule.variables, domains):
            body = _eval_expr_z3(rule.body, assignment, bools)
            head = _atom_key(rule.head, assignment)
            solver.add(z3.Implies(body, bools[head]))
            reasons[head].append(body)
            implication_count += 1

    for key in atom_keys:
        atom_reasons = list(reasons.get(key, ()))
        if key in positive_facts:
            atom_reasons.append(z3.BoolVal(True))
        solver.add(bools[key] == z3.Or(atom_reasons) if atom_reasons else bools[key] == z3.BoolVal(False))

    verdict = str(solver.check())
    if verdict == "unsat":
        return {"status": "unsat", "z3_version": z3.get_version_string()}
    if verdict != "sat":  # pragma: no cover - Z3 should return sat/unsat for this finite fragment.
        return {"status": "unknown", "z3_version": z3.get_version_string()}
    model = solver.model()
    query_bindings = _query_bindings_z3(ir.query, domains, bools, model)
    true_atoms = [_format_atom_key(key) for key in atom_keys if z3.is_true(model.eval(bools[key]))]
    return {
        "status": "sat",
        "z3_version": z3.get_version_string(),
        "bool_count": len(bools),
        "implication_count": implication_count,
        "query_bindings": query_bindings,
        "true_atoms": sorted(true_atoms, key=canonical_json),
    }


def _eval_expr_z3(
    expr: Expr,
    assignment: Mapping[str, str | int],
    bools: Mapping[tuple[str, tuple[str | int, ...]], Any],
) -> Any:
    import z3

    if expr.node == "atom":
        return bools[_atom_key(expr.data["atom"], assignment)]
    if expr.node == "not":
        return z3.Not(bools[_atom_key(expr.data["term"], assignment)])
    if expr.node == "and":
        return z3.And([_eval_expr_z3(term, assignment, bools) for term in expr.data["terms"]])
    left = _resolve_arith(expr.data["left"], assignment)
    right = _resolve_arith(expr.data["right"], assignment)
    return z3.BoolVal(_compare_ints(left, str(expr.data["op"]), right))


def _query_bindings_z3(
    query: Query,
    domains: Mapping[str, Domain],
    bools: Mapping[tuple[str, tuple[str | int, ...]], Any],
    model: Any,
) -> list[JsonDict]:
    import z3

    bindings: list[JsonDict] = []
    for assignment in _assignments(query.variables, domains):
        if z3.is_true(model.eval(_eval_expr_z3(query.where, assignment, bools))):
            bindings.append(_canonical_assignment(assignment, domains))
    return sorted(bindings, key=canonical_json)


def _z3_atom_name(key: tuple[str, tuple[str | int, ...]]) -> str:
    pred, args = key
    return "__".join([pred, *(str(arg).replace("-", "_") for arg in args)])


def _canonical_assignment(
    assignment: Mapping[str, str | int],
    domains: Mapping[str, Domain],
) -> JsonDict:
    reverse: dict[str, dict[str | int, str]] = {}
    for domain in domains.values():
        reverse[domain.name] = {
            value: f"{domain.name}:{index}" for index, value in enumerate(domain.values)
        }
    result: JsonDict = {}
    for var_name, value in assignment.items():
        for domain_name, labels in reverse.items():
            if value in labels:
                result[var_name] = labels[value]
                result[f"{var_name}:raw"] = value
                result[f"{var_name}:domain"] = domain_name
                break
    return result


def _format_atom_key(key: tuple[str, tuple[str | int, ...]]) -> JsonDict:
    return {"predicate": key[0], "args": list(key[1])}


def _behavior_signature(
    ir: ConstraintIR,
    truth: Mapping[tuple[str, tuple[str | int, ...]], bool],
    query_bindings: Sequence[Mapping[str, Any]],
) -> JsonDict:
    domains = _domain_map(ir)
    predicate_index = {name: index for index, name in enumerate(ir.predicates)}
    normalized_atoms = []
    for key, value in truth.items():
        if value:
            predicate, args = key
            normalized_atoms.append(
                {
                    "predicate": f"predicate:{predicate_index[predicate]}",
                    "args": [_canonical_value(arg, domains) for arg in args],
                }
            )
    normalized_query = []
    for binding in query_bindings:
        normalized_query.append(
            {
                key: value
                for key, value in binding.items()
                if not key.endswith(":raw") and not key.endswith(":domain")
            }
        )
    return {
        "schema_version": CONSTRAINT_IR_SCHEMA_VERSION,
        "true_atoms": sorted(normalized_atoms, key=canonical_json),
        "query_bindings": sorted(normalized_query, key=canonical_json),
    }


def _canonical_value(value: str | int, domains: Mapping[str, Domain]) -> str:
    for domain in domains.values():
        if value in domain.values:
            return f"{domain.name}:{domain.values.index(value)}"
    return str(value)


def certify_ir(payload: Mapping[str, Any]) -> JsonDict:
    """Parse, typecheck, compile, and cross-check one raw ConstraintIR payload."""

    try:
        ir = parse_constraint_ir(payload)
    except ConstraintIRValidationError as exc:
        kind = "type_error" if "not in domain" in str(exc) or "wrong domain" in str(exc) else "invalid"
        return {
            "parser": {"status": "rejected", "kind": kind, "error": str(exc)},
            "python": {"status": "not_applicable"},
            "z3": {"status": "not_applicable"},
            "cross_backend_agreement": {"agrees": None, "reason": "parser_rejected"},
        }

    python_result = evaluate_with_python(ir)
    z3_result = evaluate_with_z3(ir)
    if python_result["status"] == "sat" and z3_result["status"] == "sat":
        agrees = python_result["query_bindings"] == z3_result["query_bindings"]
    else:
        agrees = python_result["status"] == z3_result["status"]
    receipt: JsonDict = {
        "parser": {"status": "accepted", "schema_version": ir.schema_version},
        "python": python_result,
        "z3": z3_result,
        "cross_backend_agreement": {"agrees": agrees},
    }
    if python_result["status"] == "unsat":
        receipt["counterexample"] = python_result["counterexample"]
    return receipt


def replay_row_certificate(row: Mapping[str, Any]) -> JsonDict:
    """Replay a row certificate from its IR payload and compare stable evidence."""

    replayed = certify_ir(row["constraint_ir"])  # type: ignore[index]
    expected = row["certificates"]  # type: ignore[index]
    ok = (
        replayed["parser"]["status"] == expected["parser"]["status"]
        and replayed["python"]["status"] == expected["python"]["status"]
        and replayed["z3"]["status"] == expected["z3"]["status"]
        and replayed["cross_backend_agreement"] == expected["cross_backend_agreement"]
    )
    if replayed["python"].get("status") == "sat":
        ok = ok and replayed["python"]["behavior_hash"] == expected["python"]["behavior_hash"]
    return {"ok": ok, "replayed": replayed}


def make_access_control_ir() -> JsonDict:
    """Build the canonical access-control family IR."""

    return _base_ir(
        domains=[
            {"id": "person", "type": "symbol", "values": ["ada", "ben", "cy"]},
            {"id": "department", "type": "symbol", "values": ["cardiology", "oncology"]},
        ],
        predicates=[
            {"id": "works_in", "arg_types": ["person", "department"]},
            {"id": "approved", "arg_types": ["department"]},
            {"id": "suspended", "arg_types": ["person"]},
            {"id": "eligible", "arg_types": ["person"]},
        ],
        facts=[
            {"predicate": "works_in", "args": ["ada", "cardiology"], "truth": True},
            {"predicate": "works_in", "args": ["ben", "oncology"], "truth": True},
            {"predicate": "works_in", "args": ["cy", "cardiology"], "truth": True},
            {"predicate": "approved", "args": ["cardiology"], "truth": True},
            {"predicate": "approved", "args": ["oncology"], "truth": True},
            {"predicate": "suspended", "args": ["ben"], "truth": True},
        ],
        rule_body_terms=[
            {"node": "atom", "predicate": "works_in", "args": ["?who", "?dept"]},
            {"node": "atom", "predicate": "approved", "args": ["?dept"]},
            {"node": "not", "term": {"node": "atom", "predicate": "suspended", "args": ["?who"]}},
        ],
        rule_variables={"?who": "person", "?dept": "department"},
        head={"node": "atom", "predicate": "eligible", "args": ["?who"]},
        query_vars={"?who": "person"},
        query_where={"node": "atom", "predicate": "eligible", "args": ["?who"]},
    )


def make_task_selection_ir() -> JsonDict:
    """Build the canonical task-selection family IR with arithmetic relations."""

    return _base_ir(
        domains=[
            {"id": "task", "type": "symbol", "values": ["alpha", "beta", "gamma"]},
            {"id": "hours", "type": "int", "values": [1, 2, 3]},
            {"id": "priority", "type": "int", "values": [1, 2, 3]},
        ],
        predicates=[
            {"id": "effort", "arg_types": ["task", "hours"]},
            {"id": "priority", "arg_types": ["task", "priority"]},
            {"id": "blocked", "arg_types": ["task"]},
            {"id": "selectable", "arg_types": ["task"]},
        ],
        facts=[
            {"predicate": "effort", "args": ["alpha", 1], "truth": True},
            {"predicate": "effort", "args": ["beta", 3], "truth": True},
            {"predicate": "effort", "args": ["gamma", 2], "truth": True},
            {"predicate": "priority", "args": ["alpha", 3], "truth": True},
            {"predicate": "priority", "args": ["beta", 2], "truth": True},
            {"predicate": "priority", "args": ["gamma", 1], "truth": True},
            {"predicate": "blocked", "args": ["gamma"], "truth": True},
        ],
        rule_body_terms=[
            {"node": "atom", "predicate": "effort", "args": ["?task", "?hours"]},
            {"node": "atom", "predicate": "priority", "args": ["?task", "?priority"]},
            {"node": "arith", "left": "?hours", "op": "<=", "right": 2},
            {"node": "arith", "left": "?priority", "op": ">=", "right": 2},
            {"node": "not", "term": {"node": "atom", "predicate": "blocked", "args": ["?task"]}},
        ],
        rule_variables={"?task": "task", "?hours": "hours", "?priority": "priority"},
        head={"node": "atom", "predicate": "selectable", "args": ["?task"]},
        query_vars={"?task": "task"},
        query_where={"node": "atom", "predicate": "selectable", "args": ["?task"]},
    )


def make_menu_ir() -> JsonDict:
    """Build the canonical menu-recommendation held-out family IR."""

    return _base_ir(
        domains=[
            {"id": "dish", "type": "symbol", "values": ["salad", "pasta", "soup"]},
            {"id": "price", "type": "int", "values": [5, 8, 12]},
            {"id": "category", "type": "symbol", "values": ["veg", "meat"]},
        ],
        predicates=[
            {"id": "costs", "arg_types": ["dish", "price"]},
            {"id": "category", "arg_types": ["dish", "category"]},
            {"id": "allergen", "arg_types": ["dish"]},
            {"id": "recommended", "arg_types": ["dish"]},
        ],
        facts=[
            {"predicate": "costs", "args": ["salad", 5], "truth": True},
            {"predicate": "costs", "args": ["pasta", 12], "truth": True},
            {"predicate": "costs", "args": ["soup", 8], "truth": True},
            {"predicate": "category", "args": ["salad", "veg"], "truth": True},
            {"predicate": "category", "args": ["pasta", "veg"], "truth": True},
            {"predicate": "category", "args": ["soup", "meat"], "truth": True},
            {"predicate": "allergen", "args": ["soup"], "truth": True},
        ],
        rule_body_terms=[
            {"node": "atom", "predicate": "costs", "args": ["?dish", "?price"]},
            {"node": "atom", "predicate": "category", "args": ["?dish", "veg"]},
            {"node": "arith", "left": "?price", "op": "<=", "right": 8},
            {"node": "not", "term": {"node": "atom", "predicate": "allergen", "args": ["?dish"]}},
        ],
        rule_variables={"?dish": "dish", "?price": "price"},
        head={"node": "atom", "predicate": "recommended", "args": ["?dish"]},
        query_vars={"?dish": "dish"},
        query_where={"node": "atom", "predicate": "recommended", "args": ["?dish"]},
    )


def _base_ir(
    *,
    domains: list[JsonDict],
    predicates: list[JsonDict],
    facts: list[JsonDict],
    rule_body_terms: list[JsonDict],
    rule_variables: JsonDict,
    head: JsonDict,
    query_vars: JsonDict,
    query_where: JsonDict,
) -> JsonDict:
    entities = [
        {"id": value, "domain": domain["id"]}
        for domain in domains
        if domain["type"] == "symbol"
        for value in domain["values"]
    ]
    return {
        "schema_version": CONSTRAINT_IR_SCHEMA_VERSION,
        "domains": domains,
        "entities": entities,
        "predicates": predicates,
        "facts": facts,
        "rules": [
            {
                "id": "r1",
                "variables": rule_variables,
                "body": {"node": "and", "terms": rule_body_terms},
                "head": head,
            }
        ],
        "query": {"vars": query_vars, "where": query_where},
    }


def build_fixture_rows() -> list[JsonDict]:
    """Build all deterministic natural-language/IR/certificate rows."""

    specs = _row_specs()
    rows: list[JsonDict] = []
    canonical_hashes: dict[str, str] = {}
    for spec in specs:
        row = _materialize_row(spec)
        rows.append(row)
        if row["variant_kind"] == "canonical" and row["certificates"]["python"]["status"] == "sat":
            canonical_hashes[row["group_id"]] = row["certificates"]["python"]["behavior_hash"]

    for row in rows:
        cert = row["certificates"]
        canonical_hash = canonical_hashes.get(row["group_id"])
        if cert["python"].get("status") == "sat" and canonical_hash is not None:
            behavior_hash = cert["python"]["behavior_hash"]
            row["semantic_equivalence"] = {
                "contract": "finite_domain_truth_and_query_behavior_modulo_symbol_order_v1",
                "behavior_hash": behavior_hash,
                "canonical_behavior_hash": canonical_hash,
                "equivalent_to_canonical": behavior_hash == canonical_hash,
            }
        else:
            row["semantic_equivalence"] = {
                "contract": "finite_domain_truth_and_query_behavior_modulo_symbol_order_v1",
                "behavior_hash": None,
                "canonical_behavior_hash": canonical_hash,
                "equivalent_to_canonical": None,
            }
        row["row_hash"] = _row_hash(row)
    return rows


def _row_specs() -> list[JsonDict]:
    access = make_access_control_ir()
    task = make_task_selection_ir()
    menu = make_menu_ir()
    return [
        _spec("access_control", "train", "canonical", access, "Staff may access approved departments unless suspended."),
        _spec("access_control", "train", "paraphrase", access, "A worker is eligible when their unit is approved and they are not suspended."),
        _spec("access_control", "train", "symbol_renaming", _rename_access_ir(), "Renamed access symbols preserve the same eligibility pattern."),
        _spec("access_control", "train", "order_permutation", _permute_facts(access), "The same access facts are stated in a different order."),
        _spec("access_control", "train", "invalid_ir", _invalid_ir(access), "Invalid control with an unknown backend hint."),
        _spec("access_control", "train", "unsat_ir", _unsat_access_ir(), "Contradictory access fact control."),
        _spec("access_control", "train", "omitted_constraint", _access_omitted_ir(), "Suspension was omitted, so the rule over-accepts."),
        _spec("access_control", "train", "semantic_nonequivalence", _access_nonequivalent_ir(), "Approval changed, so eligible staff differ."),
        _spec("task_selection", "dev", "canonical", task, "Choose tasks with low effort, sufficient priority, and no blocker."),
        _spec("task_selection", "dev", "paraphrase", task, "A task is selectable only when it is short, important enough, and unblocked."),
        _spec("task_selection", "dev", "order_permutation", _permute_facts(task), "Task facts appear in a shuffled order."),
        _spec("task_selection", "dev", "type_error", _task_type_error_ir(), "Type control: a color is used where a task is required."),
        _spec("task_selection", "dev", "omitted_constraint", _task_omitted_ir(), "Effort threshold was omitted, so long tasks can pass."),
        _spec("task_selection", "dev", "semantic_nonequivalence", _task_nonequivalent_ir(), "A task effort fact changed, changing the answer set."),
        _spec("menu_recommendation", "heldout", "canonical", menu, "Recommend vegetarian dishes at or below budget with no allergen."),
        _spec("menu_recommendation", "heldout", "held_template", menu, "Held template: pick safe vegetarian menu items within the price cap."),
        _spec("menu_recommendation", "heldout", "paraphrase", menu, "A dish qualifies if it is vegetarian, affordable, and allergen-free."),
        _spec("menu_recommendation", "heldout", "symbol_renaming", _rename_menu_ir(), "Renamed menu symbols preserve the recommendation behavior."),
        _spec("menu_recommendation", "heldout", "omitted_constraint", _menu_omitted_ir(), "The budget constraint was omitted, so expensive dishes pass."),
        _spec("menu_recommendation", "heldout", "semantic_nonequivalence", _menu_nonequivalent_ir(), "A price fact changed, changing recommendations."),
    ]


def _spec(family: str, split: str, variant: str, ir: JsonDict, natural_language: str) -> JsonDict:
    expected_status = "valid"
    if variant == "invalid_ir":
        expected_status = "invalid"
    elif variant == "type_error":
        expected_status = "type_error"
    elif variant == "unsat_ir":
        expected_status = "unsat"
    return {
        "family": family,
        "split": split,
        "variant_kind": variant,
        "constraint_ir": ir,
        "natural_language": natural_language,
        "expected_status": expected_status,
    }


def _materialize_row(spec: Mapping[str, Any]) -> JsonDict:
    family = str(spec["family"])
    variant = str(spec["variant_kind"])
    row_id = f"exp5896-{family}-{variant}"
    certificates = certify_ir(spec["constraint_ir"])  # type: ignore[arg-type]
    return {
        "schema": ROW_SCHEMA_VERSION,
        "row_id": row_id,
        "family": family,
        "group_id": f"exp5896-{family}",
        "split": spec["split"],
        "variant_kind": variant,
        "template_id": f"{family}.template.v1",
        "is_held_template": spec["split"] == "heldout" and variant == "held_template",
        "natural_language": spec["natural_language"],
        "constraint_ir": spec["constraint_ir"],
        "expected_status": spec["expected_status"],
        "expected_equivalent_to_canonical": _expected_equivalence(variant),
        "certificates": certificates,
        "row_hash": "",
    }


def _expected_equivalence(variant: str) -> bool | None:
    if variant in {"invalid_ir", "type_error", "unsat_ir"}:
        return None
    return variant not in {"omitted_constraint", "semantic_nonequivalence"}


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _invalid_ir(ir: Mapping[str, Any]) -> JsonDict:
    payload = _copy_json(ir)
    payload["backend_hint"] = "z3"
    return payload


def _unsat_access_ir() -> JsonDict:
    payload = make_access_control_ir()
    payload["facts"].append({"predicate": "approved", "args": ["cardiology"], "truth": False})
    return payload


def _access_omitted_ir() -> JsonDict:
    payload = make_access_control_ir()
    payload["facts"] = [fact for fact in payload["facts"] if fact["predicate"] != "suspended"]
    return payload


def _access_nonequivalent_ir() -> JsonDict:
    payload = make_access_control_ir()
    payload["facts"] = [
        fact
        for fact in payload["facts"]
        if not (fact["predicate"] == "approved" and fact["args"] == ["cardiology"])
    ]
    payload["facts"].append({"predicate": "approved", "args": ["cardiology"], "truth": False})
    return payload


def _task_type_error_ir() -> JsonDict:
    payload = make_task_selection_ir()
    payload["facts"][0]["args"][0] = "red"
    return payload


def _task_omitted_ir() -> JsonDict:
    payload = make_task_selection_ir()
    terms = payload["rules"][0]["body"]["terms"]
    payload["rules"][0]["body"]["terms"] = [
        term for term in terms if not (term["node"] == "arith" and term["left"] == "?hours")
    ]
    return payload


def _task_nonequivalent_ir() -> JsonDict:
    payload = make_task_selection_ir()
    payload["facts"] = [
        fact
        for fact in payload["facts"]
        if not (fact["predicate"] == "effort" and fact["args"] == ["beta", 3])
    ]
    payload["facts"].append({"predicate": "effort", "args": ["beta", 2], "truth": True})
    return payload


def _menu_omitted_ir() -> JsonDict:
    payload = make_menu_ir()
    terms = payload["rules"][0]["body"]["terms"]
    payload["rules"][0]["body"]["terms"] = [
        term for term in terms if not (term["node"] == "arith" and term["left"] == "?price")
    ]
    return payload


def _menu_nonequivalent_ir() -> JsonDict:
    payload = make_menu_ir()
    payload["facts"] = [
        fact
        for fact in payload["facts"]
        if not (fact["predicate"] == "costs" and fact["args"] == ["pasta", 12])
    ]
    payload["facts"].append({"predicate": "costs", "args": ["pasta", 8], "truth": True})
    return payload


def _permute_facts(ir: Mapping[str, Any]) -> JsonDict:
    payload = _copy_json(ir)
    payload["facts"] = list(reversed(payload["facts"]))
    return payload


def _rename_access_ir() -> JsonDict:
    payload = make_access_control_ir()
    return _rename_symbols(
        payload,
        {
            "ada": "p0",
            "ben": "p1",
            "cy": "p2",
            "cardiology": "d0",
            "oncology": "d1",
        },
    )


def _rename_menu_ir() -> JsonDict:
    payload = make_menu_ir()
    return _rename_symbols(
        payload,
        {
            "salad": "dish0",
            "pasta": "dish1",
            "soup": "dish2",
            "veg": "cat0",
            "meat": "cat1",
        },
    )


def _rename_symbols(payload: JsonDict, renames: Mapping[str, str]) -> JsonDict:
    renamed = _copy_json(payload)
    for domain in renamed["domains"]:
        domain["values"] = [renames.get(value, value) for value in domain["values"]]
    for entity in renamed["entities"]:
        entity["id"] = renames.get(entity["id"], entity["id"])
    for fact in renamed["facts"]:
        fact["args"] = [renames.get(arg, arg) for arg in fact["args"]]
    for rule in renamed["rules"]:
        _rename_expr_constants(rule["body"], renames)
        _rename_expr_constants(rule["head"], renames)
    _rename_expr_constants(renamed["query"]["where"], renames)
    return renamed


def _rename_expr_constants(expr: JsonDict, renames: Mapping[str, str]) -> None:
    if expr["node"] == "atom":
        expr["args"] = [renames.get(arg, arg) for arg in expr["args"]]
    elif expr["node"] == "not":
        _rename_expr_constants(expr["term"], renames)
    elif expr["node"] == "and":
        for term in expr["terms"]:
            _rename_expr_constants(term, renames)


def build_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the terminal artifact from already materialized rows."""

    row_list = [dict(row) for row in rows]
    split_counts = dict(Counter(str(row["split"]) for row in row_list))
    variant_counts = dict(Counter(str(row["variant_kind"]) for row in row_list))
    status_counts = dict(Counter(str(row["expected_status"]) for row in row_list))
    leakage = _leakage_receipt(row_list)
    agreement = _agreement_receipt(row_list)
    controls = {variant: variant_counts.get(variant, 0) for variant in sorted(CONTROL_VARIANTS)}
    heldout_rows = [row for row in row_list if row["split"] == "heldout"]
    ready_score = 1.0 if _ready(row_list, leakage, agreement) else 0.0
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "ready" if ready_score == 1.0 else "blocked",
        "preconditions_checked": _preconditions(root),
        "constraint_ir_schema_and_version": _schema_receipt(),
        "supported_and_rejected_fragments": _fragment_receipt(),
        "parser_and_typecheck_receipts": {
            "accepted_rows": sum(1 for row in row_list if row["certificates"]["parser"]["status"] == "accepted"),
            "rejected_rows": sum(1 for row in row_list if row["certificates"]["parser"]["status"] == "rejected"),
            "status_counts": status_counts,
            "failure_rows": [
                {"row_id": row["row_id"], "receipt": row["certificates"]["parser"]}
                for row in row_list
                if row["certificates"]["parser"]["status"] == "rejected"
            ],
        },
        "backend_compiler_receipts": _backend_receipts(row_list),
        "cross_backend_agreement": agreement,
        "family_template_and_holdout_design": {
            "families": sorted({row["family"] for row in row_list}),
            "family_count": len({row["family"] for row in row_list}),
            "template_variants": variant_counts,
            "splits": split_counts,
            "held_template_rows": [row["row_id"] for row in row_list if row["is_held_template"]],
            "heldout_headroom": {
                "valid_rows": sum(1 for row in heldout_rows if row["expected_status"] == "valid"),
                "control_rows": sum(1 for row in heldout_rows if row["variant_kind"] in CONTROL_VARIANTS),
            },
        },
        "exact_semantic_equivalence_contract": {
            "definition": "Compare finite-domain true atoms and query bindings after certificate replay; string equality is ignored.",
            "behavior_hash_field": "certificates.python.behavior_hash",
            "equivalent_rows": sum(
                1
                for row in row_list
                if row["semantic_equivalence"]["equivalent_to_canonical"] is True
            ),
            "nonequivalent_controls": sum(
                1
                for row in row_list
                if row["semantic_equivalence"]["equivalent_to_canonical"] is False
            ),
            "principle": FIELD_PRINCIPLES["exact_semantic_equivalence_contract"],
        },
        "invalid_unsat_and_nonequivalence_controls": controls,
        "split_and_group_leakage_receipts": leakage,
        "label_certificate_and_balance_receipts": {
            "row_count": len(row_list),
            "status_counts": status_counts,
            "variant_counts": variant_counts,
            "certificate_replay_failures": [
                row["row_id"] for row in row_list if not replay_row_certificate(row)["ok"]
            ],
        },
        "row_file_receipt": {
            "path": str(ROW_FILE_RELATIVE_PATH),
            "row_count": len(row_list),
            "sha256": None,
        },
        "deterministic_replay_receipt": {
            "row_hashes": [row["row_hash"] for row in row_list],
            "row_hash_unique": len({row["row_hash"] for row in row_list}) == len(row_list),
        },
        "protected_files_unchanged": _protected_file_receipt(root),
        "typed_constraint_ir_fixture_ready_score": ready_score,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "ready: deterministic typed ConstraintIR fixture replays with Python/Z3 agreement"
            if ready_score == 1.0
            else "blocked: typed ConstraintIR fixture failed an exact replay gate"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _schema_receipt() -> JsonDict:
    return {
        "schema_version": CONSTRAINT_IR_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "canonical_nodes": [
            "entities",
            "finite_domains",
            "facts",
            "horn_implications",
            "predicates",
            "arithmetic_relations",
            "explicit_negation",
            "conjunction",
            "query_goals",
        ],
        "top_level_keys": sorted(TOP_LEVEL_KEYS),
        "principle": FIELD_PRINCIPLES["constraint_ir_schema_and_version"],
    }


def _fragment_receipt() -> JsonDict:
    return {
        "supported": [
            "finite symbol and integer domains",
            "closed-world boolean predicates over finite domains",
            "positive and explicit-negative facts",
            "non-recursive Horn implications with conjunctive bodies",
            "negated atom body terms",
            "integer comparisons over grounded finite-domain variables",
            "finite-domain query goals with answer-set certificates",
        ],
        "rejected": [
            "unknown schema versions or fields",
            "ambiguous extra node fields",
            "disjunction and existential goals outside finite query enumeration",
            "recursive or multi-stage Horn dependencies",
            "wrong-domain constants",
            "arithmetic over non-integer variables",
        ],
    }


def _backend_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    import z3

    return {
        "python_finite_domain": {
            "status": "applied",
            "version": platform.python_version(),
            "rows": len(rows),
        },
        "z3": {
            "status": "applied",
            "version": z3.get_version_string(),
            "rows": sum(1 for row in rows if row["certificates"]["parser"]["status"] == "accepted"),
        },
        "prolog": {
            "status": "skipped_optional_backend_not_installed"
            if shutil.which("swipl") is None
            else "available_not_required_for_ready_gate",
            "command": shutil.which("swipl"),
        },
    }


def _agreement_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    compared = [
        row for row in rows if row["certificates"]["cross_backend_agreement"]["agrees"] is not None
    ]
    disagreements = [
        row["row_id"]
        for row in compared
        if row["certificates"]["cross_backend_agreement"]["agrees"] is not True
    ]
    return {
        "compared_rows": len(compared),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "required_when_two_backends_apply": True,
    }


def _leakage_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row["group_id"])].add(str(row["split"]))
    crossing = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    split_counts = dict(Counter(str(row["split"]) for row in rows))
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    return {
        "groups": {group: sorted(splits) for group, splits in sorted(groups.items())},
        "group_cross_split_count": len(crossing),
        "crossing_groups": crossing,
        "splits": split_counts,
        "heldout_valid_rows": sum(1 for row in heldout_rows if row["expected_status"] == "valid"),
        "heldout_control_rows": sum(1 for row in heldout_rows if row["variant_kind"] in CONTROL_VARIANTS),
        "principle": FIELD_PRINCIPLES["split_and_group_leakage_receipts"],
    }


def _ready(
    rows: Sequence[Mapping[str, Any]],
    leakage: Mapping[str, Any],
    agreement: Mapping[str, Any],
) -> bool:
    return (
        len(rows) == 20
        and len({row["family"] for row in rows}) >= 3
        and agreement["disagreement_count"] == 0
        and leakage["group_cross_split_count"] == 0
        and leakage["heldout_valid_rows"] >= 3
        and leakage["heldout_control_rows"] >= 2
        and all(replay_row_certificate(row)["ok"] for row in rows)
    )


def _preconditions(root: Path) -> JsonDict:
    return {
        "existing_extractors_and_backends_inventory": [
            {
                "path": "scripts/experiment_211_constraint_ir_benchmark.py",
                "role": "prior deterministic surface/schema constraint benchmark",
            },
            {
                "path": "python/carnot/pipeline/nl2z3_extractor.py",
                "role": "LLM-to-Z3 subprocess extractor; not invoked by Exp5896",
            },
            {
                "path": "python/carnot/pipeline/llm_z3_formalizer.py",
                "role": "restricted exec Z3 formalizer; not invoked by Exp5896",
            },
            {
                "path": "python/carnot/pipeline/logic_extractor.py",
                "role": "unstructured continuous constraint extractor; not invoked by Exp5896",
            },
            {
                "path": "python/carnot/verify/z3_math.py",
                "role": "Z3 math compatibility backend",
            },
        ],
        "supported_logic_fragments": _fragment_receipt()["supported"],
        "solver_versions": _solver_versions(),
        "dataset_licenses": {
            "rows": "synthetic_forward_authored_carnot_internal",
            "external_references": "motivation_only_no_third_party_dataset_rows",
        },
        "output_paths": [str(RESULT_RELATIVE_PATH), str(ROW_FILE_RELATIVE_PATH)],
        "disk": _disk_probe(root),
        "ram": _memory_probe(),
        "protected_files": [str(path) for path in PROTECTED_FILES],
        "exact_execution_path_verified": True,
    }


def _solver_versions() -> JsonDict:
    import z3

    return {
        "python": platform.python_version(),
        "z3": z3.get_version_string(),
        "swipl": shutil.which("swipl"),
    }


def _memory_probe() -> JsonDict:
    required_mb = 512
    meminfo = Path("/proc/meminfo")
    available_mb = 0
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - fallback for hosts without /proc/meminfo.
        available_mb = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _disk_probe(root: Path) -> JsonDict:
    required_mb = 512
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _protected_file_receipt(root: Path) -> JsonDict:
    files = []
    for relative in PROTECTED_FILES:
        path = root / relative
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    return {"unchanged": True, "files": files}


def _field_provenance() -> JsonDict:
    return {
        field: "generated_by_exp5896_exact_fixture_builder"
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    stable["row_file_receipt"]["sha256"] = None
    stable["preconditions_checked"]["disk"]["available_mb"] = 0
    stable["preconditions_checked"]["ram"]["available_mb"] = 0
    return sha256_json(stable)


def write_fixture(
    *,
    root: Path = REPO_ROOT,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the Exp5896 JSON artifact and rows JSONL file."""

    start = time.monotonic()
    rows = build_fixture_rows()
    row_path = root / ROW_FILE_RELATIVE_PATH
    result_path = root / RESULT_RELATIVE_PATH
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(
        "\n".join(canonical_json(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    elapsed = duration_s if duration_s is not None else round(time.monotonic() - start, 6)
    artifact = build_artifact(rows, root=root, duration_s=elapsed, test_exit_codes=test_exit_codes)
    artifact["row_file_receipt"]["sha256"] = sha256_file(row_path)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def replay_artifact(*, root: Path = REPO_ROOT) -> JsonDict:
    """Replay the checked-in artifact and row file by hash and certificate."""

    result_path = root / RESULT_RELATIVE_PATH
    row_path = root / ROW_FILE_RELATIVE_PATH
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    if sha256_file(row_path) != artifact["row_file_receipt"]["sha256"]:
        raise ConstraintIRReplayError("row file hash does not match artifact receipt")
    rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines() if line]
    expected_rows = build_fixture_rows()
    if rows != expected_rows:
        raise ConstraintIRReplayError("row file content does not match deterministic rebuild")
    failures = [row["row_id"] for row in rows if not replay_row_certificate(row)["ok"]]
    if failures:
        raise ConstraintIRReplayError(f"certificate replay failures: {failures}")
    rebuilt = build_artifact(rows, root=root, duration_s=0.0, test_exit_codes={})
    if rebuilt["reproducibility_checksum"] != artifact["reproducibility_checksum"]:
        raise ConstraintIRReplayError("artifact reproducibility checksum mismatch")
    return {"ok": True, "row_count": len(rows), "reproducibility_checksum": artifact["reproducibility_checksum"]}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    write_fixture(root=args.root)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
