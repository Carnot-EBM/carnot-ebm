"""Build the frozen Exp6661 trigger-switched tail fixture.

The fixture separates output syntax from exact task meaning. One generic JSON
grammar can shape a short tail, but three independent checker functions decide
whether the transported certificate is correct. This task loads no model.

Spec: REQ-CONSTRAINT-6661 and SCENARIO-CONSTRAINT-6661-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from carnot import experiment_6604_exact_two_level_plan_corpus as plan_fixture


JsonDict = dict[str, Any]
CommandRunner = Callable[[list[str], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260827"
RESULT_PATH = Path("results/experiment_6661_triggered_tail_fixture.json")
MODULE_PATH = Path("python/carnot/experiment_6661_triggered_tail_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_6661_triggered_tail_fixture.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")

SOURCE_CORPUS_PATHS = (
    Path("results/experiment_6649_exact_certificate_proposal_corpus.json"),
    Path("results/experiment_6650_twin_prefix_verifier_map.json"),
    Path("results/experiment_5923_sota_schema_supported_constraintir_ab.json"),
    Path("results/experiment_6604_exact_two_level_plan_corpus.json"),
    Path("results/experiment_6590_qwen36_constraint_first_stream.json"),
    Path("results/experiment_6591_gemma4_31b_constraint_first_stream.json"),
    Path("python/carnot/experiment_6604_exact_two_level_plan_corpus.py"),
    Path("python/carnot/inference/grammar.py"),
    Path("python/carnot/reporting/sota_gguf_tokenizer_runtime_receipt_3338.py"),
)
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

INFERENCE_SUBSTRATE = "cpu_fixture_and_exact_checker_no_llm"
TRIGGER_TOKEN = "<|CARNOT_SYNTAX_TAIL|>"
NATURAL_MARKER = "FINAL CERTIFICATE:"
RANDOM_SEED = 6_661_027
EXPECTED_TASK_COUNT = 18
FAMILY_ORDER = ("scheduling", "graph_constraints", "arithmetic_logic")
ARM_ORDER = ("natural", "immediate_json", "triggered_tail")
ATTACK_TYPES = (
    "answer_permutation",
    "label_renaming",
    "task_id_removal",
    "grammar_only_generation",
    "trigger_collision",
    "premature_trigger",
    "missing_trigger",
    "malformed_tail",
    "unknown_fields",
    "semantically_wrong_syntactically_valid_tail",
)
EXPECTED_ATTACK_ROW_COUNT = EXPECTED_TASK_COUNT * len(ARM_ORDER) * len(ATTACK_TYPES)

PARSER_VERSIONS = {
    "natural": "carnot.natural_certificate_parser.v1",
    "immediate_json": "carnot.strict_immediate_json_parser.v1",
    "triggered_tail": "carnot.strict_triggered_tail_parser.v1",
}
TAIL_SCHEMA: JsonDict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"certificate": {"type": "string"}},
    "required": ["certificate"],
    "additionalProperties": False,
}
SYNTAX_ONLY_GBNF = r"""
root ::= "{" ws "\"certificate\"" ws ":" ws string "}" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
ws ::= [ \t\n\r]*
""".strip()

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "frozen_task_manifest",
    "arm_contracts",
    "syntax_only_grammar_receipt",
    "exact_checker_rows",
    "leakage_attack_rows",
    "triggered_tail_fixture_ready",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The terminal state records whether the deterministic fixture completed.",
    "honest_verdict": "The verdict reports fixture readiness and makes no model-quality claim.",
    "verdict_class": "The closed class makes ready infrastructure null evidence.",
    "gate_check_summary": "The first failed row, checker, parser, or leakage gate stays local.",
    "frozen_task_manifest": "Each task binds visible input, exact target, checker, seed, and hashes.",
    "arm_contracts": "Prompts, parsers, schemas, triggers, and budgets freeze before inference.",
    "syntax_only_grammar_receipt": "The receipt proves the grammar carries syntax but no answer meaning.",
    "exact_checker_rows": "Positive and negative controls test each independent exact authority.",
    "leakage_attack_rows": "Every required mutation remains visible for every task and arm.",
    "triggered_tail_fixture_ready": "One Boolean reduces complete rows without model judgment.",
    "per_unit_rows": "Tasks, arms, controls, and attacks remain independently recheckable.",
    "aggregate_row_recomputation": "Expected row keys and outcomes rebuild readiness from raw rows.",
    "preconditions_checked": "Input hashes, tools, resources, and the no-model substrate are measured.",
    "protected_files_unchanged": "Before and after hashes protect the roadmap and conductor.",
    "inference_substrate": "The declared substrate permits fixture and checker work but no model call.",
    "verifier_is_oracle": "Exact fixture labels define readiness, not later model quality.",
    "field_provenance": "Each field names its source, parser, function, and content hash.",
    "random_seed": "The fixed seed pins task and attack order.",
    "duration_s": "Monotonic time measures the real fixture build.",
    "tests_run": "Commands, exits, durations, and summaries reproduce verification.",
    "reproducibility_checksum": "The canonical final content hash detects any artifact change.",
}

VERIFICATION_COMMANDS = (
    (
        ".venv/bin/coverage",
        "run",
        "--rcfile=/dev/null",
        f"--include={MODULE_PATH.as_posix()}",
        "-m",
        "pytest",
        TEST_PATH.as_posix(),
        "-q",
        "--no-cov",
        "-n",
        "0",
        "-o",
        "addopts=",
    ),
    (
        ".venv/bin/coverage",
        "report",
        "--rcfile=/dev/null",
        f"--include={MODULE_PATH.as_posix()}",
        "--fail-under=100",
        "--show-missing",
    ),
    (".venv/bin/ruff", "check", MODULE_PATH.as_posix(), TEST_PATH.as_posix()),
    (".venv/bin/ruff", "format", "--check", MODULE_PATH.as_posix(), TEST_PATH.as_posix()),
    (
        ".venv/bin/python",
        "scripts/check_spec_coverage.py",
        TEST_PATH.as_posix(),
    ),
    (".venv/bin/pytest", "tests/python", "-q"),
)


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order and no optional whitespace."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed content hash so digest identity is explicit."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON instead of interpreter-specific object text."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash one required local file or report its absence without coercion."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def _without_hash(value: Mapping[str, Any], field: str) -> JsonDict:
    return {key: deepcopy(item) for key, item in value.items() if key != field}


def arm_contract_hash(contract: Mapping[str, Any]) -> str:
    """Hash an arm contract without trusting its stored self-hash."""

    return sha256_json(_without_hash(contract, "contract_sha256"))


def fixture_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one task fixture row without its stored digest."""

    return sha256_json(_without_hash(row, "row_sha256"))


def attack_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one adversarial row without its stored digest."""

    return sha256_json(_without_hash(row, "row_sha256"))


def _checker_control_hash(row: Mapping[str, Any]) -> str:
    return sha256_json(_without_hash(row, "row_sha256"))


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash every final field except the checksum that stores this digest."""

    return sha256_json(_without_hash(payload, "reproducibility_checksum"))


def _parse_assignments(certificate: str) -> tuple[dict[str, int] | None, str | None]:
    if not isinstance(certificate, str) or not certificate:
        return None, "empty_or_non_text_certificate"
    assignments: dict[str, int] = {}
    for part in certificate.split(";"):
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)=(-?[0-9]+)", part)
        if match is None:
            return None, "malformed_assignment"
        label, raw_value = match.groups()
        if label in assignments:
            return None, "duplicate_assignment"
        assignments[label] = int(raw_value)
    return assignments, None


def check_scheduling_certificate(task: Mapping[str, Any], certificate: str) -> JsonDict:
    """Run the independent Exp6604 executor without decoder compiler input."""

    outcome = plan_fixture.IndependentExactExecutor().execute(
        task["checker_input"], certificate.replace(";", "\n")
    )
    return {
        "exact_valid": outcome["valid"] is True,
        "reason": outcome["reason"],
        "detail": deepcopy(outcome.get("detail", [])),
        "authority": plan_fixture.EXECUTOR_VERSION,
    }


def check_graph_certificate(task: Mapping[str, Any], certificate: str) -> JsonDict:
    """Check full graph coloring assignments from graph facts alone."""

    assignments, parse_error = _parse_assignments(certificate)
    definition = task["checker_input"]
    if assignments is None:
        return {
            "exact_valid": False,
            "reason": parse_error,
            "detail": [],
            "authority": "carnot.independent_graph_coloring_checker.v1",
        }
    nodes = [str(node) for node in definition["nodes"]]
    if set(assignments) != set(nodes):
        return {
            "exact_valid": False,
            "reason": "assignment_domain_mismatch",
            "detail": sorted(set(nodes) ^ set(assignments)),
            "authority": "carnot.independent_graph_coloring_checker.v1",
        }
    color_count = int(definition["color_count"])
    out_of_range = [node for node in nodes if assignments[node] not in range(1, color_count + 1)]
    if out_of_range:
        return {
            "exact_valid": False,
            "reason": "color_out_of_range",
            "detail": out_of_range,
            "authority": "carnot.independent_graph_coloring_checker.v1",
        }
    conflicts = [
        [left, right]
        for left, right in definition["edges"]
        if assignments[str(left)] == assignments[str(right)]
    ]
    return {
        "exact_valid": not conflicts,
        "reason": "valid_coloring" if not conflicts else "edge_color_conflict",
        "detail": conflicts,
        "authority": "carnot.independent_graph_coloring_checker.v1",
    }


def check_arithmetic_certificate(task: Mapping[str, Any], certificate: str) -> JsonDict:
    """Evaluate all integer equations without comparing with a stored answer."""

    assignments, parse_error = _parse_assignments(certificate)
    definition = task["checker_input"]
    if assignments is None:
        return {
            "exact_valid": False,
            "reason": parse_error,
            "detail": [],
            "authority": "carnot.independent_integer_equation_checker.v1",
        }
    variables = [str(name) for name in definition["variables"]]
    if set(assignments) != set(variables):
        return {
            "exact_valid": False,
            "reason": "assignment_domain_mismatch",
            "detail": sorted(set(variables) ^ set(assignments)),
            "authority": "carnot.independent_integer_equation_checker.v1",
        }
    lower, upper = (int(value) for value in definition["bounds"])
    out_of_range = [name for name in variables if not lower <= assignments[name] <= upper]
    if out_of_range:
        return {
            "exact_valid": False,
            "reason": "integer_out_of_range",
            "detail": out_of_range,
            "authority": "carnot.independent_integer_equation_checker.v1",
        }
    failures = []
    for index, equation in enumerate(definition["equations"]):
        observed = sum(
            int(coefficient) * assignments[str(variable)]
            for variable, coefficient in equation["coefficients"].items()
        )
        if observed != int(equation["rhs"]):
            failures.append({"equation": index, "observed": observed, "rhs": equation["rhs"]})
    return {
        "exact_valid": not failures,
        "reason": "all_equations_satisfied" if not failures else "equation_violation",
        "detail": failures,
        "authority": "carnot.independent_integer_equation_checker.v1",
    }


CHECKERS: dict[str, Callable[[Mapping[str, Any], str], JsonDict]] = {
    "scheduling": check_scheduling_certificate,
    "graph_constraints": check_graph_certificate,
    "arithmetic_logic": check_arithmetic_certificate,
}
CHECKER_VERSIONS = {
    "scheduling": plan_fixture.EXECUTOR_VERSION,
    "graph_constraints": "carnot.independent_graph_coloring_checker.v1",
    "arithmetic_logic": "carnot.independent_integer_equation_checker.v1",
}


def _checker_identity(family: str) -> JsonDict:
    function = CHECKERS[family]
    source = inspect.getsource(function).encode("utf-8")
    return {
        "name": CHECKER_VERSIONS[family],
        "function": f"{__name__}.{function.__name__}",
        "sha256": sha256_bytes(source),
        "executable": True,
        "independent_authority": True,
        "grammar_is_authority": False,
        "finite_answer_id_transport": False,
    }


def _task_with_hash(task: JsonDict) -> JsonDict:
    task["task_sha256"] = sha256_json(task)
    return task


def _schedule_tasks() -> list[JsonDict]:
    selected = [task for task in plan_fixture.generate_plan_tasks() if task["known_feasible"]][:6]
    rows: list[JsonDict] = []
    for index, source in enumerate(selected):
        prompt_lines = str(source["model_prompt_bytes"]).splitlines()[2:]
        rows.append(
            _task_with_hash(
                {
                    "task_id": f"schedule-{index:02d}",
                    "family": "scheduling",
                    "prompt": (
                        "Construct one executable schedule. Use canonical action calls only. "
                        "Put the calls in one certificate separated by semicolons.\n"
                        + "\n".join(prompt_lines)
                    ),
                    "target": str(source["gold_witness"]).replace("\n", ";"),
                    "checker": _checker_identity("scheduling"),
                    "seed": RANDOM_SEED + index,
                    "source_definition_sha256": str(source["source_sha256"]),
                    "checker_input": deepcopy(source),
                }
            )
        )
    return rows


GRAPH_DEFINITIONS = (
    {
        "nodes": ["a", "b", "c", "d", "e"],
        "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]],
        "color_count": 2,
        "target": "a=1;b=2;c=1;d=2;e=1",
    },
    {
        "nodes": ["p", "q", "r", "s", "t", "v"],
        "edges": [["p", "q"], ["q", "r"], ["r", "s"], ["s", "t"], ["t", "v"], ["v", "p"]],
        "color_count": 2,
        "target": "p=1;q=2;r=1;s=2;t=1;v=2",
    },
    {
        "nodes": ["red", "green", "blue"],
        "edges": [["red", "green"], ["green", "blue"], ["blue", "red"]],
        "color_count": 3,
        "target": "red=1;green=2;blue=3",
    },
    {
        "nodes": ["n1", "n2", "n3", "n4", "n5"],
        "edges": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"], ["n4", "n5"], ["n5", "n1"]],
        "color_count": 3,
        "target": "n1=1;n2=2;n3=1;n4=2;n5=3",
    },
    {
        "nodes": ["w", "x", "y", "z"],
        "edges": [["w", "x"], ["w", "y"], ["w", "z"], ["x", "y"], ["x", "z"], ["y", "z"]],
        "color_count": 4,
        "target": "w=1;x=2;y=3;z=4",
    },
    {
        "nodes": ["l1", "l2", "l3", "r1", "r2", "r3"],
        "edges": [[left, right] for left in ("l1", "l2", "l3") for right in ("r1", "r2", "r3")],
        "color_count": 2,
        "target": "l1=1;l2=1;l3=1;r1=2;r2=2;r3=2",
    },
)


def _graph_tasks() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, source in enumerate(GRAPH_DEFINITIONS):
        definition = {key: deepcopy(value) for key, value in source.items() if key != "target"}
        prompt = (
            f"Color every graph node with an integer from 1 through {definition['color_count']}. "
            "Adjacent nodes must have different colors. Return every node assignment.\n"
            f"Nodes: {canonical_json(definition['nodes'])}\n"
            f"Edges: {canonical_json(definition['edges'])}"
        )
        rows.append(
            _task_with_hash(
                {
                    "task_id": f"graph-{index:02d}",
                    "family": "graph_constraints",
                    "prompt": prompt,
                    "target": source["target"],
                    "checker": _checker_identity("graph_constraints"),
                    "seed": RANDOM_SEED + 100 + index,
                    "source_definition_sha256": sha256_json(definition),
                    "checker_input": definition,
                }
            )
        )
    return rows


ARITHMETIC_DEFINITIONS = (
    ("x=4;y=3", ["x", "y"], [({"x": 1, "y": 1}, 7), ({"x": 1, "y": -1}, 1)]),
    ("a=3;b=3", ["a", "b"], [({"a": 2, "b": 1}, 9), ({"a": 1, "b": -1}, 0)]),
    (
        "p=2;q=2;r=2",
        ["p", "q", "r"],
        [({"p": 1, "q": 1, "r": 1}, 6), ({"p": 1, "q": -1}, 0), ({"q": 1, "r": -1}, 0)],
    ),
    ("m=3;n=2", ["m", "n"], [({"m": 3, "n": 1}, 11), ({"m": 1, "n": 1}, 5)]),
    ("h=4;i=2", ["h", "i"], [({"h": 1, "i": 2}, 8), ({"h": 1, "i": -1}, 2)]),
    (
        "j=4;k=3;l=2",
        ["j", "k", "l"],
        [({"j": 1, "k": 1, "l": 1}, 9), ({"j": 1, "k": -1}, 1), ({"k": 1, "l": -1}, 1)],
    ),
)


def _equation_text(coefficients: Mapping[str, int], rhs: int) -> str:
    terms = [f"{coefficient}*{name}" for name, coefficient in coefficients.items()]
    return " + ".join(terms) + f" = {rhs}"


def _arithmetic_tasks() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, (target, variables, raw_equations) in enumerate(ARITHMETIC_DEFINITIONS):
        equations = [
            {"coefficients": dict(coefficients), "rhs": rhs} for coefficients, rhs in raw_equations
        ]
        definition = {"variables": list(variables), "equations": equations, "bounds": [-20, 20]}
        prompt = (
            "Find an integer assignment that satisfies every equation. Return each variable once.\n"
            + "\n".join(
                _equation_text(equation["coefficients"], int(equation["rhs"]))
                for equation in equations
            )
        )
        rows.append(
            _task_with_hash(
                {
                    "task_id": f"arithmetic-{index:02d}",
                    "family": "arithmetic_logic",
                    "prompt": prompt,
                    "target": target,
                    "checker": _checker_identity("arithmetic_logic"),
                    "seed": RANDOM_SEED + 200 + index,
                    "source_definition_sha256": sha256_json(definition),
                    "checker_input": definition,
                }
            )
        )
    return rows


def build_frozen_task_manifest() -> list[JsonDict]:
    """Build 18 stable task rows in the preregistered family order."""

    return [*_schedule_tasks(), *_graph_tasks(), *_arithmetic_tasks()]


def check_certificate(task: Mapping[str, Any], certificate: str) -> JsonDict:
    """Dispatch only by frozen family and never by a candidate answer ID."""

    checker = CHECKERS.get(str(task.get("family")))
    if checker is None:
        return {
            "exact_valid": False,
            "reason": "unknown_family",
            "detail": [],
            "authority": None,
        }
    return checker(task, certificate)


def wrong_certificate(task: Mapping[str, Any]) -> str:
    """Create one syntax-valid certificate that its exact checker rejects."""

    target = str(task["target"])
    family = str(task["family"])
    if family == "scheduling":
        calls = target.split(";")
        return ";".join([calls[1], calls[0], *calls[2:]])
    assignments, error = _parse_assignments(target)
    if assignments is None:  # pragma: no cover - frozen targets are checked during construction.
        raise ValueError(error)
    labels = list(assignments)
    if family == "graph_constraints":
        left, right = task["checker_input"]["edges"][0]
        assignments[str(right)] = assignments[str(left)]
    else:
        assignments[labels[0]] += 1
    return ";".join(f"{label}={assignments[label]}" for label in labels)


def build_arm_contracts() -> dict[str, JsonDict]:
    """Freeze prompts, parser identities, schemas, and matched token budgets."""

    contracts: dict[str, JsonDict] = {
        "natural": {
            "arm": "natural",
            "prompt_template": (
                "{task_prompt}\nReason freely. End with one line in the form "
                f"{NATURAL_MARKER} <certificate>."
            ),
            "parser": f"{__name__}._parse_natural",
            "parser_version": PARSER_VERSIONS["natural"],
            "schema": None,
            "trigger_token": None,
            "reasoning_token_budget": 256,
            "tail_token_budget": 0,
            "total_token_budget": 256,
            "syntax_enforcement_start": None,
        },
        "immediate_json": {
            "arm": "immediate_json",
            "prompt_template": (
                "{task_prompt}\nReturn only one JSON object with exactly one string field "
                'named "certificate".'
            ),
            "parser": f"{__name__}._parse_immediate_json",
            "parser_version": PARSER_VERSIONS["immediate_json"],
            "schema": deepcopy(TAIL_SCHEMA),
            "trigger_token": None,
            "reasoning_token_budget": 0,
            "tail_token_budget": 256,
            "total_token_budget": 256,
            "syntax_enforcement_start": 0,
        },
        "triggered_tail": {
            "arm": "triggered_tail",
            "prompt_template": (
                "{task_prompt}\nReason freely first. Then emit the exact trigger "
                f"{TRIGGER_TOKEN} once. After it, return only one JSON object with exactly "
                'one string field named "certificate".'
            ),
            "parser": f"{__name__}._parse_triggered_tail",
            "parser_version": PARSER_VERSIONS["triggered_tail"],
            "schema": deepcopy(TAIL_SCHEMA),
            "trigger_token": TRIGGER_TOKEN,
            "reasoning_token_budget": 192,
            "tail_token_budget": 64,
            "total_token_budget": 256,
            "syntax_enforcement_start": "after_exact_trigger",
        },
    }
    for contract in contracts.values():
        contract["contract_sha256"] = arm_contract_hash(contract)
    return contracts


def _parse_failure(reason: str, trigger_count: int = 0) -> JsonDict:
    return {
        "parsed": False,
        "failure": reason,
        "certificate": None,
        "trigger_count": trigger_count,
    }


def _parse_natural(output: str) -> JsonDict:
    if TRIGGER_TOKEN in output:
        return _parse_failure("trigger_forbidden")
    markers = [line for line in output.splitlines() if line.startswith(NATURAL_MARKER)]
    if not markers:
        return _parse_failure("natural_marker_missing")
    if len(markers) != 1:
        return _parse_failure("natural_marker_count")
    certificate = markers[0][len(NATURAL_MARKER) :].strip()
    if not certificate:
        return _parse_failure("empty_certificate")
    return {"parsed": True, "failure": None, "certificate": certificate, "trigger_count": 0}


class _DuplicateJsonField(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> JsonDict:
    result: JsonDict = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonField(key)
        result[key] = value
    return result


def _parse_json_tail(output: str, trigger_count: int) -> JsonDict:
    try:
        payload = json.loads(output, object_pairs_hook=_unique_json_object)
    except _DuplicateJsonField:
        return _parse_failure("duplicate_fields", trigger_count)
    except (json.JSONDecodeError, TypeError):
        return _parse_failure("json_malformed", trigger_count)
    if not isinstance(payload, Mapping):
        return _parse_failure("tail_not_object", trigger_count)
    if set(payload) != {"certificate"}:
        return _parse_failure("unknown_fields", trigger_count)
    certificate = payload["certificate"]
    if not isinstance(certificate, str):
        return _parse_failure("wrong_primitive_type", trigger_count)
    if not certificate:
        return _parse_failure("empty_certificate", trigger_count)
    return {
        "parsed": True,
        "failure": None,
        "certificate": certificate,
        "trigger_count": trigger_count,
    }


def _parse_immediate_json(output: str) -> JsonDict:
    if TRIGGER_TOKEN in output:
        return _parse_failure("trigger_forbidden")
    return _parse_json_tail(output, 0)


def _parse_triggered_tail(output: str) -> JsonDict:
    trigger_count = output.count(TRIGGER_TOKEN)
    if trigger_count == 0:
        return _parse_failure("missing_trigger")
    if trigger_count != 1:
        return _parse_failure("trigger_count", trigger_count)
    reasoning, tail = output.split(TRIGGER_TOKEN, 1)
    if not reasoning.strip():
        return _parse_failure("premature_trigger", trigger_count)
    return _parse_json_tail(tail.strip(), trigger_count)


def parse_arm_output(arm: str, output: Any) -> JsonDict:
    """Parse one arm without coercing bytes, objects, or unknown arm names."""

    if not isinstance(output, str):
        return _parse_failure("output_not_text")
    parsers = {
        "natural": _parse_natural,
        "immediate_json": _parse_immediate_json,
        "triggered_tail": _parse_triggered_tail,
    }
    parser = parsers.get(arm)
    return parser(output) if parser is not None else _parse_failure("unknown_arm")


def render_known_output(arm: str, certificate: str) -> str:
    """Render a known control through the exact frozen arm syntax."""

    tail = canonical_json({"certificate": certificate})
    if arm == "natural":
        return f"Reasoning complete.\n{NATURAL_MARKER} {certificate}"
    if arm == "immediate_json":
        return tail
    if arm == "triggered_tail":
        return f"Reasoning complete.\n{TRIGGER_TOKEN}\n{tail}"
    raise ValueError(f"unknown arm: {arm}")


def _semantic_labels(task: Mapping[str, Any]) -> list[str]:
    family = str(task["family"])
    definition = task["checker_input"]
    if family == "scheduling":
        return [
            str(value) for values in definition["argument_grammar"].values() for value in values
        ]
    if family == "graph_constraints":
        return [str(value) for value in definition["nodes"]]
    return [str(value) for value in definition["variables"]]


def _replace_strings(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        pattern = "|".join(re.escape(old) for old in sorted(replacements, key=len, reverse=True))
        return re.sub(pattern, lambda match: replacements[match.group(0)], value)
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, Mapping):
        return {
            replacements.get(str(key), str(key)): _replace_strings(item, replacements)
            for key, item in value.items()
        }
    return deepcopy(value)


def _label_renamed_variant(task: Mapping[str, Any]) -> JsonDict:
    labels = _semantic_labels(task)
    replacements = {label: f"renamed_{index}" for index, label in enumerate(labels)}
    variant = deepcopy(dict(task))
    for field in ("prompt", "target", "checker_input"):
        variant[field] = _replace_strings(variant[field], replacements)
    variant["task_sha256"] = sha256_json(_without_hash(variant, "task_sha256"))
    return variant


def build_syntax_only_grammar_receipt(manifest: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Prove one generic grammar is unchanged by all frozen task semantics."""

    grammar = SYNTAX_ONLY_GBNF
    task_ids_present = [
        str(task["task_id"]) for task in manifest if str(task["task_id"]) in grammar
    ]
    targets_present = [str(task["task_id"]) for task in manifest if str(task["target"]) in grammar]
    labels_present = sorted(
        {
            label
            for task in manifest
            for label in _semantic_labels(task)
            if label and json.dumps(label) in grammar
        }
    )
    sample = '{"certificate":""}'
    grammar_only_successes = sum(
        check_certificate(task, "")["exact_valid"] is True for task in manifest
    )
    renamed_labels_hold = all(
        check_certificate(_label_renamed_variant(task), _label_renamed_variant(task)["target"])[
            "exact_valid"
        ]
        is True
        for task in manifest
    )
    answer_semantics_absent = not (
        task_ids_present or targets_present or labels_present or grammar_only_successes
    )
    return {
        "backend": "llama_cpp_gbnf_syntax_only_v1",
        "grammar": grammar,
        "grammar_sha256": sha256_bytes(grammar.encode("utf-8")),
        "schema": deepcopy(TAIL_SCHEMA),
        "schema_sha256": sha256_json(TAIL_SCHEMA),
        "allowed_syntax": {
            "field_names": ["certificate"],
            "primitive_types": {"certificate": "string"},
        },
        "semantic_sources_read_by_grammar_builder": [],
        "task_ids_present": task_ids_present,
        "targets_present": targets_present,
        "labels_present": labels_present,
        "finite_answer_enumeration": False,
        "grammar_only_sample": sample,
        "grammar_only_exact_success_count": grammar_only_successes,
        "syntax_mutation_exact_label_invariance": renamed_labels_hold,
        "answer_semantics_absent": answer_semantics_absent and renamed_labels_hold,
        "proof_sha256": sha256_json(
            {
                "grammar_sha256": sha256_bytes(grammar.encode("utf-8")),
                "schema_sha256": sha256_json(TAIL_SCHEMA),
                "task_ids_present": task_ids_present,
                "targets_present": targets_present,
                "labels_present": labels_present,
                "grammar_only_exact_success_count": grammar_only_successes,
                "syntax_mutation_exact_label_invariance": renamed_labels_hold,
            }
        ),
    }


def build_fixture_rows(
    manifest: Sequence[Mapping[str, Any]], arm_contracts: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Build one exact-positive transport fixture row for every task."""

    rows: list[JsonDict] = []
    for task in manifest:
        arm_rows: dict[str, JsonDict] = {}
        for arm in ARM_ORDER:
            output = render_known_output(arm, str(task["target"]))
            parsed = parse_arm_output(arm, output)
            exact = check_certificate(task, str(parsed["certificate"]))
            prompt = str(arm_contracts[arm]["prompt_template"]).format(task_prompt=task["prompt"])
            arm_rows[arm] = {
                "arm": arm,
                "arm_contract_sha256": arm_contracts[arm]["contract_sha256"],
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "known_output": output,
                "known_output_sha256": sha256_bytes(output.encode("utf-8")),
                "parse_result": parsed,
                "exact_result": exact,
            }
        row: JsonDict = {
            "row_kind": "task_fixture",
            "task_id": task["task_id"],
            "family": task["family"],
            "task_sha256": task["task_sha256"],
            "arm_rows": arm_rows,
        }
        row["row_sha256"] = fixture_row_hash(row)
        rows.append(row)
    return rows


def build_exact_checker_rows(manifest: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run one positive and one negative control for every task checker."""

    rows: list[JsonDict] = []
    for task in manifest:
        for kind, candidate, expected in (
            ("known_positive", str(task["target"]), True),
            ("known_negative", wrong_certificate(task), False),
        ):
            result = check_certificate(task, candidate)
            row: JsonDict = {
                "row_kind": "checker_control",
                "task_id": task["task_id"],
                "family": task["family"],
                "control_kind": kind,
                "candidate": candidate,
                "candidate_sha256": sha256_bytes(candidate.encode("utf-8")),
                "checker": deepcopy(task["checker"]),
                "expected_exact_valid": expected,
                "observed_exact_valid": result["exact_valid"],
                "exact_result": result,
                "passed": result["exact_valid"] is expected,
            }
            row["row_sha256"] = _checker_control_hash(row)
            rows.append(row)
    return rows


def _transport_observation(task: Mapping[str, Any], arm: str, output: str) -> JsonDict:
    parsed = parse_arm_output(arm, output)
    exact_valid = None
    if parsed["parsed"] is True:
        exact_valid = check_certificate(task, str(parsed["certificate"]))["exact_valid"]
    return {
        "parsed": parsed["parsed"],
        "parse_failure": parsed["failure"],
        "exact_valid": exact_valid,
    }


def _attack_observation(
    task: Mapping[str, Any], arm: str, attack_type: str, grammar_hash: str
) -> tuple[JsonDict, JsonDict, bool]:
    target = str(task["target"])
    wrong = wrong_certificate(task)
    if attack_type in {"answer_permutation", "semantically_wrong_syntactically_valid_tail"}:
        observed = _transport_observation(task, arm, render_known_output(arm, wrong))
        expected = {"parsed": True, "exact_valid": False}
        leaked = observed["exact_valid"] is True
    elif attack_type == "label_renaming":
        renamed = _label_renamed_variant(task)
        renamed_exact = check_certificate(renamed, str(renamed["target"]))["exact_valid"]
        observed = {
            "parsed": parse_arm_output(arm, render_known_output(arm, str(renamed["target"])))[
                "parsed"
            ],
            "exact_label_unchanged": renamed_exact
            == check_certificate(task, target)["exact_valid"],
            "grammar_hash_unchanged": grammar_hash
            == sha256_bytes(SYNTAX_ONLY_GBNF.encode("utf-8")),
        }
        expected = {
            "parsed": True,
            "exact_label_unchanged": True,
            "grammar_hash_unchanged": True,
        }
        leaked = not (observed["exact_label_unchanged"] and observed["grammar_hash_unchanged"])
    elif attack_type == "task_id_removal":
        prompt = str(task["prompt"])
        observed = {
            **_transport_observation(task, arm, render_known_output(arm, target)),
            "task_id_present_before": str(task["task_id"]) in prompt,
            "task_id_present_after": str(task["task_id"])
            in prompt.replace(str(task["task_id"]), ""),
        }
        expected = {
            "parsed": True,
            "exact_valid": True,
            "task_id_present_before": False,
            "task_id_present_after": False,
        }
        leaked = observed["task_id_present_before"] is True
    elif attack_type == "grammar_only_generation":
        sample = '{"certificate":""}'
        if arm == "triggered_tail":
            sample = f"Reasoning without task semantics.\n{TRIGGER_TOKEN}\n{sample}"
        observed = _transport_observation(task, arm, sample)
        observed["answer_recovered"] = observed["exact_valid"] is True
        expected = {"answer_recovered": False}
        leaked = observed["answer_recovered"] is True
    elif attack_type == "trigger_collision":
        output = (
            f"reason {TRIGGER_TOKEN} collision {TRIGGER_TOKEN} "
            + canonical_json({"certificate": target})
            if arm == "triggered_tail"
            else f"{TRIGGER_TOKEN}\n" + render_known_output(arm, target)
        )
        observed = _transport_observation(task, arm, output)
        expected = {"parsed": False, "exact_valid": None}
        leaked = False
    elif attack_type == "premature_trigger":
        output = f"{TRIGGER_TOKEN}" + canonical_json({"certificate": target})
        observed = _transport_observation(task, arm, output)
        expected = {"parsed": False, "exact_valid": None}
        leaked = False
    elif attack_type == "missing_trigger":
        if arm == "triggered_tail":
            output = canonical_json({"certificate": target})
            expected = {"parsed": False, "exact_valid": None}
        else:
            output = render_known_output(arm, target)
            expected = {"parsed": True, "exact_valid": True}
        observed = _transport_observation(task, arm, output)
        leaked = False
    elif attack_type == "malformed_tail":
        malformed = "FINAL CERTIFICATE missing colon"
        if arm == "immediate_json":
            malformed = '{"certificate":'
        elif arm == "triggered_tail":
            malformed = f"reason\n{TRIGGER_TOKEN}\n" + '{"certificate":'
        observed = _transport_observation(task, arm, malformed)
        expected = {"parsed": False, "exact_valid": None}
        leaked = False
    elif attack_type == "unknown_fields":
        output = canonical_json({"certificate": target, "unknown": "field"})
        if arm == "triggered_tail":
            output = f"reason\n{TRIGGER_TOKEN}\n{output}"
        observed = _transport_observation(task, arm, output)
        expected = {"parsed": False, "exact_valid": None}
        leaked = False
    else:  # pragma: no cover - ATTACK_TYPES is a closed constant.
        raise ValueError(attack_type)
    return observed, expected, leaked


def build_leakage_attack_rows(
    manifest: Sequence[Mapping[str, Any]], arm_contracts: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Retain every required attack for each deterministic task-arm pair."""

    del arm_contracts
    grammar_hash = sha256_bytes(SYNTAX_ONLY_GBNF.encode("utf-8"))
    rows: list[JsonDict] = []
    for task in manifest:
        for arm in ARM_ORDER:
            for attack_type in ATTACK_TYPES:
                observed, expected, leaked = _attack_observation(
                    task, arm, attack_type, grammar_hash
                )
                passed = all(observed.get(key) == value for key, value in expected.items())
                row: JsonDict = {
                    "row_kind": "leakage_attack",
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "arm": arm,
                    "attack_type": attack_type,
                    "expected": expected,
                    "observed": observed,
                    "leakage_detected": leaked,
                    "passed": passed and not leaked,
                }
                row["row_sha256"] = attack_row_hash(row)
                rows.append(row)
    return rows


def recompute_aggregate_rows(
    *,
    manifest: Sequence[Mapping[str, Any]],
    arm_contracts: Mapping[str, Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
    exact_checker_rows: Sequence[Mapping[str, Any]],
    leakage_attack_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild readiness from complete expected row keys and row outcomes."""

    expected_tasks = {str(task["task_id"]) for task in manifest}
    expected_controls = {
        (task_id, kind)
        for task_id in expected_tasks
        for kind in ("known_positive", "known_negative")
    }
    expected_attacks = {
        (task_id, arm, attack)
        for task_id in expected_tasks
        for arm in ARM_ORDER
        for attack in ATTACK_TYPES
    }
    observed_tasks = {str(row.get("task_id")) for row in fixture_rows}
    observed_controls = {
        (str(row.get("task_id")), str(row.get("control_kind"))) for row in exact_checker_rows
    }
    observed_attacks = {
        (str(row.get("task_id")), str(row.get("arm")), str(row.get("attack_type")))
        for row in leakage_attack_rows
    }
    checks = {
        "manifest_task_count": len(manifest) == EXPECTED_TASK_COUNT,
        "family_balance": {
            family: sum(task.get("family") == family for task in manifest)
            for family in FAMILY_ORDER
        }
        == {family: 6 for family in FAMILY_ORDER},
        "arm_contract_keys": set(arm_contracts) == set(ARM_ORDER)
        and len(arm_contracts) == len(ARM_ORDER),
        "fixture_row_keys": observed_tasks == expected_tasks
        and len(fixture_rows) == len(expected_tasks),
        "fixture_outcomes": all(
            row.get("row_sha256") == fixture_row_hash(row)
            and all(
                arm_row.get("parse_result", {}).get("parsed") is True
                and arm_row.get("exact_result", {}).get("exact_valid") is True
                for arm_row in row.get("arm_rows", {}).values()
            )
            and set(row.get("arm_rows", {})) == set(ARM_ORDER)
            and len(row.get("arm_rows", {})) == len(ARM_ORDER)
            for row in fixture_rows
        ),
        "checker_control_keys": observed_controls == expected_controls
        and len(exact_checker_rows) == len(expected_controls),
        "checker_control_outcomes": all(
            row.get("passed") is True
            and row.get("observed_exact_valid") == row.get("expected_exact_valid")
            and row.get("row_sha256") == _checker_control_hash(row)
            for row in exact_checker_rows
        ),
        "attack_row_keys": observed_attacks == expected_attacks
        and len(leakage_attack_rows) == len(expected_attacks),
        "attack_outcomes": all(
            row.get("passed") is True
            and row.get("leakage_detected") is False
            and row.get("row_sha256") == attack_row_hash(row)
            for row in leakage_attack_rows
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    leakage_findings = sum(row.get("leakage_detected") is True for row in leakage_attack_rows)
    return {
        "ready": not failed and leakage_findings == 0,
        "checks": checks,
        "failed_checks": failed,
        "counts": {
            "tasks": len(manifest),
            "arm_contracts": len(arm_contracts),
            "fixture_rows": len(fixture_rows),
            "checker_controls": len(exact_checker_rows),
            "attack_rows": len(leakage_attack_rows),
            "expected_attack_rows": EXPECTED_ATTACK_ROW_COUNT,
            "leakage_findings": leakage_findings,
        },
    }


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash the active roadmap and conductor before any fixture work."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def protected_files_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    """Compare protected bytes after work so integrity is independently visible."""

    after = protected_hashes(root)
    return {"before": dict(before), "after": after, "unchanged": after == dict(before)}


def _cpu_description() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()  # pragma: no cover - non-Linux fallback.


def _ram_total_bytes() -> int:
    if hasattr(os, "sysconf"):
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    return 0  # pragma: no cover - supported Linux hosts provide sysconf.


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _llama_grammar_receipt() -> JsonDict:
    importable = importlib.util.find_spec("llama_cpp") is not None
    compiled = False
    error: str | None = None
    if importable:
        try:
            from llama_cpp import LlamaGrammar

            LlamaGrammar.from_string(SYNTAX_ONLY_GBNF, verbose=False)
            compiled = True
        except Exception as exc:  # pragma: no cover - a broken local backend is a blocker.
            error = f"{type(exc).__name__}: {exc}"
    return {
        "llama_cpp_importable": importable,
        "llama_cpp_python_version": _package_version("llama-cpp-python"),
        "llama_grammar_class": "llama_cpp.LlamaGrammar",
        "grammar_compiled": compiled,
        "grammar_error": error,
        "model_instantiated": False,
        "embedded_tokenizer_loaded": False,
    }


def collect_preconditions(root: Path, manifest: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure source hashes, exact identities, resources, and the no-model path."""

    disk = shutil.disk_usage(root)
    input_paths = (*PROTECTED_PATHS, *SOURCE_CORPUS_PATHS, SPEC_PATH, MODULE_PATH, TEST_PATH)
    input_hashes = {path.as_posix(): sha256_file(root / path) for path in input_paths}
    return {
        "planning_date": RUN_DATE,
        "input_hashes": input_hashes,
        "all_inputs_present": all(value != "missing" for value in input_hashes.values()),
        "task_manifest_sha256": sha256_json(list(manifest)),
        "checker_compiler_identities": {
            family: _checker_identity(family) for family in FAMILY_ORDER
        },
        "parser_versions": dict(PARSER_VERSIONS),
        "parser_runtime": {
            "python": platform.python_version(),
            "json": getattr(json, "__version__", "stdlib"),
        },
        "cpu": {"description": _cpu_description(), "logical_count": os.cpu_count() or 1},
        "ram": {"total_bytes": _ram_total_bytes()},
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "llama_cpp_helpers": _llama_grammar_receipt(),
        "embedded_tokenizer_helper": {
            "path": SOURCE_CORPUS_PATHS[-1].as_posix(),
            "sha256": sha256_file(root / SOURCE_CORPUS_PATHS[-1]),
            "vocab_only_model_loaded": False,
            "auto_tokenizer_used": False,
        },
        "no_llm_substrate": {
            "declared": INFERENCE_SUBSTRATE,
            "model_loaded": False,
            "model_inference_called": False,
            "grammar_helper_only": True,
        },
        "random_seed": RANDOM_SEED,
        "e2e_plan": {
            "path": "ops/e2e-test-plan.md",
            "applicable_ids": [],
            "reason": "The listed E2E cases cover model, binding, and learning pipelines. This fixture uses no model or runtime pipeline.",
        },
    }


def _field_provenance(root: Path) -> dict[str, JsonDict]:
    module_hash = sha256_file(root / MODULE_PATH)
    sources = {
        "frozen_task_manifest": "build_frozen_task_manifest",
        "arm_contracts": "build_arm_contracts",
        "syntax_only_grammar_receipt": "build_syntax_only_grammar_receipt",
        "exact_checker_rows": "build_exact_checker_rows",
        "leakage_attack_rows": "build_leakage_attack_rows",
        "aggregate_row_recomputation": "recompute_aggregate_rows",
        "protected_files_unchanged": "protected_files_receipt",
        "preconditions_checked": "collect_preconditions",
        "tests_run": "run_verification_commands",
    }
    return {
        field: {
            "source_path": MODULE_PATH.as_posix(),
            "parser": "python_json_strict_v1"
            if field != "arm_contracts"
            else "frozen_contract_builder_v1",
            "function": sources.get(field, "build_artifact"),
            "sha256": module_hash,
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _all_tests_pass(tests_run: Sequence[Mapping[str, Any]]) -> bool:
    return bool(tests_run) and all(row.get("exit_code") == 0 for row in tests_run)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    protected_before: Mapping[str, str] | None = None,
) -> JsonDict:
    """Build a terminal fixture artifact from deterministic raw rows."""

    started = time.monotonic()
    before = dict(protected_before or protected_hashes(root))
    manifest = build_frozen_task_manifest()
    arms = build_arm_contracts()
    grammar = build_syntax_only_grammar_receipt(manifest)
    fixtures = build_fixture_rows(manifest, arms)
    controls = build_exact_checker_rows(manifest)
    attacks = build_leakage_attack_rows(manifest, arms)
    aggregate = recompute_aggregate_rows(
        manifest=manifest,
        arm_contracts=arms,
        fixture_rows=fixtures,
        exact_checker_rows=controls,
        leakage_attack_rows=attacks,
    )
    preconditions = collect_preconditions(root, manifest)
    protected = protected_files_receipt(root, before)
    test_rows = [deepcopy(dict(row)) for row in tests_run]
    gate_checks = {
        "aggregate_rows": aggregate["ready"] is True,
        "semantic_free_grammar": grammar["answer_semantics_absent"] is True,
        "grammar_compiles": preconditions["llama_cpp_helpers"]["grammar_compiled"] is True,
        "inputs_present": preconditions["all_inputs_present"] is True,
        "tests": _all_tests_pass(test_rows),
        "protected_files": protected["unchanged"] is True,
    }
    failed = [name for name, passed in gate_checks.items() if not passed]
    ready = not failed
    first_failed = failed[0] if failed else None
    per_unit_rows = [
        *({"row_kind": "task", **deepcopy(task)} for task in manifest),
        *(
            {"row_kind": "arm_contract", "arm": arm, **deepcopy(contract)}
            for arm, contract in arms.items()
        ),
        *(deepcopy(row) for row in controls),
        *(deepcopy(row) for row in attacks),
    ]
    artifact: JsonDict = {
        "schema": "carnot.experiment_6661.triggered_tail_fixture.v1",
        "run_date": date,
        "spec_traces": [
            "REQ-CONSTRAINT-6661",
            "SCENARIO-CONSTRAINT-6661-DELAYED-SYNTAX",
            "SCENARIO-CONSTRAINT-6661-SEMANTIC-FREE-GRAMMAR",
            "SCENARIO-CONSTRAINT-6661-IMMUTABLE-MANIFEST",
            "SCENARIO-CONSTRAINT-6661-FAIL-CLOSED-PARSERS",
            "SCENARIO-CONSTRAINT-6661-ATTACK-AND-READINESS",
        ],
        "status": "complete" if ready else "blocked_fixture_contract",
        "honest_verdict": (
            "complete: triggered-tail fixture infrastructure is ready; this null result makes no model-quality claim"
            if ready
            else f"blocked_triggered_tail_fixture: {first_failed} failed"
        ),
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": {
            "passed": ready,
            "checks": gate_checks,
            "failed_checks": failed,
            "first_failed_check": first_failed,
            "observed": {
                "aggregate_failed_checks": aggregate["failed_checks"],
                "grammar_answer_semantics_absent": grammar["answer_semantics_absent"],
                "grammar_compile_error": preconditions["llama_cpp_helpers"]["grammar_error"],
                "missing_inputs": [
                    path
                    for path, digest in preconditions["input_hashes"].items()
                    if digest == "missing"
                ],
                "failed_test_commands": [
                    row.get("command") for row in test_rows if row.get("exit_code") != 0
                ],
                "protected_unchanged": protected["unchanged"],
            },
        },
        "frozen_task_manifest": manifest,
        "arm_contracts": arms,
        "syntax_only_grammar_receipt": grammar,
        "fixture_rows": fixtures,
        "exact_checker_rows": controls,
        "leakage_attack_rows": attacks,
        "triggered_tail_fixture_ready": ready,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(root),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.monotonic() - started, 6
        ),
        "tests_run": test_rows,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Reject incomplete rows, inconsistent gates, leakage, or content drift."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        return ["missing_required_fields"]
    errors: list[str] = []
    manifest = payload["frozen_task_manifest"]
    if not isinstance(manifest, list) or len(manifest) != EXPECTED_TASK_COUNT:
        errors.append("task_count")
    attacks = payload["leakage_attack_rows"]
    if not isinstance(attacks, list) or len(attacks) != EXPECTED_ATTACK_ROW_COUNT:
        errors.append("attack_row_count")
    controls = payload["exact_checker_rows"]
    if not isinstance(controls, list) or any(row.get("passed") is not True for row in controls):
        errors.append("checker_control_failed")
    if not isinstance(attacks, list) or any(row.get("passed") is not True for row in attacks):
        errors.append("attack_failed")
    try:
        recomputed = recompute_aggregate_rows(
            manifest=manifest,
            arm_contracts=payload["arm_contracts"],
            fixture_rows=payload.get("fixture_rows", []),
            exact_checker_rows=controls,
            leakage_attack_rows=attacks,
        )
    except (KeyError, TypeError, ValueError):
        recomputed = {"ready": False}
        errors.append("aggregate_recomputation_failed")
    if recomputed != payload["aggregate_row_recomputation"]:
        errors.append("aggregate_recomputation_mismatch")
    gate_ready = (
        recomputed.get("ready") is True
        and payload["syntax_only_grammar_receipt"].get("answer_semantics_absent") is True
        and payload["preconditions_checked"].get("all_inputs_present") is True
        and payload["preconditions_checked"].get("llama_cpp_helpers", {}).get("grammar_compiled")
        is True
        and _all_tests_pass(payload["tests_run"])
        and payload["protected_files_unchanged"].get("unchanged") is True
    )
    if payload["triggered_tail_fixture_ready"] is not gate_ready:
        errors.append("readiness_mismatch")
    if gate_ready and payload["verdict_class"] != "null":
        errors.append("verdict_class_mismatch")
    if payload["protected_files_unchanged"].get("unchanged") is not True:
        errors.append("protected_files_changed")
    if (
        not _all_tests_pass(payload["tests_run"])
        and payload["triggered_tail_fixture_ready"] is True
    ):
        errors.append("test_command_failed")
    provenance = payload["field_provenance"]
    if set(REQUIRED_ARTIFACT_FIELDS) - set(provenance) or any(
        not {"source_path", "parser", "function", "sha256", "principle"} <= set(row)
        for row in provenance.values()
    ):
        errors.append("field_provenance_missing")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload["verifier_is_oracle"] is not True:
        errors.append("oracle_boundary_mismatch")
    if payload["reproducibility_checksum"] != artifact_checksum(payload):
        errors.append("checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Sync complete JSON before one atomic replacement of the terminal path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _default_command_runner(command: list[str], cwd: Path) -> JsonDict:
    started = time.monotonic()
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    lines = output.splitlines()
    return {
        "command": " ".join(command),
        "exit_code": proc.returncode,
        "summary": "\n".join(lines[-12:]) if lines else "no output",
        "output_sha256": sha256_bytes(output.encode("utf-8")),
        "duration_s": round(time.monotonic() - started, 6),
    }


def run_verification_commands(
    root: Path, *, command_runner: CommandRunner | None = None
) -> list[JsonDict]:
    """Run each preregistered verification command and retain exact exits."""

    runner = command_runner or _default_command_runner
    return [runner(list(command), root) for command in VERIFICATION_COMMANDS]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Measure tests, build the fixture, validate it, and write it atomically."""

    started = time.monotonic()
    protected_before = protected_hashes(root)
    measured_tests = (
        [deepcopy(dict(row)) for row in tests_run]
        if tests_run is not None
        else run_verification_commands(root)
    )
    artifact = build_artifact(
        root=root,
        date=date,
        duration_s=time.monotonic() - started,
        tests_run=measured_tests,
        protected_before=protected_before,
    )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise ValueError("invalid Exp6661 artifact: " + ",".join(validation_errors))
    target = output_path or root / RESULT_PATH
    write_artifact_atomic(target, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    artifact = run(date=args.date, root=REPO_ROOT, output_path=output)
    print(
        canonical_json(
            {
                "status": artifact["status"],
                "triggered_tail_fixture_ready": artifact["triggered_tail_fixture_ready"],
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
