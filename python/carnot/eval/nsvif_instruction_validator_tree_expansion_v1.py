"""Exp 3017 NSVIF instruction validator-tree expansion.

Spec refs: REQ-VERIFY-3017, SCENARIO-VERIFY-3017.

This harness treats instruction-following verification as constraint
satisfaction.  It accepts only local executable checks as authority: JSON and
string runtime checks, Python AST checks, bounded Python runtime invariants,
and Z3 replay.  A single semantic-boundary node is included to make the
non-executable edge visible, but it is marked non-authoritative and never
decides whether a candidate passes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Z3 is a project dependency; absence is blocked honestly.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_FILENAME = "experiment_3017_nsvif_instruction_validator_tree_expansion_v1.json"
VALIDATOR_MANIFEST_REL_PATH = Path(
    "results/nsvif_instruction_validator_tree_expansion_3017/validator_manifest.jsonl"
)
Z3_TRANSCRIPT_REL_DIR = Path(
    "results/nsvif_instruction_validator_tree_expansion_3017/z3_transcripts"
)
RUNTIME_TRANSCRIPT_REL_DIR = Path(
    "results/nsvif_instruction_validator_tree_expansion_3017/runtime_transcripts"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME
MIN_INSTRUCTION_ITEMS = 20
COVERAGE_FLOOR = 0.90
INFERENCE_SUBSTRATE = "deterministic_nsvif_instruction_validator_tree_corpus"
NON_AUTHORITATIVE_AUTHORITY = "semantic_boundary_non_authoritative"
EXACT_AUTHORITIES = frozenset(
    {
        "runtime_json_parser",
        "runtime_string_checker",
        "python_ast_parser",
        "python_runtime_executor",
        "z3_solver",
    }
)
TERMINAL_SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "instruction_validator_tree_ready",
        "validator_manifest_path",
        "n_instruction_items",
        "n_validator_trees",
        "exact_check_coverage",
        "all_authoritative_nodes_exact_checked",
        "z3_transcript_paths",
        "runtime_transcript_paths",
        "rejected_items",
        "llm_judge_used",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class InstructionItem:
    """One deterministic instruction-following item with good and bad outputs."""

    item_id: str
    category: str
    instruction: str
    known_good_candidate: str
    known_bad_candidate: str
    nodes: tuple[JsonDict, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Output locations and clock hooks for deterministic Exp 3017 runs."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    manifest_path: Path | None = None
    z3_transcript_dir: Path | None = None
    runtime_transcript_dir: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / VALIDATOR_MANIFEST_REL_PATH

    def resolved_z3_transcript_dir(self) -> Path:
        return self.z3_transcript_dir or self.repo_root / Z3_TRANSCRIPT_REL_DIR

    def resolved_runtime_transcript_dir(self) -> Path:
        return self.runtime_transcript_dir or self.repo_root / RUNTIME_TRANSCRIPT_REL_DIR


def build_instruction_items(limit: int = MIN_INSTRUCTION_ITEMS) -> list[InstructionItem]:
    """Build the fixed 20-item instruction-following validator corpus."""

    items = [
        _item(
            "if-3017-001",
            "required_fields",
            "Return JSON with answer, confidence, and tags; answer must be SAFE.",
            {"answer": "SAFE", "confidence": 0.8, "tags": ["audit", "exact"]},
            {"answer": "SAFE", "tags": ["audit"]},
            [
                _json_required("answer", "confidence", "tags"),
                _json_equals("answer", "SAFE"),
                _json_number_between("confidence", 0.0, 1.0),
                _semantic_boundary("polite_tone"),
            ],
        ),
        _item(
            "if-3017-002",
            "required_fields",
            "Return JSON with name, email, and age; age must be from 18 to 65.",
            {"name": "Ada", "email": "ada@example.test", "age": 32},
            [],
            [_json_required("name", "email", "age"), _json_number_between("age", 18, 65)],
        ),
        _item(
            "if-3017-003",
            "forbidden_tokens",
            "Return JSON answer text without the tokens maybe or guess.",
            {"answer": "The checked result is stable."},
            {"answer": "Maybe this is a guess."},
            [_json_required("answer"), _json_forbidden_tokens("answer", "maybe", "guess")],
        ),
        _string_item(
            "if-3017-004",
            "forbidden_tokens",
            "Return a plain status line with no TODO marker.",
            "status: verified and final",
            "status: verified TODO",
            [_string_forbidden_tokens("TODO", "maybe")],
        ),
        _item(
            "if-3017-005",
            "ordering_constraints",
            "Return JSON steps ordered as parse, check, emit.",
            {"steps": ["parse", "check", "emit"]},
            {"steps": ["check", "parse", "emit"]},
            [_json_required("steps"), _json_list_order("steps", "parse", "check", "emit")],
        ),
        _string_item(
            "if-3017-006",
            "ordering_constraints",
            "Mention alpha before beta before gamma.",
            "alpha then beta then gamma",
            "alpha then gamma then beta",
            [_string_ordered_substrings("alpha", "beta", "gamma")],
        ),
        _item(
            "if-3017-007",
            "numeric_bounds",
            "Return JSON with score from 0 through 10.",
            {"score": 7},
            {"score": 12},
            [_json_required("score"), _json_number_between("score", 0, 10)],
        ),
        _item(
            "if-3017-008",
            "numeric_bounds",
            "Return JSON with temperature_c from -10 through 40.",
            {"temperature_c": 21.5},
            {"temperature_c": -12},
            [_json_required("temperature_c"), _json_number_between("temperature_c", -10, 40)],
        ),
        _item(
            "if-3017-009",
            "simple_transformations",
            "Return JSON whose output is the uppercase form of input.",
            {"input": "mars", "output": "MARS"},
            {"input": "mars", "output": "Mars"},
            [_json_required("input", "output"), _json_transform("input", "output", "uppercase")],
        ),
        _item(
            "if-3017-010",
            "simple_transformations",
            "Return JSON whose slug is the lowercase dash form of title.",
            {"title": "Red Team Check", "slug": "red-team-check"},
            {"title": "Red Team Check", "slug": "red_team_check"},
            [_json_required("title", "slug"), _json_transform("title", "slug", "slug")],
        ),
        _item(
            "if-3017-011",
            "simple_transformations",
            "Return JSON whose output reverses input.",
            {"input": "abcde", "output": "edcba"},
            {"input": "abcde", "output": "abcde"},
            [_json_required("input", "output"), _json_transform("input", "output", "reverse")],
        ),
        _item(
            "if-3017-012",
            "z3_relations",
            "Return integer x and y such that x + y = 10 and x < y.",
            {"x": 4, "y": 6},
            {"x": 5, "y": 5},
            [_z3_relation("x", "y", constraints=[("sum_eq", "x", "y", 10), ("lt", "x", "y", 0)])],
        ),
        _item(
            "if-3017-013",
            "z3_relations",
            "Return integer low and high such that 0 <= low < high <= 9.",
            {"low": 2, "high": 8},
            {"low": 8, "high": 2},
            [
                _z3_relation(
                    "low",
                    "high",
                    constraints=[
                        ("ge", "low", "", 0),
                        ("lt", "low", "high", 0),
                        ("le", "high", "", 9),
                    ],
                )
            ],
        ),
        _item(
            "if-3017-014",
            "z3_relations",
            "Return integer a and b such that a - b = 3 and b >= 0.",
            {"a": 8, "b": 5},
            {"a": 8, "b": 6},
            [_z3_relation("a", "b", constraints=[("diff_eq", "a", "b", 3), ("ge", "b", "", 0)])],
        ),
        _item(
            "if-3017-015",
            "z3_relations",
            "Return integer n such that n = 4 and n <= 4.",
            {"n": 4},
            {"n": 5},
            [_z3_relation("n", constraints=[("eq", "n", "", 4), ("le", "n", "", 4)])],
        ),
        _python_item(
            "if-3017-016",
            "python_ast",
            "Define normalize_slug(text) with one parameter and no imports.",
            "def normalize_slug(text):\n    return text.strip().lower().replace(' ', '-')\n",
            "def normalize_slug(text, extra):\n    return text\n",
            "normalize_slug",
            1,
            [("Hello World", "hello-world"), ("  Red Team  ", "red-team")],
        ),
        _python_item(
            "if-3017-017",
            "runtime_invariants",
            "Define clamp_score(x) with no imports; outputs clamp into 0..10.",
            "def clamp_score(x):\n    return max(0, min(10, x))\n",
            "def clamp_score(x):\n    return x\n",
            "clamp_score",
            1,
            [(-3, 0), (5, 5), (12, 10)],
        ),
        _python_item(
            "if-3017-018",
            "runtime_invariants",
            "Define is_even(n) with no imports; return True only for even integers.",
            "def is_even(n):\n    return n % 2 == 0\n",
            "import math\ndef is_even(n):\n    return True\n",
            "is_even",
            1,
            [(2, True), (3, False)],
        ),
        _python_item(
            "if-3017-019",
            "runtime_invariants",
            "Define first_letter(text) with no imports; return uppercase first letter.",
            "def first_letter(text):\n    return text[0].upper()\n",
            "def first_letter(text):\n    return text[0]\n",
            "first_letter",
            1,
            [("alpha", "A"), ("Beta", "B")],
        ),
        _python_item(
            "if-3017-020",
            "python_ast",
            "Define add_one(n) with one parameter and valid Python syntax.",
            "def add_one(n):\n    return n + 1\n",
            "def add_one(:\n    return 1\n",
            "add_one",
            1,
            [(4, 5), (-1, 0)],
        ),
    ]
    return items[:limit]


def build_rejected_items() -> list[JsonDict]:
    """Return rejected generated items with explicit non-authority reasons."""

    return [
        {
            "item_id": "rejected-ambiguous-friendly-tone",
            "rejection_reason": "ambiguous_instruction",
            "detail": "friendly tone has no deterministic acceptance boundary",
        },
        {
            "item_id": "rejected-random-id",
            "rejection_reason": "nondeterministic_validator",
            "detail": "candidate required a fresh random identifier",
        },
        {
            "item_id": "rejected-looks-helpful-label",
            "rejection_reason": "llm_only_label",
            "detail": "acceptance would have depended on an LLM helpfulness judgment",
        },
    ]


def build_validator_tree(item: InstructionItem) -> JsonDict:
    """Return an inspectable validator tree for one instruction item."""

    nodes = [dict(node, node_id=f"{item.item_id}:{node['kind']}:{index}") for index, node in enumerate(item.nodes)]
    return {
        "tree_id": item.item_id,
        "category": item.category,
        "instruction_sha256": sha256_text(item.instruction),
        "root": {"op": "all", "children": [node["node_id"] for node in nodes]},
        "nodes": nodes,
    }


def evaluate_validator_tree(
    validator_tree: Mapping[str, Any],
    candidate_text: str,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Execute every validator node and compute the authoritative verdict."""

    node_results = [
        _evaluate_node(node, candidate_text, z3_module=z3_module)
        for node in validator_tree["nodes"]
    ]
    failing = [
        str(row["node_id"])
        for row in node_results
        if row["authoritative"] and not row["accepted"]
    ]
    reasons = list(
        dict.fromkeys(
            str(row["rejection_reason"])
            for row in node_results
            if row["authoritative"] and row.get("rejection_reason")
        )
    )
    return {
        "accepted": not failing,
        "failing_node_ids": failing,
        "rejection_reasons": reasons,
        "node_results": node_results,
        "llm_judge_used": False,
    }


def run_experiment(config: ExperimentConfig | None = None, *, z3_module: Any = _z3) -> JsonDict:
    """Build, execute, validate, and persist Exp 3017 artifacts."""

    active = config or ExperimentConfig()
    started = active.start_time()
    manifest_rows: list[JsonDict] = []
    for item in build_instruction_items():
        tree = build_validator_tree(item)
        good = evaluate_validator_tree(tree, item.known_good_candidate, z3_module=z3_module)
        bad = evaluate_validator_tree(tree, item.known_bad_candidate, z3_module=z3_module)
        runtime_info = _write_transcript(
            active.resolved_runtime_transcript_dir(),
            item.item_id,
            _runtime_transcript(item, tree, good, bad),
        )
        z3_info: JsonDict = {}
        if _tree_has_z3_node(tree):
            z3_info = _write_transcript(
                active.resolved_z3_transcript_dir(),
                item.item_id,
                _z3_transcript(item, good, bad, z3_module=z3_module),
            )
        manifest_rows.append(_manifest_row(active.repo_root, item, tree, good, bad, runtime_info, z3_info))

    _write_jsonl(active.resolved_manifest_path(), manifest_rows)
    artifact = build_artifact(
        active,
        manifest_rows,
        duration_s=round(active.clock() - started, 6),
        rejected_items=build_rejected_items(),
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def build_artifact(
    config: ExperimentConfig,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    rejected_items: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal artifact and compute the exact authority gates."""

    z3_paths = [str(row["z3_transcript_path"]) for row in manifest_rows if row.get("z3_transcript_path")]
    runtime_paths = [str(row["runtime_transcript_path"]) for row in manifest_rows]
    n_items = len(manifest_rows)
    n_trees = len({row["validator_tree"]["tree_id"] for row in manifest_rows})
    coverage = _exact_check_coverage(manifest_rows)
    all_authoritative_exact = bool(manifest_rows) and all(
        row["all_authoritative_nodes_exact_checked"] for row in manifest_rows
    )
    all_good_pass = bool(manifest_rows) and all(row["known_good_validation"]["accepted"] for row in manifest_rows)
    all_bad_reject = bool(manifest_rows) and all(not row["known_bad_validation"]["accepted"] for row in manifest_rows)
    llm_used = any(bool(row.get("llm_judge_used")) for row in manifest_rows)
    ready = (
        n_items >= MIN_INSTRUCTION_ITEMS
        and n_trees == n_items
        and COVERAGE_FLOOR <= coverage < 1.0
        and all_authoritative_exact
        and all_good_pass
        and all_bad_reject
        and len(runtime_paths) == n_items
        and bool(z3_paths)
        and bool(rejected_items)
        and not llm_used
    )
    return {
        "schema": "carnot.nsvif_instruction_validator_tree_expansion.v1",
        "artifact": "experiment_3017_nsvif_instruction_validator_tree_expansion_v1",
        "run_date": RUN_DATE,
        "instruction_validator_tree_ready": ready,
        "validator_manifest_path": str(_relative_to(config.repo_root, config.resolved_manifest_path())),
        "n_instruction_items": n_items,
        "n_validator_trees": n_trees,
        "exact_check_coverage": coverage,
        "all_authoritative_nodes_exact_checked": all_authoritative_exact,
        "z3_transcript_paths": z3_paths,
        "runtime_transcript_paths": runtime_paths,
        "rejected_items": [dict(row) for row in rejected_items],
        "llm_judge_used": llm_used,
        "honest_verdict": (
            "complete: NSVIF instruction validator-tree corpus exact-checked"
            if ready
            else "blocked: instruction validator-tree corpus did not clear exact gates"
        ),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts": source_artifact_status(config.repo_root),
        "manifest_sha256": sha256_file(config.resolved_manifest_path()),
        "field_provenance": field_provenance(),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3017 artifact violates its terminal contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if artifact.get("llm_judge_used") is not False:
        raise ValueError("llm_judge_used must remain false")
    if int(artifact.get("n_instruction_items") or 0) < MIN_INSTRUCTION_ITEMS:
        raise ValueError("instruction corpus requires at least 20 items")
    if artifact.get("n_validator_trees") != artifact.get("n_instruction_items"):
        raise ValueError("n_validator_trees must equal n_instruction_items")
    coverage = float(artifact.get("exact_check_coverage") or 0.0)
    if not COVERAGE_FLOOR <= coverage < 1.0:
        raise ValueError("exact_check_coverage must expose exact checks plus semantic boundary")
    if artifact.get("all_authoritative_nodes_exact_checked") is not True:
        raise ValueError("all_authoritative_nodes_exact_checked must be true")
    if not artifact.get("z3_transcript_paths") or not artifact.get("runtime_transcript_paths"):
        raise ValueError("transcript paths must be present")
    if not artifact.get("rejected_items"):
        raise ValueError("rejected_items must be surfaced")
    if artifact.get("instruction_validator_tree_ready") is not True:
        raise ValueError("instruction_validator_tree_ready must be true")


def load_manifest(path: Path) -> list[JsonDict]:
    """Load an inspectable validator manifest JSONL file."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_artifact_status(repo_root: Path) -> JsonDict:
    """Summarize source artifacts that anchor the Exp 3017 line."""

    results = repo_root / "results"
    return {
        "exp2994": _summarize_json(
            results / "experiment_2994_prompt_validator_dialogue_schema_v1.json",
            ("prompt_validator_protocol_ready", "exact_verifier_authority_preserved"),
        ),
        "exp3005": _summarize_json(
            results / "experiment_3005_solver_to_validator_tree_expansion_v1.json",
            ("validator_tree_expanded", "all_trees_exact_checked", "n_solver_items"),
        ),
    }


def field_provenance() -> JsonDict:
    """Explain why each required terminal field exists."""

    return {
        "instruction_validator_tree_ready": {
            "principle": "Downstream frontier certificate must gate on a real corpus.",
            "satisfied_by": "20 accepted instruction items with exact authoritative nodes",
        },
        "exact_check_coverage": {
            "principle": "Semantic-only labels must not be hidden.",
            "satisfied_by": "one visible non-authoritative semantic-boundary node",
        },
        "llm_judge_used": {
            "principle": "LLM judgments must not become verifiers.",
            "satisfied_by": False,
        },
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _item(
    item_id: str,
    category: str,
    instruction: str,
    good: Any,
    bad: Any,
    nodes: Sequence[Mapping[str, Any]],
) -> InstructionItem:
    return InstructionItem(
        item_id=item_id,
        category=category,
        instruction=instruction,
        known_good_candidate=json.dumps(good, sort_keys=True),
        known_bad_candidate=json.dumps(bad, sort_keys=True),
        nodes=tuple(dict(node) for node in nodes),
    )


def _string_item(
    item_id: str,
    category: str,
    instruction: str,
    good: str,
    bad: str,
    nodes: Sequence[Mapping[str, Any]],
) -> InstructionItem:
    return InstructionItem(item_id, category, instruction, good, bad, tuple(dict(node) for node in nodes))


def _python_item(
    item_id: str,
    category: str,
    instruction: str,
    good: str,
    bad: str,
    function_name: str,
    parameter_count: int,
    cases: Sequence[tuple[Any, Any]],
) -> InstructionItem:
    return InstructionItem(
        item_id=item_id,
        category=category,
        instruction=instruction,
        known_good_candidate=good,
        known_bad_candidate=bad,
        nodes=(
            _python_signature(function_name, parameter_count),
            _python_no_imports(),
            _python_runtime_cases(function_name, cases),
        ),
    )


def _base_node(kind: str, authority: str, **payload: Any) -> JsonDict:
    return {"kind": kind, "authority": authority, "authoritative": True, **payload}


def _json_required(*fields: str) -> JsonDict:
    return _base_node("json_required_fields", "runtime_json_parser", fields=list(fields))


def _json_equals(field: str, expected: Any) -> JsonDict:
    return _base_node("json_field_equals", "runtime_json_parser", field=field, expected=expected)


def _json_number_between(field: str, minimum: float, maximum: float) -> JsonDict:
    return _base_node("json_number_between", "runtime_json_parser", field=field, minimum=minimum, maximum=maximum)


def _json_forbidden_tokens(field: str, *tokens: str) -> JsonDict:
    return _base_node("json_forbidden_tokens", "runtime_json_parser", field=field, forbidden=list(tokens))


def _json_list_order(field: str, *expected: str) -> JsonDict:
    return _base_node("json_list_order", "runtime_json_parser", field=field, expected=list(expected))


def _json_transform(input_field: str, output_field: str, transform: str) -> JsonDict:
    return _base_node(
        "json_transform",
        "runtime_json_parser",
        input_field=input_field,
        output_field=output_field,
        transform=transform,
    )


def _string_forbidden_tokens(*tokens: str) -> JsonDict:
    return _base_node("string_forbidden_tokens", "runtime_string_checker", forbidden=list(tokens))


def _string_ordered_substrings(*substrings: str) -> JsonDict:
    return _base_node("string_ordered_substrings", "runtime_string_checker", substrings=list(substrings))


def _z3_relation(*variables: str, constraints: Sequence[tuple[str, str, str, int]]) -> JsonDict:
    return _base_node(
        "z3_linear_relation",
        "z3_solver",
        variables=list(variables),
        constraints=[
            {"op": op, "left": left, "right": right, "value": value}
            for op, left, right, value in constraints
        ],
    )


def _python_signature(function_name: str, parameter_count: int) -> JsonDict:
    return _base_node(
        "python_function_signature",
        "python_ast_parser",
        function_name=function_name,
        parameter_count=parameter_count,
    )


def _python_no_imports() -> JsonDict:
    return _base_node("python_no_imports", "python_ast_parser")


def _python_runtime_cases(function_name: str, cases: Sequence[tuple[Any, Any]]) -> JsonDict:
    return _base_node(
        "python_runtime_cases",
        "python_runtime_executor",
        function_name=function_name,
        cases=[{"input": case_input, "expected": expected} for case_input, expected in cases],
    )


def _semantic_boundary(label: str) -> JsonDict:
    return {
        "kind": "semantic_boundary",
        "authority": NON_AUTHORITATIVE_AUTHORITY,
        "authoritative": False,
        "boundary_label": label,
        "exact_checked": False,
        "note": "Semantic tone is logged but excluded from verifier authority.",
    }


def _evaluate_node(node: Mapping[str, Any], candidate_text: str, *, z3_module: Any) -> JsonDict:
    kind = str(node["kind"])
    evaluators = {
        "json_required_fields": _evaluate_json_required_fields,
        "json_field_equals": _evaluate_json_field_equals,
        "json_number_between": _evaluate_json_number_between,
        "json_forbidden_tokens": _evaluate_json_forbidden_tokens,
        "json_list_order": _evaluate_json_list_order,
        "json_transform": _evaluate_json_transform,
        "string_forbidden_tokens": _evaluate_string_forbidden_tokens,
        "string_ordered_substrings": _evaluate_string_ordered_substrings,
        "python_function_signature": _evaluate_python_function_signature,
        "python_no_imports": _evaluate_python_no_imports,
        "python_runtime_cases": _evaluate_python_runtime_cases,
        "semantic_boundary": _evaluate_semantic_boundary,
    }
    if kind == "z3_linear_relation":
        return _evaluate_z3_linear_relation(node, candidate_text, z3_module)
    return evaluators[kind](node, candidate_text)


def _evaluate_json_required_fields(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    missing = [field for field in node["fields"] if field not in payload]
    return _node_result(node, not missing, None if not missing else "missing_required_field")


def _evaluate_json_field_equals(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    field = str(node["field"])
    if reason:
        return _node_result(node, False, reason)
    return _node_result(node, payload.get(field) == node["expected"], None if payload.get(field) == node["expected"] else "field_value_mismatch")


def _evaluate_json_number_between(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    field = str(node["field"])
    if reason:
        return _node_result(node, False, reason)
    value = payload.get(field)
    accepted = isinstance(value, int | float) and float(node["minimum"]) <= float(value) <= float(node["maximum"])
    return _node_result(node, accepted, None if accepted else "numeric_range_violation")


def _evaluate_json_forbidden_tokens(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    text = str(payload.get(str(node["field"]), "")).lower()
    forbidden = [token.lower() for token in node["forbidden"]]
    accepted = not any(token in text for token in forbidden)
    return _node_result(node, accepted, None if accepted else "forbidden_token_present")


def _evaluate_json_list_order(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    accepted = payload.get(str(node["field"])) == node["expected"]
    return _node_result(node, accepted, None if accepted else "ordering_constraint_violation")


def _evaluate_json_transform(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    source = str(payload.get(str(node["input_field"]), ""))
    transforms = {
        "uppercase": source.upper(),
        "reverse": source[::-1],
        "slug": re.sub(r"[^a-z0-9]+", "-", source.strip().lower()).strip("-"),
    }
    accepted = payload.get(str(node["output_field"])) == transforms[str(node["transform"])]
    return _node_result(node, accepted, None if accepted else "transform_mismatch")


def _evaluate_string_forbidden_tokens(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    accepted = not any(str(token) in candidate_text for token in node["forbidden"])
    return _node_result(node, accepted, None if accepted else "forbidden_token_present")


def _evaluate_string_ordered_substrings(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    cursor = -1
    for substring in node["substrings"]:
        position = candidate_text.find(str(substring), cursor + 1)
        if position < 0:
            return _node_result(node, False, "ordering_constraint_violation")
        cursor = position
    return _node_result(node, True, None)


def _evaluate_z3_linear_relation(node: Mapping[str, Any], candidate_text: str, z3_module: Any) -> JsonDict:
    if z3_module is None:  # pragma: no cover
        return _node_result(node, False, "z3_unavailable", z3_result={"z3_executed": False})
    payload, reason = _parse_json_object(candidate_text)
    if reason:
        return _node_result(node, False, reason, z3_result={"z3_executed": False})
    if not all(isinstance(payload.get(var), int) for var in node["variables"]):
        return _node_result(node, False, "missing_integer_assignment", z3_result={"z3_executed": False})
    variables = {var: z3_module.Int(var) for var in node["variables"]}
    solver = z3_module.Solver()
    for var, symbol in variables.items():
        solver.add(symbol == int(payload[var]))
    for constraint in node["constraints"]:
        solver.add(_z3_constraint(variables, constraint))
    status = str(solver.check())
    accepted = status == "sat"
    return _node_result(
        node,
        accepted,
        None if accepted else "z3_unsatisfied",
        z3_result={"z3_executed": True, "actual_solver_status": status},
    )


def _z3_constraint(variables: Mapping[str, Any], constraint: Mapping[str, Any]) -> Any:
    op = str(constraint["op"])
    left = variables[str(constraint["left"])]
    right_name = str(constraint.get("right") or "")
    right = variables[right_name] if right_name else int(constraint["value"])
    value = int(constraint["value"])
    if op == "sum_eq":
        return left + variables[right_name] == value
    if op == "diff_eq":
        return left - variables[right_name] == value
    if op == "lt":
        return left < right
    if op == "ge":
        return left >= right
    if op == "le":
        return left <= right
    return left == value


def _evaluate_python_function_signature(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    tree, reason = _parse_python_ast(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    function = _first_function(tree, str(node["function_name"]))
    accepted = function is not None and len(function.args.args) == int(node["parameter_count"])
    return _node_result(node, accepted, None if accepted else "function_signature_mismatch")


def _evaluate_python_no_imports(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    tree, reason = _parse_python_ast(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    accepted = not any(isinstance(item, ast.Import | ast.ImportFrom) for item in ast.walk(tree))
    return _node_result(node, accepted, None if accepted else "import_statement_disallowed")


def _evaluate_python_runtime_cases(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    namespace: dict[str, Any] = {}
    try:
        exec(compile(candidate_text, "<candidate>", "exec"), _safe_globals(), namespace)
        function = namespace[str(node["function_name"])]
        accepted = all(function(case["input"]) == case["expected"] for case in node["cases"])
    except Exception as exc:
        return _node_result(node, False, f"runtime_execution_error:{type(exc).__name__}")
    return _node_result(node, accepted, None if accepted else "runtime_invariant_violation")


def _evaluate_semantic_boundary(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    del candidate_text
    return _node_result(node, True, None, authoritative=False, exact_checked=False)


def _parse_json_object(candidate_text: str) -> tuple[JsonDict, str | None]:
    try:
        payload = json.loads(candidate_text)
    except json.JSONDecodeError:
        return {}, "json_parse_error"
    if not isinstance(payload, dict):
        return {}, "json_parse_error"
    return payload, None


def _parse_python_ast(candidate_text: str) -> tuple[ast.Module | None, str | None]:
    try:
        return ast.parse(candidate_text), None
    except SyntaxError:
        return None, "python_syntax_error"


def _first_function(tree: ast.Module | None, function_name: str) -> ast.FunctionDef | None:
    for item in tree.body if tree else []:
        if isinstance(item, ast.FunctionDef) and item.name == function_name:
            return item
    return None


def _safe_globals() -> JsonDict:
    return {"__builtins__": {"abs": abs, "bool": bool, "len": len, "max": max, "min": min, "str": str}}


def _node_result(
    node: Mapping[str, Any],
    accepted: bool,
    reason: str | None,
    *,
    authoritative: bool | None = None,
    exact_checked: bool | None = None,
    z3_result: Mapping[str, Any] | None = None,
) -> JsonDict:
    is_authoritative = bool(node.get("authoritative", True)) if authoritative is None else authoritative
    result = {
        "node_id": node["node_id"],
        "kind": node["kind"],
        "authority": node["authority"],
        "authoritative": is_authoritative,
        "exact_checked": (node["authority"] in EXACT_AUTHORITIES and is_authoritative) if exact_checked is None else exact_checked,
        "accepted": accepted,
        "rejection_reason": reason,
    }
    if z3_result is not None:
        result["z3_result"] = dict(z3_result)
    return result


def _tree_has_z3_node(tree: Mapping[str, Any]) -> bool:
    return any(node["authority"] == "z3_solver" for node in tree["nodes"])


def _runtime_transcript(
    item: InstructionItem,
    tree: Mapping[str, Any],
    good: Mapping[str, Any],
    bad: Mapping[str, Any],
) -> JsonDict:
    return {
        "item_id": item.item_id,
        "category": item.category,
        "validator_tree_sha256": sha256_text(json.dumps(tree, sort_keys=True)),
        "known_good_candidate_sha256": sha256_text(item.known_good_candidate),
        "known_bad_candidate_sha256": sha256_text(item.known_bad_candidate),
        "known_good_validation": good,
        "known_bad_validation": bad,
        "llm_judge_used": False,
    }


def _z3_transcript(
    item: InstructionItem,
    good: Mapping[str, Any],
    bad: Mapping[str, Any],
    *,
    z3_module: Any,
) -> JsonDict:
    return {
        "item_id": item.item_id,
        "z3_version": z3_module.get_version_string() if z3_module is not None else None,
        "known_good_z3_results": _z3_node_results(good),
        "known_bad_z3_results": _z3_node_results(bad),
        "llm_judge_used": False,
    }


def _z3_node_results(feedback: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(row) for row in feedback["node_results"] if row["authority"] == "z3_solver"]


def _write_transcript(directory: Path, item_id: str, payload: Mapping[str, Any]) -> JsonDict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{item_id}.json"
    _write_json(path, payload)
    return {"path": path, "sha256": sha256_file(path)}


def _manifest_row(
    repo_root: Path,
    item: InstructionItem,
    tree: Mapping[str, Any],
    good: Mapping[str, Any],
    bad: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
    z3_info: Mapping[str, Any],
) -> JsonDict:
    return {
        "item_id": item.item_id,
        "category": item.category,
        "instruction_sha256": sha256_text(item.instruction),
        "validator_tree": dict(tree),
        "known_good_candidate_sha256": sha256_text(item.known_good_candidate),
        "known_bad_candidate_sha256": sha256_text(item.known_bad_candidate),
        "known_good_validation": dict(good),
        "known_bad_validation": dict(bad),
        "runtime_transcript_path": str(_relative_to(repo_root, Path(runtime_info["path"]))),
        "runtime_transcript_sha256": str(runtime_info["sha256"]),
        "z3_transcript_path": str(_relative_to(repo_root, Path(z3_info["path"]))) if z3_info else "",
        "z3_transcript_sha256": str(z3_info["sha256"]) if z3_info else "",
        "all_authoritative_nodes_exact_checked": _row_authoritative_exact(tree, good, bad),
        "llm_judge_used": False,
    }


def _row_authoritative_exact(
    tree: Mapping[str, Any],
    good: Mapping[str, Any],
    bad: Mapping[str, Any],
) -> bool:
    authoritative_nodes = [node for node in tree["nodes"] if node.get("authoritative", True)]
    result_rows = good["node_results"] + bad["node_results"]
    return bool(authoritative_nodes) and all(
        node["authority"] in EXACT_AUTHORITIES
        and any(row["node_id"] == node["node_id"] and row["exact_checked"] for row in result_rows)
        for node in authoritative_nodes
    )


def _exact_check_coverage(rows: Sequence[Mapping[str, Any]]) -> float:
    total = 0
    exact = 0
    for row in rows:
        for node in row["validator_tree"]["nodes"]:
            total += 1
            if node.get("authoritative", True) and node["authority"] in EXACT_AUTHORITIES:
                exact += 1
    return round(exact / total, 6) if total else 0.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:  # pragma: no cover - explicit external paths are not used by this experiment.
        return path


def _summarize_json(path: Path, keys: Sequence[str]) -> JsonDict:
    if not path.exists():
        return {"present": False, "path": str(path), "fields": {}}
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    return {
        "present": True,
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "fields": {key: payload.get(key) for key in keys},
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for the deterministic Exp 3017 harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    artifact = run_experiment(ExperimentConfig(output_path=Path(args.output)))
    print(
        "[exp3017] "
        f"ready={artifact['instruction_validator_tree_ready']} "
        f"items={artifact['n_instruction_items']} "
        f"coverage={artifact['exact_check_coverage']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
