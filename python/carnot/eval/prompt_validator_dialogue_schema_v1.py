"""Exp 2994 prompt-to-validator dialogue schema harness.

Spec refs: REQ-VERIFY-2994, SCENARIO-VERIFY-2994.

This module deliberately keeps the language-model role outside the acceptance
path.  It uses a tiny allow-listed prompt grammar to build validator trees, then
uses local JSON parsing, Python AST parsing, and Z3 execution as the authority.
That separation matters because downstream self-learning can safely consume a
feedback object only when the feedback came from exact runtime checks rather
than from another model's opinion about whether the candidate looks correct.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import z3 as _z3
except Exception:  # pragma: no cover - z3-solver is a project dependency.
    _z3 = None


JsonDict = dict[str, Any]

RUN_DATE = "20260524"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2994_prompt_validator_dialogue_schema_v1.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME
PROTOCOL_DOC_PATH = "openspec/change-proposals/prompt-validator-dialogue-schema-v1.md"
DETERMINISTIC_HARNESS_PATH = "python/carnot/eval/prompt_validator_dialogue_schema_v1.py"
INFERENCE_SUBSTRATE = "deterministic_prompt_validator_harness"

EXACT_AUTHORITIES = frozenset(
    {"runtime_json_parser", "python_ast_parser", "z3_solver"}
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "prompt_validator_protocol_ready",
        "protocol_doc_path",
        "deterministic_harness_path",
        "n_validator_tree_fixtures",
        "exact_verifier_authority_preserved",
        "static_transition_representation_designed",
        "no_speed_claim_made",
        "validation_commands",
        "honest_verdict",
    }
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for deterministic artifact generation.

    The clock is injectable so tests can prove duration accounting without
    sleeping.  The repo root is used only for source-artifact provenance; the
    protocol and harness paths stay anchored to this checked-out repository.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    validation_commands: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


def build_validator_tree_fixtures() -> list[JsonDict]:
    """Return the fixed V1 prompt fixtures used by the local protocol gate."""

    return [
        {
            "fixture_id": "json-final-answer-confidence",
            "prompt": (
                "The candidate must be a JSON object with field `final_answer` "
                'equal to "SAFE" and field `confidence` between 0.0 and 1.0.'
            ),
            "known_good": json.dumps({"final_answer": "SAFE", "confidence": 0.75}),
            "known_bad": json.dumps({"final_answer": "SAFE", "confidence": 1.5}),
        },
        {
            "fixture_id": "python-normalize-slug-ast",
            "prompt": (
                "The candidate must define Python function `normalize_slug` with "
                "exactly 1 parameter and must not contain import statements."
            ),
            "known_good": "def normalize_slug(text):\n    return text.strip().lower()\n",
            "known_bad": "def normalize_slug(left, right):\n    return left\n",
        },
        {
            "fixture_id": "z3-linear-integer-assignment",
            "prompt": (
                "The candidate must provide integer assignments `x` and `y` in "
                "JSON such that `x + y = 10` and `x > y`."
            ),
            "known_good": json.dumps({"x": 6, "y": 4}),
            "known_bad": json.dumps({"x": 5, "y": 5}),
        },
    ]


def compile_prompt_to_validator_tree(prompt: str, *, constraint_id: str) -> JsonDict:
    """Compile one allow-listed natural-language prompt into a validator tree."""

    normalized = " ".join(prompt.split())
    if "final_answer" in normalized and "confidence" in normalized:
        return _compiled(constraint_id, _compile_json_prompt(normalized, constraint_id))
    if "normalize_slug" in normalized and "import statements" in normalized:
        return _compiled(constraint_id, _compile_python_prompt(normalized, constraint_id))
    if "x + y" in normalized and "x > y" in normalized:
        return _compiled(constraint_id, _compile_z3_prompt(normalized, constraint_id))
    return {
        "constraint_id": constraint_id,
        "compiled": False,
        "validator_tree": None,
        "rejection_reasons": ["unsupported_prompt_pattern"],
    }


def evaluate_validator_tree(
    validator_tree: Mapping[str, Any],
    candidate_text: str,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Run all exact-check nodes and return the deterministic feedback object."""

    node_results = [
        _evaluate_node(node, candidate_text, z3_module=z3_module)
        for node in validator_tree.get("nodes", [])
    ]
    failing_node_ids = [
        str(result["node_id"]) for result in node_results if not result["accepted"]
    ]
    rejection_reasons = list(
        dict.fromkeys(
            str(result["rejection_reason"])
            for result in node_results
            if result.get("rejection_reason")
        )
    )
    return {
        "accepted": not failing_node_ids,
        "failing_node_ids": failing_node_ids,
        "rejection_reasons": rejection_reasons,
        "node_results": node_results,
        "llm_judge_used": False,
    }


def static_transition_representation() -> JsonDict:
    """Describe a sparse/static automaton representation without a speed claim."""

    return {
        "name": "validator_tree_static_sparse_transition_table",
        "representation": {
            "row_offsets": [0, 3, 4, 5, 5],
            "labels": ["json_object", "python_ast", "z3_assignment", "leaf", "accept"],
            "targets": [1, 2, 3, 4, 4],
            "accepting_states": [4],
        },
        "designed": True,
        "speed_claim": None,
        "speed_claim_made": False,
        "note": (
            "STATIC inspires the sparse transition shape only; this artifact "
            "does not benchmark or claim acceleration."
        ),
    }


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 2994 terminal artifact from deterministic local fixtures."""

    active = config or ExperimentConfig()
    started = active.start_time()
    fixture_rows = [_compile_and_evaluate_fixture(fixture) for fixture in build_validator_tree_fixtures()]
    unsupported = compile_prompt_to_validator_tree(
        "Write a pleasant haiku and decide whether it feels correct.",
        constraint_id="unsupported-haiku-judge",
    )
    static_table = static_transition_representation()
    protocol_exists = (REPO_ROOT / PROTOCOL_DOC_PATH).exists()
    exact_authority = _exact_authority_preserved(fixture_rows)
    all_good_pass = all(row["known_good_feedback"]["accepted"] for row in fixture_rows)
    all_bad_reject = all(not row["known_bad_feedback"]["accepted"] for row in fixture_rows)
    compiled_count = sum(row["compiled"] for row in fixture_rows)
    no_speed_claim = static_table["speed_claim"] is None and not static_table["speed_claim_made"]
    ready = (
        protocol_exists
        and compiled_count >= 3
        and all_good_pass
        and all_bad_reject
        and exact_authority
        and unsupported["rejection_reasons"] == ["unsupported_prompt_pattern"]
        and static_table["designed"]
        and no_speed_claim
    )
    artifact = {
        "schema": "carnot.prompt_validator_dialogue_schema.v1",
        "artifact": "experiment_2994_prompt_validator_dialogue_schema_v1",
        "run_date": RUN_DATE,
        "prompt_validator_protocol_ready": ready,
        "protocol_doc_path": PROTOCOL_DOC_PATH,
        "deterministic_harness_path": DETERMINISTIC_HARNESS_PATH,
        "n_validator_tree_fixtures": compiled_count,
        "exact_verifier_authority_preserved": exact_authority,
        "static_transition_representation_designed": bool(static_table["designed"]),
        "no_speed_claim_made": no_speed_claim,
        "validation_commands": list(active.validation_commands),
        "honest_verdict": (
            "complete: prompt-validator dialogue protocol ready with deterministic exact checks"
            if ready
            else "flagged: prompt-validator dialogue protocol is not ready"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(active.clock() - started, 6),
        "llm_inference_run": False,
        "live_llm_judge_used": False,
        "validator_tree_fixtures": fixture_rows,
        "unsupported_prompt_rejection": unsupported,
        "static_transition_representation": static_table,
        "source_artifacts": source_artifact_status(active.repo_root),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and write the deterministic Exp 2994 artifact."""

    active = config or ExperimentConfig()
    artifact = build_artifact(active)
    path = active.artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the terminal fields that downstream self-learning gates rely on."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(payload)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if payload["prompt_validator_protocol_ready"] is not True:
        raise ValueError("prompt_validator_protocol_ready must be true")
    if payload["exact_verifier_authority_preserved"] is not True:
        raise ValueError("exact_verifier_authority_preserved must be true")
    if payload["static_transition_representation_designed"] is not True:
        raise ValueError("static_transition_representation_designed must be true")
    if payload["no_speed_claim_made"] is not True:
        raise ValueError("no_speed_claim_made must be true")
    if not str(payload["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def source_artifact_status(repo_root: Path) -> JsonDict:
    """Summarize the prior MCS, monitor, and solver-provenance artifacts."""

    results = repo_root / "results"
    return {
        "exp2979": _summarize_json(
            results / "experiment_2979_solver_feedback_mcs_frontier_v1.json",
            ("honest_verdict", "mcs_feedback_schema_ready", "frontier_upgrade_ready"),
        ),
        "exp2981": _summarize_json(
            results / "experiment_2981_interwhen_partial_monitor_promotion_v2.json",
            ("honest_verdict", "partial_monitor_promoted", "full_streaming_verification_claim"),
        ),
        "exp2992": _summarize_json(
            results / "experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json",
            ("honest_verdict", "solver_provenance_reproduced", "formalization_clean"),
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running the deterministic harness locally."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--validation-command", action="append", default=[])
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    artifact = write_artifact(
        ExperimentConfig(
            output_path=Path(args.output),
            validation_commands=list(args.validation_command),
        )
    )
    print(
        "[exp2994] "
        f"ready={artifact['prompt_validator_protocol_ready']} "
        f"fixtures={artifact['n_validator_tree_fixtures']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


def _compile_json_prompt(prompt: str, constraint_id: str) -> JsonDict:
    expected = re.search(r'final_answer`? equal to "([^"]+)"', prompt)
    number = r"([0-9]+(?:\.[0-9]+)?)"
    bounds = re.search(rf"confidence`? between {number} and {number}", prompt)
    return _tree(
        constraint_id,
        [
            {
                "node_id": f"{constraint_id}:final_answer",
                "kind": "json_field_equals",
                "authority": "runtime_json_parser",
                "field": "final_answer",
                "expected": expected.group(1) if expected else "",
            },
            {
                "node_id": f"{constraint_id}:confidence",
                "kind": "json_number_between",
                "authority": "runtime_json_parser",
                "field": "confidence",
                "minimum": float(bounds.group(1)) if bounds else 0.0,
                "maximum": float(bounds.group(2)) if bounds else 0.0,
            },
        ],
    )


def _compile_python_prompt(prompt: str, constraint_id: str) -> JsonDict:
    signature = re.search(r"function `?([A-Za-z_][A-Za-z0-9_]*)`? with exactly ([0-9]+)", prompt)
    return _tree(
        constraint_id,
        [
            {
                "node_id": f"{constraint_id}:signature",
                "kind": "python_function_signature",
                "authority": "python_ast_parser",
                "function_name": signature.group(1) if signature else "",
                "parameter_count": int(signature.group(2)) if signature else 0,
            },
            {
                "node_id": f"{constraint_id}:no_imports",
                "kind": "python_no_imports",
                "authority": "python_ast_parser",
            },
        ],
    )


def _compile_z3_prompt(prompt: str, constraint_id: str) -> JsonDict:
    target_sum = re.search(r"x \+ y = ([0-9]+)", prompt)
    return _tree(
        constraint_id,
        [
            {
                "node_id": f"{constraint_id}:linear_constraints",
                "kind": "z3_linear_integer_relation",
                "authority": "z3_solver",
                "variables": ["x", "y"],
                "sum_equals": int(target_sum.group(1)) if target_sum else 0,
                "strict_greater": ["x", "y"],
            }
        ],
    )


def _compiled(constraint_id: str, validator_tree: JsonDict) -> JsonDict:
    return {
        "constraint_id": constraint_id,
        "compiled": True,
        "validator_tree": validator_tree,
        "rejection_reasons": [],
    }


def _tree(constraint_id: str, nodes: list[JsonDict]) -> JsonDict:
    return {
        "tree_id": constraint_id,
        "root": {"op": "all", "children": [node["node_id"] for node in nodes]},
        "nodes": nodes,
    }


def _evaluate_node(node: Mapping[str, Any], candidate_text: str, *, z3_module: Any) -> JsonDict:
    kind = str(node["kind"])
    if kind == "json_field_equals":
        return _evaluate_json_field_equals(node, candidate_text)
    if kind == "json_number_between":
        return _evaluate_json_number_between(node, candidate_text)
    if kind == "python_function_signature":
        return _evaluate_python_function_signature(node, candidate_text)
    if kind == "python_no_imports":
        return _evaluate_python_no_imports(node, candidate_text)
    if kind == "z3_linear_integer_relation":
        return _evaluate_z3_linear_integer_relation(node, candidate_text, z3_module)
    raise ValueError(f"unknown validator node kind: {kind}")  # pragma: no cover


def _evaluate_json_field_equals(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    field = str(node["field"])
    if reason:
        return _node_result(node, False, reason)
    if field not in payload:
        return _node_result(node, False, "missing_required_field")
    if payload[field] != node["expected"]:
        return _node_result(node, False, "field_value_mismatch")
    return _node_result(node, True, None)


def _evaluate_json_number_between(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    payload, reason = _parse_json_object(candidate_text)
    field = str(node["field"])
    if reason:
        return _node_result(node, False, reason)
    if field not in payload:
        return _node_result(node, False, "missing_required_field")
    value = payload[field]
    if not isinstance(value, int | float) or not float(node["minimum"]) <= float(value) <= float(
        node["maximum"]
    ):
        return _node_result(node, False, "numeric_range_violation")
    return _node_result(node, True, None)


def _evaluate_python_function_signature(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    tree, reason = _parse_python_ast(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    function = _first_function(tree, str(node["function_name"]))
    if function is None or len(function.args.args) != int(node["parameter_count"]):
        return _node_result(node, False, "function_signature_mismatch")
    return _node_result(node, True, None)


def _evaluate_python_no_imports(node: Mapping[str, Any], candidate_text: str) -> JsonDict:
    tree, reason = _parse_python_ast(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    has_import = any(isinstance(item, ast.Import | ast.ImportFrom) for item in ast.walk(tree))
    if has_import:
        return _node_result(node, False, "import_statement_disallowed")
    return _node_result(node, True, None)


def _evaluate_z3_linear_integer_relation(
    node: Mapping[str, Any],
    candidate_text: str,
    z3_module: Any,
) -> JsonDict:
    if z3_module is None:
        return _node_result(node, False, "z3_unavailable")
    payload, reason = _parse_json_object(candidate_text)
    if reason:
        return _node_result(node, False, reason)
    x_value = payload.get("x")
    y_value = payload.get("y")
    if not isinstance(x_value, int) or not isinstance(y_value, int):
        return _node_result(node, False, "missing_required_field")
    x_var = z3_module.Int("x")
    y_var = z3_module.Int("y")
    solver = z3_module.Solver()
    solver.add(x_var == x_value)
    solver.add(y_var == y_value)
    solver.add(x_var + y_var == int(node["sum_equals"]))
    solver.add(x_var > y_var)
    if solver.check() != z3_module.sat:
        return _node_result(node, False, "z3_unsatisfied")
    return _node_result(node, True, None)


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


def _node_result(node: Mapping[str, Any], accepted: bool, reason: str | None) -> JsonDict:
    return {
        "node_id": node["node_id"],
        "kind": node["kind"],
        "authority": node["authority"],
        "accepted": accepted,
        "rejection_reason": reason,
    }


def _compile_and_evaluate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    compiled = compile_prompt_to_validator_tree(
        str(fixture["prompt"]),
        constraint_id=str(fixture["fixture_id"]),
    )
    validator_tree = compiled["validator_tree"]
    good = evaluate_validator_tree(validator_tree, str(fixture["known_good"]))
    bad = evaluate_validator_tree(validator_tree, str(fixture["known_bad"]))
    return {
        "fixture_id": fixture["fixture_id"],
        "prompt": fixture["prompt"],
        "compiled": compiled["compiled"],
        "validator_tree": validator_tree,
        "known_good_feedback": good,
        "known_bad_feedback": bad,
    }


def _exact_authority_preserved(fixture_rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in fixture_rows:
        if row["known_good_feedback"]["llm_judge_used"] or row["known_bad_feedback"]["llm_judge_used"]:
            return False
        for node in row["validator_tree"]["nodes"]:
            if node["authority"] not in EXACT_AUTHORITIES:
                return False
    return True


def _summarize_json(path: Path, keys: Sequence[str]) -> JsonDict:
    if not path.exists():
        return {"present": False, "path": str(path), "fields": {}}
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive IO guard.
        return {"present": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "present": True,
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "fields": {key: payload.get(key) for key in keys},
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
