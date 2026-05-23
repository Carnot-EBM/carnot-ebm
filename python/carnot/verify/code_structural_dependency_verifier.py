"""Exp 2890 static structural-dependency verifier for code-corpus rows.

This verifier is intentionally narrow and deterministic. It does not execute
candidate code and does not generate code. Instead, it turns MBPP/HumanEval
manifest rows into small structural contracts and checks candidate source with
the Python AST. The goal is reusable matrix-v7 metadata: a row can show exactly
which structural prerequisite failed before a sandbox or tool-in-loop repair
cycle spends more work.

Spec: REQ-CODE-2890, SCENARIO-CODE-2890.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2890_code_structural_dependency_verifier_v1.json"
CONTRACT_SCHEMA_VERSION = "code-structural-dependency-contract/v1"

EXP2879_REL_PATH = Path("results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json")
CROSS_CORPUS_MATRIX_V6_REL_PATH = Path("results/experiment_2880_cross_corpus_matrix_v6.json")
EXP2889_REL_PATH = Path("results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json")

CODE_CORPORA = ("mbpp", "humaneval")
FORBIDDEN_IMPORTS = ("os", "pathlib", "shutil", "socket", "subprocess", "sys")
FORBIDDEN_SIDE_EFFECTS = ("__import__", "compile", "eval", "exec", "exit", "input", "open", "quit")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "structural_dependency_verifier_ready",
    "source_artifacts",
    "contract_schema_version",
    "n_contracts_built",
    "n_rows_verified",
    "violation_types",
    "localization_examples",
    "generated_outputs_consumed",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

REQUIRED_CONTRACT_FIELDS = (
    "contract_schema_version",
    "contract_id",
    "corpus",
    "stable_id",
    "required_inputs",
    "function_signature",
    "dependency_edges",
    "forbidden_imports",
    "forbidden_side_effects",
    "test_prerequisites",
    "output_obligations",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict; complete only when reference rows satisfy static contracts.",
    "structural_dependency_verifier_ready": (
        "True only when at least one contract is built and reference/canonical checks are clean."
    ),
    "source_artifacts": "Exp 2879, matrix v6, optional Exp 2889, and manifest files used as evidence.",
    "contract_schema_version": "Versioned static contract schema for matrix-v7 reuse.",
    "n_contracts_built": "Unique MBPP/HumanEval manifest rows converted into structural contracts.",
    "n_rows_verified": "Candidate source checks performed; includes references and generated outputs.",
    "violation_types": "Counts deterministic AST contract violations only, not unsupported rows.",
    "localization_examples": "Small examples with row id, candidate kind, field, violation, and source line.",
    "generated_outputs_consumed": "True only when the checked-in Exp 2889 row artifact is present and non-empty.",
    "tests_run": "Commands used to validate this verifier and artifact.",
    "duration_s": "Measured wall-clock runtime; no padding.",
}


@dataclass(frozen=True)
class CodeStructuralContract:
    """Serializable structural contract for one MBPP/HumanEval-like task row.

    The contract captures only static facts that can be checked before code is
    executed: the function expected by tests, its inputs, allowed imports, and
    the obligation to return a value. The executable checker stays in this
    module so artifacts remain data rather than trusted code.
    """

    contract_id: str
    corpus: str
    stable_id: str
    row_sha256: str
    manifest_path: str
    required_inputs: tuple[str, ...]
    function_name: str
    dependency_edges: tuple[tuple[str, str, str], ...]
    forbidden_imports: tuple[str, ...] = FORBIDDEN_IMPORTS
    forbidden_side_effects: tuple[str, ...] = FORBIDDEN_SIDE_EFFECTS
    test_prerequisites: Mapping[str, Any] = field(default_factory=dict)
    output_obligations: tuple[str, ...] = ("defines_entry_point", "returns_value")
    strict_parameter_names: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2890 artifact builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp2879_path: Path = EXP2879_REL_PATH
    cross_corpus_matrix_path: Path = CROSS_CORPUS_MATRIX_V6_REL_PATH
    exp2889_path: Path = EXP2889_REL_PATH
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable JSON digest used to bind contracts to manifest rows."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_contract_from_manifest_row(
    corpus: str,
    row: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> CodeStructuralContract:
    """Build one static contract from an MBPP or HumanEval manifest row."""

    normalized = _normalize_corpus(corpus)
    stable_id = str(row.get("stable_id") or "")
    function_name = _expected_function_name(normalized, row)
    source = _reference_source(normalized, row)
    function = _find_function(
        _parse_reference_source(source),
        function_name,
        fallback_to_first=True,
    )
    if function is None:
        parameters: tuple[str, ...] = ()
    else:
        parameters = _function_parameters(function)
    n_tests, test_kind, tests_reference_function = _test_prerequisites(normalized, row, function_name)
    edges = tuple(
        (f"input:{name}", f"function:{function_name}", "argument_to_function")
        for name in parameters
    ) + (
        # The function-to-tests edge is what lets downstream matrix consumers
        # distinguish a signature failure from a missing local oracle.
        (f"function:{function_name}", f"tests:{stable_id}", "function_under_test"),
    )
    return CodeStructuralContract(
        contract_id=f"contract:{normalized}:{stable_id}",
        corpus=normalized,
        stable_id=stable_id,
        row_sha256=stable_json_sha256(row),
        manifest_path=str(manifest_path),
        required_inputs=parameters,
        function_name=function_name,
        dependency_edges=edges,
        test_prerequisites={
            "n_tests": n_tests,
            "test_kind": test_kind,
            "tests_reference_function": tests_reference_function,
        },
        strict_parameter_names=(normalized == "humaneval"),
    )


def contract_to_json(contract: CodeStructuralContract) -> JsonDict:
    """Serialize one contract definition for artifacts and tests."""

    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "corpus": contract.corpus,
        "stable_id": contract.stable_id,
        "row_sha256": contract.row_sha256,
        "manifest_path": contract.manifest_path,
        "required_inputs": list(contract.required_inputs),
        "function_signature": {
            "name": contract.function_name,
            "parameters": list(contract.required_inputs),
        },
        "dependency_edges": [list(edge) for edge in contract.dependency_edges],
        "forbidden_imports": list(contract.forbidden_imports),
        "forbidden_side_effects": list(contract.forbidden_side_effects),
        "test_prerequisites": dict(contract.test_prerequisites),
        "output_obligations": list(contract.output_obligations),
        "strict_parameter_names": contract.strict_parameter_names,
    }


def validate_contract_json(row: Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for one serialized contract."""

    errors = [f"missing:{field}" for field in REQUIRED_CONTRACT_FIELDS if field not in row]
    if row.get("contract_schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append("invalid:contract_schema_version")
    signature = row.get("function_signature")
    if not isinstance(signature, Mapping) or not signature.get("name"):
        errors.append("invalid:function_signature")
    if not row.get("dependency_edges"):
        errors.append("invalid:dependency_edges")
    return errors


def verify_candidate_source(
    contract: CodeStructuralContract,
    source: str,
    candidate_kind: str,
    *,
    candidate_id: str | None = None,
) -> JsonDict:
    """Statically verify candidate source against one structural contract."""

    row_id = f"{contract.corpus}:{contract.stable_id}"
    result: JsonDict = {
        "row_id": row_id,
        "candidate_id": candidate_id or f"{row_id}:{candidate_kind}",
        "candidate_kind": candidate_kind,
        "contract_id": contract.contract_id,
        "corpus": _display_corpus(contract.corpus),
        "stable_id": contract.stable_id,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "code_present": bool(source.strip()),
        "ast_parseable": False,
        "passed": False,
        "violations": [],
        "unsupported_reasons": [],
    }
    if not source.strip():
        result["unsupported_reasons"] = ["empty_candidate_source"]
        return result

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result["violations"] = [
            _violation(
                "parse_error",
                "function_signature",
                f"Python AST parse failed: {exc.msg}",
                line=exc.lineno,
                col=exc.offset,
                source=source,
            )
        ]
        return result

    result["ast_parseable"] = True
    violations: list[JsonDict] = []
    violations.extend(_forbidden_import_violations(tree, contract, source))

    function = _find_function(tree, contract.function_name)
    if function is None:
        violations.append(
            _violation(
                "missing_function_definition",
                "function_signature",
                f"Expected function {contract.function_name!r} was not defined.",
                line=_first_line(tree),
                col=1,
                source=source,
            )
        )
        result["violations"] = violations
        return _finish_result(result, violations)

    candidate_parameters = _function_parameters(function)
    if len(candidate_parameters) != len(contract.required_inputs) or (
        contract.strict_parameter_names and candidate_parameters != contract.required_inputs
    ):
        violations.append(
            _violation(
                "signature_mismatch",
                "function_signature",
                (
                    "Expected parameters "
                    f"{list(contract.required_inputs)!r}, found {list(candidate_parameters)!r}."
                ),
                line=function.lineno,
                col=function.col_offset + 1,
                source=source,
            )
        )
    if len(candidate_parameters) < len(contract.required_inputs):
        violations.append(
            _violation(
                "missing_dependency_edge",
                "dependency_edges",
                "Candidate signature cannot satisfy every required input-to-function edge.",
                line=function.lineno,
                col=function.col_offset + 1,
                source=source,
            )
        )

    violations.extend(_forbidden_side_effect_violations(function, contract, source))
    if not _has_value_return(function):
        violations.append(
            _violation(
                "missing_return_obligation",
                "output_obligations",
                f"Function {contract.function_name!r} has no return statement with a value.",
                line=function.lineno,
                col=function.col_offset + 1,
                source=source,
            )
        )
    if contract.test_prerequisites.get("n_tests", 0) < 1:
        violations.append(
            _violation(
                "missing_test_prerequisite",
                "test_prerequisites",
                "Contract has no local test prerequisite for this row.",
                line=function.lineno,
                col=function.col_offset + 1,
                source=source,
            )
        )

    return _finish_result(result, violations)


def build_experiment_artifact(config: ExperimentConfig = ExperimentConfig()) -> JsonDict:
    """Build the Exp 2890 structural-dependency verifier artifact."""

    started = config.start_time()
    exp2879_path = _repo_path(config.repo_root, config.exp2879_path)
    matrix_path = _repo_path(config.repo_root, config.cross_corpus_matrix_path)
    exp2889_path = _repo_path(config.repo_root, config.exp2889_path)
    exp2879 = _read_json_if_exists(exp2879_path) or {}
    exp2889 = _read_json_if_exists(exp2889_path) or {}

    manifest_paths = _manifest_paths(exp2879, exp2889, config.repo_root)
    manifest_rows = _load_manifest_rows(manifest_paths)
    contract_keys = _selected_contract_keys(exp2879, exp2889)
    contracts: list[CodeStructuralContract] = []
    unsupported_contracts: list[JsonDict] = []
    for corpus, stable_id in contract_keys:
        row = manifest_rows.get(corpus, {}).get(stable_id)
        if row is None:
            unsupported_contracts.append(
                {
                    "corpus": _display_corpus(corpus),
                    "stable_id": stable_id,
                    "unsupported_reason": "manifest_row_not_found",
                }
            )
            continue
        contracts.append(
            build_contract_from_manifest_row(
                corpus,
                row,
                manifest_path=manifest_paths.get(corpus, ""),
            )
        )

    by_key = {(contract.corpus, contract.stable_id): contract for contract in contracts}
    verification_rows: list[JsonDict] = []
    for contract in contracts:
        row = manifest_rows[contract.corpus][contract.stable_id]
        verification_rows.append(
            verify_candidate_source(
                contract,
                _reference_source(contract.corpus, row),
                "reference",
            )
        )

    generated_outputs = list(exp2889.get("row_results") or [])
    for row_result in generated_outputs:
        corpus = _normalize_corpus(row_result.get("corpus"))
        stable_id = str(row_result.get("stable_id") or "")
        contract = by_key.get((corpus, stable_id))
        if contract is None:
            continue
        verification_rows.append(
            verify_candidate_source(
                contract,
                str(row_result.get("extracted_code") or ""),
                "generated_exp2889",
                candidate_id=f"{corpus}:{stable_id}:generated_exp2889",
            )
        )

    contract_rows = [contract_to_json(contract) for contract in contracts]
    contract_schema_errors = [
        f"{row.get('contract_id', 'unknown')}:{','.join(errors)}"
        for row in contract_rows
        if (errors := validate_contract_json(row))
    ]
    violation_types = Counter(
        violation["violation_type"]
        for row in verification_rows
        for violation in row.get("violations", [])
    )
    unsupported_reasons = Counter(
        reason
        for row in verification_rows
        for reason in row.get("unsupported_reasons", [])
    )
    reference_rows = [row for row in verification_rows if row["candidate_kind"] == "reference"]
    reference_clean = bool(reference_rows) and all(row["passed"] for row in reference_rows)
    ready = bool(contracts and verification_rows and reference_clean and not contract_schema_errors)

    source_artifacts, source_sha = _source_artifacts(
        config.repo_root,
        exp2879_path,
        matrix_path,
        exp2889_path if exp2889_path.exists() else None,
        manifest_paths,
    )
    artifact: JsonDict = {
        "artifact": "experiment_2890_code_structural_dependency_verifier_v1",
        "schema": "carnot.code_structural_dependency_verifier.v1",
        "honest_verdict": (
            "complete: MBPP/HumanEval structural dependency verifier metadata ready"
            if ready
            else "blocked_structural_dependency_verifier_not_ready"
        ),
        "structural_dependency_verifier_ready": ready,
        "source_artifacts": source_artifacts,
        "source_artifact_sha256": source_sha,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contracts": contract_rows,
        "contract_schema_errors": contract_schema_errors,
        "unsupported_contracts": unsupported_contracts,
        "n_contracts_built": len(contracts),
        "n_rows_verified": len(verification_rows),
        "verification_rows": verification_rows,
        "violation_types": dict(sorted(violation_types.items())),
        "unsupported_reasons": dict(sorted(unsupported_reasons.items())),
        "localization_examples": _localization_examples(verification_rows),
        "generated_outputs_consumed": bool(generated_outputs),
        "headline_metric_claim_made": False,
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, config.clock() - started),
    }
    return artifact


def write_experiment_artifact(config: ExperimentConfig = ExperimentConfig()) -> JsonDict:
    """Build and persist the Exp 2890 artifact under ``results/``."""

    artifact = build_experiment_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _finish_result(result: JsonDict, violations: list[JsonDict]) -> JsonDict:
    result["violations"] = violations
    result["passed"] = not violations and not result["unsupported_reasons"]
    return result


def _violation(
    violation_type: str,
    contract_field: str,
    message: str,
    *,
    line: int | None,
    col: int | None,
    source: str,
) -> JsonDict:
    return {
        "violation_type": violation_type,
        "contract_field": contract_field,
        "message": message,
        "line": line,
        "col": col,
        "snippet": _line_snippet(source, line),
    }


def _forbidden_import_violations(
    tree: ast.AST,
    contract: CodeStructuralContract,
    source: str,
) -> list[JsonDict]:
    forbidden = set(contract.forbidden_imports)
    violations: list[JsonDict] = []
    for node in ast.walk(tree):
        module_names: list[str] = []
        if isinstance(node, ast.Import):
            module_names = [alias.name.split(".", maxsplit=1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_names = [node.module.split(".", maxsplit=1)[0]]
        for module_name in module_names:
            if module_name in forbidden:
                violations.append(
                    _violation(
                        "forbidden_import",
                        "forbidden_imports",
                        f"Import of {module_name!r} is forbidden for benchmark candidates.",
                        line=node.lineno,
                        col=node.col_offset + 1,
                        source=source,
                    )
                )
    return violations


def _forbidden_side_effect_violations(
    function: ast.FunctionDef,
    contract: CodeStructuralContract,
    source: str,
) -> list[JsonDict]:
    forbidden = set(contract.forbidden_side_effects)
    violations: list[JsonDict] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name in forbidden:
            violations.append(
                _violation(
                    "forbidden_side_effect",
                    "forbidden_side_effects",
                    f"Call to {call_name!r} is forbidden for benchmark candidates.",
                    line=node.lineno,
                    col=node.col_offset + 1,
                    source=source,
                )
            )
    return violations


def _has_value_return(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Return) and node.value is not None for node in ast.walk(function)
    )


def _expected_function_name(corpus: str, row: Mapping[str, Any]) -> str:
    if corpus == "humaneval":
        return str(row.get("entry_point") or "")
    return _function_name_from_tests(row.get("tests") or ()) or _first_function_name(
        str(row.get("canonical_code") or "")
    )


def _function_name_from_tests(tests: Any) -> str:
    if isinstance(tests, str):
        test_items = [tests]
    else:
        test_items = [str(test) for test in tests or ()]
    for test in test_items:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id != "candidate":
                    return node.func.id
    return ""


def _first_function_name(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return ""


def _parse_reference_source(source: str) -> ast.Module:
    return ast.parse(source)


def _reference_source(corpus: str, row: Mapping[str, Any]) -> str:
    if corpus == "humaneval":
        return f"{row.get('prompt') or ''}{row.get('canonical_solution') or ''}"
    return str(row.get("canonical_code") or "")


def _find_function(
    tree: ast.AST,
    function_name: str,
    *,
    fallback_to_first: bool = False,
) -> ast.FunctionDef | None:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    if not fallback_to_first:
        return None
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _function_parameters(function: ast.FunctionDef) -> tuple[str, ...]:
    args = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    names = [arg.arg for arg in args]
    if function.args.vararg is not None:
        names.append(f"*{function.args.vararg.arg}")
    if function.args.kwarg is not None:
        names.append(f"**{function.args.kwarg.arg}")
    return tuple(names)


def _test_prerequisites(
    corpus: str,
    row: Mapping[str, Any],
    function_name: str,
) -> tuple[int, str, bool]:
    tests = row.get("tests")
    if corpus == "humaneval":
        test_text = str(tests or "")
        return test_text.count("assert "), "official_check", function_name in test_text
    test_items = [str(test) for test in tests or ()] if isinstance(tests, list) else []
    return len(test_items), "assert_tests", any(function_name in test for test in test_items)


def _manifest_paths(exp2879: Mapping[str, Any], exp2889: Mapping[str, Any], repo_root: Path) -> dict[str, Path]:
    raw_paths = dict(exp2879.get("manifest_paths") or {})
    raw_paths.update(dict(exp2889.get("manifest_paths") or {}))
    return {
        corpus: _repo_path(repo_root, Path(str(raw_paths[corpus])))
        for corpus in CODE_CORPORA
        if raw_paths.get(corpus)
    }


def _load_manifest_rows(manifest_paths: Mapping[str, Path]) -> dict[str, dict[str, JsonDict]]:
    rows: dict[str, dict[str, JsonDict]] = {}
    for corpus, path in manifest_paths.items():
        rows[corpus] = {str(row.get("stable_id") or ""): row for row in _read_jsonl(path)}
    return rows


def _selected_contract_keys(
    exp2879: Mapping[str, Any],
    exp2889: Mapping[str, Any],
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in exp2879.get("pilot_rows") or []:
        _append_key(keys, seen, row)
    for row in exp2889.get("row_results") or []:
        _append_key(keys, seen, row)
    return keys


def _append_key(
    keys: list[tuple[str, str]],
    seen: set[tuple[str, str]],
    row: Mapping[str, Any],
) -> None:
    corpus = _normalize_corpus(row.get("corpus"))
    stable_id = str(row.get("stable_id") or "")
    key = (corpus, stable_id)
    if corpus in CODE_CORPORA and stable_id and key not in seen:
        keys.append(key)
        seen.add(key)


def _source_artifacts(
    repo_root: Path,
    exp2879_path: Path,
    matrix_path: Path,
    exp2889_path: Path | None,
    manifest_paths: Mapping[str, Path],
) -> tuple[list[str], dict[str, str]]:
    paths = [exp2879_path, matrix_path]
    if exp2889_path is not None:
        paths.append(exp2889_path)
    paths.extend(manifest_paths[corpus] for corpus in CODE_CORPORA if corpus in manifest_paths)
    names = [_source_name(repo_root, path) for path in paths]
    return names, {
        name: _sha256(path) for name, path in zip(names, paths, strict=True) if path.exists()
    }


def _localization_examples(rows: Iterable[Mapping[str, Any]], limit: int = 8) -> list[JsonDict]:
    examples: list[JsonDict] = []
    for row in rows:
        for violation in row.get("violations", []):
            examples.append(
                {
                    "corpus": row.get("corpus"),
                    "stable_id": row.get("stable_id"),
                    "candidate_kind": row.get("candidate_kind"),
                    "violation_type": violation.get("violation_type"),
                    "contract_field": violation.get("contract_field"),
                    "line": violation.get("line"),
                    "message": violation.get("message"),
                    "snippet": violation.get("snippet"),
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _first_line(tree: ast.AST) -> int | None:
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line is not None:
            return int(line)
    return None


def _line_snippet(source: str, line: int | None) -> str:
    if line is None:
        return ""
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()
    return ""


def _normalize_corpus(corpus: Any) -> str:
    text = str(corpus or "").strip().lower()
    if text == "mbpp":
        return "mbpp"
    if text in {"humaneval", "human_eval", "human eval"}:
        return "humaneval"
    return text


def _display_corpus(corpus: str) -> str:
    return "MBPP" if corpus == "mbpp" else "HumanEval"


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _source_name(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_if_exists(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "CROSS_CORPUS_MATRIX_V6_REL_PATH",
    "EXP2879_REL_PATH",
    "EXP2889_REL_PATH",
    "ExperimentConfig",
    "FIELD_PRINCIPLES",
    "FORBIDDEN_IMPORTS",
    "FORBIDDEN_SIDE_EFFECTS",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "CodeStructuralContract",
    "build_contract_from_manifest_row",
    "build_experiment_artifact",
    "contract_to_json",
    "stable_json_sha256",
    "validate_contract_json",
    "verify_candidate_source",
    "write_experiment_artifact",
]
