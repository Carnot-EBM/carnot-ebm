"""Exp 2911 deterministic taxonomy verifier for Exp 2910 code candidates.

The verifier consumes the checked-in Exp 2910 generated-code artifact and adds
static failure labels that the original pass/fail row could not separate. It
does not generate code and does not execute untrusted candidates again; Exp 2910
already recorded sandbox task-test outcomes, and this module preserves those
outcomes while re-running deterministic AST parsing and static checks.

Spec: REQ-CODE-2911, SCENARIO-CODE-2911.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2911_code_hallucination_taxonomy_verifier_v1.json"
UPSTREAM_CODEGEN_ARTIFACT = Path("results/experiment_2910_sota_code_generation_corrigendum_v2.json")
INFERENCE_SUBSTRATE = "deterministic_verifier"

CODE_HALLUCINATION_CATEGORIES = (
    "invented_import",
    "undefined_name",
    "invented_attribute_or_method",
    "invalid_argument",
)
TAXONOMY_CATEGORIES = (
    *CODE_HALLUCINATION_CATEGORIES,
    "syntax_error",
    "runtime_error",
    "true_test_failure",
)
FILTERED_OUTCOME_CATEGORIES = (
    *CODE_HALLUCINATION_CATEGORIES,
    "syntax_error",
    "runtime_error",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "code_hallucination_verifier_ready",
    "upstream_codegen_artifact",
    "taxonomy_categories",
    "per_candidate_labels",
    "invented_import_rate",
    "undefined_name_rate",
    "invented_attribute_or_method_rate",
    "invalid_argument_rate",
    "syntax_error_rate",
    "runtime_error_rate",
    "pass_rate_after_taxonomy_filter",
    "verifier_source_paths",
    "inference_substrate",
    "duration_s",
    "run_date",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal verdict; blocked when the Exp 2910 corrigendum is absent.",
    "code_hallucination_verifier_ready": "True only after every Exp 2910 candidate is labeled.",
    "taxonomy_categories": "Orthogonal labels; one candidate may carry several categories.",
    "per_candidate_labels": "One row per Exp 2910 candidate with static and task-test labels.",
    "pass_rate_after_taxonomy_filter": (
        "Pass rate after removing syntax, runtime, and static code-hallucination failures."
    ),
    "inference_substrate": "Always deterministic_verifier; no live generation is performed.",
    "duration_s": "Measured wall-clock runtime; no padding.",
}

_TYPE_NAME_MAP = {
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "int": int,
    "float": float,
    "bool": bool,
    "bytes": bytes,
}
_BUILTIN_NAMES = set(dir(builtins))
_MODULE_CACHE: dict[str, ModuleType | None] = {}


@dataclass(frozen=True)
class StaticTaxonomyResult:
    """Static labels and findings emitted by AST analysis for one candidate."""

    labels: tuple[str, ...]
    findings: tuple[JsonDict, ...]
    syntax_success: bool
    syntax_error: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2911 artifact builder."""

    repo_root: Path = REPO_ROOT
    upstream_path: Path = UPSTREAM_CODEGEN_ARTIFACT
    output_path: Path | None = None
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def upstream_artifact_path(self) -> Path:
        return _repo_path(self.repo_root, self.upstream_path)


_DEFAULT_CONFIG = ExperimentConfig()


def classify_source(source: str) -> StaticTaxonomyResult:
    """Classify static code hallucination categories in one Python source string."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return StaticTaxonomyResult(
            labels=("syntax_error",),
            findings=(
                _finding(
                    "syntax_error",
                    "Python AST parse failed.",
                    line=exc.lineno,
                    col=exc.offset,
                    symbol=exc.msg,
                ),
            ),
            syntax_success=False,
            syntax_error=exc.msg,
        )

    import_aliases, import_findings = _import_aliases_and_findings(tree)
    module_symbols = _module_defined_names(tree) | set(import_aliases) | _BUILTIN_NAMES
    local_functions = _local_function_signatures(tree)
    type_env = _type_environment(tree, import_aliases)

    findings: list[JsonDict] = list(import_findings)
    findings.extend(_undefined_name_findings(tree, module_symbols, source))
    findings.extend(_attribute_findings(tree, import_aliases, type_env))
    findings.extend(_invalid_argument_findings(tree, import_aliases, type_env, local_functions))
    labels = tuple(
        category
        for category in TAXONOMY_CATEGORIES
        if any(f["category"] == category for f in findings)
    )
    return StaticTaxonomyResult(labels=labels, findings=tuple(findings), syntax_success=True)


def build_experiment_artifact(config: ExperimentConfig = _DEFAULT_CONFIG) -> JsonDict:
    """Build the Exp 2911 taxonomy artifact from the checked-in Exp 2910 row."""

    started = config.start_time()
    upstream_path = config.upstream_artifact_path()
    upstream = _read_json_if_ready(upstream_path)
    if upstream is None:
        return _blocked_artifact(config, started, "Exp 2910 artifact is absent or not ready.")

    candidate_rows = list(upstream.get("candidate_results") or [])
    per_task_results = list(upstream.get("per_task_results") or [])
    per_candidate_labels = [_candidate_label(config.repo_root, row) for row in candidate_rows]
    ready = bool(candidate_rows and per_task_results)
    category_rates = {
        category: _label_rate(per_candidate_labels, category) for category in TAXONOMY_CATEGORIES
    }
    filtered = [
        row
        for row in per_candidate_labels
        if not any(category in row["labels"] for category in FILTERED_OUTCOME_CATEGORIES)
    ]
    pass_rate_after_filter = _pass_rate(filtered)

    artifact: JsonDict = {
        "artifact": "experiment_2911_code_hallucination_taxonomy_verifier_v1",
        "schema": "carnot.code_hallucination_taxonomy_verifier.v1",
        "honest_verdict": (
            "complete: Exp 2910 code candidates labeled with deterministic taxonomy"
            if ready
            else "blocked_codegen_corrigendum_missing"
        ),
        "code_hallucination_verifier_ready": ready,
        "upstream_codegen_artifact": str(UPSTREAM_CODEGEN_ARTIFACT),
        "taxonomy_categories": list(TAXONOMY_CATEGORIES),
        "code_hallucination_categories": list(CODE_HALLUCINATION_CATEGORIES),
        "per_candidate_labels": per_candidate_labels,
        "invented_import_rate": category_rates["invented_import"],
        "undefined_name_rate": category_rates["undefined_name"],
        "invented_attribute_or_method_rate": category_rates["invented_attribute_or_method"],
        "invalid_argument_rate": category_rates["invalid_argument"],
        "syntax_error_rate": category_rates["syntax_error"],
        "runtime_error_rate": category_rates["runtime_error"],
        "true_test_failure_rate": category_rates["true_test_failure"],
        "pass_rate_after_taxonomy_filter": pass_rate_after_filter,
        "taxonomy_filter_definition": list(FILTERED_OUTCOME_CATEGORIES),
        "summary_by_model": _group_summary(per_candidate_labels, "model_hf_id"),
        "summary_by_corpus": _group_summary(per_candidate_labels, "corpus"),
        "summary_by_task": _group_summary(per_candidate_labels, "task_key"),
        "summary_by_pass_status": _group_summary(per_candidate_labels, "pass_status"),
        "upstream_per_task_result_count": len(per_task_results),
        "upstream_candidate_count": len(candidate_rows),
        "verifier_source_paths": _verifier_source_paths(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, config.clock() - started),
    }
    return artifact


def write_experiment_artifact(config: ExperimentConfig = _DEFAULT_CONFIG) -> JsonDict:
    """Build and persist the Exp 2911 artifact under ``results/``."""

    artifact = build_experiment_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run_experiment(config: ExperimentConfig = _DEFAULT_CONFIG) -> JsonDict:
    """Entrypoint used by the small experiment script."""

    return write_experiment_artifact(config)


def _candidate_label(repo_root: Path, row: Mapping[str, Any]) -> JsonDict:
    source = str(row.get("extracted_code") or "")
    static = classify_source(source)
    labels = set(static.labels)
    if _is_true_test_failure(row):
        labels.add("true_test_failure")
    if _is_runtime_error(row, static):
        labels.add("runtime_error")
    if row.get("passed"):
        labels.add("passed")
    ordered_labels = [label for label in (*TAXONOMY_CATEGORIES, "passed") if label in labels]
    raw_text, raw_loaded = _load_raw_response(repo_root, row)
    return {
        "corpus": str(row.get("corpus") or ""),
        "stable_id": str(row.get("stable_id") or ""),
        "task_key": f"{row.get('corpus') or ''}:{row.get('stable_id') or ''}",
        "candidate_index": int(row.get("candidate_index") or 0),
        "random_seed": int(row.get("random_seed") or 0),
        "model_hf_id": str(row.get("model_hf_id") or ""),
        "passed": bool(row.get("passed")),
        "pass_status": "passed" if row.get("passed") else "failed",
        "labels": ordered_labels,
        "findings": list(static.findings),
        "syntax_success": bool(static.syntax_success),
        "syntax_error": static.syntax_error,
        "task_test_outcome_source": "exp2910_sandbox_candidate_results",
        "error_type": row.get("error_type"),
        "error_message": str(row.get("error_message") or ""),
        "row_status": str(row.get("row_status") or ""),
        "raw_response_path": str(row.get("raw_response_path") or ""),
        "raw_response_loaded": raw_loaded,
        "raw_response_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }


def _blocked_artifact(config: ExperimentConfig, started: float, reason: str) -> JsonDict:
    return {
        "artifact": "experiment_2911_code_hallucination_taxonomy_verifier_v1",
        "schema": "carnot.code_hallucination_taxonomy_verifier.v1",
        "honest_verdict": "blocked_codegen_corrigendum_missing",
        "code_hallucination_verifier_ready": False,
        "blocked_reason": reason,
        "upstream_codegen_artifact": str(UPSTREAM_CODEGEN_ARTIFACT),
        "taxonomy_categories": list(TAXONOMY_CATEGORIES),
        "code_hallucination_categories": list(CODE_HALLUCINATION_CATEGORIES),
        "per_candidate_labels": [],
        "invented_import_rate": 0.0,
        "undefined_name_rate": 0.0,
        "invented_attribute_or_method_rate": 0.0,
        "invalid_argument_rate": 0.0,
        "syntax_error_rate": 0.0,
        "runtime_error_rate": 0.0,
        "true_test_failure_rate": 0.0,
        "pass_rate_after_taxonomy_filter": 0.0,
        "taxonomy_filter_definition": list(FILTERED_OUTCOME_CATEGORIES),
        "summary_by_model": {},
        "summary_by_corpus": {},
        "summary_by_task": {},
        "summary_by_pass_status": {},
        "upstream_per_task_result_count": 0,
        "upstream_candidate_count": 0,
        "verifier_source_paths": _verifier_source_paths(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, config.clock() - started),
    }


def _import_aliases_and_findings(tree: ast.AST) -> tuple[dict[str, str], list[JsonDict]]:
    aliases: dict[str, str] = {}
    findings: list[JsonDict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                local = alias.asname or root
                aliases[local] = root
                if not _module_available(alias.name):
                    findings.append(
                        _finding(
                            "invented_import",
                            f"Imported module {alias.name!r} is unavailable.",
                            line=node.lineno,
                            col=node.col_offset + 1,
                            symbol=alias.name,
                        )
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", maxsplit=1)[0]
            if not _module_available(node.module):
                findings.append(
                    _finding(
                        "invented_import",
                        f"Imported module {node.module!r} is unavailable.",
                        line=node.lineno,
                        col=node.col_offset + 1,
                        symbol=node.module,
                    )
                )
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = root
                if not _imported_member_available(node.module, alias.name):
                    findings.append(
                        _finding(
                            "invented_import",
                            f"Imported member {node.module}.{alias.name} is unavailable.",
                            line=node.lineno,
                            col=node.col_offset + 1,
                            symbol=f"{node.module}.{alias.name}",
                        )
                    )
    return aliases, findings


def _undefined_name_findings(
    tree: ast.AST,
    module_symbols: set[str],
    source: str,
) -> list[JsonDict]:
    findings: list[JsonDict] = []
    for scope in _function_scopes(tree):
        defined = set(module_symbols) | _function_defined_names(scope)
        for node in ast.walk(scope):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in defined
            ):
                findings.append(
                    _finding(
                        "undefined_name",
                        f"Name {node.id!r} is read but not defined in the candidate.",
                        line=node.lineno,
                        col=node.col_offset + 1,
                        symbol=node.id,
                        snippet=_line_snippet(source, node.lineno),
                    )
                )
    return _dedupe_findings(findings)


def _attribute_findings(
    tree: ast.AST,
    import_aliases: Mapping[str, str],
    type_env: Mapping[str, type],
) -> list[JsonDict]:
    findings: list[JsonDict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        receiver = _resolve_receiver(node.value, import_aliases, type_env)
        if receiver is None:
            continue
        owner_name, owner = receiver
        if hasattr(owner, node.attr):
            continue
        findings.append(
            _finding(
                "invented_attribute_or_method",
                f"Attribute or method {owner_name}.{node.attr} is unavailable.",
                line=node.lineno,
                col=node.col_offset + 1,
                symbol=f"{owner_name}.{node.attr}",
            )
        )
    return _dedupe_findings(findings)


def _invalid_argument_findings(
    tree: ast.AST,
    import_aliases: Mapping[str, str],
    type_env: Mapping[str, type],
    local_functions: Mapping[str, ast.FunctionDef],
) -> list[JsonDict]:
    findings: list[JsonDict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        finding = _invalid_call_finding(node, import_aliases, type_env, local_functions)
        if finding is not None:
            findings.append(finding)
    return _dedupe_findings(findings)


def _invalid_call_finding(
    node: ast.Call,
    import_aliases: Mapping[str, str],
    type_env: Mapping[str, type],
    local_functions: Mapping[str, ast.FunctionDef],
) -> JsonDict | None:
    call_name = _call_name(node.func)
    if isinstance(node.func, ast.Name) and node.func.id in local_functions:
        reason = _local_signature_error(local_functions[node.func.id], node)
        return _call_finding(call_name, reason, node) if reason else None
    target, include_self = _resolve_callable(node.func, import_aliases, type_env)
    if target is None:
        return None
    reason = _signature_bind_error(target, node, include_self=include_self)
    return _call_finding(call_name, reason, node) if reason else None


def _resolve_callable(
    func: ast.expr,
    import_aliases: Mapping[str, str],
    type_env: Mapping[str, type],
) -> tuple[Callable[..., Any] | Any | None, bool]:
    if isinstance(func, ast.Name) and hasattr(builtins, func.id):
        return getattr(builtins, func.id), False
    if not isinstance(func, ast.Attribute):
        return None, False
    receiver = _resolve_receiver(func.value, import_aliases, type_env)
    if receiver is None:
        return None, False
    _owner_name, owner = receiver
    if not hasattr(owner, func.attr):
        return None, False
    return getattr(owner, func.attr), isinstance(owner, type)


def _resolve_receiver(
    node: ast.AST,
    import_aliases: Mapping[str, str],
    type_env: Mapping[str, type],
) -> tuple[str, ModuleType | type] | None:
    if isinstance(node, ast.Name):
        if node.id in import_aliases:
            module = _safe_import_module(import_aliases[node.id])
            return (import_aliases[node.id], module) if module is not None else None
        if node.id in type_env:
            return type_env[node.id].__name__, type_env[node.id]
    inferred = _literal_type(node)
    if inferred is not None:
        return inferred.__name__, inferred
    return None


def _signature_bind_error(target: Any, node: ast.Call, *, include_self: bool) -> str:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return ""
    args = [object()] * len(node.args)
    if include_self:
        args.insert(0, object())
    kwargs = {kw.arg: object() for kw in node.keywords if kw.arg is not None}
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        return str(exc)
    return ""


def _local_signature_error(function: ast.FunctionDef, node: ast.Call) -> str:
    positional_params = [*function.args.posonlyargs, *function.args.args]
    required = len(positional_params) - len(function.args.defaults)
    max_positional = len(positional_params)
    n_positional = len(node.args)
    if n_positional < required:
        return f"missing required positional arguments for {function.name}"
    if function.args.vararg is None and n_positional > max_positional:
        return f"too many positional arguments for {function.name}"
    allowed_keywords = {arg.arg for arg in [*function.args.args, *function.args.kwonlyargs]}
    for keyword in node.keywords:
        if (
            keyword.arg is not None
            and keyword.arg not in allowed_keywords
            and function.args.kwarg is None
        ):
            return f"unexpected keyword argument {keyword.arg!r} for {function.name}"
    return ""


def _call_finding(call_name: str, reason: str, node: ast.Call) -> JsonDict:
    return _finding(
        "invalid_argument",
        f"Call {call_name} has an incompatible signature: {reason}",
        line=node.lineno,
        col=node.col_offset + 1,
        symbol=call_name,
    )


def _type_environment(tree: ast.AST, import_aliases: Mapping[str, str]) -> dict[str, type]:
    env: dict[str, type] = {}
    for function in _function_scopes(tree):
        for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]:
            inferred = _annotation_type(arg.annotation)
            if inferred is not None:
                env[arg.arg] = inferred
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                inferred = _literal_type(node.value)
                for target in node.targets:
                    for name in _target_names(target):
                        if inferred is not None and name not in import_aliases:
                            env[name] = inferred
            elif isinstance(node, ast.AnnAssign):
                inferred = _annotation_type(node.annotation) or _literal_type(node.value)
                if inferred is not None:
                    for name in _target_names(node.target):
                        env[name] = inferred
    return env


def _annotation_type(node: ast.AST | None) -> type | None:
    if isinstance(node, ast.Name):
        return _TYPE_NAME_MAP.get(node.id)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return _TYPE_NAME_MAP.get(node.value.id)
    return None


def _literal_type(node: ast.AST | None) -> type | None:
    if isinstance(node, ast.Constant) and node.value is not None:
        value_type = type(node.value)
        return value_type if value_type in _TYPE_NAME_MAP.values() else None
    if isinstance(node, ast.List):
        return list
    if isinstance(node, ast.Dict):
        return dict
    if isinstance(node, ast.Set):
        return set
    if isinstance(node, ast.Tuple):
        return tuple
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return _TYPE_NAME_MAP.get(node.func.id)
    return None


def _module_defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_target_names(node.target))
    return names


def _function_scopes(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _function_defined_names(function: ast.FunctionDef) -> set[str]:
    names = {arg.arg for arg in function.args.posonlyargs}
    names.update(arg.arg for arg in function.args.args)
    names.update(arg.arg for arg in function.args.kwonlyargs)
    if function.args.vararg is not None:
        names.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        names.add(function.args.kwarg.arg)
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.For, ast.AsyncFor)):
            names.update(_target_names(node.target))
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                names.update(_target_names(generator.target))
    return names


def _local_function_signatures(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in node.elts:
            names.update(_target_names(elt))
        return names
    return set()


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _imported_member_available(module_name: str, member: str) -> bool:
    if member == "*":
        return True
    module = _safe_import_module(module_name)
    if module is not None and hasattr(module, member):
        return True
    return _module_available(f"{module_name}.{member}")


def _safe_import_module(module_name: str) -> ModuleType | None:
    root = module_name.split(".", maxsplit=1)[0]
    if root not in getattr(sys, "stdlib_module_names", set()) and root not in {"typing"}:
        return None
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except Exception:
            _MODULE_CACHE[module_name] = None
    return _MODULE_CACHE[module_name]


def _is_true_test_failure(row: Mapping[str, Any]) -> bool:
    return not row.get("passed") and row.get("error_type") == "AssertionError"


def _is_runtime_error(row: Mapping[str, Any], static: StaticTaxonomyResult) -> bool:
    if row.get("passed") or not row.get("executed"):
        return False
    if bool(row.get("timed_out")):
        return True
    error_type = row.get("error_type")
    if error_type in (None, "AssertionError", "SyntaxError"):
        return False
    return static.syntax_success


def _read_json_if_ready(path: Path) -> JsonDict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("codegen_corrigendum_ready") is True else None


def _load_raw_response(repo_root: Path, row: Mapping[str, Any]) -> tuple[str, bool]:
    raw_path = str(row.get("raw_response_path") or "")
    if raw_path:
        path = _repo_path(repo_root, Path(raw_path))
        if path.is_file():
            return path.read_text(encoding="utf-8"), True
    return str(row.get("raw_response") or ""), False


def _label_rate(rows: Sequence[Mapping[str, Any]], label: str) -> float:
    return (sum(1 for row in rows if label in row.get("labels", ())) / len(rows)) if rows else 0.0


def _pass_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return (sum(1 for row in rows if row.get("passed")) / len(rows)) if rows else 0.0


def _group_summary(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "")].append(row)
    return {key: _summary(group) for key, group in sorted(groups.items())}


def _summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter(
        label for row in rows for label in row.get("labels", ()) if label in TAXONOMY_CATEGORIES
    )
    return {
        "n_candidates": len(rows),
        "n_passed": sum(1 for row in rows if row.get("passed")),
        "n_failed": sum(1 for row in rows if not row.get("passed")),
        "pass_rate": _pass_rate(rows),
        "category_counts": {category: counts.get(category, 0) for category in TAXONOMY_CATEGORIES},
        "category_rates": {
            category: _label_rate(rows, category) for category in TAXONOMY_CATEGORIES
        },
    }


def _finding(
    category: str,
    message: str,
    *,
    line: int | None,
    col: int | None,
    symbol: str,
    snippet: str = "",
) -> JsonDict:
    return {
        "category": category,
        "message": message,
        "line": line,
        "col": col,
        "symbol": symbol,
        "snippet": snippet,
    }


def _dedupe_findings(findings: Iterable[JsonDict]) -> list[JsonDict]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[JsonDict] = []
    for finding in findings:
        key = (
            finding.get("category"),
            finding.get("line"),
            finding.get("col"),
            finding.get("symbol"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(finding)
    return deduped


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return type(node).__name__


def _line_snippet(source: str, line: int | None) -> str:
    if line is None or line < 1:
        return ""
    lines = source.splitlines()
    return lines[line - 1].strip() if line <= len(lines) else ""


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _verifier_source_paths() -> list[str]:
    return [
        "python/carnot/eval/code_hallucination_taxonomy_verifier.py",
        "scripts/experiment_2911_code_hallucination_taxonomy_verifier.py",
        "tests/python/test_experiment_2911_code_hallucination_taxonomy_verifier.py",
    ]


__all__ = [
    "CODE_HALLUCINATION_CATEGORIES",
    "ExperimentConfig",
    "INFERENCE_SUBSTRATE",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "StaticTaxonomyResult",
    "TAXONOMY_CATEGORIES",
    "UPSTREAM_CODEGEN_ARTIFACT",
    "build_experiment_artifact",
    "classify_source",
    "run_experiment",
    "write_experiment_artifact",
]
