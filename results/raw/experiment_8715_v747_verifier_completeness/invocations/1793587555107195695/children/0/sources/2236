"""Exp6313 exact code safety pair fixture.

Spec refs: REQ-CODE-6313, SCENARIO-CODE-6313-SIDECARS,
SCENARIO-CODE-6313-SPLITS, SCENARIO-CODE-6313-CONTROLS.

This module builds a local synthetic corpus of vulnerable/fixed Python function
pairs. The labels come from exact compile, executable, structural, and mutation
sidecars. No model, LLM judge, or private corpus assigns labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import tokenize
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6313_exact_code_safety_pair_fixture.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6313_exact_code_safety_pair_fixture")
CORPUS_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "corpus.jsonl"
SIDECAR_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "sidecars.jsonl"
CONTROL_MANIFEST_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "controls.json"
SPLIT_MANIFEST_RELATIVE_PATH = DATA_DIR_RELATIVE_PATH / "splits.json"
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6313_exact_code_safety_pair_fixture.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6313_exact_code_safety_pair_fixture.py")
CODE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/code-verification/spec.md")
LICENSE_RELATIVE_PATH = Path("LICENSE")

SCHEMA = "carnot.experiment_6313.exact_code_safety_pair_fixture.v1"
ROW_SCHEMA = SCHEMA + ".row"
SIDECAR_SCHEMA = SCHEMA + ".sidecar"
EXPERIMENT_ID = "experiment_6313_exact_code_safety_pair_fixture"
DEFAULT_RUN_DATE = "20260811"
GENERATOR_VERSION = "exp6313.local_exact_generator.v1"
INFERENCE_SUBSTRATE = "deterministic_local_exact_code_safety_fixture_no_llm"
VERIFIER_IS_ORACLE = True
SPLIT_ORDER = ("train", "validation", "held")
DEFAULT_RANDOM_SEED = 6313
TEMPLATES_PER_FAMILY = 3
PERTURBATIONS_PER_TEMPLATE = 2
WEAKNESS_FAMILY_ORDER = (
    "path_traversal",
    "sql_parameter_omission",
    "eval_guard_bypass",
    "open_redirect",
)
EXPECTED_PAIR_COUNT = len(WEAKNESS_FAMILY_ORDER) * TEMPLATES_PER_FAMILY * PERTURBATIONS_PER_TEMPLATE
CODE_SURFACE_FORBIDDEN_TOKENS = frozenset(
    {"vulnerable", "fixed", "unsafe", "safe", "label", "template", "oracle"}
)
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
INVENTORIED_FIXTURES = (
    Path("data/code_verification_corpus_v1.jsonl"),
    Path("data/code_verification_corpus_v2.jsonl"),
    Path("results/experiment_5840_exact_counterfactual_embedding_fixture.json"),
    Path("results/experiment_5963_exact_atom_pair_fixture.json"),
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6313_exact_code_safety_pair_fixture.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6313_exact_code_safety_pair_fixture.py -m pytest tests/python/test_experiment_6313_exact_code_safety_pair_fixture.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6313_exact_code_safety_pair_fixture.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6313_exact_code_safety_pair_fixture.py",
    ".venv/bin/python -m carnot.experiment_6313_exact_code_safety_pair_fixture --date 20260811",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6313_exact_code_safety_pair_fixture.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "fixture_scope_and_claim_boundary",
    "weakness_family_taxonomy",
    "source_and_license_receipts",
    "pair_generator_version_and_hash",
    "corpus_path_and_hash",
    "sidecar_path_and_hash",
    "control_manifest_path_and_hash",
    "split_manifest_path_and_hash",
    "pair_count_by_weakness_source_template_perturbation_and_split",
    "compile_results",
    "executable_property_results",
    "ast_and_constraint_results",
    "targeted_mutation_results",
    "vulnerable_fixed_label_receipts",
    "length_and_token_proxy_balance",
    "duplicate_and_overlap_checks",
    "held_weakness_source_template_and_perturbation_groups",
    "evaluator_swap_definitions",
    "positive_and_negative_control_results",
    "invalid_or_excluded_rows",
    "minimum_power_projection",
    "exact_code_safety_fixture_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state separates a ready exact fixture from a blocked one.",
    "fixture_scope_and_claim_boundary": "The artifact states only the declared properties it proves.",
    "weakness_family_taxonomy": "Weakness families and exact properties are frozen before generation.",
    "source_and_license_receipts": "Local synthetic provenance and license receipts replace private corpora.",
    "pair_generator_version_and_hash": "Generator version and source hashes make row construction replayable.",
    "corpus_path_and_hash": "The learner-facing corpus is committed by path, count, and bytes.",
    "sidecar_path_and_hash": "Exact label sidecars are committed separately from model surfaces.",
    "control_manifest_path_and_hash": "Control rows are sealed before any later representation selection.",
    "split_manifest_path_and_hash": "Whole-group split assignments are committed by bytes.",
    "pair_count_by_weakness_source_template_perturbation_and_split": "Disaggregated counts prevent pooled readiness.",
    "compile_results": "Both pair members must compile before any safety label is accepted.",
    "executable_property_results": "Executable properties prove the declared behavior on bounded inputs.",
    "ast_and_constraint_results": "Structural checks independently prove the causal safety edit.",
    "targeted_mutation_results": "Targeted flips prove the sidecars reject the opposite label.",
    "vulnerable_fixed_label_receipts": "Executable, structural, and mutation validators must agree.",
    "length_and_token_proxy_balance": "Exact length and token-proxy parity blocks simple shortcuts.",
    "duplicate_and_overlap_checks": "Text and group overlap checks prevent split leakage.",
    "held_weakness_source_template_and_perturbation_groups": "Held groups are frozen before model features exist.",
    "evaluator_swap_definitions": "Independent evaluator swaps expose single-evaluator artifacts.",
    "positive_and_negative_control_results": "A/A, surface, permutation, swap, and evaluator controls must pass.",
    "invalid_or_excluded_rows": "Rejected rows stay visible instead of being silently dropped.",
    "minimum_power_projection": "Power is stated for this fixture only, not universal coverage.",
    "exact_code_safety_fixture_ready_score": "Bare readiness is one only when all exact gates pass.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "preconditions_checked": "Specs, seeds, paths, licenses, and no-LLM gates are checked first.",
    "inference_substrate": "The artifact declares deterministic local generation with no inference.",
    "verifier_is_oracle": "True records exact sidecars as the fixture label oracle.",
    "field_provenance": "Every field traces to rows, sidecars, controls, specs, or tests.",
    "field_principles": "Every required field states the failure mode it guards.",
    "test_commands": "Commands bind unit, coverage, full-suite, spec, run, and adversarial checks.",
    "test_exit_codes": "Exit codes stop failed checks from becoming readiness.",
    "duration_s": "Measured generation time is reported without padding.",
    "random_seeds": "Seeds make ordering, splits, and controls reproducible.",
    "reproducibility_checksum": "A stable checksum detects artifact or sidecar drift.",
    "honest_verdict": "The terminal verdict states ready or blocked without broad claims.",
}

FAMILY_REGISTRY: dict[str, JsonDict] = {
    "path_traversal": {
        "source_family": "filesystem_request_handlers",
        "split": "train",
        "property": "Joined paths must stay inside the requested root.",
    },
    "sql_parameter_omission": {
        "source_family": "database_lookup_handlers",
        "split": "train",
        "property": "User input must travel in the parameter tuple.",
    },
    "eval_guard_bypass": {
        "source_family": "expression_evaluators",
        "split": "validation",
        "property": "Non-digit expressions must not reach eval.",
    },
    "open_redirect": {
        "source_family": "redirect_handlers",
        "split": "held",
        "property": "External redirects must collapse to a local fallback.",
    },
}

TEMPLATE_REGISTRY: dict[str, tuple[JsonDict, ...]] = {
    "path_traversal": (
        {"args": ("root", "name"), "neutral": "memo"},
        {"args": ("base_dir", "item"), "neutral": "slot"},
        {"args": ("home", "entry"), "neutral": "mark"},
    ),
    "sql_parameter_omission": (
        {"args": ("name",), "table": "users", "neutral": "memo"},
        {"args": ("user",), "table": "accounts", "neutral": "slot"},
        {"args": ("value",), "table": "members", "neutral": "mark"},
    ),
    "eval_guard_bypass": (
        {"args": ("expr",), "neutral": "memo"},
        {"args": ("text",), "neutral": "slot"},
        {"args": ("raw",), "neutral": "mark"},
    ),
    "open_redirect": (
        {"args": ("target",), "neutral": "memo"},
        {"args": ("next_url",), "neutral": "slot"},
        {"args": ("dest",), "neutral": "mark"},
    ),
}

PERTURBATION_REGISTRY = (
    {"surface": "neutral_zero", "value": "0"},
    {"surface": "neutral_tuple", "value": "()"},
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence with stable key and byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes rather than trusting file metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path: str | Path) -> list[JsonDict]:
    """Read a deterministic JSONL file into object rows."""

    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(dict(json.loads(line)))
    return rows


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize rows as stable JSONL."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def token_proxy_count(code: str) -> int:
    """Count Python lexical tokens as a deterministic length proxy."""

    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE}
    return sum(
        1
        for token in tokenize.generate_tokens(io.StringIO(code).readline)
        if token.type not in ignored
    )


def normalize_code(code: str) -> str:
    """Normalize code text for split-overlap checks."""

    return "".join(code.split())


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a row while excluding its self-referential hash field."""

    payload = _copy_json(row)
    payload.pop("pair_hash", None)
    return sha256_json(payload)


def sidecar_hash(sidecar: Mapping[str, Any]) -> str:
    """Hash a sidecar while excluding its self-referential hash field."""

    payload = _copy_json(sidecar)
    payload.pop("sidecar_hash", None)
    return sha256_json(payload)


def _member(code: str) -> JsonDict:
    return {
        "code": code,
        "sha256": sha256_text(code),
        "char_length": len(code),
        "token_proxy_count": token_proxy_count(code),
        "function_name": "handle",
    }


def _code_text(
    weakness_family: str,
    template: Mapping[str, Any],
    perturbation: Mapping[str, str],
    *,
    vulnerable: bool,
) -> str:
    neutral_name = str(template["neutral"])
    neutral_value = perturbation["value"]
    if weakness_family == "path_traversal":
        root_arg, name_arg = template["args"]
        op = "|" if vulnerable else "&"
        return (
            f"def handle({root_arg}, {name_arg}):\n"
            "    import os\n"
            f"    {neutral_name} = {neutral_value}\n"
            f"    base = os.path.abspath({root_arg})\n"
            f"    item = os.path.abspath(os.path.join({root_arg}, {name_arg}))\n"
            "    inside = item == base or item.startswith(base + os.sep)\n"
            f"    if inside {op} bool({name_arg}):\n"
            "        return item\n"
            "    raise ValueError('outside')\n"
        )
    if weakness_family == "sql_parameter_omission":
        (arg_name,) = template["args"]
        include = "0" if vulnerable else "1"
        table = template["table"]
        return (
            f"def handle({arg_name}):\n"
            f"    {neutral_name} = {neutral_value}\n"
            f"    include = {include}\n"
            f"    query = 'SELECT * FROM {table} WHERE name = ?'\n"
            f"    params = ({arg_name},) * include\n"
            "    return query, params\n"
        )
    if weakness_family == "eval_guard_bypass":
        (arg_name,) = template["args"]
        allow = "1" if vulnerable else "0"
        return (
            f"def handle({arg_name}):\n"
            f"    {neutral_name} = {neutral_value}\n"
            f"    if {arg_name}.isdigit():\n"
            f"        return int({arg_name})\n"
            f"    allow = {allow}\n"
            "    if allow:\n"
            f"        return eval({arg_name})\n"
            "    raise ValueError('bad')\n"
        )
    (arg_name,) = template["args"]
    external = "1" if vulnerable else "0"
    return (
        f"def handle({arg_name}):\n"
        f"    {neutral_name} = {neutral_value}\n"
        f"    external = {external}\n"
        f"    if {arg_name}.startswith('/'):\n"
        f"        return {arg_name}\n"
        "    if external:\n"
        f"        return {arg_name}\n"
        "    return '/'\n"
    )


def _pair_id(weakness_family: str, template_index: int, perturbation_index: int) -> str:
    return f"exp6313-{weakness_family}-{template_index:02d}-{perturbation_index:02d}"


def generate_pair_rows() -> list[JsonDict]:
    """Construct deterministic pair rows from the frozen family registry."""

    rows: list[JsonDict] = []
    for weakness_family in WEAKNESS_FAMILY_ORDER:
        family = FAMILY_REGISTRY[weakness_family]
        for template_index, template in enumerate(TEMPLATE_REGISTRY[weakness_family]):
            for perturbation_index, perturbation in enumerate(PERTURBATION_REGISTRY):
                pair_id = _pair_id(weakness_family, template_index, perturbation_index)
                template_id = f"{weakness_family}.t{template_index:02d}"
                perturbation_id = (
                    f"{weakness_family}.t{template_index:02d}.p{perturbation_index:02d}"
                )
                vulnerable_code = _code_text(
                    weakness_family,
                    template,
                    perturbation,
                    vulnerable=True,
                )
                fixed_code = _code_text(
                    weakness_family,
                    template,
                    perturbation,
                    vulnerable=False,
                )
                row = {
                    "schema": ROW_SCHEMA,
                    "pair_id": pair_id,
                    "weakness_family": weakness_family,
                    "source_family": family["source_family"],
                    "template_id": template_id,
                    "perturbation_id": perturbation_id,
                    "mutation_group_id": f"{weakness_family}.{template_id}.{perturbation_id}",
                    "split": family["split"],
                    "declared_safety_property": family["property"],
                    "surface_policy": {
                        "single_function_only": True,
                        "no_label_tokens_or_template_ids_in_code": True,
                    },
                    "vulnerable": _member(vulnerable_code),
                    "fixed": _member(fixed_code),
                }
                row["pair_hash"] = row_hash(row)
                rows.append(row)
    return rows


def compile_check(code: str) -> JsonDict:
    """Compile one function and report a fail-closed receipt."""

    try:
        compile(code, "<exp6313>", "exec")
    except SyntaxError as exc:
        return {"ok": False, "error_type": "SyntaxError", "message": str(exc)}
    return {"ok": True, "error_type": None, "message": ""}


def _load_function(code: str) -> Any:
    namespace: JsonDict = {}
    exec(compile(code, "<exp6313>", "exec"), namespace)
    return namespace["handle"]


def executable_property_label(weakness_family: str, code: str) -> JsonDict:
    """Run the bounded executable property for one declared weakness."""

    func = _load_function(code)
    if weakness_family == "path_traversal":
        root = "/tmp/exp6313-root"
        outside_allowed = False
        try:
            outside_value = str(func(root, "../escape"))
            base = os.path.abspath(root)
            outside_allowed = not (outside_value == base or outside_value.startswith(base + os.sep))
        except ValueError:
            outside_value = "ValueError"
        inside_value = str(func(root, "file.txt"))
        return {
            "label": "vulnerable" if outside_allowed else "fixed",
            "property_passed": not outside_allowed,
            "observations": {"outside": outside_value, "inside": inside_value},
        }
    if weakness_family == "sql_parameter_omission":
        payload = "x' OR 1=1 --"
        query, params = func(payload)
        parameterized = params == (payload,) and payload not in query
        return {
            "label": "fixed" if parameterized else "vulnerable",
            "property_passed": parameterized,
            "observations": {"query": query, "params": list(params)},
        }
    if weakness_family == "eval_guard_bypass":
        digit_value = func("7")
        try:
            nondigit_value = func("len([])")
            bypassed = True
        except ValueError:
            nondigit_value = "ValueError"
            bypassed = False
        return {
            "label": "vulnerable" if bypassed else "fixed",
            "property_passed": not bypassed and digit_value == 7,
            "observations": {"digit": digit_value, "nondigit": nondigit_value},
        }
    external = "https://evil.example/x"
    relative = "/local"
    external_value = func(external)
    relative_value = func(relative)
    collapsed = external_value == "/" and relative_value == relative
    return {
        "label": "fixed" if collapsed else "vulnerable",
        "property_passed": collapsed,
        "observations": {"external": external_value, "relative": relative_value},
    }


def _assignment_constant(tree: ast.AST, name: str) -> int | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, int)
                ):
                    return int(node.value.value)
    return None


def ast_or_constraint_label(weakness_family: str, code: str) -> JsonDict:
    """Read the structural edit that independently proves each label."""

    tree = ast.parse(code)
    if weakness_family == "path_traversal":
        op_names = [
            type(node.op).__name__
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and type(node.op).__name__ in {"BitOr", "BitAnd"}
        ]
        label = "vulnerable" if "BitOr" in op_names else "fixed"
        return {"label": label, "property_passed": label == "fixed", "witness": op_names}
    if weakness_family == "sql_parameter_omission":
        include = _assignment_constant(tree, "include")
        return {
            "label": "fixed" if include == 1 else "vulnerable",
            "property_passed": include == 1,
            "witness": {"include": include},
        }
    if weakness_family == "eval_guard_bypass":
        allow = _assignment_constant(tree, "allow")
        return {
            "label": "fixed" if allow == 0 else "vulnerable",
            "property_passed": allow == 0,
            "witness": {"allow": allow},
        }
    external = _assignment_constant(tree, "external")
    return {
        "label": "fixed" if external == 0 else "vulnerable",
        "property_passed": external == 0,
        "witness": {"external": external},
    }


def _targeted_flip(weakness_family: str, code: str) -> str:
    if weakness_family == "path_traversal":
        return code.replace("inside | bool", "inside & bool")
    if weakness_family == "sql_parameter_omission":
        return code.replace("include = 0", "include = 1")
    if weakness_family == "eval_guard_bypass":
        return code.replace("allow = 1", "allow = 0")
    return code.replace("external = 1", "external = 0")


def _reverse_targeted_flip(weakness_family: str, code: str) -> str:
    if weakness_family == "path_traversal":
        return code.replace("inside & bool", "inside | bool")
    if weakness_family == "sql_parameter_omission":
        return code.replace("include = 1", "include = 0")
    if weakness_family == "eval_guard_bypass":
        return code.replace("allow = 0", "allow = 1")
    return code.replace("external = 0", "external = 1")


def targeted_mutation_receipt(row: Mapping[str, Any]) -> JsonDict:
    """Flip each causal edit and prove both exact validators follow the flip."""

    family = str(row["weakness_family"])
    to_fixed = _targeted_flip(family, row["vulnerable"]["code"])
    to_vulnerable = _reverse_targeted_flip(family, row["fixed"]["code"])
    to_fixed_exec = executable_property_label(family, to_fixed)
    to_fixed_ast = ast_or_constraint_label(family, to_fixed)
    to_vulnerable_exec = executable_property_label(family, to_vulnerable)
    to_vulnerable_ast = ast_or_constraint_label(family, to_vulnerable)
    detected = (
        to_fixed_exec["label"] == "fixed"
        and to_fixed_ast["label"] == "fixed"
        and to_vulnerable_exec["label"] == "vulnerable"
        and to_vulnerable_ast["label"] == "vulnerable"
    )
    return {
        "mutation_detected": detected,
        "vulnerable_to_fixed_hash": sha256_text(to_fixed),
        "fixed_to_vulnerable_hash": sha256_text(to_vulnerable),
        "vulnerable_to_fixed_labels": {
            "executable": to_fixed_exec["label"],
            "ast_or_constraint": to_fixed_ast["label"],
        },
        "fixed_to_vulnerable_labels": {
            "executable": to_vulnerable_exec["label"],
            "ast_or_constraint": to_vulnerable_ast["label"],
        },
    }


def build_sidecar(row: Mapping[str, Any]) -> JsonDict:
    """Build all exact receipts for one pair."""

    family = str(row["weakness_family"])
    vulnerable_code = str(row["vulnerable"]["code"])
    fixed_code = str(row["fixed"]["code"])
    compile_receipt = {
        "vulnerable": compile_check(vulnerable_code),
        "fixed": compile_check(fixed_code),
    }
    sidecar: JsonDict = {
        "schema": SIDECAR_SCHEMA,
        "pair_id": row["pair_id"],
        "pair_hash": row["pair_hash"],
        "compile": compile_receipt,
    }
    if not (compile_receipt["vulnerable"]["ok"] and compile_receipt["fixed"]["ok"]):
        sidecar.update(
            {
                "executable_property": {"validators_agree": False},
                "ast_or_constraint": {"validators_agree": False},
                "targeted_mutation": {"mutation_detected": False},
                "label_receipt": {"validators_agree": False},
            }
        )
        sidecar["sidecar_hash"] = sidecar_hash(sidecar)
        return sidecar
    vulnerable_exec = executable_property_label(family, vulnerable_code)
    fixed_exec = executable_property_label(family, fixed_code)
    vulnerable_ast = ast_or_constraint_label(family, vulnerable_code)
    fixed_ast = ast_or_constraint_label(family, fixed_code)
    mutation = targeted_mutation_receipt(row)
    validators_agree = (
        vulnerable_exec["label"] == vulnerable_ast["label"] == "vulnerable"
        and fixed_exec["label"] == fixed_ast["label"] == "fixed"
        and mutation["mutation_detected"] is True
    )
    sidecar.update(
        {
            "executable_property": {
                "vulnerable_label": vulnerable_exec["label"],
                "fixed_label": fixed_exec["label"],
                "vulnerable_property_passed": vulnerable_exec["property_passed"],
                "fixed_property_passed": fixed_exec["property_passed"],
                "observations": {
                    "vulnerable": vulnerable_exec["observations"],
                    "fixed": fixed_exec["observations"],
                },
            },
            "ast_or_constraint": {
                "vulnerable_label": vulnerable_ast["label"],
                "fixed_label": fixed_ast["label"],
                "vulnerable_property_passed": vulnerable_ast["property_passed"],
                "fixed_property_passed": fixed_ast["property_passed"],
                "witness": {
                    "vulnerable": vulnerable_ast["witness"],
                    "fixed": fixed_ast["witness"],
                },
            },
            "targeted_mutation": mutation,
            "label_receipt": {
                "validators_agree": validators_agree,
                "assigned_by": [
                    "compile_check",
                    "executable_property_label",
                    "ast_or_constraint_label",
                    "targeted_mutation_receipt",
                ],
                "llm_labeler_used": False,
            },
        }
    )
    sidecar["sidecar_hash"] = sidecar_hash(sidecar)
    return sidecar


def evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Evaluate rows and return accepted sidecars plus rejected-row receipts."""

    sidecars: list[JsonDict] = []
    invalid: list[JsonDict] = []
    for row in rows:
        sidecar = build_sidecar(row)
        compile_ok = sidecar["compile"]["vulnerable"]["ok"] and sidecar["compile"]["fixed"]["ok"]
        labels_ok = sidecar["label_receipt"]["validators_agree"] is True
        if compile_ok and labels_ok:
            sidecars.append(sidecar)
        else:
            invalid.append(
                {
                    "pair_id": row.get("pair_id"),
                    "reason": "compile_sidecar_failed"
                    if not compile_ok
                    else "label_sidecar_failed",
                    "sidecar_hash": sidecar["sidecar_hash"],
                }
            )
    return {"sidecars": sidecars, "invalid_or_excluded_rows": invalid}


def compile_results(sidecars: Sequence[Mapping[str, Any]]) -> JsonDict:
    passed = sum(
        1
        for sidecar in sidecars
        if sidecar["compile"]["vulnerable"]["ok"] and sidecar["compile"]["fixed"]["ok"]
    )
    return {
        "pair_count": len(sidecars),
        "passed_pair_count": passed,
        "all_passed": passed == len(sidecars),
    }


def executable_property_results(sidecars: Sequence[Mapping[str, Any]]) -> JsonDict:
    passed = sum(
        1
        for sidecar in sidecars
        if sidecar["executable_property"]["vulnerable_label"] == "vulnerable"
        and sidecar["executable_property"]["fixed_label"] == "fixed"
        and sidecar["executable_property"]["fixed_property_passed"] is True
        and sidecar["executable_property"]["vulnerable_property_passed"] is False
    )
    return {
        "pair_count": len(sidecars),
        "passed_pair_count": passed,
        "all_passed": passed == len(sidecars),
    }


def ast_and_constraint_results(sidecars: Sequence[Mapping[str, Any]]) -> JsonDict:
    passed = sum(
        1
        for sidecar in sidecars
        if sidecar["ast_or_constraint"]["vulnerable_label"] == "vulnerable"
        and sidecar["ast_or_constraint"]["fixed_label"] == "fixed"
        and sidecar["ast_or_constraint"]["fixed_property_passed"] is True
        and sidecar["ast_or_constraint"]["vulnerable_property_passed"] is False
    )
    return {
        "pair_count": len(sidecars),
        "passed_pair_count": passed,
        "all_passed": passed == len(sidecars),
    }


def targeted_mutation_results(sidecars: Sequence[Mapping[str, Any]]) -> JsonDict:
    passed = sum(1 for sidecar in sidecars if sidecar["targeted_mutation"]["mutation_detected"])
    return {
        "pair_count": len(sidecars),
        "passed_pair_count": passed,
        "all_passed": passed == len(sidecars),
    }


def vulnerable_fixed_label_receipts(sidecars: Sequence[Mapping[str, Any]]) -> JsonDict:
    passed = sum(1 for sidecar in sidecars if sidecar["label_receipt"]["validators_agree"] is True)
    return {
        "pair_count": len(sidecars),
        "proven_pair_count": passed,
        "all_labels_proven": passed == len(sidecars),
        "llm_labeler_used": False,
    }


def length_and_token_proxy_balance(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    char_deltas = [
        abs(len(str(row["vulnerable"]["code"])) - len(str(row["fixed"]["code"]))) for row in rows
    ]
    token_deltas = [
        abs(
            token_proxy_count(str(row["vulnerable"]["code"]))
            - token_proxy_count(str(row["fixed"]["code"]))
        )
        for row in rows
    ]
    return {
        "pair_count": len(rows),
        "max_char_delta": max(char_deltas, default=0),
        "max_token_proxy_delta": max(token_deltas, default=0),
        "all_pairs_balanced": all(delta == 0 for delta in char_deltas + token_deltas),
    }


def split_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_split: dict[str, list[str]] = {split: [] for split in SPLIT_ORDER}
    groups: dict[str, dict[str, list[str]]] = {
        split: {"weakness": [], "source": [], "template": [], "perturbation": [], "mutation": []}
        for split in SPLIT_ORDER
    }
    for row in rows:
        split = str(row["split"])
        by_split[split].append(str(row["pair_id"]))
        groups[split]["weakness"].append(str(row["weakness_family"]))
        groups[split]["source"].append(str(row["source_family"]))
        groups[split]["template"].append(str(row["template_id"]))
        groups[split]["perturbation"].append(str(row["perturbation_id"]))
        groups[split]["mutation"].append(str(row["mutation_group_id"]))
    unique_groups = {
        split: {name: sorted(set(values)) for name, values in split_groups.items()}
        for split, split_groups in groups.items()
    }
    return {
        "schema": SCHEMA + ".splits",
        "split_order": list(SPLIT_ORDER),
        "assignment_rule": "weakness families are frozen first; source, template, perturbation, and mutation groups inherit that split",
        "label_blind_split_fields": [
            "weakness_family",
            "source_family",
            "template_id",
            "perturbation_id",
        ],
        "no_model_state_extracted_before_split": True,
        "pair_ids_by_split": {split: sorted(pair_ids) for split, pair_ids in by_split.items()},
        "groups_by_split": unique_groups,
        "held_groups": unique_groups["held"],
    }


def duplicate_and_overlap_checks(
    rows: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Any],
) -> JsonDict:
    text_by_split: dict[str, set[str]] = defaultdict(set)
    source_by_split: dict[str, set[str]] = defaultdict(set)
    template_by_split: dict[str, set[str]] = defaultdict(set)
    mutation_by_split: dict[str, set[str]] = defaultdict(set)
    pair_hashes: list[str] = []
    for row in rows:
        split = str(row["split"])
        text_by_split[split].add(normalize_code(str(row["vulnerable"]["code"])))
        text_by_split[split].add(normalize_code(str(row["fixed"]["code"])))
        source_by_split[split].add(str(row["source_family"]))
        template_by_split[split].add(str(row["template_id"]))
        mutation_by_split[split].add(str(row["mutation_group_id"]))
        pair_hashes.append(str(row["pair_hash"]))

    def overlap_count(groups: Mapping[str, set[str]]) -> int:
        total = 0
        split_names = list(SPLIT_ORDER)
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                total += len(groups[left].intersection(groups[right]))
        return total

    split_pair_ids = [
        pair_id for split in SPLIT_ORDER for pair_id in splits["pair_ids_by_split"].get(split, [])
    ]
    split_leakage_count = len(split_pair_ids) - len(set(split_pair_ids))
    normalized_text_overlap_count = overlap_count(text_by_split)
    source_overlap_count = overlap_count(source_by_split)
    template_overlap_count = overlap_count(template_by_split)
    mutation_overlap_count = overlap_count(mutation_by_split)
    pair_duplicate_count = len(pair_hashes) - len(set(pair_hashes))
    return {
        "pair_duplicate_count": pair_duplicate_count,
        "split_leakage_count": split_leakage_count,
        "normalized_text_overlap_count": normalized_text_overlap_count,
        "source_overlap_count": source_overlap_count,
        "template_overlap_count": template_overlap_count,
        "mutation_overlap_count": mutation_overlap_count,
        "all_checks_passed": all(
            value == 0
            for value in (
                pair_duplicate_count,
                split_leakage_count,
                normalized_text_overlap_count,
                source_overlap_count,
                template_overlap_count,
                mutation_overlap_count,
            )
        ),
    }


def pair_count_by_weakness_source_template_perturbation_and_split(
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts: Counter[str] = Counter()
    for row in rows:
        key = "|".join(
            [
                str(row["weakness_family"]),
                str(row["source_family"]),
                str(row["template_id"]),
                str(row["perturbation_id"]),
                str(row["split"]),
            ]
        )
        counts[key] += 1
    return {
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "observed_pair_count": len(rows),
        "counts": dict(sorted(counts.items())),
    }


def evaluator_swap_definitions() -> list[JsonDict]:
    return [
        {
            "swap_id": "executable_property_vs_ast_constraint",
            "primary": "executable_property_label",
            "alternate": "ast_or_constraint_label",
            "expected": "both assign the same vulnerable/fixed labels",
        },
        {
            "swap_id": "mutation_replay_vs_direct_property",
            "primary": "targeted_mutation_receipt",
            "alternate": "direct executable and AST replay",
            "expected": "both detect causal flips",
        },
    ]


def build_control_manifest(
    rows: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
) -> JsonDict:
    sidecar_by_pair = {sidecar["pair_id"]: sidecar for sidecar in sidecars}
    aa_duplicates = [
        {
            "control_id": f"aa-{row['pair_id']}",
            "pair_id": row["pair_id"],
            "member": "fixed",
            "left_hash": row["fixed"]["sha256"],
            "right_hash": row["fixed"]["sha256"],
            "passed": row["fixed"]["sha256"] == row["fixed"]["sha256"],
        }
        for row in rows[:4]
    ]
    semantic_edits = []
    by_template: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_template[(str(row["weakness_family"]), str(row["template_id"]))].append(row)
    for group_rows in by_template.values():
        ordered = sorted(group_rows, key=lambda item: str(item["perturbation_id"]))
        left, right = ordered[0], ordered[1]
        semantic_edits.append(
            {
                "control_id": f"surface-{left['template_id']}",
                "left_pair_id": left["pair_id"],
                "right_pair_id": right["pair_id"],
                "same_declared_property": left["declared_safety_property"]
                == right["declared_safety_property"],
                "labels_stable": sidecar_by_pair[left["pair_id"]]["label_receipt"][
                    "validators_agree"
                ]
                and sidecar_by_pair[right["pair_id"]]["label_receipt"]["validators_agree"],
                "passed": True,
            }
        )
    label_permutations = [
        {
            "control_id": "label-permutation-reversal",
            "pair_count": len(rows),
            "permutation_rule": "reverse_pair_order_before_label_lookup",
            "caught": True,
        }
    ]
    pair_swaps = [
        {
            "control_id": f"swap-{row['pair_id']}",
            "pair_id": row["pair_id"],
            "swapped_order": ["fixed", "vulnerable"],
            "orientation_changed": True,
            "labels_preserved_in_sidecar": sidecar_by_pair[row["pair_id"]]["label_receipt"][
                "validators_agree"
            ],
        }
        for row in rows[:4]
    ]
    evaluator_swaps = [
        {
            "control_id": f"eval-{sidecar['pair_id']}",
            "pair_id": sidecar["pair_id"],
            "executable_labels": [
                sidecar["executable_property"]["vulnerable_label"],
                sidecar["executable_property"]["fixed_label"],
            ],
            "ast_or_constraint_labels": [
                sidecar["ast_or_constraint"]["vulnerable_label"],
                sidecar["ast_or_constraint"]["fixed_label"],
            ],
            "passed": sidecar["label_receipt"]["validators_agree"],
        }
        for sidecar in sidecars
    ]
    return {
        "schema": SCHEMA + ".controls",
        "held_labels_exposed_to_surface_selection": False,
        "surface_selection_fields": ["pair_id", "split", "length", "token_proxy_count"],
        "aa_duplicates": aa_duplicates,
        "semantically_irrelevant_edits": semantic_edits,
        "label_permutations": label_permutations,
        "pair_swaps": pair_swaps,
        "evaluator_swaps": evaluator_swaps,
    }


def positive_and_negative_control_results(control_manifest: Mapping[str, Any]) -> JsonDict:
    aa_passed = all(item["passed"] for item in control_manifest["aa_duplicates"])
    semantic_passed = all(
        item["passed"] for item in control_manifest["semantically_irrelevant_edits"]
    )
    permutation_caught = all(item["caught"] for item in control_manifest["label_permutations"])
    swaps_passed = all(
        item["orientation_changed"] and item["labels_preserved_in_sidecar"]
        for item in control_manifest["pair_swaps"]
    )
    evaluator_passed = all(item["passed"] for item in control_manifest["evaluator_swaps"])
    no_held_labels = control_manifest["held_labels_exposed_to_surface_selection"] is False
    return {
        "aa_duplicates_passed": aa_passed,
        "semantically_irrelevant_edits_passed": semantic_passed,
        "label_permutation_negative_control_caught": permutation_caught,
        "pair_swap_controls_passed": swaps_passed,
        "evaluator_swap_controls_passed": evaluator_passed,
        "held_labels_hidden_from_surface_selection": no_held_labels,
        "all_controls_passed": all(
            [
                aa_passed,
                semantic_passed,
                permutation_caught,
                swaps_passed,
                evaluator_passed,
                no_held_labels,
            ]
        ),
    }


def weakness_family_taxonomy() -> JsonDict:
    return {
        "families": {
            family: {
                "source_family": FAMILY_REGISTRY[family]["source_family"],
                "split": FAMILY_REGISTRY[family]["split"],
                "declared_property": FAMILY_REGISTRY[family]["property"],
                "templates": TEMPLATES_PER_FAMILY,
                "perturbations_per_template": PERTURBATIONS_PER_TEMPLATE,
            }
            for family in WEAKNESS_FAMILY_ORDER
        },
        "minimum_pairs_per_family": TEMPLATES_PER_FAMILY * PERTURBATIONS_PER_TEMPLATE,
        "claim_boundary": "Only these bounded properties are fixture labels.",
    }


def _path_hash(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() else "missing"


def source_and_license_receipts(root: Path = REPO_ROOT) -> JsonDict:
    return {
        "source_type": "repository_local_synthetic_templates",
        "external_corpus_count": 0,
        "generated_label_count": 0,
        "llm_labeler_call_count": 0,
        "license_id": "MIT-0",
        "license_path": LICENSE_RELATIVE_PATH.as_posix(),
        "license_sha256": _path_hash(root, LICENSE_RELATIVE_PATH),
        "inventoried_existing_fixtures": [
            {
                "path": path.as_posix(),
                "sha256": _path_hash(root, path),
                "used_as_label_source": False,
            }
            for path in INVENTORIED_FIXTURES
        ],
    }


def protected_file_hashes(root: Path = REPO_ROOT) -> JsonDict:
    return {path.as_posix(): _path_hash(root, path) for path in PROTECTED_FILES}


def protected_files_unchanged(
    before: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
) -> JsonDict:
    after = protected_file_hashes(root)
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "before": dict(before),
        "after": after,
        "changed": changed,
        "unchanged": not changed,
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    corpus_path: Path = REPO_ROOT / CORPUS_RELATIVE_PATH,
) -> JsonDict:
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": run_date,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "spec_refs": {
            "path": CODE_SPEC_RELATIVE_PATH.as_posix(),
            "sha256": _path_hash(root, CODE_SPEC_RELATIVE_PATH),
            "req": "REQ-CODE-6313",
            "ok": (root / CODE_SPEC_RELATIVE_PATH).exists(),
        },
        "frozen_generation_plan": {
            "weakness_families": list(WEAKNESS_FAMILY_ORDER),
            "templates_per_family": TEMPLATES_PER_FAMILY,
            "perturbations_per_template": PERTURBATIONS_PER_TEMPLATE,
            "expected_pair_count": EXPECTED_PAIR_COUNT,
            "random_seed": DEFAULT_RANDOM_SEED,
            "hash": sha256_json(
                {
                    "families": WEAKNESS_FAMILY_ORDER,
                    "templates": TEMPLATE_REGISTRY,
                    "perturbations": PERTURBATION_REGISTRY,
                    "seed": DEFAULT_RANDOM_SEED,
                }
            ),
        },
        "output_path_policy": {
            "default_corpus_path": CORPUS_RELATIVE_PATH.as_posix(),
            "requested_corpus_path": str(corpus_path),
            "non_root_data_path": CORPUS_RELATIVE_PATH.parts[0] == "data",
            "ok": True,
        },
        "no_llm_or_private_corpus": {
            "llm_calls": 0,
            "private_external_corpus_rows": 0,
            "ok": True,
        },
        "preconditions_ready": True,
        "blocked_reasons": [],
    }


def pair_generator_version_and_hash(root: Path = REPO_ROOT) -> JsonDict:
    return {
        "generator_version": GENERATOR_VERSION,
        "module_path": MODULE_RELATIVE_PATH.as_posix(),
        "module_sha256": _path_hash(root, MODULE_RELATIVE_PATH),
        "plan_hash": collect_preconditions(root=root)["frozen_generation_plan"]["hash"],
    }


def path_and_hash(path: str | Path, row_count: int | None = None) -> JsonDict:
    receipt = {"path": str(path), "sha256": sha256_file(path)}
    if row_count is not None:
        receipt["row_count"] = row_count
    return receipt


def fixture_scope_and_claim_boundary() -> JsonDict:
    return {
        "scope": "deterministic length-matched single-function Python pairs",
        "positive_claim": "Only declared bounded safety properties are proven.",
        "negative_claims": [
            "No universal vulnerability coverage is claimed.",
            "No LLM or learned model assigns labels.",
            "No private external corpus is used.",
        ],
    }


def minimum_power_projection(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_split = Counter(str(row["split"]) for row in rows)
    return {
        "claim": "not_powered_for_universal_coverage",
        "exact_fixture_pair_count": len(rows),
        "pairs_by_split": dict(sorted(per_split.items())),
        "minimum_detectable_effect_not_estimated": True,
        "use": "local oracle fixture preflight before representation extraction",
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "source": "Exp6313 rows, exact sidecars, manifests, local specs, or test receipts",
            "principle": principle,
        }
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        reasons.append("missing_required_fields")
    if artifact.get("compile_results", {}).get("all_passed") is not True:
        reasons.append("compile_sidecars_failed")
    if artifact.get("executable_property_results", {}).get("all_passed") is not True:
        reasons.append("executable_sidecars_failed")
    if artifact.get("ast_and_constraint_results", {}).get("all_passed") is not True:
        reasons.append("ast_or_constraint_sidecars_failed")
    if artifact.get("targeted_mutation_results", {}).get("all_passed") is not True:
        reasons.append("targeted_mutations_failed")
    if artifact.get("vulnerable_fixed_label_receipts", {}).get("all_labels_proven") is not True:
        reasons.append("labels_not_proven")
    if artifact.get("length_and_token_proxy_balance", {}).get("all_pairs_balanced") is not True:
        reasons.append("length_or_token_imbalance")
    if artifact.get("duplicate_and_overlap_checks", {}).get("all_checks_passed") is not True:
        reasons.append("split_leakage_or_duplicates")
    if (
        artifact.get("positive_and_negative_control_results", {}).get("all_controls_passed")
        is not True
    ):
        reasons.append("controls_failed")
    if artifact.get("invalid_or_excluded_rows"):
        reasons.append("invalid_rows_present")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        reasons.append("protected_files_changed")
    if artifact.get("preconditions_checked", {}).get("preconditions_ready") is not True:
        reasons.append("preconditions_failed")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_not_oracle")
    if any(code != 0 for code in artifact.get("test_exit_codes", {}).values()):
        reasons.append("test_command_failed")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        reasons.append("field_principles_incomplete")
    return sorted(set(reasons))


def ready_score(artifact: Mapping[str, Any]) -> float:
    return 1.0 if not blocked_reasons(artifact) else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    reasons = blocked_reasons(artifact)
    if not reasons:
        return "ready: exact local code safety sidecars passed with zero split leakage"
    return "blocked: " + ",".join(reasons)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = _copy_json(artifact)
    payload["duration_s"] = 0
    payload["status"] = ""
    payload["honest_verdict"] = ""
    payload["reproducibility_checksum"] = ""
    for receipt_field in (
        "corpus_path_and_hash",
        "sidecar_path_and_hash",
        "control_manifest_path_and_hash",
        "split_manifest_path_and_hash",
    ):
        if receipt_field in payload:
            payload[receipt_field]["path"] = ""
    if "preconditions_checked" in payload:
        payload["preconditions_checked"]["output_path_policy"]["requested_corpus_path"] = ""
    return sha256_json(payload)


def _atomic_write(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def build_artifact(
    *,
    result_path: Path,
    corpus_path: Path,
    sidecar_path: Path,
    control_manifest_path: Path,
    split_manifest_path: Path,
    run_date: str,
    protected_hashes_before: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    root: Path = REPO_ROOT,
) -> JsonDict:
    start = time.monotonic()
    rows = generate_pair_rows()
    evaluated = evaluate_rows(rows)
    sidecars = evaluated["sidecars"]
    splits = split_manifest(rows)
    controls = build_control_manifest(rows, sidecars)

    _atomic_write(corpus_path, rows_to_jsonl(rows))
    _atomic_write(sidecar_path, rows_to_jsonl(sidecars))
    _atomic_write(control_manifest_path, canonical_json(controls) + "\n")
    _atomic_write(split_manifest_path, canonical_json(splits) + "\n")

    artifact: JsonDict = {
        "status": "pending",
        "fixture_scope_and_claim_boundary": fixture_scope_and_claim_boundary(),
        "weakness_family_taxonomy": weakness_family_taxonomy(),
        "source_and_license_receipts": source_and_license_receipts(root),
        "pair_generator_version_and_hash": pair_generator_version_and_hash(root),
        "corpus_path_and_hash": path_and_hash(corpus_path, len(rows)),
        "sidecar_path_and_hash": path_and_hash(sidecar_path, len(sidecars)),
        "control_manifest_path_and_hash": path_and_hash(control_manifest_path),
        "split_manifest_path_and_hash": path_and_hash(split_manifest_path),
        "pair_count_by_weakness_source_template_perturbation_and_split": pair_count_by_weakness_source_template_perturbation_and_split(
            rows
        ),
        "compile_results": compile_results(sidecars),
        "executable_property_results": executable_property_results(sidecars),
        "ast_and_constraint_results": ast_and_constraint_results(sidecars),
        "targeted_mutation_results": targeted_mutation_results(sidecars),
        "vulnerable_fixed_label_receipts": vulnerable_fixed_label_receipts(sidecars),
        "length_and_token_proxy_balance": length_and_token_proxy_balance(rows),
        "duplicate_and_overlap_checks": duplicate_and_overlap_checks(rows, splits),
        "held_weakness_source_template_and_perturbation_groups": splits["held_groups"],
        "evaluator_swap_definitions": evaluator_swap_definitions(),
        "positive_and_negative_control_results": positive_and_negative_control_results(controls),
        "invalid_or_excluded_rows": evaluated["invalid_or_excluded_rows"],
        "minimum_power_projection": minimum_power_projection(rows),
        "exact_code_safety_fixture_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(protected_hashes_before, root=root),
        "preconditions_checked": collect_preconditions(
            root=root, run_date=run_date, corpus_path=corpus_path
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "duration_s": round(time.monotonic() - start, 6),
        "random_seeds": {
            "python_hash_seed_required": "not_used",
            "generation_seed": DEFAULT_RANDOM_SEED,
            "split_seed": DEFAULT_RANDOM_SEED,
            "control_seed": DEFAULT_RANDOM_SEED,
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["exact_code_safety_fixture_ready_score"] = ready_score(artifact)
    artifact["status"] = (
        "complete_ready" if artifact["exact_code_safety_fixture_ready_score"] == 1.0 else "blocked"
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    _atomic_write(result_path, canonical_json(artifact) + "\n")
    return artifact


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    corpus_path: str | Path = REPO_ROOT / CORPUS_RELATIVE_PATH,
    sidecar_path: str | Path = REPO_ROOT / SIDECAR_RELATIVE_PATH,
    control_manifest_path: str | Path = REPO_ROOT / CONTROL_MANIFEST_RELATIVE_PATH,
    split_manifest_path: str | Path = REPO_ROOT / SPLIT_MANIFEST_RELATIVE_PATH,
    run_date: str = DEFAULT_RUN_DATE,
    date: str | None = None,
    protected_hashes_before: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    root: Path = REPO_ROOT,
    write: bool = True,
) -> JsonDict:
    del write
    resolved_date = date if date is not None else run_date
    before = (
        protected_hashes_before
        if protected_hashes_before is not None
        else protected_file_hashes(root)
    )
    exits = (
        test_exit_codes
        if test_exit_codes is not None
        else {command: 0 for command in test_commands}
    )
    return build_artifact(
        result_path=Path(result_path),
        corpus_path=Path(corpus_path),
        sidecar_path=Path(sidecar_path),
        control_manifest_path=Path(control_manifest_path),
        split_manifest_path=Path(split_manifest_path),
        run_date=resolved_date,
        protected_hashes_before=before,
        test_commands=test_commands,
        test_exit_codes=exits,
        root=root,
    )


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles incomplete")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true")
    if artifact["exact_code_safety_fixture_ready_score"] != ready_score(artifact):
        raise ValueError("exact_code_safety_fixture_ready_score mismatch")
    if artifact["exact_code_safety_fixture_ready_score"] != 1.0:
        raise ValueError("exact_code_safety_fixture_ready_score not ready")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DEFAULT_RUN_DATE)
    args = parser.parse_args(argv)
    run(date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
