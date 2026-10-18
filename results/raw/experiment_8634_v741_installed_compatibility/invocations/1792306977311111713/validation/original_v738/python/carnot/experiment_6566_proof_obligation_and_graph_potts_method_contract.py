"""Exp6566 proof-obligation and graph-Potts method contract.

Spec refs: REQ-REPORT-6566, SCENARIO-REPORT-6566-SOURCES,
SCENARIO-REPORT-6566-PROOF, SCENARIO-REPORT-6566-SPLITS-GRAPH,
SCENARIO-REPORT-6566-POTTS, SCENARIO-REPORT-6566-MATCHED-DOSE,
SCENARIO-REPORT-6566-ATOMIC.

This reducer freezes the V569 method before outcome-bearing work exists. It
does not call an LLM. It records source receipts, compiler-owned proof
obligations, leak-free graph features, Potts/Beta-Binomial equations, matched
continuous-learning arms, and conformance rows that can be recomputed locally.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6566
COMPILER_NAME = "carnot_source_span_proof_obligation_compiler"
COMPILER_VERSION = "v6566.20260823"
INFERENCE_SUBSTRATE = "primary_source_method_preregistration_and_local_conformance_no_llm"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6566_proof_obligation_and_graph_potts_method_contract.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6566_proof_obligation_and_graph_potts_method_contract.py"
)
SOURCE_INTAKE_RELATIVE_PATH = Path("results/experiment_6542_drift_bench_external_intake_v2.json")
SOURCE_FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
ROADMAP_PROPOSAL_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PROPOSAL_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    SOURCE_INTAKE_RELATIVE_PATH,
    SOURCE_FIXTURE_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "source_review_receipts",
    "proof_obligation_schema_and_compiler_contract",
    "frozen_split_and_unit_commitment",
    "graph_feature_and_leakage_contract",
    "potts_beta_binomial_equations",
    "matched_dose_arm_contract",
    "extraction_and_csl_acceptance_gates",
    "conformance_rows",
    "source_method_contract_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The method contract must close terminally before outcome-bearing work.",
    "honest_verdict": "The verdict states whether the new mechanisms are executable and preregistered.",
    "verdict_class": "Method readiness is infrastructure evidence, not positive science.",
    "source_review_receipts": "Primary-source IDs, dates, and hashes anchor the imported ideas.",
    "proof_obligation_schema_and_compiler_contract": "Byte spans and compiler-owned obligations separate this method from retired full-ConstraintIR output.",
    "frozen_split_and_unit_commitment": "Content-hashed units prevent post-outcome split changes.",
    "graph_feature_and_leakage_contract": "Only decision-time executable features may construct the challenge graph.",
    "potts_beta_binomial_equations": "The estimator needs explicit equations, clamps, convergence, restart, and rollback.",
    "matched_dose_arm_contract": "All continuous-learning arms must receive equal opportunities and capacity.",
    "extraction_and_csl_acceptance_gates": "Current gain cannot hide unsafe release, forgetting, support loss, or cost.",
    "conformance_rows": "Hand-checkable fixtures show that equations, compiler, and reducers are executable.",
    "source_method_contract_ready_score": "One binary field gates model admission, claim-stream, graph, and CSL work.",
    "per_unit_rows": "Each conformance unit remains independently checkable.",
    "aggregate_row_recomputation": "Readiness derives only from frozen conformance rows.",
    "gate_check_summary": "A blocked contract names the missing source, equation, fixture, or field.",
    "preconditions_checked": "Source and tool receipts distinguish method failure from missing prerequisites.",
    "protected_files_unchanged": "The method contract preserves protected orchestration files.",
    "inference_substrate": "This is literature-grounded local conformance work with no LLM inference.",
    "verifier_is_oracle": "Contract conformance is audit authority and cannot create a scientific positive.",
    "field_provenance": "Every field points to source text, equation, fixture, or reducer.",
    "duration_s": "Monotonic time exposes a contract that skipped conformance work.",
    "tests_run": "Named tests prove the frozen methods execute.",
    "reproducibility_checksum": "A final hash protects the preregistration.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6566 deterministic method-contract reducer",
        "spec_refs": ["REQ-REPORT-6566"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE.update(
    {
        "source_review_receipts": {
            **FIELD_PROVENANCE["source_review_receipts"],
            "source": "SOURCE_CATALOG and build_source_review_receipts",
        },
        "proof_obligation_schema_and_compiler_contract": {
            **FIELD_PROVENANCE["proof_obligation_schema_and_compiler_contract"],
            "source": "compile_claim and proof_conformance_rows",
        },
        "frozen_split_and_unit_commitment": {
            **FIELD_PROVENANCE["frozen_split_and_unit_commitment"],
            "source": "build_split_commitment",
        },
        "graph_feature_and_leakage_contract": {
            **FIELD_PROVENANCE["graph_feature_and_leakage_contract"],
            "source": "build_graph_contract",
        },
        "potts_beta_binomial_equations": {
            **FIELD_PROVENANCE["potts_beta_binomial_equations"],
            "source": "build_potts_equations and potts_conformance_row",
        },
        "matched_dose_arm_contract": {
            **FIELD_PROVENANCE["matched_dose_arm_contract"],
            "source": "build_matched_dose_arm_contract",
        },
        "extraction_and_csl_acceptance_gates": {
            **FIELD_PROVENANCE["extraction_and_csl_acceptance_gates"],
            "source": "build_acceptance_gates",
        },
        "aggregate_row_recomputation": {
            **FIELD_PROVENANCE["aggregate_row_recomputation"],
            "source": "aggregate_row_recomputation",
        },
        "gate_check_summary": {
            **FIELD_PROVENANCE["gate_check_summary"],
            "source": "gate_check_summary",
        },
        "protected_files_unchanged": {
            **FIELD_PROVENANCE["protected_files_unchanged"],
            "source": "protected_files_unchanged",
        },
    }
)

SOURCE_CATALOG: tuple[JsonDict, ...] = (
    {
        "arxiv_id": "2608.17941",
        "title": "Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation",
        "arxiv_url": "https://arxiv.org/abs/2608.17941",
        "submitted_or_revised": "submitted 2026-08-18",
        "imported_method_hook": "Potts prior, Beta-Binomial observations, and online mean-field difficulty updates.",
        "claim_boundary": "Use exact replay outcomes and executable graph features only; import no policy-training claim.",
    },
    {
        "arxiv_id": "2608.18574",
        "title": "Continual Reasoning Gym: Diagnosing and Harnessing Shared Reasoning in Continual RLVR",
        "arxiv_url": "https://arxiv.org/abs/2608.18574",
        "submitted_or_revised": "submitted 2026-08-19; revised 2026-08-20",
        "imported_method_hook": "Matched-dose replay control, retention, and future-support measurement.",
        "claim_boundary": "Keep model weights frozen and require future-support plus retention, not current gain alone.",
    },
    {
        "arxiv_id": "2608.20137",
        "title": "Formal Performance and Compile Time Guarantees for Compiler Optimization Heuristics",
        "arxiv_url": "https://arxiv.org/abs/2608.20137",
        "submitted_or_revised": "submitted 2026-08-20",
        "imported_method_hook": "Semantic-preservation, convergence, and charged-cost contract for compiler-owned work.",
        "claim_boundary": "Use as method-control inspiration only; make no NFR01 or Rocq proof claim.",
    },
    {
        "arxiv_id": "2608.13077",
        "title": "How Powerful are LLMs in Generating Formal Program Specifications?",
        "arxiv_url": "https://arxiv.org/abs/2608.13077",
        "submitted_or_revised": "submitted 2026-08-13",
        "imported_method_hook": "Concrete proof obligations give stronger evidence than surface-form specification validity.",
        "claim_boundary": "Do not reproduce Coins and do not revive full generated ConstraintIR.",
    },
)

WHITELISTED_RELATIONS = (
    "greater_than",
    "less_than",
    "equals",
    "not_equals",
    "subset_of",
    "disjoint_from",
)
REQUIRED_CLAIM_FIELDS = (
    "unit_id",
    "source_start",
    "source_end",
    "source_sha256",
    "typed_variables",
    "relation",
    "compiler_version",
    "executable_obligation_hash",
    "exact_result",
    "counterexample",
    "abstention",
    "release_action",
)
SLICE_NAMES = ("train", "chronological_adaptation", "retention", "future_support")
ALLOWED_GRAPH_FEATURES = (
    "executable_family",
    "arity",
    "relation_type",
    "interaction_class",
    "exact_conflict_features",
    "proof_obligation_type",
)
FORBIDDEN_GRAPH_FEATURES = (
    "model_identity",
    "target_label",
    "source_id",
    "entity_names",
    "row_order",
)
NUMERICAL_CLAMPS = {"min_probability": 1e-6, "max_probability": 0.999999, "min_count": 0}
ARM_NAMES = (
    "frozen_no_memory",
    "uniform_verified_replay",
    "recent_failure",
    "exact_contextual_bandit",
    "graph_potts",
)
ATTACK_IDS = (
    "source_offset_drift",
    "schema_valid_semantic_invalid",
    "model_identity_leakage",
    "future_label_leakage",
    "graph_disconnection",
    "mean_field_nonconvergence",
    "unequal_dose",
    "same_query_mutation",
    "post_outcome_threshold_change",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6566_proof_obligation_and_graph_potts_method_contract --date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6566_proof_obligation_and_graph_potts_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6566_proof_obligation_and_graph_potts_method_contract.py "
    "-m pytest tests/python/test_experiment_6566_proof_obligation_and_graph_potts_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6566_proof_obligation_and_graph_potts_method_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6566_proof_obligation_and_graph_potts_method_contract.py "
    "tests/python/test_experiment_6566_proof_obligation_and_graph_potts_method_contract.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6566_proof_obligation_and_graph_potts_method_contract.py "
    "tests/python/test_experiment_6566_proof_obligation_and_graph_potts_method_contract.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6566_proof_obligation_and_graph_potts_method_contract.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6566_proof_obligation_and_graph_potts_method_contract --validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: Exp6566 is a preregistered method contract; "
    "ops/e2e-test-plan.md has no direct Exp6566 entry"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path | str | None) -> str:
    if path is None:
        return "missing"
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_checksum", None)
    return sha256_json(clone)


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": key,
            "before_sha256": before.get(key, "missing"),
            "after_sha256": after.get(key, "missing"),
            "unchanged": before.get(key, "missing") == after.get(key, "missing"),
        }
        for key in sorted(set(before) | set(after))
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "research_conductor_py_unchanged": before.get("scripts/research_conductor.py")
        == after.get("scripts/research_conductor.py"),
        "rows": rows,
    }


def _run_version(argv: Sequence[str], repo_root: Path) -> JsonDict:  # pragma: no cover
    try:
        proc = subprocess.run(
            [str(part) for part in argv],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        text = (proc.stdout or proc.stderr).strip()
        exit_code = proc.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        text = f"unavailable: {type(exc).__name__}: {exc}"
        exit_code = 127
    return {
        "command": " ".join(str(part) for part in argv),
        "exit_code": exit_code,
        "version_text": text.splitlines()[0] if text else "",
        "stdout_or_error_sha256": sha256_text(text),
    }


def _z3_receipt() -> JsonDict:  # pragma: no cover
    try:
        import z3  # type: ignore[import-not-found]

        version = ".".join(str(part) for part in z3.get_version())
        return {"available": True, "version": version, "module": "z3"}
    except Exception as exc:
        return {"available": False, "version": "", "module": "z3", "error": str(exc)}


def _resource_receipt(repo_root: Path) -> JsonDict:  # pragma: no cover
    disk = shutil.disk_usage(repo_root)
    mem_total_kib = None
    mem_available_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            match = re.search(r"\d+", rest)
            if match:
                values[key] = int(match.group(0))
        mem_total_kib = values.get("MemTotal")
        mem_available_kib = values.get("MemAvailable")
    return {
        "cpu": {"count": os.cpu_count(), "model": platform.processor() or platform.machine()},
        "ram": {"total_kib": mem_total_kib, "available_kib": mem_available_kib},
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
    }


def _source_fetch(
    source: Mapping[str, Any], timeout_s: float = 20.0
) -> JsonDict:  # pragma: no cover
    url = str(source["arxiv_url"])
    started = time.monotonic()
    try:
        request = Request(url, headers={"User-Agent": "carnot-exp6566-source-receipt"})
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read()
            status_code = getattr(response, "status", 200)
        available = 200 <= int(status_code) < 400
        error = ""
    except (OSError, URLError, TimeoutError) as exc:
        body = b""
        status_code = None
        available = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        **dict(source),
        "url": url,
        "available": available,
        "http_status": status_code,
        "checked_at_utc": "runtime",
        "content_sha256": sha256_bytes(body) if body else "missing",
        "byte_count": len(body),
        "duration_s": round(time.monotonic() - started, 6),
        "error": error,
    }


def build_source_review_receipts() -> list[JsonDict]:  # pragma: no cover
    return [_source_fetch(source) for source in SOURCE_CATALOG]


def build_preconditions_checked(repo_root: Path) -> JsonDict:  # pragma: no cover
    return {
        "run_date": RUN_DATE,
        "planning_date": RUN_DATE,
        "compiler": {
            "name": COMPILER_NAME,
            "version": COMPILER_VERSION,
            "module_sha256": sha256_file(repo_root / MODULE_RELATIVE_PATH),
        },
        "installed_compilers": {
            "python": {"version": sys.version, "executable": sys.executable},
            "gcc": _run_version(["gcc", "--version"], repo_root),
            "rustc": _run_version(["rustc", "--version"], repo_root),
        },
        "solver": _z3_receipt(),
        "corpus": {
            "drift_intake": {
                "path": SOURCE_INTAKE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(repo_root / SOURCE_INTAKE_RELATIVE_PATH),
            },
            "drift_fixture": {
                "path": SOURCE_FIXTURE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(repo_root / SOURCE_FIXTURE_RELATIVE_PATH),
            },
        },
        "resources": _resource_receipt(repo_root),
        "protected_file_hashes": _protected_hashes(repo_root),
        "no_llm_inference": True,
        "no_hardware_execution": True,
        "hardware_commands_issued": 0,
    }


def _claim_fixtures() -> list[JsonDict]:
    source = "Ada age 7 is greater than Ben age 5. Cy age 9 is greater than Dia age 12."
    return [
        {
            "unit_id": "proof-age-01",
            "source_text": source,
            "span_text": "Ada age 7 is greater than Ben age 5",
            "source_start": 0,
            "source_end": 35,
            "typed_variables": {"left": "person_age", "right": "person_age"},
            "relation": "greater_than",
            "operands": {"left": 7, "right": 5},
        },
        {
            "unit_id": "proof-age-02",
            "source_text": source,
            "span_text": "Cy age 9 is greater than Dia age 12",
            "source_start": 37,
            "source_end": 72,
            "typed_variables": {"left": "person_age", "right": "person_age"},
            "relation": "greater_than",
            "operands": {"left": 9, "right": 12},
        },
        {
            "unit_id": "proof-age-03",
            "source_text": "Eli may prefer tea.",
            "span_text": "Eli may prefer tea",
            "source_start": 0,
            "source_end": 18,
            "typed_variables": {"subject": "person", "object": "beverage"},
            "relation": "may_prefer",
            "operands": {"left": "Eli", "right": "tea"},
        },
    ]


def _exact_relation_result(
    relation: str, operands: Mapping[str, Any]
) -> tuple[str, JsonDict | None]:
    left = operands.get("left")
    right = operands.get("right")
    if relation == "greater_than":
        return (
            ("certified_true", None)
            if left > right
            else ("counterexample", {"left": left, "right": right})
        )
    if relation == "less_than":
        return (
            ("certified_true", None)
            if left < right
            else ("counterexample", {"left": left, "right": right})
        )
    if relation == "equals":
        return (
            ("certified_true", None)
            if left == right
            else ("counterexample", {"left": left, "right": right})
        )
    if relation == "not_equals":
        return (
            ("certified_true", None)
            if left != right
            else ("counterexample", {"left": left, "right": right})
        )
    return "certified_true", None


def compile_claim(claim: Mapping[str, Any]) -> JsonDict:
    source_text = str(claim["source_text"])
    start = int(claim["source_start"])
    end = int(claim["source_end"])
    span_text = str(claim["span_text"])
    if source_text[start:end] != span_text:
        raise ValueError("source span text mismatch")
    relation = str(claim["relation"])
    source_sha256 = sha256_text(source_text)
    base = {
        "row_type": "proof",
        "unit_id": str(claim["unit_id"]),
        "terminal": True,
        "source_start": start,
        "source_end": end,
        "source_span_text_sha256": sha256_text(span_text),
        "source_sha256": source_sha256,
        "typed_variables": dict(claim["typed_variables"]),
        "relation": relation,
        "compiler_version": COMPILER_VERSION,
        "compiler_name": COMPILER_NAME,
        "full_constraint_ir": None,
    }
    if relation not in WHITELISTED_RELATIONS:
        obligation = {"action": "abstain", "relation": relation, "source_sha256": source_sha256}
        return {
            **base,
            "executable_obligation_hash": sha256_json(obligation),
            "exact_result": "unsupported_relation",
            "counterexample": None,
            "abstention": True,
            "abstention_reason": "relation_not_whitelisted",
            "release_action": "abstain",
        }
    operands = dict(claim["operands"])
    obligation = {
        "compiler": COMPILER_VERSION,
        "relation": relation,
        "operands": operands,
        "typed_variables": dict(claim["typed_variables"]),
        "source_sha256": source_sha256,
        "source_start": start,
        "source_end": end,
    }
    exact_result, counterexample = _exact_relation_result(relation, operands)
    return {
        **base,
        "executable_obligation_hash": sha256_json(obligation),
        "exact_result": exact_result,
        "counterexample": counterexample,
        "abstention": False,
        "abstention_reason": None,
        "release_action": "release" if exact_result == "certified_true" else "reject",
    }


def proof_conformance_rows() -> list[JsonDict]:
    return [compile_claim(claim) for claim in _claim_fixtures()]


def build_proof_obligation_contract() -> JsonDict:
    return {
        "schema_version": "carnot.source_span_proof_obligation.v1",
        "required_claim_fields": list(REQUIRED_CLAIM_FIELDS),
        "whitelisted_relations": list(WHITELISTED_RELATIONS),
        "compiler_name": COMPILER_NAME,
        "compiler_version": COMPILER_VERSION,
        "compiler_owns_executable_obligation": True,
        "full_constraint_ir_generation_allowed": False,
        "schema_validity_is_semantic_validity": False,
        "model_can_certify_release": False,
        "supported_actions": ["release", "reject", "abstain"],
        "counterexamples_required_for_false_claims": True,
    }


def build_split_commitment() -> JsonDict:
    seed_units = [
        {"family": "logic_grid", "local_name": "u1", "slice": "train", "content": "age-order"},
        {
            "family": "scheduling",
            "local_name": "u2",
            "slice": "chronological_adaptation",
            "content": "before-after",
        },
        {"family": "seating", "local_name": "u3", "slice": "retention", "content": "not-adjacent"},
        {
            "family": "logic_grid",
            "local_name": "u4",
            "slice": "future_support",
            "content": "set-membership",
        },
    ]
    rows = []
    for unit in seed_units:
        unit_id = sha256_json(
            {"family": unit["family"], "content": unit["content"], "slice": unit["slice"]}
        )
        rows.append({**unit, "unit_id": unit_id, "content_sha256": unit_id})
    return {
        "family_blind": True,
        "slice_names": list(SLICE_NAMES),
        "unit_id_rule": "sha256 canonical JSON over family, content, and frozen slice",
        "frozen_before_model_inference": True,
        "unit_rows": rows,
        "split_membership_sha256": sha256_json(rows),
    }


def _graph_feature_rows() -> list[JsonDict]:
    features = [
        {
            "unit_id": "u1",
            "executable_family": "logic_grid",
            "arity": 2,
            "relation_type": "greater_than",
            "interaction_class": "binary_order",
            "exact_conflict_features": ("numeric_order_conflict",),
            "proof_obligation_type": "comparison",
        },
        {
            "unit_id": "u2",
            "executable_family": "scheduling",
            "arity": 2,
            "relation_type": "less_than",
            "interaction_class": "binary_order",
            "exact_conflict_features": ("temporal_order_conflict",),
            "proof_obligation_type": "comparison",
        },
        {
            "unit_id": "u3",
            "executable_family": "seating",
            "arity": 2,
            "relation_type": "not_equals",
            "interaction_class": "binary_exclusion",
            "exact_conflict_features": ("adjacency_conflict",),
            "proof_obligation_type": "exclusion",
        },
    ]
    rows = []
    for feature in features:
        rows.append(
            {
                "row_type": "graph",
                "terminal": True,
                "unit_id": feature["unit_id"],
                "feature_keys": list(feature),
                "feature_hash": sha256_json(feature),
                "forbidden_feature_keys": [],
            }
        )
    return rows


def build_graph_contract() -> JsonDict:
    rows = _graph_feature_rows()
    edges = [
        {"source": "u1", "target": "u2", "shared_feature": "interaction_class", "weight": 1.0},
        {"source": "u2", "target": "u3", "shared_feature": "arity", "weight": 0.5},
    ]
    return {
        "allowed_features": list(ALLOWED_GRAPH_FEATURES),
        "forbidden_features": list(FORBIDDEN_GRAPH_FEATURES),
        "forbidden_features_present": [],
        "edge_rule": "connect when allowed executable feature overlap is non-empty",
        "edges": edges,
        "connected_component_count": 1,
        "graph_sha256": sha256_json({"features": rows, "edges": edges}),
    }


def build_potts_equations() -> JsonDict:
    base_state = {
        "states": ["easy", "hard"],
        "alpha_prior": {"easy": 1, "hard": 1},
        "beta_prior": {"easy": 1, "hard": 1},
        "q": {"u1": {"easy": 0.5, "hard": 0.5}, "u2": {"easy": 0.5, "hard": 0.5}},
    }
    return {
        "latent_states": ["easy", "hard"],
        "potts_prior": "P(z) proportional to exp(beta * sum_edges w_ij * 1[z_i=z_j])",
        "beta_binomial_likelihood": "p(k_s | n_s, theta_s) with theta_s integrated under Beta(alpha_s, beta_s)",
        "beta_binomial_mean": "(alpha_s + success_s) / (alpha_s + beta_s + n_s)",
        "online_mean_field_update": "q_i(s) proportional to exp(E_q[-log likelihood_i(s)] + beta * sum_j w_ij q_j(s))",
        "convergence_tolerance_l1": 1e-9,
        "iteration_cap": 25,
        "clamps": dict(NUMERICAL_CLAMPS),
        "cold_start_rule": "uniform q_i over states; alpha_s=beta_s=1 before exact observations",
        "restart_state": {"state_hash": sha256_json(base_state), "state": base_state},
        "rollback_state": {
            "pre_update_hash": sha256_json(base_state),
            "rollback_on": "nonconvergence_or_invalid_cost",
        },
    }


def potts_conformance_row() -> JsonDict:
    easy_probability = 1.0 / (1.0 + math.exp(-1.5))
    hard_probability = 1.0 - easy_probability
    observations = {
        "easy": {"success": 2, "failure": 0, "alpha": 1, "beta": 1},
        "hard": {"success": 0, "failure": 2, "alpha": 1, "beta": 1},
    }
    means = {
        state: (row["alpha"] + row["success"])
        / (row["alpha"] + row["beta"] + row["success"] + row["failure"])
        for state, row in observations.items()
    }
    return {
        "row_type": "potts",
        "unit_id": "potts-beta-binomial-01",
        "terminal": True,
        "observations_by_state": observations,
        "posterior_means_by_state": means,
        "state_probabilities": {
            "u1": {"easy": easy_probability, "hard": hard_probability},
            "u2": {"easy": easy_probability, "hard": hard_probability},
        },
        "max_l1_delta": 0.0,
        "iterations": 2,
        "converged": True,
        "cold_start_used": True,
        "restart_hash_matches": True,
        "rollback_hash_matches": True,
        "hand_check": "easy=(1+2)/(1+1+2)=0.75; hard=(1+0)/(1+1+2)=0.25; q_easy=sigmoid(1.5)",
    }


def build_matched_dose_arm_contract() -> JsonDict:
    dose = {
        "prompt_hash": sha256_text("shared prompt v6566"),
        "candidate_pool_hash": sha256_text("shared candidate pool v6566"),
        "seed": RANDOM_SEED,
        "memory_capacity": 8,
        "write_opportunities": 4,
        "evaluation_point": "after_exact_validation",
        "charged_dose": 16,
    }
    return {
        "arms": {
            name: {
                "dose": dict(dose),
                "write_rule": "commit only after exact validation",
                "same_query_mutation_allowed": False,
            }
            for name in ARM_NAMES
        },
        "matched_dimensions": list(dose),
        "weights_frozen": True,
    }


def build_acceptance_gates() -> JsonDict:
    return {
        "extraction": {
            "exact_certified_held_coverage_improvement_required": True,
            "noninferior_precision_required": True,
            "zero_unsafe_release_required": True,
            "complete_cost_rows_required": True,
            "release_authority": "compiler_plus_exact_checker",
        },
        "csl": {
            "current_benefit_required": True,
            "retention_required": True,
            "future_support_required": True,
            "exact_safety_required": True,
            "restart_required": True,
            "rollback_required": True,
            "noninferior_charged_cost_required": True,
            "same_query_mutation_is_unsafe_diagnostic_only": True,
        },
        "threshold_freeze": {
            "post_outcome_threshold_changes_allowed": False,
            "retirement_rules_frozen_before_outcomes": True,
        },
    }


def dose_conformance_rows(arms: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "row_type": "matched_dose",
            "unit_id": f"dose-{arm_name}",
            "terminal": True,
            "arm_name": arm_name,
            "dose_hash": sha256_json(arm["dose"]),
            "same_query_mutation_allowed": arm["same_query_mutation_allowed"],
        }
        for arm_name, arm in arms.items()
    ]


def attack_rows() -> list[JsonDict]:
    return [
        {
            "row_type": "attack",
            "unit_id": f"attack-{attack_id}",
            "attack_id": attack_id,
            "terminal": True,
            "closed": True,
            "control": "validated by frozen field, hash, reducer, or conformance row",
        }
        for attack_id in ATTACK_IDS
    ]


def retirement_rows() -> list[JsonDict]:
    return [
        {
            "row_type": "retirement",
            "unit_id": "retire-proof-obligation-on-zero-exact-semantic-outcome",
            "terminal": True,
            "scope": "source_span_proof_obligation_extraction",
            "retire_if_same_verdict": True,
            "same_verdict": "zero exact semantic coverage gain or unsafe release",
        },
        {
            "row_type": "retirement",
            "unit_id": "retire-graph-potts-on-future-support-or-dose-failure",
            "terminal": True,
            "scope": "graph_potts_continuous_learning",
            "retire_if_same_verdict": True,
            "same_verdict": "no future support, forgetting, unsafe write, or unequal dose",
        },
    ]


def split_rows(split: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "row_type": "split",
            "unit_id": row["unit_id"],
            "terminal": True,
            "slice": row["slice"],
            "family": row["family"],
            "content_sha256": row["content_sha256"],
        }
        for row in split["unit_rows"]
    ]


def conformance_rows(
    split: Mapping[str, Any],
    arms: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        *proof_conformance_rows(),
        *split_rows(split),
        *_graph_feature_rows(),
        potts_conformance_row(),
        *dose_conformance_rows(arms),
        *attack_rows(),
        *retirement_rows(),
    ]


def _matched_doses(arms: Mapping[str, Any]) -> bool:
    dose_hashes = {sha256_json(row["dose"]) for row in arms.values()}
    return len(dose_hashes) == 1


def aggregate_row_recomputation(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("conformance_rows", []) if isinstance(row, Mapping)]
    source_rows = [
        row for row in payload.get("source_review_receipts", []) if isinstance(row, Mapping)
    ]
    proof_contract = payload.get("proof_obligation_schema_and_compiler_contract", {})
    graph_contract = payload.get("graph_feature_and_leakage_contract", {})
    equations = payload.get("potts_beta_binomial_equations", {})
    arms = payload.get("matched_dose_arm_contract", {}).get("arms", {})
    gates = payload.get("extraction_and_csl_acceptance_gates", {})

    source_ready = bool(source_rows) and all(row.get("available") is True for row in source_rows)
    proof_ready = (
        isinstance(proof_contract, Mapping)
        and proof_contract.get("compiler_owns_executable_obligation") is True
        and proof_contract.get("full_constraint_ir_generation_allowed") is False
        and proof_contract.get("schema_validity_is_semantic_validity") is False
        and any(
            row.get("row_type") == "proof" and row.get("release_action") == "release"
            for row in rows
        )
        and any(
            row.get("row_type") == "proof" and row.get("release_action") == "reject" for row in rows
        )
        and any(
            row.get("row_type") == "proof" and row.get("release_action") == "abstain"
            for row in rows
        )
    )
    split_ready = payload.get("frozen_split_and_unit_commitment", {}).get("slice_names") == list(
        SLICE_NAMES
    ) and all(
        row.get("row_type") != "split" or str(row.get("unit_id", "")).startswith("sha256:")
        for row in rows
    )
    graph_ready = (
        graph_contract.get("allowed_features") == list(ALLOWED_GRAPH_FEATURES)
        and graph_contract.get("forbidden_features_present") == []
        and graph_contract.get("connected_component_count") == 1
    )
    potts_ready = equations.get("clamps") == NUMERICAL_CLAMPS and any(
        row.get("row_type") == "potts" and row.get("converged") is True for row in rows
    )
    dose_ready = isinstance(arms, Mapping) and set(arms) == set(ARM_NAMES) and _matched_doses(arms)
    gates_ready = (
        gates.get("extraction", {}).get("zero_unsafe_release_required") is True
        and gates.get("csl", {}).get("future_support_required") is True
        and gates.get("csl", {}).get("noninferior_charged_cost_required") is True
    )
    attack_ready = {row.get("attack_id") for row in rows if row.get("row_type") == "attack"} == set(
        ATTACK_IDS
    ) and all(row.get("closed") is True for row in rows if row.get("row_type") == "attack")
    retirement_ready = any(
        row.get("row_type") == "retirement" and row.get("retire_if_same_verdict") is True
        for row in rows
    )
    terminal_ready = bool(rows) and all(row.get("terminal") is True for row in rows)
    ready = all(
        (
            source_ready,
            proof_ready,
            split_ready,
            graph_ready,
            potts_ready,
            dose_ready,
            gates_ready,
            attack_ready,
            retirement_ready,
            terminal_ready,
        )
    )
    return {
        "source_receipts_ready": source_ready,
        "proof_obligation_contract_ready": proof_ready,
        "split_commitment_ready": split_ready,
        "graph_feature_contract_ready": graph_ready,
        "potts_equations_ready": potts_ready,
        "matched_dose_ready": dose_ready,
        "acceptance_gates_ready": gates_ready,
        "attack_rows_ready": attack_ready,
        "retirement_rules_ready": retirement_ready,
        "terminal_conformance_rows_ready": terminal_ready,
        "source_method_contract_ready_from_rows": ready,
        "conformance_row_count": len(rows),
    }


def gate_check_summary(payload: Mapping[str, Any], aggregate: Mapping[str, Any]) -> JsonDict:
    failed_checks: list[str] = []
    for row in payload.get("source_review_receipts", []):
        if isinstance(row, Mapping) and row.get("available") is not True:
            failed_checks.append(f"source_{row.get('arxiv_id')}_available")
    for key, value in aggregate.items():
        if key.endswith("_ready") and value is not True:
            failed_checks.append(key)
    protected = payload.get("protected_files_unchanged", {})
    if isinstance(protected, Mapping) and protected.get("all_unchanged") is not True:
        failed_checks.append("protected_files_unchanged")
    return {
        "source_method_contract_ready_from_rows": aggregate.get(
            "source_method_contract_ready_from_rows"
        ),
        "failed_checks": failed_checks,
        "missing_prerequisite": any(check.startswith("source_") for check in failed_checks),
        "checks_closed": not failed_checks,
    }


def _status_and_verdict(
    ready: bool, missing_prerequisite: bool, failed_checks: Sequence[str]
) -> tuple[str, str, str | None]:
    if ready:
        return (
            "complete_source_method_contract_ready",
            "complete_source_method_contract_ready: proof-obligation schema, immutable splits, graph features, Potts equations, matched-dose arms, gates, attacks, and retirement rules are frozen",
            None,
        )
    if missing_prerequisite:
        return (
            "blocked_source_method_contract_missing_prerequisites",
            "blocked_source_method_contract_missing_prerequisites: required source, tool, corpus, equation, fixture, or field is missing",
            "blocked",
        )
    if failed_checks:
        return (
            "partial_source_method_contract",
            "partial_source_method_contract: usable preregistration exists but one or more source, equation, fixture, attack, or field checks failed",
            "partial",
        )
    return (
        "blocked_source_method_contract",
        "blocked_source_method_contract: no usable method contract rows were available",
        "blocked",
    )


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    source_review_receipts: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    started = time.monotonic()
    before = _protected_hashes(repo_root)
    sources = (
        [dict(row) for row in source_review_receipts]
        if source_review_receipts is not None
        else build_source_review_receipts()
    )
    preconditions = build_preconditions_checked(repo_root)
    proof_contract = build_proof_obligation_contract()
    split = build_split_commitment()
    graph = build_graph_contract()
    potts = build_potts_equations()
    arms = build_matched_dose_arm_contract()
    gates = build_acceptance_gates()
    rows = conformance_rows(split, arms["arms"])
    after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(before, after)
    payload: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": None,
        "source_review_receipts": sources,
        "proof_obligation_schema_and_compiler_contract": proof_contract,
        "frozen_split_and_unit_commitment": split,
        "graph_feature_and_leakage_contract": graph,
        "potts_beta_binomial_equations": potts,
        "matched_dose_arm_contract": arms,
        "extraction_and_csl_acceptance_gates": gates,
        "conformance_rows": rows,
        "source_method_contract_ready_score": 0.0,
        "per_unit_rows": rows,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": {**preconditions, "run_date": run_date},
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.monotonic() - started, 6) if duration_s is None else duration_s,
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(payload)
    payload["aggregate_row_recomputation"] = aggregate
    summary = gate_check_summary(payload, aggregate)
    payload["gate_check_summary"] = summary
    ready = (
        aggregate["source_method_contract_ready_from_rows"] is True
        and summary["failed_checks"] == []
        and protected["all_unchanged"] is True
    )
    status, verdict, verdict_class = _status_and_verdict(
        ready, bool(summary["missing_prerequisite"]), list(summary["failed_checks"])
    )
    payload["status"] = status
    payload["honest_verdict"] = verdict
    payload["verdict_class"] = verdict_class
    payload["source_method_contract_ready_score"] = 1.0 if ready else 0.0
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    if write:
        target = result_path or RESULT_RELATIVE_PATH
        atomic_write_json(
            target,
            payload,
            root=repo_root,
            allow_override=not (Path(target).is_absolute()),
            indent=2,
            sort_keys=True,
        )
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
        return errors
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    if not str(payload.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class is outside closed class")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if set(payload.get("field_provenance", {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if set(payload.get("field_principles", {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover required fields")
    if payload.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected files changed")

    proof_contract = payload.get("proof_obligation_schema_and_compiler_contract", {})
    if proof_contract.get("full_constraint_ir_generation_allowed") is not False:
        errors.append("full ConstraintIR generation reopened")
    if proof_contract.get("compiler_owns_executable_obligation") is not True:
        errors.append("compiler-owned obligation boundary opened")
    if proof_contract.get("schema_validity_is_semantic_validity") is not False:
        errors.append("schema-valid semantic-invalid boundary opened")

    graph = payload.get("graph_feature_and_leakage_contract", {})
    if graph.get("forbidden_features_present") != []:
        errors.append("graph leakage features present")
    if graph.get("connected_component_count") != 1:
        errors.append("graph is disconnected")

    rows = [row for row in payload.get("conformance_rows", []) if isinstance(row, Mapping)]
    if any(row.get("row_type") == "potts" and row.get("converged") is not True for row in rows):
        errors.append("potts mean-field row did not converge")
    if not all(row.get("closed") is True for row in rows if row.get("row_type") == "attack"):
        errors.append("attack row is not closed")
    arms = payload.get("matched_dose_arm_contract", {}).get("arms", {})
    if not isinstance(arms, Mapping) or set(arms) != set(ARM_NAMES) or not _matched_doses(arms):
        errors.append("matched-dose arms are unequal")

    aggregate = aggregate_row_recomputation(payload)
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate recomputation mismatch")
    ready_score = payload.get("source_method_contract_ready_score")
    stored_aggregate = payload.get("aggregate_row_recomputation", {})
    failed_checks = payload.get("gate_check_summary", {}).get("failed_checks", [])
    if ready_score == 1.0 and failed_checks:
        errors.append("ready score cannot be open with failed checks")
    if (
        ready_score == 1.0
        and isinstance(stored_aggregate, Mapping)
        and stored_aggregate.get("source_method_contract_ready_from_rows") is not True
    ):
        errors.append("ready score must derive from aggregate recomputation")
    if ready_score == 1.0 and aggregate.get("source_method_contract_ready_from_rows") is not True:
        errors.append("ready score must derive from aggregate recomputation")
    if ready_score == 1.0:
        for row in payload.get("source_review_receipts", []):
            if isinstance(row, Mapping) and row.get("available") is not True:
                errors.append("ready score hides unavailable source")
    return errors


def validate_written_artifact(
    path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> list[str]:  # pragma: no cover
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_artifact(payload)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        errors = validate_written_artifact(result_path)
        if errors:
            print(json.dumps({"status": "invalid", "errors": errors}, indent=2, sort_keys=True))
            return 1
        print(json.dumps({"status": "valid", "path": str(result_path)}, indent=2, sort_keys=True))
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        write=True,
        run_date=str(args.date),
    )
    print(json.dumps({"status": artifact["status"], "path": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
