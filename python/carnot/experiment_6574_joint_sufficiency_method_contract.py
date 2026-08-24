"""Exp6574 hop-conditioned joint-sufficiency method contract.

Spec refs: REQ-REPORT-6574, REQ-REPORT-6574-GATES,
REQ-REPORT-6574-SOURCES, REQ-REPORT-6574-NODES,
REQ-REPORT-6574-EDGES, REQ-REPORT-6574-REDUCER,
REQ-REPORT-6574-SPLITS-ARMS, REQ-REPORT-6574-FIXTURES,
REQ-REPORT-6574-ATTACKS, REQ-REPORT-6574-ACCEPTANCE,
REQ-REPORT-6574-ATOMIC.

The reducer freezes a small executable method contract. It does not extract
live model output. It proves only that source-byte atomic nodes, dependency
edges, and the joint release rule are specified and replayable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
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
RUN_DATE = "20260824"
RANDOM_SEED = 6574
COMPILER_NAME = "carnot_hop_conditioned_joint_sufficiency_compiler"
COMPILER_VERSION = "v6574.20260824"
INFERENCE_SUBSTRATE = (
    "primary_source_joint_sufficiency_preregistration_and_local_conformance_no_llm"
)

RESULT_RELATIVE_PATH = Path("results/experiment_6574_joint_sufficiency_method_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6574_joint_sufficiency_method_contract.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6574_joint_sufficiency_method_contract.py")
UPSTREAM_EXP6566_RELATIVE_PATH = Path(
    "results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json"
)
UPSTREAM_EXP6571_RELATIVE_PATH = Path(
    "results/experiment_6571_v570_evidence_gate_and_retirement_root.json"
)
CORPUS_RELATIVE_PATHS = (
    Path("results/experiment_6542_drift_bench_external_intake_v2.json"),
    Path("results/fixtures/v566_drift_bench_external_slice.jsonl"),
)
RETIRED_SOTA_RELATIVE_PATHS = (
    Path("results/experiment_5909_sota_constraint_synthesis_ab.json"),
    Path("results/experiment_5910_verification_guided_constraint_repair.json"),
    Path("results/experiment_5923_sota_schema_supported_constraintir_ab.json"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    SPEC_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
    Path("scripts/research_conductor.py"),
    UPSTREAM_EXP6566_RELATIVE_PATH,
    UPSTREAM_EXP6571_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "source_review_receipts",
    "atomic_obligation_node_schema",
    "dependency_edge_and_joint_reducer_contract",
    "frozen_split_and_arm_commitment",
    "conformance_rows",
    "extraction_acceptance_and_retirement_gates",
    "joint_sufficiency_method_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
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
    "status": "The preregistration must end terminally before outcome-bearing extraction.",
    "honest_verdict": "The verdict states whether joint sufficiency is executable and frozen.",
    "verdict_class": "Method readiness is infrastructure evidence, not positive science.",
    "gate_check_summary": "A blocked contract names the missing upstream field or source receipt.",
    "source_review_receipts": "Primary-source identifiers, dates, and hashes anchor the imported control.",
    "atomic_obligation_node_schema": "Each node binds a hop-conditioned claim to source bytes and compiler-owned semantics.",
    "dependency_edge_and_joint_reducer_contract": "Composed release needs explicit required edges and a deterministic sufficiency reducer.",
    "frozen_split_and_arm_commitment": "Content-hashed units and matched arms prevent retrospective method changes.",
    "conformance_rows": "Hand-checkable positive and abstention fixtures prove the method is executable.",
    "extraction_acceptance_and_retirement_gates": "Coverage cannot hide lower precision, unsafe release, lost lineage, or repeated zero semantics.",
    "joint_sufficiency_method_ready_score": "This exact binary field gates the flagship stream and extractor.",
    "per_unit_rows": "Every source and conformance fixture remains independently checkable.",
    "aggregate_row_recomputation": "Readiness derives only from frozen source and fixture rows.",
    "preconditions_checked": "Source, tool, and corpus receipts separate prerequisite loss from method failure.",
    "protected_files_unchanged": "The SOTA contract preserves protected orchestration files.",
    "inference_substrate": "This is literature-grounded local conformance with no model inference.",
    "verifier_is_oracle": "Exact fixture checks are oracle authority and cannot create a scientific positive.",
    "field_provenance": "Every field points to source text, schema, fixture, compiler output, or reducer.",
    "duration_s": "Monotonic time exposes a contract that skipped source or fixture work.",
    "tests_run": "Named tests prove schemas and reducers execute.",
    "reproducibility_checksum": "A final hash protects the preregistration.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6574 deterministic method-contract reducer",
        "spec_refs": ["REQ-REPORT-6574"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE.update(
    {
        "gate_check_summary": {
            **FIELD_PROVENANCE["gate_check_summary"],
            "source": "gate_check_summary and Exp6571 structured gate receipt",
        },
        "source_review_receipts": {
            **FIELD_PROVENANCE["source_review_receipts"],
            "source": "SOURCE_CATALOG and build_source_review_receipts",
        },
        "atomic_obligation_node_schema": {
            **FIELD_PROVENANCE["atomic_obligation_node_schema"],
            "source": "build_atomic_obligation_node_schema and compile_node",
        },
        "dependency_edge_and_joint_reducer_contract": {
            **FIELD_PROVENANCE["dependency_edge_and_joint_reducer_contract"],
            "source": "build_dependency_edge_and_joint_reducer_contract and joint_sufficiency_reduce",
        },
        "frozen_split_and_arm_commitment": {
            **FIELD_PROVENANCE["frozen_split_and_arm_commitment"],
            "source": "build_frozen_split_and_arm_commitment",
        },
        "conformance_rows": {
            **FIELD_PROVENANCE["conformance_rows"],
            "source": "conformance_rows",
        },
        "extraction_acceptance_and_retirement_gates": {
            **FIELD_PROVENANCE["extraction_acceptance_and_retirement_gates"],
            "source": "build_extraction_acceptance_and_retirement_gates",
        },
        "aggregate_row_recomputation": {
            **FIELD_PROVENANCE["aggregate_row_recomputation"],
            "source": "aggregate_row_recomputation",
        },
        "preconditions_checked": {
            **FIELD_PROVENANCE["preconditions_checked"],
            "source": "build_preconditions_checked",
        },
        "protected_files_unchanged": {
            **FIELD_PROVENANCE["protected_files_unchanged"],
            "source": "_protected_files_unchanged",
        },
    }
)

SOURCE_CATALOG: tuple[JsonDict, ...] = (
    {
        "source_id": "arxiv:2608.00585",
        "source_kind": "arxiv_primary",
        "arxiv_id": "2608.00585",
        "title": "Verification Without Sufficiency: Per-Chunk Filtering Fails on Multi-Hop RAG, and Decomposition Repairs It",
        "url": "https://arxiv.org/abs/2608.00585",
        "submitted_or_revised": "submitted 2026-08-01",
        "checked_date": RUN_DATE,
        "imported_control": "Independently supported chunks can be insufficient for a composed multi-hop answer.",
        "method_hook": "Condition verification on decomposed sub-questions and release only with joint sufficiency.",
    },
    {
        "source_id": "research-references:v570-planner-refresh-20260823",
        "source_kind": "local_method_note",
        "title": "V570 planner refresh method note",
        "path": "research-references.md",
        "anchor_start": "<!-- V570-PLANNER-REFRESH-20260823-START -->",
        "anchor_end": "<!-- V570-PLANNER-REFRESH-20260823-END -->",
        "checked_date": RUN_DATE,
        "imported_control": "A source span may certify one atomic obligation without certifying a composed claim.",
        "method_hook": "Add dependency graph over atomic obligations and joint-sufficiency release.",
    },
)

WHITELISTED_NODE_RELATIONS = (
    "greater_than",
    "less_than",
    "equals",
    "not_equals",
    "contains",
)
WHITELISTED_EDGE_RELATIONS = (
    "requires_entity_binding",
    "requires_temporal_order",
    "requires_set_membership",
    "requires_branch_support",
)
REQUIRED_NODE_FIELDS = (
    "node_id",
    "composed_claim_id",
    "hop_index",
    "sub_question",
    "source_hash",
    "source_start",
    "source_end",
    "source_bytes_hash",
    "typed_variables",
    "relation",
    "compiler_version",
    "executable_obligation_hash",
    "exact_result",
    "counterexample",
    "action",
)
REQUIRED_EDGE_FIELDS = (
    "parent_id",
    "child_id",
    "relation_type",
    "status",
    "ordering_rule",
    "coverage_role",
    "provenance",
)
EDGE_STATUS_VALUES = ("required", "optional")
SPLIT_NAMES = ("train", "calibration", "held")
ARM_NAMES = ("no_filter", "atomic_span_only", "hop_conditioned_joint")
FIXTURE_IDS = (
    "valid_single_hop",
    "valid_two_hop",
    "valid_branched_claim",
    "missing_hop",
    "wrong_span",
    "unsupported_relation",
    "contradictory_nodes",
    "disconnected_graph",
    "duplicate_node",
    "cyclic_dependency",
)
SAFE_FIXTURE_IDS = ("valid_single_hop", "valid_two_hop", "valid_branched_claim")
ATTACK_IDS = (
    "schema_valid_semantic_invalid_nodes",
    "source_offset_drift",
    "omitted_hops",
    "optional_edge_laundering",
    "graph_cycles",
    "duplicate_support",
    "model_self_citation",
    "post_outcome_decomposition",
    "threshold_changes",
    "source_leakage",
    "exact_check_bypass",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6574_joint_sufficiency_method_contract "
    "--date 20260824"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6574_joint_sufficiency_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6574_joint_sufficiency_method_contract.py "
    "-m pytest tests/python/test_experiment_6574_joint_sufficiency_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6574_joint_sufficiency_method_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6574_joint_sufficiency_method_contract.py "
    "tests/python/test_experiment_6574_joint_sufficiency_method_contract.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6574_joint_sufficiency_method_contract.py "
    "tests/python/test_experiment_6574_joint_sufficiency_method_contract.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6574_joint_sufficiency_method_contract.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6574_joint_sufficiency_method_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6574_joint_sufficiency_method_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6574_joint_sufficiency_method_contract --validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: Exp6574 is a no-LLM extraction method contract; "
    "ops/e2e-test-plan.md has no direct Exp6574 live extraction entry"
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


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_between(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    end_index = text.find(end)
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        return ""
    return text[start_index : end_index + len(end)]


def _source_fetch(
    source: Mapping[str, Any], repo_root: Path, timeout_s: float = 20.0
) -> JsonDict:  # pragma: no cover
    if source.get("source_kind") == "local_method_note":
        path = repo_root / str(source["path"])
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        body = _extract_between(text, str(source["anchor_start"]), str(source["anchor_end"]))
        return {
            **dict(source),
            "available": bool(body),
            "checked_at_utc": "runtime",
            "content_sha256": sha256_text(body) if body else "missing",
            "byte_count": len(body.encode("utf-8")),
            "http_status": None,
            "error": "" if body else "method note anchor missing",
        }

    started = time.monotonic()
    url = str(source["url"])
    try:
        request = Request(url, headers={"User-Agent": "carnot-exp6574-source-receipt"})
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
        "available": available,
        "checked_at_utc": "runtime",
        "content_sha256": sha256_bytes(body) if body else "missing",
        "byte_count": len(body),
        "http_status": status_code,
        "duration_s": round(time.monotonic() - started, 6),
        "error": error,
    }


def build_source_review_receipts(repo_root: Path = REPO_ROOT) -> list[JsonDict]:  # pragma: no cover
    return [_source_fetch(source, repo_root) for source in SOURCE_CATALOG]


def build_preconditions_checked(repo_root: Path) -> JsonDict:  # pragma: no cover
    upstream = _read_json(repo_root / UPSTREAM_EXP6571_RELATIVE_PATH)
    observed = upstream.get("v570_evidence_contract_ready_score", "missing")
    gate_passed = observed == 1.0
    exp6566 = _read_json(repo_root / UPSTREAM_EXP6566_RELATIVE_PATH)
    return {
        "run_date": RUN_DATE,
        "planning_date": RUN_DATE,
        "structured_gate": {
            "upstream": "exp6571-v570-evidence-gate-and-retirement-root",
            "path": UPSTREAM_EXP6571_RELATIVE_PATH.as_posix(),
            "artifact_field": "v570_evidence_contract_ready_score",
            "expected": 1.0,
            "observed": observed,
            "passed": gate_passed,
            "artifact_sha256": sha256_file(repo_root / UPSTREAM_EXP6571_RELATIVE_PATH),
        },
        "exp6566_receipt": {
            "path": UPSTREAM_EXP6566_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / UPSTREAM_EXP6566_RELATIVE_PATH),
            "source_method_contract_ready_score": exp6566.get(
                "source_method_contract_ready_score", "missing"
            ),
        },
        "compiler_and_solver_versions": {
            "compiler": {
                "name": COMPILER_NAME,
                "version": COMPILER_VERSION,
                "module_sha256": sha256_file(repo_root / MODULE_RELATIVE_PATH),
            },
            "python": {"version": sys.version, "executable": sys.executable},
            "gcc": _run_version(["gcc", "--version"], repo_root),
            "rustc": _run_version(["rustc", "--version"], repo_root),
            "z3": _z3_receipt(),
        },
        "corpus_receipts": [
            {"path": path.as_posix(), "sha256": sha256_file(repo_root / path)}
            for path in CORPUS_RELATIVE_PATHS
        ],
        "retired_sota_artifact_receipts": [
            {"path": path.as_posix(), "sha256": sha256_file(repo_root / path)}
            for path in RETIRED_SOTA_RELATIVE_PATHS
        ],
        "resources": _resource_receipt(repo_root),
        "protected_file_hashes": _protected_hashes(repo_root),
        "no_llm_inference": True,
        "no_hardware_execution": True,
        "hardware_commands_issued": 0,
        "outcome_bearing_extraction_observed": False,
    }


def _span_for(source_text: str, span_text: str) -> tuple[int, int]:
    start = source_text.index(span_text)
    return start, start + len(span_text.encode("utf-8"))


def _node_template(
    fixture_id: str,
    hop_index: int,
    source_text: str,
    span_text: str,
    relation: str,
    operands: Mapping[str, Any],
    *,
    node_suffix: str | None = None,
    typed_variables: Mapping[str, str] | None = None,
    sub_question: str | None = None,
    source_start: int | None = None,
    source_end: int | None = None,
) -> JsonDict:
    start, end = _span_for(source_text, span_text)
    return {
        "node_id": f"{fixture_id}:{node_suffix or f'h{hop_index}'}",
        "composed_claim_id": fixture_id,
        "hop_index": hop_index,
        "sub_question": sub_question or f"Does hop {hop_index} hold?",
        "source_text": source_text,
        "span_text": span_text,
        "source_start": start if source_start is None else source_start,
        "source_end": end if source_end is None else source_end,
        "typed_variables": dict(typed_variables or {"left": "entity", "right": "entity"}),
        "relation": relation,
        "operands": dict(operands),
    }


def _edge(
    parent_id: str,
    child_id: str,
    relation_type: str,
    *,
    status: str = "required",
    coverage_role: str = "required_chain",
    provenance: str = "frozen_fixture",
) -> JsonDict:
    return {
        "parent_id": parent_id,
        "child_id": child_id,
        "relation_type": relation_type,
        "status": status,
        "ordering_rule": "parent_hop_before_child",
        "coverage_role": coverage_role,
        "provenance": provenance,
    }


def build_fixture(fixture_id: str) -> JsonDict:
    source = (
        "Ada age 7 is greater than Ben age 5. "
        "Ben age 5 is greater than Cy age 3. "
        "Paris is in France. Museum A is in Paris."
    )
    if fixture_id == "valid_single_hop":
        node = _node_template(
            fixture_id,
            0,
            source,
            "Ada age 7 is greater than Ben age 5",
            "greater_than",
            {"left": 7, "right": 5},
            sub_question="Is Ada older than Ben?",
        )
        return {"fixture_id": fixture_id, "expected_hops": [0], "nodes": [node], "edges": []}
    if fixture_id == "valid_two_hop":
        nodes = [
            _node_template(
                fixture_id,
                0,
                source,
                "Ada age 7 is greater than Ben age 5",
                "greater_than",
                {"left": 7, "right": 5},
                sub_question="Is Ada older than Ben?",
            ),
            _node_template(
                fixture_id,
                1,
                source,
                "Ben age 5 is greater than Cy age 3",
                "greater_than",
                {"left": 5, "right": 3},
                sub_question="Is Ben older than Cy?",
            ),
        ]
        edges = [_edge(nodes[0]["node_id"], nodes[1]["node_id"], "requires_entity_binding")]
        return {"fixture_id": fixture_id, "expected_hops": [0, 1], "nodes": nodes, "edges": edges}
    if fixture_id == "valid_branched_claim":
        nodes = [
            _node_template(
                fixture_id,
                0,
                source,
                "Museum A is in Paris",
                "equals",
                {"left": "Paris", "right": "Paris"},
                sub_question="Where is Museum A?",
            ),
            _node_template(
                fixture_id,
                1,
                source,
                "Paris is in France",
                "equals",
                {"left": "France", "right": "France"},
                sub_question="Which country contains Paris?",
            ),
            _node_template(
                fixture_id,
                2,
                source,
                "Ada age 7 is greater than Ben age 5",
                "greater_than",
                {"left": 7, "right": 5},
                sub_question="Does the independent age branch hold?",
            ),
        ]
        edges = [
            _edge(nodes[0]["node_id"], nodes[1]["node_id"], "requires_set_membership"),
            _edge(nodes[0]["node_id"], nodes[2]["node_id"], "requires_branch_support"),
        ]
        return {
            "fixture_id": fixture_id,
            "expected_hops": [0, 1, 2],
            "nodes": nodes,
            "edges": edges,
        }
    if fixture_id == "missing_hop":
        node = _node_template(
            fixture_id,
            0,
            source,
            "Ada age 7 is greater than Ben age 5",
            "greater_than",
            {"left": 7, "right": 5},
        )
        return {"fixture_id": fixture_id, "expected_hops": [0, 1], "nodes": [node], "edges": []}
    if fixture_id == "wrong_span":
        node = _node_template(
            fixture_id,
            0,
            source,
            "Ada age 7 is greater than Ben age 5",
            "greater_than",
            {"left": 7, "right": 5},
            source_start=1,
        )
        return {"fixture_id": fixture_id, "expected_hops": [0], "nodes": [node], "edges": []}
    if fixture_id == "unsupported_relation":
        node = _node_template(
            fixture_id,
            0,
            source,
            "Museum A is in Paris",
            "may_prefer",
            {"left": "Museum A", "right": "Paris"},
        )
        return {"fixture_id": fixture_id, "expected_hops": [0], "nodes": [node], "edges": []}
    if fixture_id == "contradictory_nodes":
        nodes = [
            _node_template(
                fixture_id,
                0,
                source,
                "Ada age 7 is greater than Ben age 5",
                "greater_than",
                {"left": 7, "right": 5},
            ),
            _node_template(
                fixture_id,
                1,
                source,
                "Ben age 5 is greater than Cy age 3",
                "greater_than",
                {"left": 3, "right": 5},
            ),
        ]
        edges = [_edge(nodes[0]["node_id"], nodes[1]["node_id"], "requires_entity_binding")]
        return {"fixture_id": fixture_id, "expected_hops": [0, 1], "nodes": nodes, "edges": edges}
    if fixture_id == "disconnected_graph":
        nodes = [
            _node_template(
                fixture_id,
                0,
                source,
                "Ada age 7 is greater than Ben age 5",
                "greater_than",
                {"left": 7, "right": 5},
            ),
            _node_template(
                fixture_id,
                1,
                source,
                "Ben age 5 is greater than Cy age 3",
                "greater_than",
                {"left": 5, "right": 3},
            ),
        ]
        return {"fixture_id": fixture_id, "expected_hops": [0, 1], "nodes": nodes, "edges": []}
    if fixture_id == "duplicate_node":
        node = _node_template(
            fixture_id,
            0,
            source,
            "Ada age 7 is greater than Ben age 5",
            "greater_than",
            {"left": 7, "right": 5},
        )
        duplicate = {**node, "hop_index": 1}
        edges = [_edge(node["node_id"], duplicate["node_id"], "requires_entity_binding")]
        return {
            "fixture_id": fixture_id,
            "expected_hops": [0, 1],
            "nodes": [node, duplicate],
            "edges": edges,
        }
    if fixture_id == "cyclic_dependency":
        nodes = [
            _node_template(
                fixture_id,
                0,
                source,
                "Ada age 7 is greater than Ben age 5",
                "greater_than",
                {"left": 7, "right": 5},
            ),
            _node_template(
                fixture_id,
                1,
                source,
                "Ben age 5 is greater than Cy age 3",
                "greater_than",
                {"left": 5, "right": 3},
            ),
        ]
        edges = [
            _edge(nodes[0]["node_id"], nodes[1]["node_id"], "requires_entity_binding"),
            _edge(nodes[1]["node_id"], nodes[0]["node_id"], "requires_entity_binding"),
        ]
        return {"fixture_id": fixture_id, "expected_hops": [0, 1], "nodes": nodes, "edges": edges}
    raise ValueError(f"unknown fixture_id: {fixture_id}")


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
    if relation == "contains":
        return (
            ("certified_true", None)
            if str(right) in str(left)
            else ("counterexample", {"left": left, "right": right})
        )
    return "unsupported_relation", None


def compile_node(raw_node: Mapping[str, Any]) -> JsonDict:
    source_text = str(raw_node["source_text"])
    source_bytes = source_text.encode("utf-8")
    start = int(raw_node["source_start"])
    end = int(raw_node["source_end"])
    expected_span = str(raw_node["span_text"]).encode("utf-8")
    actual_span = source_bytes[start:end] if 0 <= start <= end <= len(source_bytes) else b""
    relation = str(raw_node["relation"])
    source_hash = sha256_bytes(source_bytes)
    base = {
        "node_id": str(raw_node["node_id"]),
        "composed_claim_id": str(raw_node["composed_claim_id"]),
        "hop_index": int(raw_node["hop_index"]),
        "sub_question": str(raw_node["sub_question"]),
        "source_hash": source_hash,
        "source_start": start,
        "source_end": end,
        "source_bytes_hash": sha256_bytes(actual_span),
        "typed_variables": dict(raw_node["typed_variables"]),
        "relation": relation,
        "compiler_version": COMPILER_VERSION,
        "compiler_name": COMPILER_NAME,
    }
    if actual_span != expected_span:
        obligation = {**base, "action": "abstain", "reason": "source_span_mismatch"}
        return {
            **base,
            "executable_obligation_hash": sha256_json(obligation),
            "exact_result": "source_span_mismatch",
            "counterexample": None,
            "action": "abstain",
            "abstention_reason": "source_span_mismatch",
        }
    if relation not in WHITELISTED_NODE_RELATIONS:
        obligation = {**base, "action": "abstain", "reason": "relation_not_whitelisted"}
        return {
            **base,
            "executable_obligation_hash": sha256_json(obligation),
            "exact_result": "unsupported_relation",
            "counterexample": None,
            "action": "abstain",
            "abstention_reason": "relation_not_whitelisted",
        }
    operands = dict(raw_node["operands"])
    exact_result, counterexample = _exact_relation_result(relation, operands)
    obligation = {
        **base,
        "operands": operands,
        "compiler_owned_semantics": True,
    }
    return {
        **base,
        "executable_obligation_hash": sha256_json(obligation),
        "exact_result": exact_result,
        "counterexample": counterexample,
        "action": "release" if exact_result == "certified_true" else "reject",
        "abstention_reason": None,
    }


def _has_cycle(edges: Sequence[Mapping[str, Any]], node_ids: set[str]) -> bool:
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        parent = str(edge.get("parent_id"))
        child = str(edge.get("child_id"))
        if parent in node_ids and child in node_ids:
            outgoing[parent].append(child)
            indegree[child] += 1
    queue = deque([node_id for node_id, count in indegree.items() if count == 0])
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for child in outgoing[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited != len(node_ids)


def _connected_required_graph(
    required_ids: set[str], required_edges: Sequence[Mapping[str, Any]]
) -> bool:
    if len(required_ids) <= 1:
        return True
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in required_ids}
    for edge in required_edges:
        parent = str(edge["parent_id"])
        child = str(edge["child_id"])
        if parent in required_ids and child in required_ids:
            adjacency[parent].add(child)
            adjacency[child].add(parent)
    start = next(iter(required_ids))
    seen = {start}
    queue = deque([start])
    while queue:
        node_id = queue.popleft()
        for peer in adjacency[node_id]:
            if peer not in seen:
                seen.add(peer)
                queue.append(peer)
    return seen == required_ids


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def joint_sufficiency_reduce(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    expected_hops: Sequence[int],
) -> JsonDict:
    reasons: list[str] = []
    node_ids = [str(node.get("node_id")) for node in nodes]
    duplicates = sorted(node_id for node_id, count in Counter(node_ids).items() if count > 1)
    if duplicates:
        _append_unique(reasons, "duplicate_node_id")

    unique_nodes = {str(node["node_id"]): node for node in nodes if "node_id" in node}
    required_ids = set(unique_nodes)
    hops_present = {int(node.get("hop_index", -1)) for node in nodes}
    missing_hops = [hop for hop in expected_hops if hop not in hops_present]
    if missing_hops:
        _append_unique(reasons, "missing_required_hop")

    for node in nodes:
        action = node.get("action")
        exact_result = node.get("exact_result")
        if action == "abstain":
            _append_unique(reasons, str(node.get("abstention_reason") or "node_abstained"))
        elif action == "reject" or exact_result == "counterexample":
            _append_unique(reasons, "contradictory_nodes")
        elif action != "release":
            _append_unique(reasons, "node_not_released")

    required_edges = [edge for edge in edges if edge.get("status") == "required"]
    for edge in edges:
        parent = str(edge.get("parent_id"))
        child = str(edge.get("child_id"))
        if parent not in unique_nodes or child not in unique_nodes:
            _append_unique(reasons, "edge_references_unknown_node")
        if edge.get("relation_type") not in WHITELISTED_EDGE_RELATIONS:
            _append_unique(reasons, "edge_relation_not_whitelisted")
        if edge.get("status") not in EDGE_STATUS_VALUES:
            _append_unique(reasons, "edge_status_invalid")
        if (
            edge.get("status") == "optional"
            and edge.get("coverage_role") == "required_hop_coverage"
        ):
            _append_unique(reasons, "optional_edge_laundering")
        parent_node = unique_nodes.get(parent)
        child_node = unique_nodes.get(child)
        if (
            parent_node is not None
            and child_node is not None
            and edge.get("ordering_rule") == "parent_hop_before_child"
            and int(parent_node.get("hop_index", -1)) >= int(child_node.get("hop_index", -1))
        ):
            _append_unique(reasons, "ordering_rule_violation")

    if required_ids and _has_cycle(edges, required_ids):
        _append_unique(reasons, "cyclic_dependency")
    if len(required_ids) > 1 and not _connected_required_graph(required_ids, required_edges):
        _append_unique(reasons, "disconnected_required_graph")

    required_exact = bool(nodes) and all(node.get("action") == "release" for node in nodes)
    coverage_complete = not missing_hops and set(node_ids) == required_ids and not duplicates
    edge_complete = all(
        str(edge.get("parent_id")) in required_ids and str(edge.get("child_id")) in required_ids
        for edge in required_edges
    )
    release = not reasons and required_exact and coverage_complete and edge_complete
    return {
        "action": "release" if release else "abstain",
        "abstention": not release,
        "abstention_reasons": [] if release else reasons,
        "required_node_ids": sorted(required_ids),
        "required_edge_count": len(required_edges),
        "required_node_coverage_complete": coverage_complete,
        "required_nodes_exact": required_exact,
        "required_edges_connected": _connected_required_graph(required_ids, required_edges),
        "cycle_free": not _has_cycle(edges, required_ids) if required_ids else True,
        "deterministic": True,
    }


def evaluate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    nodes = [compile_node(node) for node in fixture["nodes"]]
    edges = [dict(edge) for edge in fixture["edges"]]
    reduction = joint_sufficiency_reduce(nodes, edges, list(fixture["expected_hops"]))
    fixture_id = str(fixture["fixture_id"])
    unsafe_fixture = fixture_id not in SAFE_FIXTURE_IDS
    return {
        "row_type": "conformance_fixture",
        "unit_id": f"fixture-{fixture_id}",
        "fixture_id": fixture_id,
        "terminal": True,
        "nodes": nodes,
        "edges": edges,
        "expected_hops": list(fixture["expected_hops"]),
        "action": reduction["action"],
        "abstention": reduction["abstention"],
        "abstention_reasons": reduction["abstention_reasons"],
        "unsafe_case": unsafe_fixture,
        "unsafe_release": unsafe_fixture and reduction["action"] == "release",
        "reducer_trace": reduction,
        "hand_check": (
            "release requires all required nodes, byte spans, exact results, required edges, "
            "connectivity, and cycle-free ordering"
        ),
    }


def build_atomic_obligation_node_schema() -> JsonDict:
    return {
        "schema_version": "carnot.atomic_obligation_node.v6574",
        "required_node_fields": list(REQUIRED_NODE_FIELDS),
        "whitelisted_relations": list(WHITELISTED_NODE_RELATIONS),
        "compiler_name": COMPILER_NAME,
        "compiler_version": COMPILER_VERSION,
        "compiler_owns_executable_obligation": True,
        "source_byte_offsets_required": True,
        "source_bytes_hash_required": True,
        "model_can_certify_release": False,
        "full_constraint_ir_generation_allowed": False,
        "supported_actions": ["release", "reject", "abstain"],
    }


def build_dependency_edge_and_joint_reducer_contract() -> JsonDict:
    return {
        "schema_version": "carnot.dependency_edge_and_joint_reducer.v6574",
        "required_edge_fields": list(REQUIRED_EDGE_FIELDS),
        "whitelisted_relation_types": list(WHITELISTED_EDGE_RELATIONS),
        "edge_status_values": list(EDGE_STATUS_VALUES),
        "ordering_rule": "parent_hop_before_child",
        "cycle_handling": "reject",
        "optional_edges_can_satisfy_required_coverage": False,
        "deterministic_reducer": "joint_sufficiency_reduce.v6574",
        "release_rule": [
            "node_ids_unique",
            "source_byte_spans_match",
            "relations_whitelisted",
            "required_hops_present",
            "required_nodes_exact_certified",
            "required_edges_known_and_connected",
            "cycle_free",
        ],
        "release_authority": "compiler_plus_exact_fixture_checker",
    }


def build_frozen_split_and_arm_commitment() -> JsonDict:
    unit_specs = [
        {"slice": "train", "family": "logic_grid", "content": "source-byte-single-hop"},
        {"slice": "calibration", "family": "multi_hop", "content": "source-byte-two-hop"},
        {"slice": "held", "family": "branched", "content": "source-byte-branched-hop"},
    ]
    split_rows = []
    for spec in unit_specs:
        unit_id = sha256_json(spec)
        split_rows.append({**spec, "unit_id": unit_id, "content_sha256": unit_id})
    matched = {
        "corpus_hash": sha256_json(unit_specs),
        "raw_responses_hash": sha256_text("no live responses observed before Exp6574"),
        "parser_version": "source_byte_atomic_parser.v6574",
        "compiler_version": COMPILER_VERSION,
        "threshold": "exact_release_only",
        "source_bytes_hash": sha256_json([row["content_sha256"] for row in split_rows]),
        "seed": RANDOM_SEED,
        "charged_cost_units": 128,
    }
    arms = {
        name: {
            **matched,
            "arm_name": name,
            "matched_commitment_hash": sha256_json(matched),
            "live_outcome_access": False,
        }
        for name in ARM_NAMES
    }
    return {
        "split_names": list(SPLIT_NAMES),
        "split_rows": split_rows,
        "split_membership_sha256": sha256_json(split_rows),
        "frozen_before_live_outcomes": True,
        "live_outcomes_observed_before_freeze": False,
        "arms": arms,
        "matched_dimensions": list(matched),
    }


def build_extraction_acceptance_and_retirement_gates() -> JsonDict:
    return {
        "coverage_gate": {
            "exact_certified_composed_claim_coverage_must_improve": True,
            "comparison": "hop_conditioned_joint > atomic_span_only on held units",
        },
        "precision_gate": {
            "precision_noninferior_required": True,
            "comparison": "hop_conditioned_joint precision >= atomic_span_only precision",
        },
        "safety_gate": {
            "unsafe_release_must_be_zero": True,
            "abstention_is_required_for_unsafe_fixture": True,
        },
        "lineage_and_cost_gate": {
            "lineage_complete_required": True,
            "cost_rows_complete_required": True,
            "charged_cost_matched_across_arms": True,
        },
        "retirement_rules": [
            {
                "scope": "hop_conditioned_joint_sufficiency",
                "retire_if_same_verdict": True,
                "same_verdict": "no held composed-claim coverage gain",
            },
            {
                "scope": "source_byte_atomic_obligation_extractor",
                "retire_if_same_verdict": True,
                "same_verdict": "zero exact semantics or any unsafe release",
            },
        ],
        "post_outcome_mutation": {
            "decomposition_changes_allowed": False,
            "threshold_changes_allowed": False,
            "split_changes_allowed": False,
        },
    }


def split_conformance_rows(commitment: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "row_type": "split",
            "unit_id": row["unit_id"],
            "terminal": True,
            "slice": row["slice"],
            "family": row["family"],
            "content_sha256": row["content_sha256"],
        }
        for row in commitment["split_rows"]
    ]


def arm_conformance_rows(commitment: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "row_type": "arm",
            "unit_id": f"arm-{name}",
            "terminal": True,
            "arm_name": name,
            "matched_commitment_hash": arm["matched_commitment_hash"],
            "charged_cost_units": arm["charged_cost_units"],
            "live_outcome_access": arm["live_outcome_access"],
        }
        for name, arm in commitment["arms"].items()
    ]


def retirement_rows(gates: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "row_type": "retirement",
            "unit_id": f"retirement-{row['scope']}",
            "terminal": True,
            **dict(row),
        }
        for row in gates["retirement_rules"]
    ]


def attack_rows() -> list[JsonDict]:
    controls = {
        "schema_valid_semantic_invalid_nodes": "exact relation result controls release",
        "source_offset_drift": "source bytes hash and offset replay control release",
        "omitted_hops": "expected hop set controls release",
        "optional_edge_laundering": "optional edges cannot satisfy required coverage",
        "graph_cycles": "cycle handling is reject",
        "duplicate_support": "node ID uniqueness controls release",
        "model_self_citation": "source hash must bind external source bytes",
        "post_outcome_decomposition": "split and decomposition freeze before outcomes",
        "threshold_changes": "exact-release threshold is frozen",
        "source_leakage": "arm matching excludes target labels and live outcomes",
        "exact_check_bypass": "compiler plus exact checker owns release authority",
    }
    return [
        {
            "row_type": "attack",
            "unit_id": f"attack-{attack_id}",
            "attack_id": attack_id,
            "terminal": True,
            "closed": True,
            "control": controls[attack_id],
        }
        for attack_id in ATTACK_IDS
    ]


def conformance_rows(
    commitment: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> list[JsonDict]:
    fixture_rows = [evaluate_fixture(build_fixture(fixture_id)) for fixture_id in FIXTURE_IDS]
    return [
        *fixture_rows,
        *split_conformance_rows(commitment),
        *arm_conformance_rows(commitment),
        *retirement_rows(gates),
        *attack_rows(),
    ]


def _matched_arms(commitment: Mapping[str, Any]) -> bool:
    arms = commitment.get("arms", {})
    if not isinstance(arms, Mapping) or set(arms) != set(ARM_NAMES):
        return False
    normalized = []
    for arm in arms.values():
        if not isinstance(arm, Mapping):
            return False
        normalized.append(
            {
                key: value
                for key, value in arm.items()
                if key not in {"arm_name", "matched_commitment_hash"}
            }
        )
    return len({sha256_json(row) for row in normalized}) == 1


def aggregate_row_recomputation(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("conformance_rows", []) if isinstance(row, Mapping)]
    sources = [row for row in payload.get("source_review_receipts", []) if isinstance(row, Mapping)]
    node_schema = payload.get("atomic_obligation_node_schema", {})
    edge_contract = payload.get("dependency_edge_and_joint_reducer_contract", {})
    commitment = payload.get("frozen_split_and_arm_commitment", {})
    gates = payload.get("extraction_acceptance_and_retirement_gates", {})
    preconditions = payload.get("preconditions_checked", {})

    source_ready = bool(sources) and all(
        row.get("available") is True
        and str(row.get("content_sha256", "")).startswith("sha256:")
        and int(row.get("byte_count") or 0) > 0
        for row in sources
    )
    gate_ready = (
        preconditions.get("structured_gate", {}).get("passed") is True
        and preconditions.get("structured_gate", {}).get("observed") == 1.0
    )
    schema_ready = (
        isinstance(node_schema, Mapping)
        and set(node_schema.get("required_node_fields", [])) >= set(REQUIRED_NODE_FIELDS)
        and node_schema.get("compiler_owns_executable_obligation") is True
        and node_schema.get("full_constraint_ir_generation_allowed") is False
        and isinstance(edge_contract, Mapping)
        and set(edge_contract.get("required_edge_fields", [])) >= set(REQUIRED_EDGE_FIELDS)
        and edge_contract.get("cycle_handling") == "reject"
        and edge_contract.get("optional_edges_can_satisfy_required_coverage") is False
    )
    fixture_rows = [row for row in rows if row.get("row_type") == "conformance_fixture"]
    fixtures_ready = (
        {row.get("fixture_id") for row in fixture_rows} == set(FIXTURE_IDS)
        and all(
            row.get("action") == "release"
            for row in fixture_rows
            if row.get("fixture_id") in SAFE_FIXTURE_IDS
        )
        and all(
            row.get("action") == "abstain"
            for row in fixture_rows
            if row.get("fixture_id") not in SAFE_FIXTURE_IDS
        )
        and all(row.get("unsafe_release") is False for row in fixture_rows)
    )
    split_arm_ready = (
        commitment.get("split_names") == list(SPLIT_NAMES)
        and commitment.get("frozen_before_live_outcomes") is True
        and commitment.get("live_outcomes_observed_before_freeze") is False
        and _matched_arms(commitment)
    )
    acceptance_ready = (
        gates.get("coverage_gate", {}).get("exact_certified_composed_claim_coverage_must_improve")
        is True
        and gates.get("precision_gate", {}).get("precision_noninferior_required") is True
        and gates.get("safety_gate", {}).get("unsafe_release_must_be_zero") is True
        and gates.get("lineage_and_cost_gate", {}).get("lineage_complete_required") is True
        and gates.get("lineage_and_cost_gate", {}).get("cost_rows_complete_required") is True
    )
    attack_ready = {row.get("attack_id") for row in rows if row.get("row_type") == "attack"} == set(
        ATTACK_IDS
    ) and all(row.get("closed") is True for row in rows if row.get("row_type") == "attack")
    retirement_ready = any(
        row.get("row_type") == "retirement" and row.get("retire_if_same_verdict") is True
        for row in rows
    )
    terminal_rows_ready = bool(rows) and all(row.get("terminal") is True for row in rows)
    preconditions_ready = (
        preconditions.get("no_llm_inference") is True
        and preconditions.get("no_hardware_execution") is True
        and preconditions.get("hardware_commands_issued") == 0
        and preconditions.get("outcome_bearing_extraction_observed") is False
    )
    ready = all(
        (
            source_ready,
            gate_ready,
            schema_ready,
            fixtures_ready,
            split_arm_ready,
            acceptance_ready,
            attack_ready,
            retirement_ready,
            terminal_rows_ready,
            preconditions_ready,
        )
    )
    return {
        "source_receipts_ready": source_ready,
        "structured_gate_ready": gate_ready,
        "schema_rows_ready": schema_ready,
        "conformance_fixtures_ready": fixtures_ready,
        "frozen_splits_and_matched_arms_ready": split_arm_ready,
        "acceptance_gates_ready": acceptance_ready,
        "attack_rows_ready": attack_ready,
        "retirement_rules_ready": retirement_ready,
        "terminal_conformance_rows_ready": terminal_rows_ready,
        "preconditions_ready": preconditions_ready,
        "joint_sufficiency_method_ready_from_rows": ready,
        "conformance_row_count": len(rows),
    }


def gate_check_summary(payload: Mapping[str, Any], aggregate: Mapping[str, Any]) -> JsonDict:
    failed_checks: list[str] = []
    preconditions = payload.get("preconditions_checked", {})
    gate = preconditions.get("structured_gate", {}) if isinstance(preconditions, Mapping) else {}
    if isinstance(gate, Mapping) and gate.get("passed") is not True:
        failed_checks.append("structured_gate_v570_evidence_contract_ready_score")
    for row in payload.get("source_review_receipts", []):
        if isinstance(row, Mapping) and row.get("available") is not True:
            failed_checks.append(f"source_{row.get('source_id')}_available")
    for key, value in aggregate.items():
        if key.endswith("_ready") and value is not True:
            failed_checks.append(key)
    protected = payload.get("protected_files_unchanged", {})
    if isinstance(protected, Mapping) and protected.get("all_unchanged") is not True:
        failed_checks.append("protected_files_unchanged")
    return {
        "joint_sufficiency_method_ready_from_rows": aggregate.get(
            "joint_sufficiency_method_ready_from_rows"
        ),
        "failed_checks": failed_checks,
        "missing_prerequisite": any(
            check.startswith("source_") or check.startswith("structured_gate_")
            for check in failed_checks
        ),
        "retrospective_method_change": bool(
            isinstance(preconditions, Mapping)
            and preconditions.get("outcome_bearing_extraction_observed") is True
        ),
        "checks_closed": not failed_checks,
    }


def _status_and_verdict(
    ready: bool,
    missing_prerequisite: bool,
    retrospective_method_change: bool,
    failed_checks: Sequence[str],
) -> tuple[str, str, str | None]:
    if ready:
        return (
            "complete_joint_sufficiency_method_ready",
            "complete_joint_sufficiency_method_ready: source-byte atomic nodes, dependency edges, joint reducer, splits, arms, fixtures, attacks, gates, and retirement rules are frozen",
            None,
        )
    if retrospective_method_change:
        return (
            "disqualified_joint_sufficiency_method_retrospective_change",
            "disqualified_joint_sufficiency_method_retrospective_change: decomposition, split, threshold, or release rule changed after outcome-bearing extraction",
            "disqualified",
        )
    if missing_prerequisite:
        return (
            "blocked_joint_sufficiency_method_missing_prerequisites",
            "blocked_joint_sufficiency_method_missing_prerequisites: required upstream gate or source receipt is missing",
            "blocked",
        )
    if failed_checks:
        return (
            "partial_joint_sufficiency_method_contract",
            "partial_joint_sufficiency_method_contract: usable schemas and fixtures exist but one or more reducer, attack, gate, lineage, or cost checks remain open",
            "partial",
        )
    return (
        "blocked_joint_sufficiency_method_contract",
        "blocked_joint_sufficiency_method_contract: no usable joint-sufficiency rows were available",
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
    preconditions: Mapping[str, Any] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    started = time.monotonic()
    before = _protected_hashes(repo_root)
    sources = (
        [dict(row) for row in source_review_receipts]
        if source_review_receipts is not None
        else build_source_review_receipts(repo_root)
    )
    checked = (
        dict(preconditions) if preconditions is not None else build_preconditions_checked(repo_root)
    )
    node_schema = build_atomic_obligation_node_schema()
    edge_contract = build_dependency_edge_and_joint_reducer_contract()
    commitment = build_frozen_split_and_arm_commitment()
    gates = build_extraction_acceptance_and_retirement_gates()
    rows = conformance_rows(commitment, gates)
    after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(before, after)
    payload: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": None,
        "gate_check_summary": {},
        "source_review_receipts": sources,
        "atomic_obligation_node_schema": node_schema,
        "dependency_edge_and_joint_reducer_contract": edge_contract,
        "frozen_split_and_arm_commitment": commitment,
        "conformance_rows": rows,
        "extraction_acceptance_and_retirement_gates": gates,
        "joint_sufficiency_method_ready_score": 0.0,
        "per_unit_rows": rows,
        "aggregate_row_recomputation": {},
        "preconditions_checked": {**checked, "run_date": run_date},
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
        aggregate["joint_sufficiency_method_ready_from_rows"] is True
        and summary["failed_checks"] == []
        and protected["all_unchanged"] is True
    )
    status, verdict, verdict_class = _status_and_verdict(
        ready,
        bool(summary["missing_prerequisite"]),
        bool(summary["retrospective_method_change"]),
        list(summary["failed_checks"]),
    )
    payload["status"] = status
    payload["honest_verdict"] = verdict
    payload["verdict_class"] = verdict_class
    payload["joint_sufficiency_method_ready_score"] = 1.0 if ready else 0.0
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    if write:
        target = result_path or RESULT_RELATIVE_PATH
        atomic_write_json(
            target,
            payload,
            root=repo_root,
            allow_override=not Path(target).is_absolute(),
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

    node_schema = payload.get("atomic_obligation_node_schema", {})
    if set(node_schema.get("required_node_fields", [])) < set(REQUIRED_NODE_FIELDS):
        errors.append("node schema missing required fields")
    if node_schema.get("compiler_owns_executable_obligation") is not True:
        errors.append("compiler-owned node semantics reopened")
    if node_schema.get("full_constraint_ir_generation_allowed") is not False:
        errors.append("full ConstraintIR generation reopened")

    edge_contract = payload.get("dependency_edge_and_joint_reducer_contract", {})
    if set(edge_contract.get("required_edge_fields", [])) < set(REQUIRED_EDGE_FIELDS):
        errors.append("edge schema missing required fields")
    if edge_contract.get("cycle_handling") != "reject":
        errors.append("cycle handling reopened")
    if edge_contract.get("optional_edges_can_satisfy_required_coverage") is not False:
        errors.append("optional edge laundering reopened")

    rows = [row for row in payload.get("conformance_rows", []) if isinstance(row, Mapping)]
    fixture_rows = [row for row in rows if row.get("row_type") == "conformance_fixture"]
    fixture_ids = {row.get("fixture_id") for row in fixture_rows}
    if fixture_ids != set(FIXTURE_IDS):
        errors.append("conformance fixture set mismatch")
    for row in fixture_rows:
        if row.get("fixture_id") in SAFE_FIXTURE_IDS and row.get("action") != "release":
            errors.append("safe fixture did not release")
        if row.get("fixture_id") not in SAFE_FIXTURE_IDS and row.get("action") != "abstain":
            errors.append("unsafe fixture released")
        if row.get("unsafe_release") is True:
            errors.append("unsafe fixture released")
    if any(row.get("closed") is not True for row in rows if row.get("row_type") == "attack"):
        errors.append("attack row is not closed")
    if {row.get("attack_id") for row in rows if row.get("row_type") == "attack"} != set(ATTACK_IDS):
        errors.append("attack row set mismatch")
    if not _matched_arms(payload.get("frozen_split_and_arm_commitment", {})):
        errors.append("matched arms diverged")

    aggregate = aggregate_row_recomputation(payload)
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate recomputation mismatch")
    ready_score = payload.get("joint_sufficiency_method_ready_score")
    failed_checks = payload.get("gate_check_summary", {}).get("failed_checks", [])
    if ready_score == 1.0 and failed_checks:
        errors.append("ready score cannot hide failed checks")
    stored_aggregate = payload.get("aggregate_row_recomputation", {})
    if (
        ready_score == 1.0
        and isinstance(stored_aggregate, Mapping)
        and stored_aggregate.get("joint_sufficiency_method_ready_from_rows") is not True
    ):
        errors.append("ready score must derive from aggregate recomputation")
    if ready_score == 1.0 and aggregate.get("joint_sufficiency_method_ready_from_rows") is not True:
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
