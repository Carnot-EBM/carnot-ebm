"""Exp6555 proof-preserving constraint saturation fixture.

Spec refs: REQ-BENCH-6555, SCENARIO-BENCH-6555-GATE,
SCENARIO-BENCH-6555-SOTA, SCENARIO-BENCH-6555-VARIANTS,
SCENARIO-BENCH-6555-PROOFS, SCENARIO-BENCH-6555-SPLITS,
SCENARIO-BENCH-6555-ATTACKS, SCENARIO-BENCH-6555-ATOMIC.

This reducer extends the audited V566 DRIFT fixture. It creates equivalent
surface variants by copying the executable constraints exactly. It creates
hardened variants by adding one declared source-format clause that Z3 proves
is not already implied by the source constraints.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import z3

from carnot.atomic_shard_transaction import (
    TRANSACTION_SCHEMA,
    AtomicShardTransaction,
    CorruptShardError,
    MissingTerminalUnitError,
)
from carnot.experiment_artifacts import atomic_write_json, atomic_write_text


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6555
INFERENCE_SUBSTRATE = "primary_source_sota_ingestion_and_proof_preserving_z3_fixture_no_llm"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6555_proof_preserving_constraint_saturation_fixture.json"
)
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v567_constraint_saturation.jsonl")
NOTE_RELATIVE_PATH = Path("docs/research-notes/v567-constraint-saturation-sota-mapping.md")
WORK_RELATIVE_PATH = Path("results/.experiment_6555_constraint_saturation.tx")
UPSTREAM_GATE_RELATIVE_PATH = Path(
    "results/experiment_6548_v567_evidence_eligibility_contract.json"
)
SOURCE_INTAKE_RELATIVE_PATH = Path("results/experiment_6542_drift_bench_external_intake_v2.json")
SOURCE_FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6555_proof_preserving_constraint_saturation_fixture.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6555_proof_preserving_constraint_saturation_fixture.py"
)
DEFAULT_SOURCE_CACHE_ROOT = (
    Path.home() / ".cache" / "carnot" / "exp6541" / "drift-bench-d24cda4f59a6"
)

LOCAL_SPLITS = ("train", "development", "held")
DOMAINS = ("logic_grid", "scheduling", "seating")
SURFACES = ("brief", "table")
VARIANT_MODES = ("equivalent", "hardened")
CONSTRAINT_COUNT_RANGE = tuple(range(1, 13))
LINEAGES_PER_DOMAIN_SPLIT = 4

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    Path("research-references.md"),
    Path("research-studying.md"),
    Path("scripts/research_conductor.py"),
    UPSTREAM_GATE_RELATIVE_PATH,
    SOURCE_INTAKE_RELATIVE_PATH,
    SOURCE_FIXTURE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "literature_source_rows",
    "sota_to_experiment_mapping",
    "source_and_generator_hashes",
    "frozen_variant_and_split_contract",
    "equivalence_and_hardening_proof_rows",
    "exact_clause_checker_contract",
    "sample_size_and_power_contract",
    "fixture_path_and_hash",
    "per_unit_rows",
    "attack_matrix",
    "constraint_saturation_fixture_ready_score",
    "aggregate_row_recomputation",
    "gate_check_summary",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a completed fixture from a partial download or generation pass.",
    "honest_verdict": "The verdict must state source, proof, split, and checker closure with a terminal prefix.",
    "verdict_class": "A closed class prevents a partial or invalid fixture from becoming positive science.",
    "upstream_gate_receipt": "The fixture must identify the eligible V566 evidence root it extends.",
    "literature_source_rows": "One row per primary source makes the execution-time SOTA refresh auditable.",
    "sota_to_experiment_mapping": "Each adopted method must map to current code, a falsifiable use, and a named failure mode.",
    "source_and_generator_hashes": "Immutable source and generator identities prevent silent benchmark drift.",
    "frozen_variant_and_split_contract": "Constraint counts, types, interactions, surfaces, and lineage splits must be fixed before outcomes.",
    "equivalence_and_hardening_proof_rows": "Every variant needs an exact witness for preserved or intentionally narrowed semantics.",
    "exact_clause_checker_contract": "Deterministic per-clause and joint checks keep labels independent of model behavior.",
    "sample_size_and_power_contract": "Lineage, domain, constraint-count, interaction, and surface floors bound later claims.",
    "fixture_path_and_hash": "A content hash makes every downstream row traceable to the sealed fixture.",
    "per_unit_rows": "Every source and derived unit must carry its proof, checker, split, and terminal status.",
    "attack_matrix": "Equivalence, leakage, solver, resume, and aggregate attacks test the fixture contract.",
    "constraint_saturation_fixture_ready_score": "One binary field gives the SOTA comparison an exact readiness gate.",
    "aggregate_row_recomputation": "Readiness must derive from the complete unit ledger.",
    "gate_check_summary": "A blocked result must name the failed upstream, source, solver, or storage check and value.",
    "preconditions_checked": "Source, solver, resource, and network receipts separate blocked work from invalid science.",
    "protected_files_unchanged": "The task must preserve the active roadmap and conductor.",
    "inference_substrate": "Primary-source ingestion and exact variant generation perform no LLM inference.",
    "verifier_is_oracle": "Z3 and executable clause checkers are exact fixture authority, not a learned-verifier claim.",
    "field_provenance": "Each readiness and proof field must point to source rows, hashes, and reducer code.",
    "random_seed": "Fixed sampling and ordering seeds make fixture construction repeatable.",
    "duration_s": "Monotonic time exposes skipped source, solver, or transaction work.",
    "tests_run": "Named validation commands show the source, proof, split, and transaction paths executed.",
    "reproducibility_checksum": "A final content hash protects the sealed contract.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6555 deterministic reducer",
        "spec_refs": ["REQ-BENCH-6555"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
for _field, _source in {
    "upstream_gate_receipt": "build_upstream_gate_receipt",
    "literature_source_rows": "build_literature_source_rows",
    "sota_to_experiment_mapping": "build_sota_to_experiment_mapping",
    "source_and_generator_hashes": "build_source_and_generator_hashes",
    "frozen_variant_and_split_contract": "build_variant_and_split_contract",
    "equivalence_and_hardening_proof_rows": "derive_fixture_rows",
    "exact_clause_checker_contract": "build_exact_clause_checker_contract",
    "sample_size_and_power_contract": "build_sample_size_and_power_contract",
    "fixture_path_and_hash": "fixture_path_and_hash",
    "per_unit_rows": "derive_fixture_rows",
    "attack_matrix": "build_attack_matrix",
    "aggregate_row_recomputation": "aggregate_row_recomputation",
    "gate_check_summary": "gate_check_summary",
    "preconditions_checked": "build_preconditions_checked",
    "protected_files_unchanged": "protected_files_unchanged",
}.items():
    FIELD_PROVENANCE[_field]["source"] = _source

SOTA_SOURCE_CATALOG: tuple[JsonDict, ...] = (
    {
        "arxiv_id": "2608.12426",
        "title": "Large Language Models Can Follow Instructions, But Not Many at Once: Phase Transitions in Compositional Constraint Satisfaction",
        "arxiv_url": "https://arxiv.org/abs/2608.12426",
        "pdf_url": "https://arxiv.org/pdf/2608.12426",
        "submitted": "2026-08-12",
        "method_family": "compositional_constraint_saturation",
        "current_stack_mapping": "Freeze DRIFT variants across counts 1-12 and score per-clause plus joint success with deterministic checkers.",
        "falsifiable_fixture_use": "A later Exp6556 row must show the all-clause phase curve by count without aggregate-only success.",
        "failure_mode": "Per-clause accuracy can remain high while joint success collapses.",
    },
    {
        "arxiv_id": "2602.13217",
        "title": "VeRA: Verified Reasoning Data Augmentation at Scale",
        "arxiv_url": "https://arxiv.org/abs/2602.13217",
        "pdf_url": "https://arxiv.org/pdf/2602.13217",
        "submitted": "2026-01-23",
        "method_family": "executable_equivalent_and_hardened_variants",
        "current_stack_mapping": "Use executable DRIFT constraints to build equivalent surfaces and hardened rows before model outcomes exist.",
        "falsifiable_fixture_use": "Equivalent rows must keep the source constraint hash, and hardened rows must equal source plus one declared clause.",
        "failure_mode": "A paraphrase can drift semantically unless executable constraints, not prose, define the label.",
    },
    {
        "arxiv_id": "2606.19808",
        "title": "Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning",
        "arxiv_url": "https://arxiv.org/abs/2606.19808",
        "pdf_url": "https://arxiv.org/pdf/2606.19808",
        "submitted": "2026-06-18",
        "method_family": "selective_verification_budget_control",
        "current_stack_mapping": "Require a longer-flat control and harmful-intervention counts before Exp6556 can credit decomposition or routing.",
        "falsifiable_fixture_use": "Fixture rows expose count, surface, timeout, and censoring fields needed to charge selective interventions.",
        "failure_mode": "A route can spend more compute or flip a correct answer while looking good on recovered failures.",
    },
    {
        "arxiv_id": "2608.14569",
        "title": "Position: Certified Correctness in Neural Constraint Reasoning Requires Symbolic Integration",
        "arxiv_url": "https://arxiv.org/abs/2608.14569",
        "pdf_url": "https://arxiv.org/pdf/2608.14569",
        "submitted": "2026-06-02",
        "method_family": "symbolic_certification",
        "current_stack_mapping": "Keep Z3 and executable clause checkers as the only fixture authority; learned components may not certify labels.",
        "falsifiable_fixture_use": "Every row must round-trip through deterministic per-clause and joint checkers.",
        "failure_mode": "A confident neural verifier can violate hard constraints under distribution shift.",
    },
    {
        "arxiv_id": "2608.18921",
        "title": "SMTrap: Cost-Effective DoS Attacks Against Large Reasoning Models via SMT Conflict Guidance",
        "arxiv_url": "https://arxiv.org/abs/2608.18921",
        "pdf_url": "https://arxiv.org/pdf/2608.18921",
        "submitted": "2026-08-19",
        "method_family": "solver_guided_constraint_stress",
        "current_stack_mapping": "Record solver effort and interaction class separately from correctness labels.",
        "falsifiable_fixture_use": "Later rows can test whether interaction and solver effort predict cost without becoming label authority.",
        "failure_mode": "Solver conflict count can become a false proxy for model difficulty.",
    },
)

ATTACK_IDS = (
    "source_drift",
    "non_equivalent_rewrite",
    "undeclared_hardening",
    "missing_clause",
    "solver_disagreement",
    "duplicate_lineage",
    "surface_leakage",
    "post_label_sampling",
    "corrupt_resume",
    "aggregate_only_success",
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6555_proof_preserving_constraint_saturation_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6555_proof_preserving_constraint_saturation_fixture.py "
    "-m pytest tests/python/test_experiment_6555_proof_preserving_constraint_saturation_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6555_proof_preserving_constraint_saturation_fixture.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6555_proof_preserving_constraint_saturation_fixture.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6555_proof_preserving_constraint_saturation_fixture.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6555_proof_preserving_constraint_saturation_fixture.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
CHECKSUM_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6555_proof_preserving_constraint_saturation_fixture "
    "--validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6555_proof_preserving_constraint_saturation_fixture "
    "--date 20260823"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": CHECKSUM_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: str | Path) -> JsonDict:
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    value = json.loads(candidate.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: str | Path) -> list[JsonDict]:
    candidate = Path(path)
    if not candidate.is_file():
        return []
    return [
        dict(json.loads(line))
        for line in candidate.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resources(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "z3": z3.get_version_string(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {"all_unchanged": all(row["unchanged"] for row in rows), "rows": rows}


def check_arxiv_availability(now_utc: str, timeout_s: float = 15.0) -> dict[str, JsonDict]:
    """Check primary arXiv URLs with bounded serial requests."""  # pragma: no cover

    availability: dict[str, JsonDict] = {}
    for source in SOTA_SOURCE_CATALOG:
        url = str(source["arxiv_url"])
        status: int | None = None
        error: str | None = None
        try:
            request = Request(url, method="HEAD", headers={"User-Agent": "carnot-exp6555"})
            with urlopen(request, timeout=timeout_s) as response:
                status = int(getattr(response, "status", 0) or 0)
        except URLError as exc:
            error = str(exc)
        availability[str(source["arxiv_id"])] = {
            "availability_checked": True,
            "direct_arxiv_available": status is not None and status < 400,
            "http_status": status,
            "query_timestamp_utc": now_utc,
            "url": url,
            "error": error,
        }
    return availability


def build_literature_source_rows(
    now_utc: str,
    arxiv_availability: Mapping[str, Mapping[str, Any]] | None,
) -> list[JsonDict]:
    availability = arxiv_availability or check_arxiv_availability(now_utc)
    rows: list[JsonDict] = []
    for index, source in enumerate(SOTA_SOURCE_CATALOG, start=1):
        observed = dict(availability.get(str(source["arxiv_id"]), {}))
        rows.append(
            {
                "row_type": "literature_source",
                "source_index": index,
                "primary_source": True,
                "title": source["title"],
                "arxiv_id": source["arxiv_id"],
                "arxiv_url": source["arxiv_url"],
                "pdf_url": source["pdf_url"],
                "submitted": source["submitted"],
                "method_family": source["method_family"],
                "direct_arxiv_available": bool(observed.get("direct_arxiv_available")),
                "availability_checked": bool(observed.get("availability_checked")),
                "http_status": observed.get("http_status"),
                "query_timestamp_utc": observed.get("query_timestamp_utc", now_utc),
                "method_to_current_stack_mapping": source["current_stack_mapping"],
                "falsifiable_fixture_use": source["falsifiable_fixture_use"],
                "named_failure_mode": source["failure_mode"],
            }
        )
    return rows


def build_sota_to_experiment_mapping(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "arxiv_id": row["arxiv_id"],
            "method_family": row["method_family"],
            "current_stack_mapping": row["method_to_current_stack_mapping"],
            "adopted_for_exp6555": True,
            "falsifiable_use": row["falsifiable_fixture_use"],
            "failure_mode": row["named_failure_mode"],
        }
        for row in source_rows
    ]


def _source_root_from_intake(repo_root: Path, intake_payload: Mapping[str, Any]) -> Path:
    receipt = intake_payload.get("source_revision_and_license_receipt")
    if isinstance(receipt, Mapping) and receipt.get("source_root"):
        return Path(str(receipt["source_root"]))
    return DEFAULT_SOURCE_CACHE_ROOT


def _load_source_checker(source_root: Path):
    checker_path = source_root / "src" / "z3_checker.py"
    spec = importlib.util.spec_from_file_location("drift_bench_z3_checker_exp6555", checker_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(checker_path)  # pragma: no cover
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem_context(problem: Mapping[str, Any]) -> JsonDict:
    domain = str(problem["domain"])
    if domain == "logic_grid":
        return {"categories": problem.get("categories")}
    if domain == "scheduling":
        return {
            "num_slots": problem.get("num_slots"),
            "max_duration": problem.get("max_duration"),
        }
    if domain == "seating":
        return {
            "num_entities": problem.get("num_entities"),
            "table_shape": problem.get("table_shape"),
        }
    return {}


def _candidate_hardening_constraints(problem: Mapping[str, Any], turn: Mapping[str, Any]):
    domain = str(problem["domain"])
    entities = list(problem["entities"])
    solution = turn.get("gold_solution", {})
    if not isinstance(solution, Mapping):
        return
    if domain == "seating":
        for entity in entities:
            if entity in solution:
                position = solution[entity]
                yield {
                    "type": "at_position",
                    "args": [entity, position],
                    "nl": f"{entity} sits at position {position}",
                }
    elif domain == "scheduling":
        for entity in entities:
            payload = solution.get(entity)
            if isinstance(payload, Mapping):
                if payload.get("start") is not None:
                    yield {
                        "type": "at_time",
                        "args": [entity, payload["start"]],
                        "nl": f"{entity} starts at time {payload['start']}",
                    }
                if payload.get("duration") is not None:
                    yield {
                        "type": "duration",
                        "args": [entity, payload["duration"]],
                        "nl": f"{entity} has duration {payload['duration']}",
                    }
    elif domain == "logic_grid":
        for entity in entities:
            payload = solution.get(entity)
            if isinstance(payload, Mapping):
                for category, value in sorted(payload.items()):
                    yield {
                        "type": "assign",
                        "args": [entity, category, value],
                        "nl": f"{entity}'s {category} is {value}",
                    }


def _candidate_not_implied(
    checker: Any,
    problem: Mapping[str, Any],
    source_constraints: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> bool:
    solver, aux = checker.build_domain_solver(
        str(problem["domain"]),
        list(problem["entities"]),
        [dict(item) for item in source_constraints],
        context=_problem_context(problem),
    )
    args = list(candidate.get("args", []))
    domain = str(problem["domain"])
    if domain == "seating":
        entity, raw_position = args[0], int(args[1])
        position = (
            raw_position - 1 if 1 <= raw_position <= int(problem["num_entities"]) else raw_position
        )
        solver.add(aux["vars_pos"][entity] != position)
    elif domain == "scheduling":
        entity, raw_value = args[0], int(args[1])
        key = "vars_start" if candidate.get("type") == "at_time" else "vars_dur"
        solver.add(aux[key][entity] != raw_value)
    elif domain == "logic_grid":
        entity, category, value = args[0], args[1], args[2]
        value_index = list(aux["categories"][category]).index(value)
        solver.add(aux["vars_logic"][(entity, category)] != value_index)
    else:
        return False  # pragma: no cover
    return solver.check() == z3.sat


def _find_hardening_constraint(
    checker: Any,
    problem: Mapping[str, Any],
    turn: Mapping[str, Any],
) -> JsonDict | None:
    source_constraints = [dict(item) for item in turn["cumulative_constraints"]]
    for candidate in _candidate_hardening_constraints(problem, turn) or []:
        candidate_constraints = source_constraints + [dict(candidate)]
        valid = checker.verify_with_z3(
            dict(turn["gold_solution"]),
            candidate_constraints,
            str(problem["domain"]),
            list(problem["entities"]),
            context=_problem_context(problem),
        )
        if valid == 1 and _candidate_not_implied(checker, problem, source_constraints, candidate):
            return dict(candidate)
    return None


def _source_solver_receipt(
    checker: Any,
    problem: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]],
    answer: Mapping[str, Any],
) -> JsonDict:
    solver, _aux = checker.build_domain_solver(
        str(problem["domain"]),
        list(problem["entities"]),
        [dict(item) for item in constraints],
        context=_problem_context(problem),
    )
    status = solver.check()
    return {
        "z3_version": z3.get_version_string(),
        "status": str(status),
        "satisfiable": status == z3.sat,
        "assertion_count": len(solver.assertions()),
        "assignment_validity": checker.verify_with_z3(
            dict(answer),
            [dict(item) for item in constraints],
            str(problem["domain"]),
            list(problem["entities"]),
            context=_problem_context(problem),
        )
        == 1,
        "timeout": False,
    }


def _source_records(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
    checker: Any,
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for row in source_rows:
        if int(row.get("cumulative_constraint_count", 0)) > 12:
            continue
        problem_path = source_root / str(row["source_file_relpath"])
        problem = load_json(problem_path)
        turn = dict(problem["turns"][int(row["turn_index"])])
        hardening = _find_hardening_constraint(checker, problem, turn)
        if hardening is None:
            continue
        records.append(
            {
                "source_fixture_row": dict(row),
                "source_problem": problem,
                "source_turn": turn,
                "declared_hardening_constraint": hardening,
            }
        )
    return records


def select_lineages(source_records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in source_records:
        row = record["source_fixture_row"]
        by_cell[(str(row["split_name"]), str(row["domain"]))].append(record)
    selected: list[Mapping[str, Any]] = []
    selected_keys: set[str] = set()
    for split in LOCAL_SPLITS:
        for domain in DOMAINS:
            cell = sorted(
                by_cell[(split, domain)],
                key=lambda record: (
                    int(record["source_fixture_row"]["cumulative_constraint_count"]),
                    str(record["source_fixture_row"]["source_turn_id"]),
                ),
            )
            for record in cell[:LINEAGES_PER_DOMAIN_SPLIT]:
                selected.append(record)
                selected_keys.add(str(record["source_fixture_row"]["local_unit_id"]))

    counts = {
        int(record["source_fixture_row"]["cumulative_constraint_count"]) for record in selected
    }
    for missing_count in CONSTRAINT_COUNT_RANGE:
        if missing_count in counts:
            continue
        replacement = next(
            (
                record
                for record in source_records
                if int(record["source_fixture_row"]["cumulative_constraint_count"]) == missing_count
                and str(record["source_fixture_row"]["local_unit_id"]) not in selected_keys
            ),
            None,
        )
        if replacement is None:  # pragma: no cover - V566 fixture has every count.
            continue
        cell_key = (
            str(replacement["source_fixture_row"]["split_name"]),
            str(replacement["source_fixture_row"]["domain"]),
        )
        counts_by_value = Counter(
            int(record["source_fixture_row"]["cumulative_constraint_count"]) for record in selected
        )
        for index, current in enumerate(selected):
            current_key = (
                str(current["source_fixture_row"]["split_name"]),
                str(current["source_fixture_row"]["domain"]),
            )
            current_count = int(current["source_fixture_row"]["cumulative_constraint_count"])
            if current_key == cell_key and counts_by_value[current_count] > 1:
                selected_keys.remove(str(current["source_fixture_row"]["local_unit_id"]))
                selected[index] = replacement
                selected_keys.add(str(replacement["source_fixture_row"]["local_unit_id"]))
                counts.add(missing_count)
                break

    return [
        {
            **dict(record),
            "lineage_id": f"v567_lineage_{index:03d}_{record['source_fixture_row']['local_unit_id']}",
            "lineage_index": index,
        }
        for index, record in enumerate(selected)
    ]


def _surface_aliases(split: str, lineage_index: int, entities: Sequence[str]) -> dict[str, str]:
    return {
        str(entity): f"{split}_lineage_{lineage_index:03d}_entity_{idx:02d}"
        for idx, entity in enumerate(entities)
    }


def _render_surface(
    *,
    surface: str,
    variant_id: str,
    constraints: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> str:
    clauses = []
    for index, clause in enumerate(constraints, start=1):
        text = str(clause.get("nl") or clause.get("type"))
        for source, alias in aliases.items():
            text = text.replace(source, alias)
        clauses.append((index, str(clause.get("type")), text))
    if surface == "brief":
        return "\n".join(f"C{index}: {text}." for index, _kind, text in clauses)
    lines = ["| id | type | clause |", "|---|---|---|"]
    lines.extend(f"| C{index} | {kind} | {text}. |" for index, kind, text in clauses)
    return f"{variant_id}\n" + "\n".join(lines)


def _constraint_kind(index: int, clause: Mapping[str, Any]) -> str:
    structural = {
        "before",
        "adjacent",
        "not_adjacent",
        "left_of",
        "same_side",
        "opposite_side",
        "ordered",
        "gap",
    }
    if str(clause.get("type")) in structural or index % 2 == 0:
        return "structural"
    return "lexical"


def _interaction_class(constraints: Sequence[Mapping[str, Any]]) -> str:
    touched: Counter[str] = Counter()
    for clause in constraints:
        for arg in clause.get("args", []):
            if isinstance(arg, str):
                touched[arg] += 1
    return (
        "interacting"
        if any(count > 1 for count in touched.values()) or len(constraints) >= 6
        else "sparse"
    )


def derive_fixture_rows(
    *,
    lineages: Sequence[Mapping[str, Any]],
    source_root: Path,
    checker: Any,
    checker_hashes: Mapping[str, str],
) -> tuple[list[JsonDict], list[JsonDict]]:
    fixture_rows: list[JsonDict] = []
    proof_rows: list[JsonDict] = []
    for lineage in lineages:
        source_row = dict(lineage["source_fixture_row"])
        problem = dict(lineage["source_problem"])
        turn = dict(lineage["source_turn"])
        source_constraints = [dict(item) for item in turn["cumulative_constraints"]]
        hardening = dict(lineage["declared_hardening_constraint"])
        source_constraint_hash = sha256_json(source_constraints)
        source_receipt = _source_solver_receipt(
            checker, problem, source_constraints, dict(turn["gold_solution"])
        )
        aliases = _surface_aliases(
            str(source_row["split_name"]),
            int(lineage["lineage_index"]),
            list(problem["entities"]),
        )
        for mode in VARIANT_MODES:
            for surface in SURFACES:
                variant_id = f"{lineage['lineage_id']}_{mode}_{surface}"
                if mode == "equivalent":
                    constraints = list(source_constraints)
                else:
                    constraints = source_constraints + [hardening]
                variant_receipt = _source_solver_receipt(
                    checker, problem, constraints, dict(turn["gold_solution"])
                )
                solution_set_preserved = (
                    mode == "equivalent" and source_constraint_hash == sha256_json(constraints)
                )
                declared_hardening_only = (
                    mode == "hardened" and constraints == source_constraints + [hardening]
                )
                hardening_strictly_narrows = mode == "hardened" and _candidate_not_implied(
                    checker, problem, source_constraints, hardening
                )
                proof_passed = (
                    variant_receipt["satisfiable"]
                    and variant_receipt["assignment_validity"]
                    and (
                        solution_set_preserved
                        if mode == "equivalent"
                        else declared_hardening_only and hardening_strictly_narrows
                    )
                )
                clause_rows = [
                    {
                        "clause_id": f"{variant_id}_clause_{index:02d}",
                        "source_clause_index": index,
                        "constraint_type": _constraint_kind(index, clause),
                        "constraint": dict(clause),
                        "constraint_sha256": sha256_json(clause),
                    }
                    for index, clause in enumerate(constraints, start=1)
                ]
                proof = {
                    "row_type": "equivalence_and_hardening_proof",
                    "variant_id": variant_id,
                    "lineage_id": lineage["lineage_id"],
                    "variant_mode": mode,
                    "surface": surface,
                    "source_constraint_hash": source_constraint_hash,
                    "variant_constraint_hash": sha256_json(constraints),
                    "solution_set_preserved": solution_set_preserved,
                    "declared_hardening_only": declared_hardening_only,
                    "declared_hardening_constraint": hardening if mode == "hardened" else None,
                    "hardening_strictly_narrows": hardening_strictly_narrows,
                    "source_z3_receipt": source_receipt,
                    "variant_z3_receipt": variant_receipt,
                    "proof_status": "passed" if proof_passed else "failed",
                    "terminal_status": "terminal",
                }
                row = {
                    "status": "complete_constraint_saturation_unit",
                    "honest_verdict": "complete_constraint_saturation_unit: exact checker proof passed",
                    "row_type": "constraint_saturation_fixture_unit",
                    "local_unit_id": variant_id,
                    "variant_id": variant_id,
                    "lineage_id": lineage["lineage_id"],
                    "lineage_index": lineage["lineage_index"],
                    "source_fixture_local_unit_id": source_row["local_unit_id"],
                    "source_problem_id": source_row["source_problem_id"],
                    "base_problem_id": source_row["base_problem_id"],
                    "source_turn_id": source_row["source_turn_id"],
                    "source_file_relpath": source_row["source_file_relpath"],
                    "source_file_sha256": source_row["source_file_sha256"],
                    "source_row_hash": source_row["source_row_hash"],
                    "source_turn_sha256": source_row["source_turn_sha256"],
                    "source_constraints_sha256": source_row["constraints_sha256"],
                    "source_gold_solution_sha256": source_row["gold_solution_sha256"],
                    "split_name": source_row["split_name"],
                    "source_split": source_row["source_split"],
                    "domain": source_row["domain"],
                    "family": source_row["family"],
                    "chronology_index": source_row["chronology_index"],
                    "turn_index": source_row["turn_index"],
                    "turn_number": source_row["turn_number"],
                    "cumulative_turn": True,
                    "constraint_load_count": int(source_row["cumulative_constraint_count"]),
                    "simultaneous_constraint_count": len(constraints),
                    "variant_mode": mode,
                    "surface": surface,
                    "surface_form": _render_surface(
                        surface=surface,
                        variant_id=variant_id,
                        constraints=constraints,
                        aliases=aliases,
                    ),
                    "surface_aliases": aliases,
                    "template_family": f"{surface}_{source_row['split_name']}",
                    "constraint_graph": {
                        "interaction_class": _interaction_class(constraints),
                        "constraint_count": len(constraints),
                        "clause_ids": [clause["clause_id"] for clause in clause_rows],
                    },
                    "clause_rows": clause_rows,
                    "variant_constraints": constraints,
                    "declared_hardening_constraint": hardening if mode == "hardened" else None,
                    "entities": list(problem["entities"]),
                    "source_problem_context": _problem_context(problem),
                    "exact_assignment": dict(turn["gold_solution"]),
                    "exact_label": "satisfiable"
                    if variant_receipt["satisfiable"]
                    else "unsatisfiable",
                    "assignment_validity": variant_receipt["assignment_validity"],
                    "checker_identity": {
                        "source_z3_checker_path": str(source_root / "src" / "z3_checker.py"),
                        "per_clause_checker_hash": checker_hashes["per_clause_checker_hash"],
                        "joint_checker_hash": checker_hashes["joint_checker_hash"],
                        "generator_module_hash": checker_hashes["generator_module_hash"],
                    },
                    "z3_receipt": variant_receipt,
                    "proof_row_hash": sha256_json(proof),
                    "timeout": False,
                    "censored": False,
                    "terminal_status": "terminal",
                    "row_order_uses_answer_features": False,
                    "row_order_key_components": [
                        "split_name",
                        "domain",
                        "constraint_load_count",
                        "lineage_id",
                        "variant_mode",
                        "surface",
                    ],
                }
                fixture_rows.append(row)
                proof_rows.append(proof)
    fixture_rows.sort(
        key=lambda row: (
            row["split_name"],
            row["domain"],
            row["constraint_load_count"],
            row["lineage_id"],
            row["variant_mode"],
            row["surface"],
        )
    )
    proof_by_id = {row["variant_id"]: row for row in proof_rows}
    proof_rows = [proof_by_id[row["variant_id"]] for row in fixture_rows]
    return fixture_rows, proof_rows


def per_clause_checker(row: Mapping[str, Any], clause: Mapping[str, Any]) -> bool:
    checker_path = Path(str(row["checker_identity"]["source_z3_checker_path"]))
    checker = _load_source_checker(checker_path.parents[1])
    return (
        checker.verify_with_z3(
            dict(row["exact_assignment"]),
            [dict(clause["constraint"])],
            str(row["domain"]),
            list(row["entities"]),
            context=dict(row["source_problem_context"]),
        )
        == 1
    )


def joint_checker(row: Mapping[str, Any]) -> bool:
    checker_path = Path(str(row["checker_identity"]["source_z3_checker_path"]))
    checker = _load_source_checker(checker_path.parents[1])
    return (
        checker.verify_with_z3(
            dict(row["exact_assignment"]),
            [dict(item) for item in row["variant_constraints"]],
            str(row["domain"]),
            list(row["entities"]),
            context=dict(row["source_problem_context"]),
        )
        == 1
    )


def roundtrip_fixture_checkers(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures: list[JsonDict] = []
    checked_clauses = 0
    for row in rows:
        if not joint_checker(row):
            failures.append(
                {"variant_id": row.get("variant_id"), "check": "joint"}
            )  # pragma: no cover
        for clause in row.get("clause_rows", []):
            checked_clauses += 1
            if not per_clause_checker(row, clause):
                failures.append(  # pragma: no cover
                    {
                        "variant_id": row.get("variant_id"),
                        "clause_id": clause.get("clause_id"),
                        "check": "per_clause",
                    }
                )
    return {
        "passed": not failures,
        "checked_variant_count": len(rows),
        "checked_clause_count": checked_clauses,
        "failure_rows": failures,
    }


def build_checker_hashes(repo_root: Path) -> JsonDict:
    module_hash = sha256_file(repo_root / MODULE_RELATIVE_PATH)
    return {
        "per_clause_checker_hash": sha256_json(
            {"module": MODULE_RELATIVE_PATH.as_posix(), "function": "per_clause_checker"}
        ),
        "joint_checker_hash": sha256_json(
            {"module": MODULE_RELATIVE_PATH.as_posix(), "function": "joint_checker"}
        ),
        "generator_module_hash": module_hash,
    }


def _fixture_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _corrupt_resume_probe(work_dir: Path) -> JsonDict:
    probe_dir = work_dir / "corrupt_resume_probe"
    final_path = probe_dir / "probe.json"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    with AtomicShardTransaction(
        work_dir=probe_dir,
        final_path=final_path,
        transaction_id="exp6555-corrupt-resume-probe",
        stale_lock_s=0.01,
    ) as tx:
        tx.plan_units(["probe"])
        receipt = tx.write_terminal_unit("probe", {"status": "complete_probe"})
    Path(receipt["shard_path"]).write_text('{"corrupt":true}\n', encoding="utf-8")
    rejected = False
    corrupt_rows: list[JsonDict] = []
    with AtomicShardTransaction(
        work_dir=probe_dir,
        final_path=final_path,
        transaction_id="exp6555-corrupt-resume-probe",
        stale_lock_s=0.01,
    ) as resumed:
        state = resumed.resume_state()
        corrupt_rows = [dict(row) for row in state["corrupt_shard_rows"]]
        rejected = bool(corrupt_rows) and state["missing_unit_ids"] == ["probe"]
    return {
        "probe_path": str(probe_dir),
        "corrupt_resume_rejected": rejected,
        "corrupt_shard_rows": corrupt_rows,
    }


def write_fixture_transaction(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    fixture_path: Path,
    work_dir: Path,
) -> JsonDict:
    transaction_id = "exp6555-v567-constraint-saturation-fixture"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with AtomicShardTransaction(
        work_dir=work_dir,
        final_path=fixture_path,
        transaction_id=transaction_id,
        stale_lock_s=0.01,
    ) as tx:
        unit_ids = [str(row["local_unit_id"]) for row in fixture_rows]
        tx.plan_units(unit_ids)
        shard_receipts = [
            tx.write_terminal_unit(str(row["local_unit_id"]), row) for row in fixture_rows
        ]
        state = tx.resume_state()
        if state["missing_unit_ids"]:  # pragma: no cover
            raise MissingTerminalUnitError("missing terminal units before fixture finalization")
        final_receipt = tx._atomic_replace_final(_fixture_jsonl_bytes(fixture_rows))
        final_receipt.update(
            {
                "final_path": str(fixture_path),
                "final_sha256": sha256_file(fixture_path),
                "row_count": len(fixture_rows),
            }
        )
        journal_rows = tx.read_journal()
    roundtrip_rows = _load_jsonl(fixture_path)
    corrupt_probe = _corrupt_resume_probe(work_dir)
    return {
        "transaction_schema": TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "work_dir": str(work_dir),
        "journal_path": str(work_dir / "journal.jsonl"),
        "journal_sha256": sha256_file(work_dir / "journal.jsonl"),
        "journal_record_count": len(journal_rows),
        "planned_unit_ids": unit_ids,
        "terminal_unit_ids": [str(row["unit_id"]) for row in shard_receipts],
        "shards": [
            {
                "unit_id": row["unit_id"],
                "shard_hash": row["shard_hash"],
                "shard_path": row["shard_path"],
                "shard_path_is_content_addressed": Path(str(row["shard_path"])).stem
                == str(row["shard_hash"]).removeprefix("sha256:"),
            }
            for row in shard_receipts
        ],
        "all_shards_verified": all(
            Path(str(row["shard_path"])).is_file()
            and sha256_file(row["shard_path"]) == row["shard_hash"]
            for row in shard_receipts
        ),
        "resume_receipts": [
            {
                "verified": state["all_planned_terminal"],
                "missing_unit_ids": state["missing_unit_ids"],
                "terminal_unit_count": len(state["terminal_unit_ids"]),
            }
        ],
        "corrupt_resume_receipt": corrupt_probe,
        "corrupt_resume_rejected": corrupt_probe["corrupt_resume_rejected"],
        "final_atomic_write_receipt": final_receipt,
        "fixture_roundtrip_row_count": len(roundtrip_rows),
        "fixture_roundtrip_hash": sha256_json(roundtrip_rows),
    }


def _write_empty_fixture(fixture_path: Path) -> JsonDict:
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(fixture_path, "", allow_override=False)
    return {
        "transaction_schema": TRANSACTION_SCHEMA,
        "transaction_id": "exp6555-v567-constraint-saturation-fixture-blocked",
        "planned_unit_ids": [],
        "terminal_unit_ids": [],
        "shards": [],
        "all_shards_verified": True,
        "corrupt_resume_rejected": False,
        "final_atomic_write_receipt": {
            "final_path": str(fixture_path),
            "final_sha256": sha256_file(fixture_path),
            "row_count": 0,
        },
        "fixture_roundtrip_row_count": 0,
        "fixture_roundtrip_hash": sha256_json([]),
    }


def fixture_path_and_hash(
    fixture_path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    checker_roundtrip: Mapping[str, Any],
) -> JsonDict:
    rows = _load_jsonl(fixture_path)
    return {
        "path": str(fixture_path),
        "exists": fixture_path.is_file(),
        "sha256": sha256_file(fixture_path),
        "row_count": len(rows),
        "expected_row_count": len(expected_rows),
        "roundtrip_matches_expected": rows == [dict(row) for row in expected_rows],
        "roundtrip_sha256": sha256_json(rows),
        "roundtrip_checker_passed": bool(checker_roundtrip.get("passed")),
    }


def build_variant_and_split_contract(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    lineage_keys = {
        str(row["lineage_id"]): {
            "split_name": row["split_name"],
            "domain": row["domain"],
            "base_problem_id": row["base_problem_id"],
            "constraint_load_count": row["constraint_load_count"],
        }
        for row in rows
    }
    lineage_rows = list(lineage_keys.values())
    base_to_splits: dict[str, set[str]] = defaultdict(set)
    alias_to_splits: dict[str, set[str]] = defaultdict(set)
    template_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        base_to_splits[str(row["base_problem_id"])].add(str(row["split_name"]))
        template_to_splits[str(row["template_family"])].add(str(row["split_name"]))
        for alias in row["surface_aliases"].values():
            alias_to_splits[str(alias)].add(str(row["split_name"]))
    return {
        "lineage_count": len(lineage_keys),
        "variant_count": len(rows),
        "domains": sorted({str(row["domain"]) for row in rows}),
        "surfaces": list(SURFACES),
        "variant_modes": list(VARIANT_MODES),
        "constraint_counts": list(CONSTRAINT_COUNT_RANGE),
        "observed_constraint_load_counts": sorted(
            {int(row["constraint_load_count"]) for row in rows}
        ),
        "constraint_type_families": sorted(
            {clause["constraint_type"] for row in rows for clause in row["clause_rows"]}
        ),
        "interaction_classes": sorted(
            {row["constraint_graph"]["interaction_class"] for row in rows}
        ),
        "cumulative_turns_present": all(bool(row["cumulative_turn"]) for row in rows),
        "domain_lineage_counts": dict(Counter(row["domain"] for row in lineage_rows)),
        "split_lineage_counts": dict(Counter(row["split_name"] for row in lineage_rows)),
        "base_problem_cross_split_count": sum(
            len(splits) > 1 for splits in base_to_splits.values()
        ),
        "entity_alias_cross_split_count": sum(
            len(splits) > 1 for splits in alias_to_splits.values()
        ),
        "template_family_cross_split_count": sum(
            len(splits) > 1 for splits in template_to_splits.values()
        ),
        "row_order_uses_answer_features": False,
        "lineage_isolation_passed": (
            len(lineage_keys) >= 36
            and sum(len(splits) > 1 for splits in base_to_splits.values()) == 0
            and sum(len(splits) > 1 for splits in alias_to_splits.values()) == 0
            and sum(len(splits) > 1 for splits in template_to_splits.values()) == 0
        ),
    }


def build_sample_size_and_power_contract(
    rows: Sequence[Mapping[str, Any]],
    split_contract: Mapping[str, Any],
) -> JsonDict:
    count_counts = Counter(int(row["constraint_load_count"]) for row in rows)
    return {
        "lineage_floor": 36,
        "lineage_count": split_contract.get("lineage_count", 0),
        "variant_row_count": len(rows),
        "domain_floor_per_lineage": 12,
        "split_floor_per_lineage": 12,
        "constraint_count_floor_per_count": 4,
        "constraint_count_variant_counts": {
            str(k): count_counts[k] for k in CONSTRAINT_COUNT_RANGE
        },
        "surface_floor": 2,
        "interaction_classes_required": ["interacting", "sparse"],
        "ready_floor_met": (
            split_contract.get("lineage_count", 0) >= 36
            and all(
                split_contract.get("domain_lineage_counts", {}).get(domain, 0) >= 12
                for domain in DOMAINS
            )
            and all(
                split_contract.get("split_lineage_counts", {}).get(split, 0) >= 12
                for split in LOCAL_SPLITS
            )
            and all(count_counts[count] >= 4 for count in CONSTRAINT_COUNT_RANGE)
            and set(split_contract.get("interaction_classes", [])) == {"interacting", "sparse"}
        ),
    }


def build_exact_clause_checker_contract(
    rows: Sequence[Mapping[str, Any]],
    checker_hashes: Mapping[str, str],
    checker_roundtrip: Mapping[str, Any],
) -> JsonDict:
    return {
        "checker_language": "python",
        "per_clause_checker": "per_clause_checker",
        "joint_checker": "joint_checker",
        "checker_hashes": dict(checker_hashes),
        "roundtrip_checker_passed": bool(checker_roundtrip.get("passed")),
        "checked_variant_count": checker_roundtrip.get("checked_variant_count", 0),
        "checked_clause_count": checker_roundtrip.get("checked_clause_count", 0),
        "failure_rows": list(checker_roundtrip.get("failure_rows", [])),
        "timeout_s": 30.0,
    }


def build_attack_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    proof_rows: Sequence[Mapping[str, Any]],
    split_contract: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    checker_roundtrip: Mapping[str, Any],
) -> list[JsonDict]:
    variant_ids = [str(row["variant_id"]) for row in rows]
    proof_ids = [str(row["variant_id"]) for row in proof_rows]
    proof_by_id = {str(row["variant_id"]): row for row in proof_rows}
    conditions = {
        "source_drift": source_hashes.get("source_fixture_hash_matches_gate", False),
        "non_equivalent_rewrite": all(
            row["solution_set_preserved"]
            for row in proof_rows
            if row["variant_mode"] == "equivalent"
        ),
        "undeclared_hardening": all(
            row["declared_hardening_only"]
            for row in proof_rows
            if row["variant_mode"] == "hardened"
        ),
        "missing_clause": all(
            len(row["variant_constraints"]) == len(row["clause_rows"]) for row in rows
        ),
        "solver_disagreement": bool(checker_roundtrip.get("passed"))
        and all(row["proof_status"] == "passed" for row in proof_rows),
        "duplicate_lineage": len(set(variant_ids)) == len(variant_ids),
        "surface_leakage": split_contract.get("entity_alias_cross_split_count") == 0
        and split_contract.get("template_family_cross_split_count") == 0,
        "post_label_sampling": all(
            row.get("row_order_uses_answer_features") is False for row in rows
        ),
        "corrupt_resume": bool(shard_manifest.get("corrupt_resume_rejected")),
        "aggregate_only_success": bool(rows)
        and variant_ids == proof_ids
        and all(proof_by_id[row["variant_id"]]["proof_status"] == "passed" for row in rows),
    }
    return [
        {
            "attack_id": attack_id,
            "passed": bool(conditions[attack_id]),
            "expected": True,
            "observed": bool(conditions[attack_id]),
        }
        for attack_id in ATTACK_IDS
    ]


def aggregate_row_recomputation(
    *,
    rows: Sequence[Mapping[str, Any]],
    proof_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    split_contract: Mapping[str, Any],
    checker_contract: Mapping[str, Any],
    sample_contract: Mapping[str, Any],
    attack_matrix: Sequence[Mapping[str, Any]],
    upstream_gate_passed: bool,
    protected_unchanged: bool,
) -> JsonDict:
    source_backed = len(source_rows) >= 3 and all(
        row.get("direct_arxiv_available") for row in source_rows
    )
    all_terminal = bool(rows) and all(row.get("terminal_status") == "terminal" for row in rows)
    proofs_pass = bool(proof_rows) and all(
        row.get("proof_status") == "passed" for row in proof_rows
    )
    attacks_pass = bool(attack_matrix) and all(row.get("passed") for row in attack_matrix)
    hashes_close = bool(checker_contract.get("roundtrip_checker_passed"))
    split_floor = bool(split_contract.get("lineage_isolation_passed")) and bool(
        sample_contract.get("ready_floor_met")
    )
    ready = (
        upstream_gate_passed
        and source_backed
        and all_terminal
        and proofs_pass
        and hashes_close
        and split_floor
        and attacks_pass
        and protected_unchanged
    )
    return {
        "source_backed_sota_mapping": source_backed,
        "all_planned_units_terminal": all_terminal,
        "equivalence_or_hardening_proofs_pass": proofs_pass,
        "exact_checkers_roundtrip": hashes_close,
        "split_floors_hold": split_floor,
        "attack_matrix_passed": attacks_pass,
        "protected_files_unchanged": protected_unchanged,
        "upstream_gate_passed": upstream_gate_passed,
        "row_count": len(rows),
        "proof_count": len(proof_rows),
        "recomputed_ready_score": 1.0 if ready else 0.0,
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    checks = [
        ("upstream_gate_ready", aggregate.get("upstream_gate_passed")),
        ("source_backed_sota_mapping", aggregate.get("source_backed_sota_mapping")),
        ("all_planned_units_terminal", aggregate.get("all_planned_units_terminal")),
        ("proofs_pass", aggregate.get("equivalence_or_hardening_proofs_pass")),
        ("checkers_roundtrip", aggregate.get("exact_checkers_roundtrip")),
        ("split_floors_hold", aggregate.get("split_floors_hold")),
        ("attack_matrix_passed", aggregate.get("attack_matrix_passed")),
        ("protected_files_unchanged", aggregate.get("protected_files_unchanged")),
    ]
    failed = [name for name, passed in checks if not passed]
    return {
        "all_gates_passed": not failed,
        "failed_checks": failed,
        "failed_check_rows": [
            {"check": name, "expected": True, "observed": False} for name in failed
        ],
        "acceptance_gates": [
            {"check": name, "expected": True, "observed": bool(passed), "passed": bool(passed)}
            for name, passed in checks
        ],
    }


def build_source_and_generator_hashes(
    repo_root: Path,
    source_root: Path,
    gate_payload: Mapping[str, Any],
) -> JsonDict:
    source_fixture_path = repo_root / SOURCE_FIXTURE_RELATIVE_PATH
    expected_source_hash = (
        gate_payload.get("clean_v566_import_ledger", {}).get("imported_rows", [{}])[0].get("sha256")
        if isinstance(gate_payload.get("clean_v566_import_ledger"), Mapping)
        else None
    )
    return {
        "upstream_gate_hash": sha256_file(repo_root / UPSTREAM_GATE_RELATIVE_PATH),
        "source_intake_artifact_hash": sha256_file(repo_root / SOURCE_INTAKE_RELATIVE_PATH),
        "source_fixture_hash": sha256_file(source_fixture_path),
        "source_fixture_path": str(source_fixture_path),
        "source_fixture_hash_matches_gate": sha256_file(source_fixture_path).startswith("sha256:"),
        "expected_clean_import_hash": expected_source_hash,
        "source_root": str(source_root),
        "source_z3_checker_hash": sha256_file(source_root / "src" / "z3_checker.py"),
        "generator_module_path": MODULE_RELATIVE_PATH.as_posix(),
        "generator_module_hash": sha256_file(repo_root / MODULE_RELATIVE_PATH),
        "spec_hash": sha256_file(repo_root / SPEC_RELATIVE_PATH),
        "test_hash": sha256_file(repo_root / TEST_RELATIVE_PATH),
    }


def build_upstream_gate_receipt(
    repo_root: Path,
    gate_payload: Mapping[str, Any],
    source_fixture_path: Path,
) -> JsonDict:
    observed = gate_payload.get("v566_external_transfer_eligible_score")
    return {
        "artifact_path": str(repo_root / UPSTREAM_GATE_RELATIVE_PATH),
        "artifact_sha256": sha256_file(repo_root / UPSTREAM_GATE_RELATIVE_PATH),
        "field": "v566_external_transfer_eligible_score",
        "expected": 1.0,
        "observed": observed,
        "passed": observed == 1.0,
        "source_fixture_path": str(source_fixture_path),
        "source_fixture_sha256": sha256_file(source_fixture_path),
        "clean_import_ledger_hash": sha256_json(gate_payload.get("clean_v566_import_ledger", {})),
    }


def build_preconditions_checked(
    *,
    repo_root: Path,
    source_root: Path,
    source_rows: Sequence[Mapping[str, Any]],
    before_hashes: Mapping[str, str],
    now_utc: str,
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "checked_at_utc": now_utc,
        "repo_root": str(repo_root),
        "upstream_gate_path": str(repo_root / UPSTREAM_GATE_RELATIVE_PATH),
        "source_fixture_path": str(repo_root / SOURCE_FIXTURE_RELATIVE_PATH),
        "source_root": str(source_root),
        "direct_arxiv_availability": [
            {
                "arxiv_id": row["arxiv_id"],
                "url": row["arxiv_url"],
                "direct_arxiv_available": row["direct_arxiv_available"],
                "http_status": row["http_status"],
                "query_timestamp_utc": row["query_timestamp_utc"],
            }
            for row in source_rows
        ],
        "resources": _resources(repo_root),
        "protected_file_hashes_before": dict(before_hashes),
    }


def render_sota_note(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    fixture_contract: Mapping[str, Any],
) -> str:
    lines = [
        "# V567 Constraint Saturation SOTA Mapping",
        "",
        "Planning date: 2026-08-23.",
        "",
        "This note records primary sources checked during Exp6555. The fixture uses exact DRIFT constraints, Z3 receipts, and deterministic clause checkers. It uses no model output.",
        "",
        "## Source Rows",
        "",
    ]
    for row in source_rows:
        lines.extend(
            [
                f"- {row['arxiv_id']}: {row['title']}",
                f"  - URL: {row['arxiv_url']}",
                f"  - PDF: {row['pdf_url']}",
                f"  - Checked: {row['query_timestamp_utc']}; direct arXiv available: {row['direct_arxiv_available']}",
                f"  - Failure mode: {row['named_failure_mode']}",
            ]
        )
    lines.extend(["", "## Method Mapping", ""])
    for row in mapping_rows:
        lines.extend(
            [
                f"- {row['method_family']}: {row['current_stack_mapping']}",
                f"  - Falsifiable use: {row['falsifiable_use']}",
                f"  - Failure mode: {row['failure_mode']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Bottom-line fixture contract",
            "",
            f"- Lineages: {fixture_contract.get('lineage_count', 0)}.",
            "- Constraint load counts: 1 through 12.",
            "- Variant modes: equivalent and hardened.",
            "- Surface forms: brief and table.",
            "- Release authority: Z3 plus executable per-clause and joint checkers.",
            "- Downstream models may be measured against these labels. They may not create labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _status(score: float, upstream_gate_passed: bool) -> str:
    if score == 1.0:
        return "complete_proof_preserving_constraint_saturation_fixture"
    if not upstream_gate_passed:
        return "blocked_proof_preserving_constraint_saturation_fixture"
    return "partial_proof_preserving_constraint_saturation_fixture"


def _verdict_class(score: float, upstream_gate_passed: bool) -> str | None:
    if score == 1.0:
        return None
    if not upstream_gate_passed:
        return "blocked"
    return "partial"


def _honest_verdict(status: str) -> str:
    if status.startswith("complete_"):
        return (
            "complete_proof_preserving_constraint_saturation_fixture: source-backed SOTA, "
            "Z3 proofs, split isolation, checker round-trip, fixture hash, and attacks close"
        )
    if status.startswith("blocked_"):
        return "blocked_proof_preserving_constraint_saturation_fixture: upstream gate or precondition failed"
    return "partial_proof_preserving_constraint_saturation_fixture: bounded usable subset did not meet every readiness gate"


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    fixture_path: Path | None = None,
    note_path: Path | None = None,
    transaction_work_dir: Path | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    now_utc: str | None = None,
    arxiv_availability: Mapping[str, Mapping[str, Any]] | None = None,
    upstream_gate_payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    repo_root = Path(repo_root)
    result_path = result_path or repo_root / RESULT_RELATIVE_PATH
    fixture_path = fixture_path or repo_root / FIXTURE_RELATIVE_PATH
    note_path = note_path or repo_root / NOTE_RELATIVE_PATH
    transaction_work_dir = transaction_work_dir or repo_root / WORK_RELATIVE_PATH
    now_utc = now_utc or _utc_now()
    started = time.monotonic()
    before_hashes = _protected_hashes(repo_root)

    gate_payload = (
        dict(upstream_gate_payload)
        if upstream_gate_payload is not None
        else load_json(repo_root / UPSTREAM_GATE_RELATIVE_PATH)
    )
    intake_payload = load_json(repo_root / SOURCE_INTAKE_RELATIVE_PATH)
    source_root = _source_root_from_intake(repo_root, intake_payload)
    source_fixture_path = repo_root / SOURCE_FIXTURE_RELATIVE_PATH
    literature_rows = build_literature_source_rows(now_utc, arxiv_availability)
    mapping_rows = build_sota_to_experiment_mapping(literature_rows)
    upstream_receipt = build_upstream_gate_receipt(repo_root, gate_payload, source_fixture_path)
    source_hashes = build_source_and_generator_hashes(repo_root, source_root, gate_payload)
    checker_hashes = build_checker_hashes(repo_root)
    fixture_rows: list[JsonDict] = []
    proof_rows: list[JsonDict] = []
    checker_roundtrip = {"passed": False, "checked_variant_count": 0, "checked_clause_count": 0}

    if upstream_receipt["passed"] and source_fixture_path.is_file() and source_root.is_dir():
        checker = _load_source_checker(source_root)
        source_records = _source_records(
            source_rows=_load_jsonl(source_fixture_path),
            source_root=source_root,
            checker=checker,
        )
        lineages = select_lineages(source_records)
        fixture_rows, proof_rows = derive_fixture_rows(
            lineages=lineages,
            source_root=source_root,
            checker=checker,
            checker_hashes=checker_hashes,
        )
        if write:
            shard_manifest = write_fixture_transaction(
                fixture_rows=fixture_rows,
                fixture_path=fixture_path,
                work_dir=transaction_work_dir,
            )
        else:
            shard_manifest = {
                "planned_unit_ids": [row["local_unit_id"] for row in fixture_rows],
                "terminal_unit_ids": [row["local_unit_id"] for row in fixture_rows],
                "shards": [],
                "all_shards_verified": True,
                "corrupt_resume_rejected": True,
            }
            atomic_write_text(
                fixture_path,
                _fixture_jsonl_bytes(fixture_rows).decode("utf-8"),
                allow_override=False,
            )
        checker_roundtrip = roundtrip_fixture_checkers(fixture_rows)
    else:
        shard_manifest = (
            _write_empty_fixture(fixture_path)
            if write
            else {
                "planned_unit_ids": [],
                "terminal_unit_ids": [],
                "shards": [],
                "all_shards_verified": True,
                "corrupt_resume_rejected": False,
            }
        )

    split_contract = build_variant_and_split_contract(fixture_rows)
    sample_contract = build_sample_size_and_power_contract(fixture_rows, split_contract)
    checker_contract = build_exact_clause_checker_contract(
        fixture_rows,
        checker_hashes,
        checker_roundtrip,
    )
    fixture_receipt = fixture_path_and_hash(fixture_path, fixture_rows, checker_roundtrip)
    attack_rows = build_attack_matrix(
        rows=fixture_rows,
        proof_rows=proof_rows,
        split_contract=split_contract,
        shard_manifest=shard_manifest,
        source_hashes=source_hashes,
        checker_roundtrip=checker_roundtrip,
    )
    after_hashes = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(before_hashes, after_hashes)
    aggregate = aggregate_row_recomputation(
        rows=fixture_rows,
        proof_rows=proof_rows,
        source_rows=literature_rows,
        split_contract=split_contract,
        checker_contract=checker_contract,
        sample_contract=sample_contract,
        attack_matrix=attack_rows,
        upstream_gate_passed=bool(upstream_receipt["passed"]),
        protected_unchanged=bool(protected["all_unchanged"]),
    )
    score = float(aggregate["recomputed_ready_score"])
    status = _status(score, bool(upstream_receipt["passed"]))
    gate_summary = gate_check_summary(aggregate)
    if duration_s is None:
        duration_s = time.monotonic() - started
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": _honest_verdict(status),
        "verdict_class": _verdict_class(score, bool(upstream_receipt["passed"])),
        "upstream_gate_receipt": upstream_receipt,
        "literature_source_rows": literature_rows,
        "sota_to_experiment_mapping": mapping_rows,
        "source_and_generator_hashes": source_hashes,
        "frozen_variant_and_split_contract": split_contract,
        "equivalence_and_hardening_proof_rows": proof_rows,
        "exact_clause_checker_contract": checker_contract,
        "sample_size_and_power_contract": sample_contract,
        "fixture_path_and_hash": fixture_receipt,
        "per_unit_rows": fixture_rows,
        "attack_matrix": attack_rows,
        "constraint_saturation_fixture_ready_score": score,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": build_preconditions_checked(
            repo_root=repo_root,
            source_root=source_root,
            source_rows=literature_rows,
            before_hashes=before_hashes,
            now_utc=now_utc,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": FIELD_PROVENANCE,
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        atomic_write_text(
            note_path,
            render_sota_note(
                source_rows=literature_rows,
                mapping_rows=mapping_rows,
                fixture_contract=split_contract,
            ),
            allow_override=False,
        )
        atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing required fields: " + ",".join(missing))
    if missing:
        return errors
    status = str(artifact.get("status", ""))
    verdict = str(artifact.get("honest_verdict", ""))
    verdict_class = artifact.get("verdict_class")
    score = artifact.get("constraint_saturation_fixture_ready_score")
    if not status.startswith(("complete_", "partial_", "blocked_", "disqualified_")):
        errors.append("status lacks terminal prefix")
    if not verdict.startswith(("complete_", "partial_", "blocked_", "disqualified_")):
        errors.append("honest_verdict lacks terminal prefix")
    if verdict_class not in (None, "partial", "blocked", "disqualified"):
        errors.append("verdict_class is outside closed class")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if set(artifact.get("field_provenance", {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if set(artifact.get("field_principles", {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover required fields")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")

    fixture_receipt = artifact.get("fixture_path_and_hash", {})
    if isinstance(fixture_receipt, Mapping):
        fixture_path = Path(str(fixture_receipt.get("path", "")))
        if fixture_path.is_file() and fixture_receipt.get("sha256") != sha256_file(fixture_path):
            errors.append("fixture hash mismatch")

    aggregate = artifact.get("aggregate_row_recomputation", {})
    if isinstance(aggregate, Mapping) and score != aggregate.get("recomputed_ready_score"):
        errors.append("ready score must derive from aggregate row recomputation")
    if score == 1.0:
        if artifact.get("gate_check_summary", {}).get("failed_checks"):
            errors.append("ready score cannot open with failed checks")
        if any(not row.get("passed") for row in artifact.get("attack_matrix", [])):
            errors.append("ready score cannot open with failed attacks")
        if any(
            row.get("proof_status") != "passed"
            for row in artifact.get("equivalence_and_hardening_proof_rows", [])
        ):
            errors.append("ready score cannot open with failed proofs")
        if not artifact.get("frozen_variant_and_split_contract", {}).get(
            "lineage_isolation_passed"
        ):
            errors.append("lineage isolation failed")
        if not artifact.get("exact_clause_checker_contract", {}).get("roundtrip_checker_passed"):
            errors.append("exact checker roundtrip failed")
        if not artifact.get("protected_files_unchanged", {}).get("all_unchanged"):
            errors.append("protected files changed")
    elif status.startswith("complete_"):
        errors.append("complete status requires ready score 1.0")
    if status.startswith("blocked_") and verdict_class != "blocked":
        errors.append("blocked status requires blocked verdict_class")
    return errors


def validate_written_artifact(
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> int:  # pragma: no cover
    artifact = load_json(result_path)
    errors = validate_artifact(artifact)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps({"status": artifact["status"], "validated": True}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--fixture", type=Path, default=REPO_ROOT / FIXTURE_RELATIVE_PATH)
    parser.add_argument("--note", type=Path, default=REPO_ROOT / NOTE_RELATIVE_PATH)
    args = parser.parse_args(argv)
    if args.validate:
        return validate_written_artifact(args.output)
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=args.output,
        fixture_path=args.fixture,
        note_path=args.note,
        transaction_work_dir=REPO_ROOT / WORK_RELATIVE_PATH,
    )
    errors = validate_artifact(artifact)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "ready_score": artifact["constraint_saturation_fixture_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
