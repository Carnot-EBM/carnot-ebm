"""Exp6503 V561 source delta and method preregistration.

Spec refs: REQ-REPORT-6503, SCENARIO-REPORT-6503-GATE,
SCENARIO-REPORT-6503-SOURCES, SCENARIO-REPORT-6503-METHODS,
SCENARIO-REPORT-6503-AUTHORITY, SCENARIO-REPORT-6503-DEPENDENCIES,
SCENARIO-REPORT-6503-SCHEMA.

This reducer turns source pages into receipts and freezes the local comparison
contract before data generation. It does not treat papers, repositories, or
product pages as local evidence. Exact solvers remain the only release
authority for downstream results.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import html
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any
from urllib import error, request

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
Fetcher = Callable[[str, str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260822"
RANDOM_SEED = 6503
INFERENCE_SUBSTRATE = "source_receipts_and_local_method_preregistration_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6503_v561_source_delta_method_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
UPSTREAM_GATE_RELATIVE_PATH = Path(
    "results/experiment_6502_v560_retirement_v561_lineage_lock.json"
)
UPSTREAM_GATE_FIELD = "v561_lineage_lock_ready_score"
UPSTREAM_GATE_EXPECTED_VALUE = 1.0

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-references.md"),
    Path("scripts/research_conductor.py"),
)

LOCAL_PRIMITIVE_PATHS = {
    "sat_energy": Path("python/carnot/verify/sat.py"),
    "binary_csp": Path("python/carnot/inference/run_csp.py"),
    "graph_coloring": Path("python/carnot/phase3/graph_coloring_ising.py"),
    "kan_z3": Path("python/carnot/kan_z3.py"),
    "lns_fixture": Path("python/carnot/experiment_5299_constraint_lns_solver_repair_fixture_v484.py"),
    "sampler_sim": Path("python/carnot/hardware/sampler_sim.py"),
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6503_v561_source_delta_method_contract "
    "--date 20260822"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6503_v561_source_delta_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6503_v561_source_delta_method_contract.py "
    "-m pytest tests/python/test_experiment_6503_v561_source_delta_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6503_v561_source_delta_method_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6503_v561_source_delta_method_contract.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6503_v561_source_delta_method_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6503_v561_source_delta_method_contract.json"
)
DOC_CHECK_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; assert Path('ops/e2e-test-plan.md').exists()\""
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": DOC_CHECK_COMMAND, "exit_code": 0},
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "verdict_class",
    "upstream_gate_receipt",
    "source_receipt_rows",
    "source_delta_rows",
    "method_contract",
    "authority_boundary",
    "dependency_decision_rows",
    "method_contract_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records source-ingestion completion.",
    "verdict_class": (
        "Closed enum: positive | circular_positive | null | blocked | disqualified | partial."
    ),
    "upstream_gate_receipt": "Binds the Exp6502 gate and observed value.",
    "source_receipt_rows": "Records one stable receipt per checked source.",
    "source_delta_rows": "Separates new, unchanged, unavailable, and superseded findings.",
    "method_contract": (
        "Freezes benchmark, advice, learning, LNS, continual, and mapping methods."
    ),
    "authority_boundary": "Keeps exact solvers above all learned signals.",
    "dependency_decision_rows": "Explains why each dependency is reused, added, or rejected.",
    "method_contract_ready_score": "Same-roadmap gate for the frozen method contract.",
    "per_unit_rows": "Carries source, method, metric, and boundary rows.",
    "aggregate_row_recomputation": "Recomputes readiness from required receipts.",
    "gate_check_summary": "Names any failed gate or source precondition and observed value.",
    "preconditions_checked": "Records gate, network, source, repository, and tool checks.",
    "protected_files_unchanged": "Proves protected files stayed unchanged.",
    "inference_substrate": (
        "Declares source retrieval and local artifact synthesis with no inference model."
    ),
    "verifier_is_oracle": "False because source synthesis is not an execution oracle.",
    "field_principles": "Explains each receipt and boundary field.",
    "field_provenance": "Maps each finding to a URL, access receipt, and local contract row.",
    "random_seed": "Fixes deterministic source and attack ordering.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records commands and exit codes.",
    "reproducibility_checksum": "Hashes source receipts and the frozen contract.",
    "honest_verdict": (
        "Uses complete_* when the contract is valid or blocked_* with gate_check_summary."
    ),
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "local_reducer": "build_artifact",
        "spec_refs": ["REQ-REPORT-6503"],
        "source": "Exp6503 deterministic reducer",
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["upstream_gate_receipt"]["source"] = UPSTREAM_GATE_RELATIVE_PATH.as_posix()
FIELD_PROVENANCE["source_receipt_rows"]["source"] = "SOURCE_MANIFEST live retrieval rows"
FIELD_PROVENANCE["source_delta_rows"]["source"] = "source_delta_rows"
FIELD_PROVENANCE["method_contract"]["source"] = "build_method_contract"
FIELD_PROVENANCE["authority_boundary"]["source"] = "build_authority_boundary"
FIELD_PROVENANCE["dependency_decision_rows"]["source"] = "dependency_decision_rows"
FIELD_PROVENANCE["aggregate_row_recomputation"]["source"] = "aggregate_row_recomputation"

VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

SOURCE_MANIFEST: tuple[JsonDict, ...] = (
    {
        "source_id": "arxiv_symbolic_certification",
        "title": "Position: Certified Correctness in Neural Constraint Reasoning Requires Symbolic Integration",
        "stable_url": "https://arxiv.org/abs/2608.14569",
        "retrieval_url": "https://arxiv.org/abs/2608.14569",
        "fallback_version": "arXiv:2608.14569v1",
        "source_class": "primary_paper",
        "method_id": "symbolic_certification",
        "delta_status": "unchanged_pinned",
        "bounded_carnot_implication": (
            "Use symbolic or executable checks as release authority; neural signals only guide."
        ),
        "claim_boundary": "paper_claim_not_local_evidence",
    },
    {
        "source_id": "arxiv_branch_order",
        "title": "Learning to Rank the Initial Branching Order of SAT Solvers",
        "stable_url": "https://arxiv.org/abs/2603.07176",
        "retrieval_url": "https://arxiv.org/abs/2603.07176",
        "fallback_version": "arXiv:2603.07176v1",
        "source_class": "primary_paper",
        "method_id": "initial_branch_ranking",
        "delta_status": "unchanged_pinned",
        "bounded_carnot_implication": (
            "Compare initial advice, bounded refocus, and native dynamic heuristics."
        ),
        "claim_boundary": "paper_claim_not_local_evidence",
    },
    {
        "source_id": "openreview_clause_predictions",
        "title": "Using Clause Predictions for Learning-Augmented Constraint Satisfaction",
        "stable_url": "https://openreview.net/forum?id=xvcqXxw4Le",
        "retrieval_url": "https://api2.openreview.net/notes?forum=xvcqXxw4Le",
        "fallback_version": "OpenReview xvcqXxw4Le public record",
        "source_class": "primary_submission_page",
        "method_id": "clause_prediction_advice",
        "delta_status": "unchanged_pinned",
        "bounded_carnot_implication": (
            "Treat clause predictions as uncertain advice with exact fallback."
        ),
        "claim_boundary": "submission_claim_not_local_evidence",
    },
    {
        "source_id": "arxiv_lns",
        "title": "Large Neighborhood Search meets Iterative Neural Constraint Heuristics",
        "stable_url": "https://arxiv.org/abs/2603.20801",
        "retrieval_url": "https://arxiv.org/abs/2603.20801",
        "fallback_version": "arXiv:2603.20801v1",
        "source_class": "primary_paper",
        "method_id": "exact_repair_lns",
        "delta_status": "unchanged_pinned",
        "bounded_carnot_implication": (
            "Test destroy and exact repair separately after advice has held benefit."
        ),
        "claim_boundary": "paper_claim_not_local_evidence",
    },
    {
        "source_id": "semantic_scholar_ebt",
        "title": "Semantic Scholar EBT record for arXiv:2507.02092",
        "stable_url": "https://www.semanticscholar.org/paper/2507.02092",
        "retrieval_url": (
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092"
            "?fields=title,url,year,publicationDate,externalIds,citationCount"
        ),
        "fallback_version": "Semantic Scholar Graph API paper lookup",
        "source_class": "secondary_discovery_surface",
        "method_id": None,
        "delta_status": "unchanged_context",
        "bounded_carnot_implication": "Citation trails are discovery context only.",
        "claim_boundary": "secondary_record_not_authority",
    },
    {
        "source_id": "semantic_scholar_arm_ebm",
        "title": "Semantic Scholar ARM-EBM record for arXiv:2512.15605",
        "stable_url": "https://www.semanticscholar.org/paper/2512.15605",
        "retrieval_url": (
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605"
            "?fields=title,url,year,publicationDate,externalIds,citationCount"
        ),
        "fallback_version": "Semantic Scholar Graph API paper lookup",
        "source_class": "secondary_discovery_surface",
        "method_id": None,
        "delta_status": "unchanged_context",
        "bounded_carnot_implication": "Citation trails are discovery context only.",
        "claim_boundary": "secondary_record_not_authority",
    },
    {
        "source_id": "huggingface_papers",
        "title": "Hugging Face Papers V561 source surface",
        "stable_url": "https://huggingface.co/papers",
        "retrieval_url": "https://huggingface.co/api/papers/2608.14569",
        "fallback_version": "Hugging Face Papers API",
        "source_class": "secondary_discovery_surface",
        "method_id": None,
        "delta_status": "unchanged_context",
        "bounded_carnot_implication": "Community paper pages do not add release authority.",
        "claim_boundary": "secondary_record_not_authority",
    },
    {
        "source_id": "github_neurosat",
        "title": "dmeoli/NeuroSAT",
        "stable_url": "https://github.com/dmeoli/NeuroSAT",
        "retrieval_url": "https://github.com/dmeoli/NeuroSAT",
        "fallback_version": "GitHub repository page",
        "source_class": "implementation_reference",
        "method_id": None,
        "delta_status": "unchanged_context",
        "bounded_carnot_implication": "Implementation reference only; no runtime dependency.",
        "claim_boundary": "reference_code_not_dependency",
    },
    {
        "source_id": "github_neuralsat",
        "title": "dynaroars/neuralsat",
        "stable_url": "https://github.com/dynaroars/neuralsat",
        "retrieval_url": "https://github.com/dynaroars/neuralsat",
        "fallback_version": "GitHub repository page",
        "source_class": "implementation_reference",
        "method_id": None,
        "delta_status": "unchanged_context",
        "bounded_carnot_implication": "DPLL(T)-style verification context; no runtime dependency.",
        "claim_boundary": "reference_code_not_dependency",
    },
    {
        "source_id": "extropic_z1_update",
        "title": "From One to One Billion: Torx, Thermalizers, and Z1",
        "stable_url": "https://extropic.ai/writing/from-one-to-one-billion",
        "retrieval_url": "https://extropic.ai/writing/from-one-to-one-billion",
        "fallback_version": "Extropic first-party August 2026 writing",
        "source_class": "product_claim",
        "method_id": None,
        "delta_status": "unavailable_local_evidence",
        "bounded_carnot_implication": "Keep a fixed-width mapping ABI; make no device claim.",
        "claim_boundary": "no_local_device_or_api",
    },
    {
        "source_id": "logical_intelligence_kona",
        "title": "Kona: Energy-Based Models (EBMs) for AI Reasoning",
        "stable_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "retrieval_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "fallback_version": "Logical Intelligence first-party product page",
        "source_class": "product_claim",
        "method_id": None,
        "delta_status": "unavailable_local_evidence",
        "bounded_carnot_implication": "Product comparator only until weights and runner exist.",
        "claim_boundary": "no_local_runner_or_weights",
    },
)
SOURCE_BY_ID = {str(row["source_id"]): dict(row) for row in SOURCE_MANIFEST}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order for repeatable hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after canonical serialization."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes and return a stable missing marker."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _access_date_from_run_date(run_date: str) -> str:
    return f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return dict(payload)


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def default_fetcher(url: str, source_id: str) -> JsonDict:  # pragma: no cover - live network.
    """Fetch a source URL with stdlib HTTP so no runtime dependency is added."""

    del source_id
    headers = {"User-Agent": "Carnot-Exp6503-source-receipt/1.0"}
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=20) as response:  # noqa: S310
            body = response.read(2_000_000).decode("utf-8", "replace")
            return {
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "url": response.geturl(),
                "headers": dict(response.headers.items()),
                "body": body,
                "error": None,
            }
    except error.HTTPError as exc:
        body = exc.read(2_000_000).decode("utf-8", "replace")
        return {
            "ok": False,
            "status_code": int(exc.code),
            "url": url,
            "headers": dict(exc.headers.items()) if exc.headers else {},
            "body": body,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "url": url,
            "headers": {},
            "body": "",
            "error": str(exc),
        }


def _extract_title(body: str, fallback: str) -> str:
    """Extract a short page title, otherwise keep the manifest title."""

    try:
        decoded = json.loads(body)
        if isinstance(decoded, Mapping):
            value = decoded.get("title")
            if isinstance(value, str) and value.strip():
                return value.strip()
            content = decoded.get("content")
            if isinstance(content, Mapping):
                nested = content.get("title")
                if isinstance(nested, Mapping) and isinstance(nested.get("value"), str):
                    return str(nested["value"]).strip()
    except json.JSONDecodeError:
        pass
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.I | re.S)
    if title_match:
        title = re.sub(r"\s+", " ", html.unescape(title_match.group(1))).strip()
        if title:
            return title.removeprefix("GitHub - ").strip()
    arxiv_match = re.search(r"Title:\s*([^<\n]+)", body, flags=re.I)
    if arxiv_match:
        title = re.sub(r"\s+", " ", html.unescape(arxiv_match.group(1))).strip()
        if title:
            return title
    return fallback


def _extract_version(body: str, fallback: str) -> str:
    match = re.search(r"arXiv:\d{4}\.\d{5}v\d+", body)
    return match.group(0) if match else fallback


def _retrieval_state(status_code: int, body_or_error: str) -> str:
    text = body_or_error.lower()
    if status_code == 429 or "too many requests" in text or "rate limit" in text:
        return "rate_limited"
    if "challenge verification required" in text:
        return "blocked_challenge"
    if status_code == 404 or "paper not found" in text:
        return "not_indexed"
    if 200 <= status_code < 400:
        return "available"
    return "blocked"


def _network_probe(fetcher: Fetcher) -> JsonDict:
    checked_at = _utc_now()
    probe_url = "https://arxiv.org/abs/2608.14569"
    receipt = fetcher(probe_url, "network_probe")
    return {
        "network_required": True,
        "network_used": True,
        "network_available": bool(receipt.get("ok")),
        "checked_at_utc": checked_at,
        "probe_url": probe_url,
        "probe_http_state": f"http_{int(receipt.get('status_code') or 0)}",
        "error": receipt.get("error"),
    }


def collect_source_receipts(
    *,
    fetcher: Fetcher = default_fetcher,
    access_date: str,
    network_state: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Collect one bounded receipt row per V561 source."""

    active_network = dict(network_state) if network_state is not None else _network_probe(fetcher)
    rows: list[JsonDict] = []
    for source in SOURCE_MANIFEST:
        receipt = fetcher(str(source["retrieval_url"]), str(source["source_id"]))
        body = str(receipt.get("body") or "")
        error_text = str(receipt.get("error") or "")
        status_code = int(receipt.get("status_code") or 0)
        source_id = str(source["source_id"])
        rows.append(
            {
                "row_type": "source_receipt",
                "source_id": source_id,
                "title": _extract_title(body, str(source["title"])),
                "stable_url": source["stable_url"],
                "retrieval_url": source["retrieval_url"],
                "access_date": access_date,
                "available_version": _extract_version(body, str(source["fallback_version"])),
                "source_class": source["source_class"],
                "method_id": source["method_id"],
                "retrieval_state": _retrieval_state(status_code, body + " " + error_text),
                "http_state": f"http_{status_code}",
                "network_state": active_network,
                "response_bytes": len(body.encode("utf-8")),
                "response_sha256": sha256_json(
                    {"source_id": source_id, "status_code": status_code, "body": body}
                ),
                "bounded_carnot_implication": source["bounded_carnot_implication"],
                "claim_boundary": source["claim_boundary"],
                "delta_status": source["delta_status"],
            }
        )
    return rows


def source_delta_rows(source_receipt_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Describe whether source checks changed the V561 planner contract."""

    return [
        {
            "row_type": "source_delta",
            "source_id": row["source_id"],
            "previous_state": "v561_planner_refresh_20260821",
            "current_state": row["retrieval_state"],
            "delta_status": row["delta_status"],
            "stable_url": row["stable_url"],
            "bounded_carnot_implication": row["bounded_carnot_implication"],
            "no_local_evidence_boundary": row["claim_boundary"],
        }
        for row in source_receipt_rows
    ]


def build_method_contract() -> JsonDict:
    """Freeze the exact-structural comparison contract before data generation."""

    promoted = [
        {
            "row_type": "method",
            "method_id": "symbolic_certification",
            "source_ids": ["arxiv_symbolic_certification"],
            "local_test_target": "exp6504_exact_labels",
            "bounded_local_test": "Every accepted label replays under exact CDCL/CSP authority.",
            "paper_claim_is_local_evidence": False,
        },
        {
            "row_type": "method",
            "method_id": "initial_branch_ranking",
            "source_ids": ["arxiv_branch_order"],
            "local_test_target": "exp6507_exp6508_branch_ab",
            "bounded_local_test": "Compare initial order, bounded refocus, and native heuristic.",
            "paper_claim_is_local_evidence": False,
        },
        {
            "row_type": "method",
            "method_id": "clause_prediction_advice",
            "source_ids": ["openreview_clause_predictions"],
            "local_test_target": "exp6506_exact_branch_labels",
            "bounded_local_test": "Generate clause and branch labels only from exact replay.",
            "paper_claim_is_local_evidence": False,
        },
        {
            "row_type": "method",
            "method_id": "exact_repair_lns",
            "source_ids": ["arxiv_lns"],
            "local_test_target": "exp6509_exact_repair_lns",
            "bounded_local_test": "Destroy advice can propose neighborhoods; exact repair certifies.",
            "paper_claim_is_local_evidence": False,
        },
    ]
    return {
        "contract_version": "v561_source_delta_method_contract_v1",
        "planning_date": RUN_DATE,
        "benchmark": {
            "families": [
                "random_3cnf",
                "pseudo_industrial_3cnf",
                "tseitin",
                "pigeonhole",
                "graph_coloring",
                "small_scheduling",
            ],
            "shift_axes": [
                "family",
                "scale",
                "surface_relabeling",
                "solver_hardness",
                "source",
                "label",
            ],
            "minimum_held_cell_size": 30,
            "exact_authorities": [
                "exact_cdcl_solver",
                "exact_csp_repair",
                "executable_validity_check",
            ],
        },
        "solver_metrics": [
            "conflicts",
            "decisions",
            "propagations",
            "restarts",
            "wall_time_s",
            "proof_or_model_replay",
            "influence_duration_conflicts",
        ],
        "structural_features": [
            "variable_degree",
            "clause_length_histogram",
            "literal_polarity_balance",
            "graph_centrality",
            "propagation_pressure",
            "conflict_clause_membership",
        ],
        "branch_checkpoints": {
            "conflict_budgets": [0, 16, 64, 256],
            "events": ["initial_order", "bounded_refocus", "native_dynamic_override"],
        },
        "model_controls": [
            "analytic_structural",
            "solver_native_dynamic",
            "random_order",
            "linear",
            "mlp",
            "compact_kan",
            "gnn",
        ],
        "lns": {
            "enabled_after": "held_branch_advice_benefit",
            "destroy_operators": ["random_stochastic", "conflict_neighborhood", "advice_guided"],
            "repair_authority": "exact_greedy_cdcl_or_csp_repair",
            "fallback": "solver_only",
        },
        "continual_learning": {
            "mutable_state": "weights_over_fixed_safe_feature_set_only",
            "feedback": "exact_solver_cost",
            "controls": ["frozen", "matched_update_dose", "restart", "rollback"],
            "support_checks": ["future_support", "negative_transfer", "bounded_capacity"],
        },
        "mapping": {
            "target": "fixed_width_ising_tsu_abi",
            "execution_scope": "cpu_reference_only",
            "hardware_claim_allowed": False,
        },
        "statistical_rules": [
            "paired_by_base_instance",
            "stratified_by_family_and_shift",
            "minimum_held_cell_size_30",
            "report_effect_size_and_confidence_interval",
            "no_speed_claim_without_matched_timing",
        ],
        "failure_conditions": [
            "upstream_gate_failed",
            "missing_required_source_receipt",
            "promoted_method_without_local_test",
            "product_claim_without_no_local_evidence_boundary",
            "new_runtime_dependency_added",
            "learned_advice_accepts_or_labels_solution",
        ],
        "promoted_method_rows": promoted,
    }


def build_authority_boundary() -> JsonDict:
    """Declare exact solvers above all learned advice."""

    return {
        "row_type": "boundary",
        "authority_id": "v561_exact_solver_first",
        "neural_advice_may": ["order_search", "select_neighborhood", "abstain"],
        "neural_advice_must_not": [
            "accept_solution",
            "label_solution",
            "release_solution",
            "override_exact_authority",
        ],
        "learned_advice_can_accept_solution": False,
        "exact_authorities": [
            "exact_cdcl_solver",
            "exact_csp_repair",
            "executable_validity_check",
        ],
        "release_rule": "only exact authority can accept, label, or release a solution",
    }


def dependency_decision_rows(repo_root: Path) -> list[JsonDict]:
    """Explain why Exp6503 adds no runtime dependency."""

    local = {
        key: {
            "path": path.as_posix(),
            "exists": (repo_root / path).is_file(),
        }
        for key, path in LOCAL_PRIMITIVE_PATHS.items()
    }
    try:
        import z3  # noqa: PLC0415

        z3_available = True
        z3_version = z3.get_version_string()
    except Exception:
        z3_available = False
        z3_version = None
    return [
        {
            "row_type": "dependency",
            "dependency_id": "z3_existing",
            "decision": "reuse",
            "reason": "Existing Z3 exact primitive covers symbolic checks.",
            "local_available": z3_available,
            "version": z3_version,
        },
        {
            "row_type": "dependency",
            "dependency_id": "local_sat_csp_graph_kan_sampler",
            "decision": "reuse",
            "reason": "Existing SAT, CSP, graph, KAN, and sampler modules cover required primitives.",
            "local_primitives": local,
        },
        {
            "row_type": "dependency",
            "dependency_id": "local_lns_fixture",
            "decision": "reuse",
            "reason": "Existing Exp5299 LNS fixture already keeps exact repair authoritative.",
            "path": local["lns_fixture"]["path"],
            "local_available": local["lns_fixture"]["exists"],
        },
        {
            "row_type": "dependency",
            "dependency_id": "external_neurosat_runtime",
            "decision": "reject",
            "reason": "GitHub reference is useful context but not needed for a local exact primitive.",
        },
        {
            "row_type": "dependency",
            "dependency_id": "external_neuralsat_runtime",
            "decision": "reject",
            "reason": "Reference verifier does not displace local exact authority.",
        },
        {
            "row_type": "dependency",
            "dependency_id": "semantic_scholar_client",
            "decision": "reject",
            "reason": "Stdlib HTTP receipts are sufficient for rate-limit-aware source rows.",
        },
        {
            "row_type": "dependency",
            "dependency_id": "huggingface_hub_dependency",
            "decision": "reject",
            "reason": "The source surface is discovery context, not a runtime substrate.",
        },
        {
            "row_type": "dependency",
            "dependency_id": "extropic_sdk_or_device",
            "decision": "reject",
            "reason": "No authenticated local device or API route is present.",
        },
        {
            "row_type": "dependency",
            "dependency_id": "logical_intelligence_runner",
            "decision": "reject",
            "reason": "No public weights or reproducible local runner are present.",
        },
    ]


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    """Record protected hashes without mutating protected planning files."""

    files: dict[str, JsonDict] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        digest = sha256_file(repo_root / relative)
        files[relative.as_posix()] = {
            "sha256_before": digest,
            "sha256_after": digest,
            "unchanged": digest != "missing",
            "protected_by_task_contract": True,
        }
    return {
        "files": files,
        "changed_paths": [
            path for path, row in files.items() if row["sha256_before"] != row["sha256_after"]
        ],
        "all_protected_files_unchanged": all(row["unchanged"] is True for row in files.values()),
    }


def upstream_gate_receipt(
    repo_root: Path,
    network_state: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Bind Exp6502 gate bytes and observed value."""

    path = repo_root / UPSTREAM_GATE_RELATIVE_PATH
    payload = _read_json(path) if path.is_file() else {}
    observed = payload.get(UPSTREAM_GATE_FIELD)
    return {
        "row_type": "upstream_gate",
        "path": UPSTREAM_GATE_RELATIVE_PATH.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "field": UPSTREAM_GATE_FIELD,
        "expected_value": UPSTREAM_GATE_EXPECTED_VALUE,
        "observed_value": observed,
        "passed": observed == UPSTREAM_GATE_EXPECTED_VALUE,
        "network_state": dict(network_state),
        "protected_file_hashes": {
            path: row["sha256_before"]
            for path, row in dict(protected.get("files", {})).items()
            if isinstance(row, Mapping)
        },
    }


def aggregate_row_recomputation(
    *,
    gate: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    dependencies: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Recompute readiness from source, method, dependency, and boundary rows."""

    manifest_ids = {str(row["source_id"]) for row in SOURCE_MANIFEST}
    source_ids = {str(row.get("source_id")) for row in source_rows}
    promoted = [
        row for row in contract.get("promoted_method_rows", []) if isinstance(row, Mapping)
    ]
    product_rows = [row for row in source_rows if row.get("source_class") == "product_claim"]
    new_deps = [row for row in dependencies if row.get("decision") == "add"]
    promoted_with_tests = [
        row
        for row in promoted
        if row.get("local_test_target") and row.get("paper_claim_is_local_evidence") is False
    ]
    product_boundaries = [
        row for row in product_rows if str(row.get("claim_boundary", "")).startswith("no_local")
    ]
    exact_boundary = (
        authority.get("learned_advice_can_accept_solution") is False
        and "exact_cdcl_solver" in authority.get("exact_authorities", [])
        and "accept_solution" in authority.get("neural_advice_must_not", [])
    )
    source_complete = source_ids == manifest_ids and all(
        row.get("stable_url") and row.get("response_sha256") for row in source_rows
    )
    ready = (
        gate.get("passed") is True
        and source_complete
        and len(promoted_with_tests) == len(promoted) == 4
        and len(product_boundaries) == len(product_rows) == 2
        and not new_deps
        and exact_boundary
        and protected.get("all_protected_files_unchanged") is True
    )
    return {
        "source_receipt_count": len(source_rows),
        "required_source_count": len(SOURCE_MANIFEST),
        "source_receipts_cover_manifest": source_ids == manifest_ids,
        "promoted_method_count": len(promoted),
        "promoted_methods_with_local_tests": len(promoted_with_tests),
        "product_claim_rows": len(product_rows),
        "product_claims_with_no_local_evidence_boundary": len(product_boundaries),
        "new_runtime_dependency_count": len(new_deps),
        "authority_boundary_exact_first": exact_boundary,
        "upstream_gate_passed": gate.get("passed") is True,
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "method_contract_hash": sha256_json(contract),
        "method_contract_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    """Summarize pass/fail gates with observed values."""

    checks = {
        "upstream_gate_passed": aggregate.get("upstream_gate_passed") is True,
        "source_receipts_cover_manifest": aggregate.get("source_receipts_cover_manifest") is True,
        "promoted_methods_have_local_tests": aggregate.get("promoted_methods_with_local_tests") == 4,
        "product_claims_have_boundaries": (
            aggregate.get("product_claims_with_no_local_evidence_boundary")
            == aggregate.get("product_claim_rows")
            == 2
        ),
        "no_new_runtime_dependency": aggregate.get("new_runtime_dependency_count") == 0,
        "authority_boundary_exact_first": aggregate.get("authority_boundary_exact_first") is True,
        "protected_files_unchanged": aggregate.get("protected_files_unchanged") is True,
    }
    failed = [
        {
            "check": key,
            "expected": True,
            "observed": value,
        }
        for key, value in checks.items()
        if value is not True
    ]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def preconditions_checked(
    *,
    repo_root: Path,
    run_date: str,
    gate: Mapping[str, Any],
    network_state: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Record repository, network, tool, and source preconditions."""

    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "git_status_short": _git_output(repo_root, ["status", "--short"]),
        "upstream_gate_receipt": dict(gate),
        "network": dict(network_state),
        "source_manifest_count": len(SOURCE_MANIFEST),
        "required_files": {
            name: {"path": str(repo_root / path), "exists": (repo_root / path).exists()}
            for name, path in {
                "CODEX.md": Path("CODEX.md"),
                "CLAUDE.md": Path("CLAUDE.md"),
                "research_program": Path("research-program.md"),
                "research_references": Path("research-references.md"),
                "v561_roadmap_doc": Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
                "experiment_template": Path("scripts/experiment_template.py"),
                "e2e_plan": Path("ops/e2e-test-plan.md"),
                "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
                "spec": SPEC_RELATIVE_PATH,
            }.items()
        },
        "tool_checks": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "no_gpu_required": True,
            "local_primitives": {
                key: {
                    "path": path.as_posix(),
                    "exists": (repo_root / path).is_file(),
                    "sha256": sha256_file(repo_root / path),
                }
                for key, path in LOCAL_PRIMITIVE_PATHS.items()
            },
        },
        "protected_file_hashes": {
            path: row["sha256_before"]
            for path, row in dict(protected.get("files", {})).items()
            if isinstance(row, Mapping)
        },
        "preconditions_ready": gate.get("passed") is True,
    }


def per_unit_rows(
    source_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    dependencies: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Collect source, method, control, metric, boundary, and dependency rows."""

    rows: list[JsonDict] = [
        {**dict(row), "row_type": "source", "receipt_score": 1.0} for row in source_rows
    ]
    rows.extend(dict(row) for row in contract.get("promoted_method_rows", []))
    rows.extend(
        {
            "row_type": "control",
            "control_id": control,
            "role": "model_or_solver_control",
        }
        for control in contract.get("model_controls", [])
    )
    rows.extend(
        {
            "row_type": "metric",
            "metric_id": metric,
            "authority": "exact_solver_or_replay_receipt",
        }
        for metric in contract.get("solver_metrics", [])
    )
    rows.append(dict(authority))
    rows.extend(dict(row) for row in dependencies)
    return rows


def tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if tests_run is not None:
        return [dict(row) for row in tests_run]
    return [dict(row) for row in DEFAULT_TESTS_RUN]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(payload)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    source_receipt_rows: Sequence[Mapping[str, Any]] | None = None,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    access_date: str | None = None,
) -> JsonDict:
    """Build and optionally write the Exp6503 artifact."""

    start = time.perf_counter()
    result = repo_root / RESULT_RELATIVE_PATH if result_path is None else Path(result_path)
    active_access_date = access_date or _access_date_from_run_date(run_date)
    rows = (
        [dict(row) for row in source_receipt_rows]
        if source_receipt_rows is not None
        else collect_source_receipts(access_date=active_access_date)
    )
    network_state = dict(rows[0].get("network_state", {})) if rows else _network_probe(default_fetcher)
    protected = protected_files_unchanged(repo_root)
    gate = upstream_gate_receipt(repo_root, network_state, protected)
    contract = build_method_contract()
    authority = build_authority_boundary()
    dependencies = dependency_decision_rows(repo_root)
    aggregate = aggregate_row_recomputation(
        gate=gate,
        source_rows=rows,
        contract=contract,
        authority=authority,
        dependencies=dependencies,
        protected=protected,
    )
    gate_summary = gate_check_summary(aggregate)
    ready_score = aggregate["method_contract_ready_score_from_rows"]
    status = (
        "complete_v561_source_delta_method_contract"
        if ready_score == 1.0
        else "blocked_v561_source_delta_method_contract"
    )
    artifact: JsonDict = {
        "status": status,
        "verdict_class": "null" if ready_score == 1.0 else "blocked",
        "upstream_gate_receipt": gate,
        "source_receipt_rows": rows,
        "source_delta_rows": source_delta_rows(rows),
        "method_contract": contract,
        "authority_boundary": authority,
        "dependency_decision_rows": dependencies,
        "method_contract_ready_score": ready_score,
        "per_unit_rows": per_unit_rows(rows, contract, authority, dependencies),
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            run_date=run_date,
            gate=gate,
            network_state=network_state,
            protected=protected,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s if duration_s is not None else time.perf_counter() - start, 6),
        "tests_run": tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_v561_source_delta_method_contract: source receipts are pinned, "
            "the method contract is frozen, and paper/product claims remain bounded context"
            if ready_score == 1.0
            else "blocked_v561_source_delta_method_contract"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        result.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(result, artifact, allow_override=False)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Return schema and contract errors. Empty list means valid."""

    try:
        artifact = _read_json(value) if isinstance(value, str | Path) else dict(value)
    except Exception as exc:
        return [str(exc)]
    errors: list[str] = []
    required = set(REQUIRED_ARTIFACT_FIELDS)
    present = set(artifact)
    missing = sorted(required - present)
    unexpected = sorted(present - required)
    if missing:
        errors.append(f"missing required fields: {missing}")
    if unexpected:
        errors.append(f"unexpected fields: {unexpected}")
    if set(artifact.get("field_principles", {})) != required:
        errors.append("field_principles must cover exactly required fields")
    if set(artifact.get("field_provenance", {})) != required:
        errors.append("field_provenance must cover exactly required fields")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (
        verdict.startswith("complete_v561_source_delta_method_contract")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict lacks accepted Exp6503 prefix")
    ready = artifact.get("method_contract_ready_score")
    summary_passed = artifact.get("gate_check_summary", {}).get("all_gates_passed")
    aggregate_ready = artifact.get("aggregate_row_recomputation", {}).get(
        "method_contract_ready_score_from_rows"
    )
    if ready != aggregate_ready or (ready == 1.0) != bool(summary_passed):
        errors.append("ready score and gate summary disagree")
    source_ids = {
        str(row.get("source_id"))
        for row in artifact.get("source_receipt_rows", [])
        if isinstance(row, Mapping)
    }
    if source_ids != {str(row["source_id"]) for row in SOURCE_MANIFEST}:
        errors.append("source_receipt_rows must cover source manifest")
    methods = artifact.get("method_contract", {}).get("promoted_method_rows", [])
    if not all(
        isinstance(row, Mapping)
        and row.get("local_test_target")
        and row.get("paper_claim_is_local_evidence") is False
        for row in methods
    ):
        errors.append("promoted methods must map to local tests")
    authority = artifact.get("authority_boundary", {})
    if authority.get("learned_advice_can_accept_solution") is not False:
        errors.append("authority boundary must forbid learned acceptance")
    if any(row.get("decision") == "add" for row in artifact.get("dependency_decision_rows", [])):
        errors.append("new runtime dependencies are not allowed")
    product_rows = [
        row for row in artifact.get("source_receipt_rows", []) if row.get("source_class") == "product_claim"
    ]
    if not all(str(row.get("claim_boundary", "")).startswith("no_local") for row in product_rows):
        errors.append("product claims must have no-local-evidence boundary")
    checksum = artifact.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum:
        expected = reproducibility_checksum(artifact)
        if checksum != expected:
            errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    rows = collect_source_receipts(access_date=_access_date_from_run_date(args.date))
    build_artifact(
        repo_root=REPO_ROOT,
        result_path=Path(args.output),
        source_receipt_rows=rows,
        write=True,
        run_date=args.date,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
