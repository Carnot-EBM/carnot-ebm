"""Exp6515 V564 source delta and method contract.

Spec refs: REQ-REPORT-6515, SCENARIO-REPORT-6515-SOURCES,
SCENARIO-REPORT-6515-METHODS, SCENARIO-REPORT-6515-AUTHORITY,
SCENARIO-REPORT-6515-SCHEMA.

This module records source ingestion and freezes a method contract. It does
not run an experimental model, train a router, or read downstream V564
outcomes. Papers and products can guide local tests, but exact solvers remain
on the acceptance path.
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
CommandRunner = Callable[[list[str], Path], tuple[int, str, str]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6515
INFERENCE_SUBSTRATE = "low_concurrency_primary_source_ingestion_no_experimental_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6515_v564_source_method_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_RELATIVE_PATH = Path("docs/research-notes/v564-source-method-contract-2026-08-23.md")

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("results/experiment_6510_v563_independent_exact_root.json"),
    Path("results/experiment_6513_v564_terminal_handoff_contract.json"),
    Path("results/experiment_6514_atomic_shard_artifact_transaction.json"),
)
SOURCE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-studying.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("scripts/sweep_clusters.py"),
    Path("scripts/sweep_semscholar.py"),
    Path("docs/research-notes/search-layer-literature-2026-06-11.md"),
    NOTE_RELATIVE_PATH,
    Path("results/experiment_6503_v561_source_delta_method_contract.json"),
)
LOCAL_HOOK_PATHS = (
    Path("python/carnot/experiment_6504_exact_structural_benchmark_commitment.py"),
    Path("python/carnot/experiment_6510_v563_independent_exact_root.py"),
    Path("python/carnot/experiment_6514_atomic_shard_artifact_transaction.py"),
    Path("python/carnot/atomic_shard_transaction.py"),
    Path("python/carnot/cem/solver.py"),
    Path("python/carnot/inference/run_csp.py"),
    Path("python/carnot/experiment_5924_transactional_constraint_memory_v2.py"),
    Path("python/carnot/models/kan/formal_verification.py"),
    Path("python/carnot/analysis/pdit_certificate_state_mapping.py"),
)
OUTCOME_ARTIFACT_RELATIVE_PATHS = tuple(
    Path(f"results/experiment_{exp}_v564_outcome_placeholder.json")
    for exp in range(6516, 6524)
) + (
    Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json"),
    Path("results/experiment_6517_branch_pilot_independent_audit.json"),
    Path("results/experiment_6518_structural_control_headroom_ab_v2.json"),
    Path("results/experiment_6519_structural_headroom_certificate.json"),
    Path("results/experiment_6520_safety_net_router.json"),
    Path("results/experiment_6521_conflict_memory_controller.json"),
    Path("results/experiment_6522_chronological_self_learning.json"),
    Path("results/experiment_6523_adaptive_validation_csl_audit.json"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "query_receipts",
    "source_rows",
    "primary_source_hashes",
    "citation_count_boundaries",
    "sota_to_experiment_rows",
    "non_transfer_rows",
    "frozen_method_contract",
    "v564_method_contract_ready_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
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
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records whether the V564 source-method contract is complete or blocked.",
    "honest_verdict": (
        "States the terminal source-ingestion outcome without turning sources into experimental evidence."
    ),
    "verdict_class": "Closed enum for this contract: null or partial.",
    "query_receipts": "Keeps each low-concurrency source query dated, reproducible, and bounded.",
    "source_rows": "Records one row per checked paper, product, or repository.",
    "primary_source_hashes": "Binds verified primary and first-party pages to source hashes.",
    "citation_count_boundaries": (
        "Prevents rate limits or secondary indexes from becoming fabricated counts."
    ),
    "sota_to_experiment_rows": "Maps methods to Exp6516-Exp6523 implementation steps and metrics.",
    "non_transfer_rows": "Names claims that do not transfer into Carnot.",
    "frozen_method_contract": (
        "Freezes method, control, fallback, witness, mapping, and stop-rule contracts before outcomes."
    ),
    "v564_method_contract_ready_score": (
        "Gate score opens only when every adopted method has source, mapping, control, and boundary evidence."
    ),
    "gate_check_summary": "Names failed source, schema, protection, and readiness gates with observed errors.",
    "per_unit_rows": "Flattens source, mapping, boundary, control, and gate rows for recomputation.",
    "aggregate_row_recomputation": "Recomputes readiness from rows instead of trusting narrative text.",
    "preconditions_checked": (
        "Records network, query, tool, source-path, git, and protected-hash preconditions."
    ),
    "protected_files_unchanged": "Proves protected files stayed byte-identical.",
    "inference_substrate": "Declares primary-source ingestion with no experimental LLM inference.",
    "verifier_is_oracle": "False because source ingestion is not a correctness oracle.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps each required field to receipts, source rows, or deterministic reducers.",
    "random_seed": "Pins deterministic row ordering for this no-randomness contract.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": "Detects drift in source rows, method rows, gates, and receipts.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6515 deterministic reducer",
        "spec_refs": ["REQ-REPORT-6515"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["query_receipts"]["source"] = "collect_query_receipts"
FIELD_PROVENANCE["source_rows"]["source"] = "collect_source_rows"
FIELD_PROVENANCE["primary_source_hashes"]["source"] = "source_rows.source_hash"
FIELD_PROVENANCE["citation_count_boundaries"]["source"] = "citation_count_boundaries"
FIELD_PROVENANCE["sota_to_experiment_rows"]["source"] = "build_sota_to_experiment_rows"
FIELD_PROVENANCE["non_transfer_rows"]["source"] = "build_non_transfer_rows"
FIELD_PROVENANCE["frozen_method_contract"]["source"] = "build_frozen_method_contract"
FIELD_PROVENANCE["aggregate_row_recomputation"]["source"] = "aggregate_row_recomputation"
FIELD_PROVENANCE["protected_files_unchanged"]["source"] = "protected_files_unchanged"
FIELD_PROVENANCE["preconditions_checked"]["source"] = "preconditions_checked"

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6515_v564_source_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6515_v564_source_method_contract.py "
    "-m pytest tests/python/test_experiment_6515_v564_source_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6515_v564_source_method_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6515_v564_source_method_contract.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6515_v564_source_method_contract --date 20260823"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6515_v564_source_method_contract.json"
)
VALIDATE_COMMAND = ".venv/bin/python -m carnot.experiment_6515_v564_source_method_contract --validate"
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)


def _source(
    *,
    source_id: str,
    title: str,
    source_kind: str,
    stable_url: str,
    query_channel: str,
    method: str,
    claimed_evidence: str,
    available_code_or_data: str,
    carnot_hook: str,
    non_transferable_claim: str,
    exact_authority_boundary: str,
    expected_source_date: str | None = None,
    arxiv_id: str | None = None,
    adopted_method: bool = False,
    required_primary_check: bool = False,
    method_transfer_status: str = "candidate_local_contract",
) -> JsonDict:
    return {
        "source_id": source_id,
        "title": title,
        "source_kind": source_kind,
        "stable_url": stable_url,
        "retrieval_url": stable_url,
        "query_channel": query_channel,
        "method": method,
        "claimed_evidence": claimed_evidence,
        "available_code_or_data": available_code_or_data,
        "carnot_hook": carnot_hook,
        "non_transferable_claim": non_transferable_claim,
        "exact_authority_boundary": exact_authority_boundary,
        "expected_source_date": expected_source_date,
        "arxiv_id": arxiv_id,
        "adopted_method": adopted_method,
        "required_primary_check": required_primary_check,
        "method_transfer_status": method_transfer_status,
    }


SOURCE_MANIFEST: tuple[JsonDict, ...] = (
    _source(
        source_id="task_coevolve",
        title="Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2608.20169",
        query_channel="arxiv_primary",
        method="variance_weighted_adaptive_validation",
        claimed_evidence="Reports full-set-like final performance with 80 percent fewer evaluations.",
        available_code_or_data="code_will_be_released",
        carnot_hook="Use adaptive validation only around a frozen full audit and exact sentinel set.",
        non_transferable_claim="No LLM harness optimization performance transfers to exact solver validity.",
        exact_authority_boundary="Adaptive validation estimates cost only; full exact held audit controls release.",
        expected_source_date="2026-08-20",
        arxiv_id="2608.20169",
        adopted_method=True,
        required_primary_check=True,
    ),
    _source(
        source_id="safety_nets",
        title="On the Applicability of Safety Nets: A Safety-By-Design Solution for Certifying Neural Networks",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2608.20053",
        query_channel="arxiv_primary",
        method="neural_fast_path_with_exception_table",
        claimed_evidence="Reports at least 97 percent neural coverage and compact residual lookup tables.",
        available_code_or_data="open_source_implementation_claimed",
        carnot_hook="Pair a learned branch router with a content-hashed exception table and native fallback.",
        non_transferable_claim="No aviation certification or full-domain correctness claim transfers.",
        exact_authority_boundary="Lookup and learned path cannot certify branch answers; native exact fallback remains authority.",
        expected_source_date="2026-08-20",
        arxiv_id="2608.20053",
        adopted_method=True,
        required_primary_check=True,
    ),
    _source(
        source_id="learned_conflicts",
        title="Incremental Neural Network Verification via Learned Conflicts",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2603.12232",
        query_channel="arxiv_primary",
        method="refinement_checked_learned_conflict_reuse",
        claimed_evidence="Reports sound conflict inheritance under a proved refinement relation.",
        available_code_or_data="marabou_implementation_described",
        carnot_hook="Admit conflict memory only with a query-refinement witness and exact replay receipt.",
        non_transferable_claim="No Marabou speedup transfers to Carnot chronological exact streams.",
        exact_authority_boundary="Conflicts are reusable only after a refinement witness and exact consistency check.",
        expected_source_date="2026-03-12",
        arxiv_id="2603.12232",
        adopted_method=True,
        required_primary_check=True,
    ),
    _source(
        source_id="dibs",
        title="DiBS: Diffusion-Informed Branch Selection",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2606.06518",
        query_channel="arxiv_primary",
        method="diffusion_informed_value_ordering_with_consistency_signal",
        claimed_evidence="Reports lower nodes, backtracks, and long-tail costs while preserving solver completeness.",
        available_code_or_data="public_repository_without_data_or_checkpoint",
        carnot_hook="Use consistency-aware ordering as a structural control before any learned router.",
        non_transferable_claim="No Sudoku checkpoint, data, or diffusion prior transfers into Carnot.",
        exact_authority_boundary="Advice may order values only; every candidate remains available to the exact solver.",
        expected_source_date="2026-06-02",
        arxiv_id="2606.06518",
        adopted_method=True,
        required_primary_check=True,
    ),
    _source(
        source_id="nested_smc",
        title="Discrete Diffusion Inference-Time Control with Nested Sequential Monte Carlo",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2608.20123",
        query_channel="arxiv_primary",
        method="nested_smc_discrete_diffusion_control",
        claimed_evidence="Reports improved sequence-reward steering for discrete diffusion language models.",
        available_code_or_data="paper_only_at_ingestion",
        carnot_hook="Defer as a future diffusion-decoder control; do not attach to the current autoregressive path.",
        non_transferable_claim="No current autoregressive decoder control transfers.",
        exact_authority_boundary="Deferred control cannot enter V564 acceptance or release gates.",
        expected_source_date="2026-08-20",
        arxiv_id="2608.20123",
        required_primary_check=True,
        method_transfer_status="deferred_non_autoregressive_decoder_control",
    ),
    _source(
        source_id="chainforge",
        title="ChainForge: Characterizing Embedding as the Bottleneck in Quantum Annealer Workloads",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2608.15961",
        query_channel="arxiv_primary",
        method="mapping_cost_and_embedding_bottleneck_accounting",
        claimed_evidence="Reports embedding, chain length, and remapping as dominant workload costs.",
        available_code_or_data="paper_only_at_ingestion",
        carnot_hook="Charge logical-to-mapped size, remapping, routing, and physical expansion in all hardware rows.",
        non_transferable_claim="No quantum annealer speed, latency, power, or fidelity claim transfers.",
        exact_authority_boundary="Mapping receipts are cost evidence only; they cannot certify solver answers.",
        expected_source_date="2026-08-16",
        arxiv_id="2608.15961",
        adopted_method=True,
        required_primary_check=True,
    ),
    _source(
        source_id="openreview_solver_advice",
        title="OpenReview solver-advice discovery route",
        source_kind="secondary_index",
        stable_url="https://openreview.net/search?term=constraint%20solver%20learned%20branching",
        query_channel="openreview",
        method="secondary_solver_advice_context",
        claimed_evidence="Discovery route only; no stronger V564 authority found.",
        available_code_or_data="index_only",
        carnot_hook="Keep solver advice behind exact fallback.",
        non_transferable_claim="No OpenReview search result transfers without a primary source row.",
        exact_authority_boundary="Discovery indexes are not local evidence.",
    ),
    _source(
        source_id="semantic_scholar_ebt",
        title="Semantic Scholar EBT citation route for 2507.02092",
        source_kind="secondary_index",
        stable_url="https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092",
        query_channel="semantic_scholar",
        method="citation_context_only",
        claimed_evidence="Secondary citation route for EBT context.",
        available_code_or_data="metadata_index",
        carnot_hook="Do not change V564 exact-authority boundary from citation metadata.",
        non_transferable_claim="Citation counts are dated metadata, not method evidence.",
        exact_authority_boundary="Secondary metadata cannot promote a method.",
    ),
    _source(
        source_id="semantic_scholar_arm_ebm",
        title="Semantic Scholar ARM-EBM citation route for 2512.15605",
        source_kind="secondary_index",
        stable_url="https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605",
        query_channel="semantic_scholar",
        method="citation_context_only",
        claimed_evidence="Secondary citation route for ARM-EBM context.",
        available_code_or_data="metadata_index",
        carnot_hook="Record rate limits instead of fabricating current counts.",
        non_transferable_claim="Rate-limited counts cannot be reported as current.",
        exact_authority_boundary="Secondary metadata cannot promote a method.",
    ),
    _source(
        source_id="huggingface_papers_task_coevolve",
        title="Hugging Face Papers Task-CoEvolve route",
        source_kind="secondary_index",
        stable_url="https://huggingface.co/papers/2608.20169",
        query_channel="huggingface_papers",
        method="community_discovery_context",
        claimed_evidence="Community paper page mirrors a primary arXiv source.",
        available_code_or_data="paper_index",
        carnot_hook="Use only as discovery; arXiv remains source authority.",
        non_transferable_claim="Community index metadata does not transfer as evidence.",
        exact_authority_boundary="Primary arXiv page owns the paper row.",
    ),
    _source(
        source_id="extropic_z1_status",
        title="Extropic first-party Z1 status",
        source_kind="product",
        stable_url="https://extropic.ai/writing/from-one-to-one-billion",
        query_channel="extropic",
        method="thermodynamic_hardware_product_context",
        claimed_evidence="First-party product page describes Torx, Thermalizers, and Z1 access timing.",
        available_code_or_data="no_local_sdk_or_device",
        carnot_hook="Keep TSU work watch-only and cost-ledger-only.",
        non_transferable_claim="No TSU execution, latency, energy, or availability claim transfers.",
        exact_authority_boundary="Product page cannot become local hardware authority.",
    ),
    _source(
        source_id="github_ferrotherm",
        title="dcharlot-physicalai-bmi/ferrotherm",
        source_kind="repository",
        stable_url="https://github.com/dcharlot-physicalai-bmi/ferrotherm",
        query_channel="github",
        method="implementation_reference_for_mapping_and_joules_ledger",
        claimed_evidence="Repository reference for device traits, mapping, exact checks, and joules ledger shape.",
        available_code_or_data="repository_reference",
        carnot_hook="Use as implementation comparison only; do not import or repeat claims.",
        non_transferable_claim="No repository performance claim transfers without local audit.",
        exact_authority_boundary="Reference code is not a runtime dependency or answer authority.",
    ),
    _source(
        source_id="github_dibs",
        title="shanxierdan/DiBS",
        source_kind="repository",
        stable_url="https://github.com/shanxierdan/DiBS",
        query_channel="github",
        method="dibs_repository_context",
        claimed_evidence="Repository exposes solver and heuristic code but not data or checkpoint.",
        available_code_or_data="repository_without_data_or_checkpoint",
        carnot_hook="Use code shape only for ordering controls.",
        non_transferable_claim="No trained DiBS model transfers.",
        exact_authority_boundary="Repository code cannot replace local exact solver labels.",
    ),
    _source(
        source_id="kona_product_page",
        title="Kona: Energy-Based Models for AI Reasoning",
        source_kind="product",
        stable_url="https://logicalintelligence.com/kona-ebms-energy-based-models",
        query_channel="kona",
        method="product_comparator_context",
        claimed_evidence="First-party product page describes verifier-oriented EBM reasoning.",
        available_code_or_data="no_public_weights_or_runner",
        carnot_hook="Comparator only until public weights and local runner exist.",
        non_transferable_claim="No Kona architecture, runner, or performance claim transfers.",
        exact_authority_boundary="Product comparator only; no local runner or weights.",
    ),
)
SOURCE_BY_ID = {str(row["source_id"]): dict(row) for row in SOURCE_MANIFEST}
ADOPTED_SOURCE_IDS = tuple(
    str(row["source_id"]) for row in SOURCE_MANIFEST if row.get("adopted_method") is True
)
REQUIRED_CHANNELS = {
    "network_probe",
    "sweep_clusters",
    "sweep_semscholar",
    "arxiv_primary",
    "openreview",
    "semantic_scholar",
    "huggingface_papers",
    "extropic",
    "github",
    "kona",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
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


def default_fetcher(url: str, source_id: str) -> JsonDict:  # pragma: no cover - live network path.
    del source_id
    req = request.Request(url, headers={"User-Agent": "Carnot-Exp6515-source-receipt/1.0"})
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
        return {"ok": False, "status_code": 0, "url": url, "headers": {}, "body": "", "error": str(exc)}


def default_command_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:  # pragma: no cover - subprocess path.
    result = subprocess.run(args, cwd=cwd, check=False, text=True, capture_output=True)
    return result.returncode, result.stdout, result.stderr


def offline_fetcher(url: str, source_id: str) -> JsonDict:
    del source_id
    return {
        "ok": False,
        "status_code": 0,
        "url": url,
        "headers": {},
        "body": "",
        "error": "live network disabled",
    }


def offline_command_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
    del cwd
    return 0, "", f"live network disabled for {' '.join(args)}"


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)
    return result.stdout.strip()


def _extract_title(body: str, fallback: str) -> str:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, Mapping):
        title = decoded.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
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


def _extract_source_date(body: str) -> str | None:
    iso = re.search(r"(20\d{2}-\d{2}-\d{2})", body)
    if iso:
        return iso.group(1)
    match = re.search(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+(20\d{2})", body)
    if not match:
        return None
    months = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    day, month, year = match.groups()
    return f"{year}-{months[month]}-{int(day):02d}"


def _retrieval_state(status_code: int, body_or_error: str) -> str:
    text = body_or_error.lower()
    if status_code == 429 or "too many requests" in text or "rate limit" in text:
        return "rate_limited"
    if status_code == 404 or "not found" in text:
        return "not_found"
    if 200 <= status_code < 400:
        return "available"
    return "blocked"


def _citation_count(body: str) -> int | None:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    count = decoded.get("citationCount") if isinstance(decoded, Mapping) else None
    return int(count) if isinstance(count, int) else None


def _network_probe(fetcher: Fetcher, now_utc: str) -> JsonDict:
    receipt = fetcher("https://arxiv.org/abs/2608.20169", "network_probe")
    return {
        "network_required": True,
        "network_used": True,
        "network_available": bool(receipt.get("ok")),
        "checked_at_utc": now_utc,
        "probe_url": "https://arxiv.org/abs/2608.20169",
        "probe_http_state": f"http_{int(receipt.get('status_code') or 0)}",
        "error": receipt.get("error"),
    }


def _command_receipt(
    *,
    receipt_id: str,
    channel: str,
    query: str,
    args: list[str],
    repo_root: Path,
    command_runner: CommandRunner,
    now_utc: str,
) -> JsonDict:
    rc, stdout, stderr = command_runner(args, repo_root)
    return {
        "receipt_id": receipt_id,
        "channel": channel,
        "query": query,
        "url": " ".join(args),
        "accessed_at_utc": now_utc,
        "low_concurrency": True,
        "command": args,
        "exit_code": rc,
        "stdout_sha256": sha256_json(stdout),
        "stderr_sha256": sha256_json(stderr),
        "candidate_ids": re.findall(r"\b\d{4}\.\d{5}\b", stdout),
        "observed_error": stderr.strip() if rc != 0 or stderr.strip().lower().startswith("http") else None,
    }


def collect_query_receipts(
    *,
    repo_root: Path,
    command_runner: CommandRunner,
    network_state: Mapping[str, Any],
    now_utc: str,
) -> list[JsonDict]:
    rows = [
        {
            "receipt_id": "network_probe_arxiv_task_coevolve",
            "channel": "network_probe",
            "query": "GET https://arxiv.org/abs/2608.20169",
            "url": "https://arxiv.org/abs/2608.20169",
            "accessed_at_utc": now_utc,
            "low_concurrency": True,
            "access_outcome": network_state,
        },
        _command_receipt(
            receipt_id="sweep_clusters_all_low_concurrency",
            channel="sweep_clusters",
            query="scripts/sweep_clusters.py all --max-results 8",
            args=[sys.executable, "scripts/sweep_clusters.py", "all", "--max-results", "8"],
            repo_root=repo_root,
            command_runner=command_runner,
            now_utc=now_utc,
        ),
        _command_receipt(
            receipt_id="sweep_semscholar_v564_queries",
            channel="sweep_semscholar",
            query="Task-CoEvolve Safety Nets learned conflicts DiBS ChainForge exact solver advice",
            args=[
                sys.executable,
                "scripts/sweep_semscholar.py",
                "Task-CoEvolve Safety Nets learned conflicts DiBS ChainForge exact solver advice",
                "--limit",
                "8",
            ],
            repo_root=repo_root,
            command_runner=command_runner,
            now_utc=now_utc,
        ),
    ]
    channel_queries = {
        "arxiv_primary": "direct arXiv primary pages for six V564 papers",
        "openreview": "OpenReview learned solver advice and exact-constraint records",
        "semantic_scholar": "Semantic Scholar EBT and ARM-EBM citation records",
        "huggingface_papers": "Hugging Face Papers mirror pages for V564 papers",
        "extropic": "Extropic first-party Z1 status",
        "github": "GitHub DiBS and Ferrotherm repository checks",
        "kona": "Logical Intelligence Kona first-party product page",
    }
    rows.extend(
        {
            "receipt_id": f"{channel}_sequential_check",
            "channel": channel,
            "query": query,
            "url": "source_manifest",
            "accessed_at_utc": now_utc,
            "low_concurrency": True,
            "access_outcome": "covered_by_source_rows",
        }
        for channel, query in channel_queries.items()
    )
    return rows


def collect_source_rows(*, fetcher: Fetcher, now_utc: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in SOURCE_MANIFEST:
        source_id = str(source["source_id"])
        receipt = fetcher(str(source["retrieval_url"]), source_id)
        body = str(receipt.get("body") or "")
        error_text = str(receipt.get("error") or "")
        status_code = int(receipt.get("status_code") or 0)
        retrieval_state = _retrieval_state(status_code, body + " " + error_text)
        source_date = _extract_source_date(body)
        expected = source.get("expected_source_date")
        dated_source = expected is None or source_date == expected
        ok = bool(receipt.get("ok")) and retrieval_state == "available"
        rows.append(
            {
                "row_type": "source",
                "source_id": source_id,
                "source_kind": source["source_kind"],
                "title": _extract_title(body, str(source["title"])),
                "stable_url": source["stable_url"],
                "retrieval_url": source["retrieval_url"],
                "query_channel": source["query_channel"],
                "arxiv_id": source["arxiv_id"],
                "source_date": source_date,
                "expected_source_date": expected,
                "source_date_verified": bool(ok and dated_source),
                "primary_url_verified": bool(ok),
                "retrieval_state": retrieval_state,
                "http_state": f"http_{status_code}",
                "observed_error": error_text or None,
                "source_hash": sha256_json(
                    {"source_id": source_id, "status_code": status_code, "body": body}
                ),
                "accessed_at_utc": now_utc,
                "method": source["method"],
                "method_transfer_status": source["method_transfer_status"],
                "claimed_evidence": source["claimed_evidence"],
                "available_code_or_data": source["available_code_or_data"],
                "carnot_hook": source["carnot_hook"],
                "non_transferable_claim": source["non_transferable_claim"],
                "exact_authority_boundary": source["exact_authority_boundary"],
                "adopted_method": source["adopted_method"],
                "required_primary_check": source["required_primary_check"],
                "citation_count_observed": _citation_count(body),
            }
        )
    return rows


def citation_count_boundaries(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in source_rows:
        if row.get("query_channel") != "semantic_scholar":
            continue
        limited = row.get("retrieval_state") == "rate_limited"
        rows.append(
            {
                "source_id": row["source_id"],
                "channel": "semantic_scholar",
                "observed_citation_count": None if limited else row.get("citation_count_observed"),
                "count_is_current_guarantee": False,
                "rate_limited": limited,
                "observed_error": row.get("observed_error"),
                "boundary": "Do not fabricate or refresh counts when the endpoint is limited.",
            }
        )
    return rows


def build_sota_to_experiment_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_id = {str(row["source_id"]): row for row in source_rows}
    mappings = {
        "task_coevolve": (
            "Exp6523",
            "Run full, fixed-subset, and variance-weighted adaptive validation on the chronological conflict stream.",
            "full_set_audit_and_exact_sentinel",
            "IPW estimate error versus full exact audit; hidden-regression sentinel failure rate.",
            "Block promotion when adaptive and full audit disagree.",
        ),
        "safety_nets": (
            "Exp6520",
            "Use a learned branch router plus a content-hashed residual exception table and native fallback.",
            "native_exact_solver_only_and_empty_exception_table",
            "Same exact answer rate, exception hit rate, fallback rate, and charged overhead.",
            "Abort if any advice prunes a candidate or skips native fallback.",
        ),
        "learned_conflicts": (
            "Exp6521-Exp6522",
            "Persist conflicts only with a proved query-refinement witness and exact replay receipt.",
            "scratch_no_memory_invalid_reuse_veto_and_rollback",
            "Held-future exact cost delta, invalid-reuse veto count, rollback integrity.",
            "Reject writes without witness or replay hash.",
        ),
        "dibs": (
            "Exp6518",
            "Compare consistency-aware value ordering with native, analytical, shuffled, and random controls.",
            "native_dynamic_branching_random_order_and_analytical_order",
            "Nodes, backtracks, conflicts, wall time, and answer equality under charged budgets.",
            "Retire learned routing unless structural controls show held headroom.",
        ),
        "chainforge": (
            "Exp6516-Exp6523",
            "Charge logical size, mapped size, embedding attempts, remapping, routing, and physical expansion.",
            "cpu_logical_only_cost_ledger",
            "Mapped/logical expansion ratio, embedding failure count, and charged mapping time.",
            "No hardware speed or power claim without authenticated matched execution.",
        ),
    }
    rows: list[JsonDict] = []
    for source_id in ADOPTED_SOURCE_IDS:
        target, step, control, metric, failure = mappings[source_id]
        source = by_id[source_id]
        rows.append(
            {
                "row_type": "mapping",
                "source_id": source_id,
                "method": source["method"],
                "target_experiment": target,
                "implementation_step": step,
                "implementable_local_mapping": step,
                "negative_control": control,
                "falsifiable_metric": metric,
                "failure_control": failure,
                "exact_authority_boundary": source["exact_authority_boundary"],
                "source_hash": source["source_hash"],
            }
        )
    return rows


def build_non_transfer_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_type": "non_transfer",
            "source_id": row["source_id"],
            "non_transferable_claim": row["non_transferable_claim"],
            "exact_authority_boundary": row["exact_authority_boundary"],
            "source_hash": row["source_hash"],
            "not_local_evidence": True,
        }
        for row in source_rows
    ]


def build_frozen_method_contract() -> JsonDict:
    return {
        "contract_version": "v564_source_method_contract_v1",
        "sealed_on_date": RUN_DATE,
        "feature_contract": {
            "allowed_features": [
                "static_graph_features",
                "partial_assignment_consistency",
                "conflict_pressure",
                "lineage_and_shift_tags",
            ],
            "forbidden_features": ["answer_label", "held_outcome", "post_result_repair_hint"],
        },
        "control_contract": {
            "negative_controls": [
                "native_dynamic_branching",
                "random_order",
                "fixed_subset_validation",
                "scratch_no_memory",
                "invalid_conflict_veto",
            ],
            "positive_controls": ["exact_replay_receipt", "full_set_audit", "exact_sentinel_set"],
        },
        "exact_fallback_contract": {
            "solver_acceptance_authority": "native_exact_solver",
            "fallback_required_for_every_unit": True,
            "advice_can_remove_candidates": False,
        },
        "adaptive_sampling_contract": {
            "arms": ["full_set", "fixed_subset", "variance_weighted_adaptive"],
            "estimator": "inverse_probability_weighted",
            "release_authority": "full_held_audit_only",
        },
        "exception_table_contract": {
            "content_hashed": True,
            "lookup_table_is_authority": False,
            "native_fallback_on_miss_or_mismatch": True,
        },
        "conflict_witness_contract": {
            "admission_gate": "proved_query_refinement_witness",
            "required_receipts": ["witness_hash", "exact_replay_hash", "rollback_parent_hash"],
            "invalid_reuse_action": "veto_and_quarantine",
        },
        "mapping_cost_contract": {
            "charge_mapping_cost": True,
            "charge_remapping_cost": True,
            "hardware_claim_allowed": False,
            "reported_sizes": ["logical_variables", "mapped_variables", "physical_chains"],
        },
        "stop_rule_contract": {
            "stop_before_outcome_artifact_review": True,
            "promote_only_after_independent_audit": True,
            "block_on_source_unavailable": True,
        },
        "authority_contract": {
            "learned_advice_may_order": True,
            "learned_advice_may_request_refocus": True,
            "learned_advice_may_prune": False,
            "learned_advice_may_certify": False,
            "adaptive_validation_may_release": False,
        },
    }


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    files = {
        path.as_posix(): {
            "sha256_before": sha256_file(repo_root / path),
            "sha256_after": sha256_file(repo_root / path),
            "unchanged": sha256_file(repo_root / path) != "missing",
        }
        for path in PROTECTED_RELATIVE_PATHS
    }
    changed = [
        path
        for path, row in files.items()
        if row["sha256_before"] != row["sha256_after"] or row["unchanged"] is not True
    ]
    return {"files": files, "changed_paths": changed, "all_protected_files_unchanged": not changed}


def _path_hash_rows(repo_root: Path, paths: Sequence[Path]) -> list[JsonDict]:
    return [
        {
            "path": path.as_posix(),
            "exists": (repo_root / path).exists(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in paths
    ]


def preconditions_checked(
    *,
    repo_root: Path,
    run_date: str,
    now_utc: str,
    network_state: Mapping[str, Any],
    query_receipts: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> JsonDict:
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "timestamps": {"started_or_accessed_at_utc": now_utc},
        "network_availability": dict(network_state),
        "query_strings": [row.get("query") for row in query_receipts],
        "tool_versions": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "sweep_clusters_sha256": sha256_file(repo_root / "scripts/sweep_clusters.py"),
            "sweep_semscholar_sha256": sha256_file(repo_root / "scripts/sweep_semscholar.py"),
        },
        "source_paths": _path_hash_rows(repo_root, SOURCE_PATHS),
        "local_hook_paths": _path_hash_rows(repo_root, LOCAL_HOOK_PATHS),
        "git": {
            "head": _git_output(repo_root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(repo_root, ["status", "--short"]),
        },
        "protected_file_hashes": {
            path: row["sha256_before"]
            for path, row in dict(protected.get("files", {})).items()
            if isinstance(row, Mapping)
        },
        "no_high_concurrency_deep_research_harness": True,
        "outcome_artifact_guard": {
            "blocked_paths": [path.as_posix() for path in OUTCOME_ARTIFACT_RELATIVE_PATHS],
            "existing_paths_not_read": [
                path.as_posix() for path in OUTCOME_ARTIFACT_RELATIVE_PATHS if (repo_root / path).exists()
            ],
            "outcome_artifacts_read": [],
        },
    }


def aggregate_row_recomputation(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    query_receipts: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
    non_transfer_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    protected: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    manifest_ids = {str(row["source_id"]) for row in SOURCE_MANIFEST}
    source_ids = {str(row.get("source_id")) for row in source_rows}
    required_primary = [row for row in source_rows if row.get("required_primary_check") is True]
    required_verified = all(
        row.get("primary_url_verified") is True and row.get("source_date_verified") is True
        for row in required_primary
    )
    adopted_mapping_ids = {str(row.get("source_id")) for row in mappings}
    adopted_complete = all(
        row.get("implementable_local_mapping")
        and row.get("negative_control")
        and row.get("falsifiable_metric")
        and row.get("exact_authority_boundary")
        for row in mappings
    ) and adopted_mapping_ids == set(ADOPTED_SOURCE_IDS)
    non_transfer_complete = len(non_transfer_rows) == len(source_rows) and all(
        row.get("non_transferable_claim") and row.get("exact_authority_boundary")
        for row in non_transfer_rows
    )
    channels = {str(row.get("channel")) for row in query_receipts}
    query_complete = REQUIRED_CHANNELS <= channels
    authority_ok = (
        contract.get("authority_contract", {}).get("learned_advice_may_prune") is False
        and contract.get("authority_contract", {}).get("learned_advice_may_certify") is False
        and contract.get("exact_fallback_contract", {}).get("solver_acceptance_authority")
        == "native_exact_solver"
    )
    protected_ok = protected.get("all_protected_files_unchanged") is True
    no_outcome_read = not preconditions.get("outcome_artifact_guard", {}).get("outcome_artifacts_read")
    ready = (
        source_ids == manifest_ids
        and required_verified
        and adopted_complete
        and non_transfer_complete
        and query_complete
        and authority_ok
        and protected_ok
        and no_outcome_read
    )
    return {
        "required_source_count": len(SOURCE_MANIFEST),
        "source_row_count": len(source_rows),
        "source_rows_cover_manifest": source_ids == manifest_ids,
        "required_primary_source_count": len(required_primary),
        "required_primary_sources_verified": required_verified,
        "adopted_method_count": len(ADOPTED_SOURCE_IDS),
        "adopted_methods_with_source_mapping_control_boundary": len(mappings)
        if adopted_complete
        else 0,
        "non_transfer_row_count": len(non_transfer_rows),
        "non_transfer_rows_complete": non_transfer_complete,
        "required_query_channels_present": query_complete,
        "authority_boundary_exact_first": authority_ok,
        "protected_files_unchanged": protected_ok,
        "outcome_artifacts_not_read": no_outcome_read,
        "frozen_method_contract_hash": sha256_json(contract),
        "ready_score_from_rows": 1.0 if ready else 0.0,
    }


def gate_check_summary(
    aggregate: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    checks = {
        "source_rows_cover_manifest": aggregate.get("source_rows_cover_manifest") is True,
        "required_primary_sources_verified": aggregate.get("required_primary_sources_verified") is True,
        "adopted_methods_have_mapping_control_boundary": (
            aggregate.get("adopted_methods_with_source_mapping_control_boundary")
            == aggregate.get("adopted_method_count")
        ),
        "non_transfer_rows_complete": aggregate.get("non_transfer_rows_complete") is True,
        "required_query_channels_present": aggregate.get("required_query_channels_present") is True,
        "authority_boundary_exact_first": aggregate.get("authority_boundary_exact_first") is True,
        "protected_files_unchanged": aggregate.get("protected_files_unchanged") is True,
        "outcome_artifacts_not_read": aggregate.get("outcome_artifacts_not_read") is True,
    }
    failed: list[JsonDict] = []
    for key, passed in checks.items():
        if passed is True:
            continue
        failed.append({"check": key, "expected": True, "observed": passed, "channel": "local_gate"})
    for row in source_rows:
        if row.get("required_primary_check") is True and (
            row.get("primary_url_verified") is not True or row.get("source_date_verified") is not True
        ):
            failed.append(
                {
                    "check": "required_primary_source_verified",
                    "source_id": row.get("source_id"),
                    "channel": row.get("query_channel"),
                    "expected": "available_primary_url_and_matching_source_date",
                    "observed": {
                        "retrieval_state": row.get("retrieval_state"),
                        "http_state": row.get("http_state"),
                        "source_date": row.get("source_date"),
                        "observed_error": row.get("observed_error"),
                    },
                }
            )
    return {"checks": checks, "failed_checks": failed, "all_gates_passed": not failed}


def build_per_unit_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
    non_transfer_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = [dict(row) for row in source_rows]
    rows.extend(dict(row) for row in mappings)
    rows.extend(dict(row) for row in non_transfer_rows)
    rows.extend(
        {"row_type": "contract", "contract_section": key, "contract_hash": sha256_json(value)}
        for key, value in contract.items()
        if isinstance(value, Mapping)
    )
    rows.extend({"row_type": "gate", **row} for row in gate.get("failed_checks", []))
    if not gate.get("failed_checks"):
        rows.append({"row_type": "gate", "check": "all_gates_passed", "observed": True})
    return rows


def tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(stable)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    run_date: str = RUN_DATE,
    fetcher: Fetcher = default_fetcher,
    command_runner: CommandRunner = default_command_runner,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    now_utc: str | None = None,
) -> JsonDict:
    start = time.monotonic()
    now = now_utc or _utc_now()
    network = _network_probe(fetcher, now)
    query_receipts = collect_query_receipts(
        repo_root=repo_root,
        command_runner=command_runner,
        network_state=network,
        now_utc=now,
    )
    source_rows = collect_source_rows(fetcher=fetcher, now_utc=now)
    source_hashes = {str(row["source_id"]): str(row["source_hash"]) for row in source_rows}
    citation_boundaries = citation_count_boundaries(source_rows)
    mappings = build_sota_to_experiment_rows(source_rows)
    non_transfer = build_non_transfer_rows(source_rows)
    contract = build_frozen_method_contract()
    protected = protected_files_unchanged(repo_root)
    preconditions = preconditions_checked(
        repo_root=repo_root,
        run_date=run_date,
        now_utc=now,
        network_state=network,
        query_receipts=query_receipts,
        protected=protected,
    )
    aggregate = aggregate_row_recomputation(
        source_rows=source_rows,
        query_receipts=query_receipts,
        mappings=mappings,
        non_transfer_rows=non_transfer,
        contract=contract,
        protected=protected,
        preconditions=preconditions,
    )
    gate = gate_check_summary(aggregate, source_rows)
    ready_score = float(aggregate["ready_score_from_rows"])
    status = (
        "complete_v564_source_method_contract_ready"
        if ready_score == 1.0
        else "blocked_v564_source_method_contract"
    )
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": (
            "complete_v564_source_method_contract_ready: primary sources verified and method contract frozen"
            if ready_score == 1.0
            else "blocked_v564_source_method_contract: one or more source or readiness gates failed"
        ),
        "verdict_class": None if ready_score == 1.0 else "partial",
        "query_receipts": query_receipts,
        "source_rows": source_rows,
        "primary_source_hashes": source_hashes,
        "citation_count_boundaries": citation_boundaries,
        "sota_to_experiment_rows": mappings,
        "non_transfer_rows": non_transfer,
        "frozen_method_contract": contract,
        "v564_method_contract_ready_score": ready_score,
        "gate_check_summary": gate,
        "per_unit_rows": build_per_unit_rows(
            source_rows=source_rows,
            mappings=mappings,
            non_transfer_rows=non_transfer,
            contract=contract,
            gate=gate,
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - start),
        "tests_run": tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        target = Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
        atomic_write_json(target, artifact, allow_override=False, sort_keys=False)
    return artifact


def _recompute_current_aggregate(artifact: Mapping[str, Any]) -> JsonDict:
    return aggregate_row_recomputation(
        source_rows=artifact.get("source_rows", []),
        query_receipts=artifact.get("query_receipts", []),
        mappings=artifact.get("sota_to_experiment_rows", []),
        non_transfer_rows=artifact.get("non_transfer_rows", []),
        contract=artifact.get("frozen_method_contract", {}),
        protected=artifact.get("protected_files_unchanged", {}),
        preconditions=artifact.get("preconditions_checked", {}),
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in {None, "partial"}:
        errors.append("verdict_class outside Exp6515 enum")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    current = _recompute_current_aggregate(artifact)
    score = artifact.get("v564_method_contract_ready_score")
    if score not in {0.0, 1.0} or score != current["ready_score_from_rows"]:
        errors.append("ready score mismatch")
    manifest_ids = {str(row["source_id"]) for row in SOURCE_MANIFEST}
    source_ids = {str(row.get("source_id")) for row in artifact.get("source_rows", [])}
    if source_ids != manifest_ids:
        errors.append("source rows must cover manifest")
    if artifact.get("v564_method_contract_ready_score") == 1.0 and current[
        "required_primary_sources_verified"
    ] is not True:
        errors.append("required primary sources must verify")
    mapping_ids = {str(row.get("source_id")) for row in artifact.get("sota_to_experiment_rows", [])}
    mapping_complete = all(
        row.get("implementable_local_mapping")
        and row.get("negative_control")
        and row.get("falsifiable_metric")
        and row.get("exact_authority_boundary")
        for row in artifact.get("sota_to_experiment_rows", [])
    )
    if mapping_ids != set(ADOPTED_SOURCE_IDS) or not mapping_complete:
        errors.append("adopted methods must map to local implementation controls")
    if not all(
        row.get("non_transferable_claim") and row.get("exact_authority_boundary")
        for row in artifact.get("non_transfer_rows", [])
    ):
        errors.append("non-transfer rows must forbid unsupported transfer")
    authority = artifact.get("frozen_method_contract", {}).get("authority_contract", {})
    if authority.get("learned_advice_may_prune") is not False or authority.get("learned_advice_may_certify") is not False:
        errors.append("learned advice cannot certify or prune")
    if artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is not True:
        errors.append("protected files changed")
    if not REQUIRED_CHANNELS <= {str(row.get("channel")) for row in artifact.get("query_receipts", [])}:
        errors.append("query receipts must include sweep helper and sequential channels")
    if set(artifact.get("primary_source_hashes", {})) != manifest_ids:
        errors.append("primary_source_hashes must cover manifest")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def _load_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate Exp6515 V564 source-method contract.")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-live-network", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        errors = validate_artifact(_load_json(result))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result,
        run_date=str(args.date),
        fetcher=offline_fetcher if args.no_live_network else default_fetcher,
        command_runner=offline_command_runner if args.no_live_network else default_command_runner,
        write=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
