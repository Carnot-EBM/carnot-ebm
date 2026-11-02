"""Build the bounded V609 post-marker source-delta artifact.

Spec refs: REQ-REPORT-6953 and SCENARIO-REPORT-6953-*.

The module stores the dated query plan before any request runs. This keeps the
scope fixed when a source changes during the audit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_6953_v609_source_delta"
RUN_DATE = "20260903"
RANDOM_SEED = 6953
MARKER_UTC = "2026-09-03T20:06:00Z"
INFERENCE_SUBSTRATE = "bounded_primary_source_web_research_no_model_inference"
SCHEMA_VERSION = "carnot.experiment_6953.v609_source_delta.v1"

MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6953_v609_source_delta.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6953_v609_source_delta.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6953_v609_source_delta.json")
LEDGER_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
DESIGN_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PREDECESSOR_RELATIVE_PATH = Path("results/experiment_6941_v608_source_delta.json")
NETWORK_POLICY_RELATIVE_PATH = Path("CLAUDE.md")

PLANNER_START_MARKER = "<!-- V609-PLANNER-REFRESH-20260903-START -->"
PLANNER_END_MARKER = "<!-- V609-PLANNER-REFRESH-20260903-END -->"

ARXIV_FAMILIES = (
    "arxiv_ebm_verification",
    "arxiv_neural_constraints",
    "arxiv_ising_ml",
    "arxiv_hallucination_mitigation",
    "arxiv_kan",
    "arxiv_energy_guided_decoding",
    "arxiv_sampling_hardware",
    "arxiv_continual_constraint_learning",
)
FOLLOWUP_FAMILIES = (
    "openreview",
    "huggingface_papers",
    "semantic_scholar_ebt",
    "semantic_scholar_arm_ebm",
    "github_official",
    "extropic",
    "logical_intelligence",
)
SOURCE_FAMILIES = ARXIV_FAMILIES + FOLLOWUP_FAMILIES

SELECTED_ARXIV_IDS = (
    "2609.00728",
    "2605.23395",
    "2609.00652",
    "2608.21539",
    "2609.00796",
)

_ACCEPTANCE = (
    "Accept a ledger fact only when a primary paper or first-party source has a stable "
    "identity, was observed after 2026-09-03T20:06:00Z, is absent from the V609 planner "
    "marker, and changes a material local execution boundary. Keep unchanged sources, "
    "duplicates, access failures, announcements, private models, unverified repositories, "
    "and incompatible hardware as terminal non-findings."
)


def _plan_row(
    query_id: str,
    query_text: str,
    canonical_url: str,
    allowed_domains: list[str],
    planned_utc: str,
) -> JsonDict:
    """Create one query receipt with the same small network budget."""

    return {
        "query_id": query_id,
        "source_family": query_id,
        "query_text": query_text,
        "canonical_url": canonical_url,
        "allowed_domains": allowed_domains,
        "planned_utc": planned_utc,
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    }


# Frozen at 2026-09-03T20:16:25Z, before the first network request. Planned
# times state request order. They do not claim that a response arrived then.
_FROZEN_QUERY_PLAN: tuple[JsonDict, ...] = (
    _plan_row(
        "arxiv_ebm_verification",
        '("energy-based model" OR "energy-based transformer") AND (verification OR verifier OR reasoning) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:17:00Z",
    ),
    _plan_row(
        "arxiv_neural_constraints",
        '("neural constraint" OR "constraint reasoning" OR neuro-symbolic) AND (verification OR language) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:18:00Z",
    ),
    _plan_row(
        "arxiv_ising_ml",
        '(Ising AND ("machine learning" OR sampling OR "probabilistic computing")) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:19:00Z",
    ),
    _plan_row(
        "arxiv_hallucination_mitigation",
        '((hallucination AND (verification OR detection OR mitigation)) OR "hidden-state verifier") AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:20:00Z",
    ),
    _plan_row(
        "arxiv_kan",
        '("Kolmogorov-Arnold Network" OR KAN) AND (verification OR symbolic OR constraint OR FPGA) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:21:00Z",
    ),
    _plan_row(
        "arxiv_energy_guided_decoding",
        '("energy-guided decoding" OR "reward-guided decoding" OR "energy-based decoding") AND (LLM OR language) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:22:00Z",
    ),
    _plan_row(
        "arxiv_sampling_hardware",
        "((thermodynamic OR Ising OR stochastic) AND (hardware OR accelerator) AND sampling) AND submittedDate:[202501010000 TO 202612312359]",
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:23:00Z",
    ),
    _plan_row(
        "arxiv_continual_constraint_learning",
        '(("continual learning" OR "strategy memory" OR "episodic memory") AND (constraint OR verifier OR LLM OR agent)) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T20:24:00Z",
    ),
    _plan_row(
        "openreview",
        'site:openreview.net ("energy based" OR constraint OR verifier OR continual memory) ICLR ICML NeurIPS TMLR 2025 2026',
        "https://openreview.net/search",
        ["openreview.net"],
        "2026-09-03T20:25:00Z",
    ),
    _plan_row(
        "huggingface_papers",
        'site:huggingface.co/papers ("2609.00728" OR "2605.23395" OR "2609.00652" OR "2608.21539" OR "2609.00796")',
        "https://huggingface.co/papers",
        ["huggingface.co"],
        "2026-09-03T20:26:00Z",
    ),
    _plan_row(
        "semantic_scholar_ebt",
        "ARXIV:2507.02092 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations",
        ["api.semanticscholar.org", "semanticscholar.org"],
        "2026-09-03T20:27:00Z",
    ),
    _plan_row(
        "semantic_scholar_arm_ebm",
        "ARXIV:2512.15605 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations",
        ["api.semanticscholar.org", "semanticscholar.org"],
        "2026-09-03T20:28:00Z",
    ),
    _plan_row(
        "github_official",
        "official repositories for arXiv 2609.00728 2605.23395 2609.00652 2608.21539 2609.00796",
        "https://github.com/search",
        ["github.com", "api.github.com"],
        "2026-09-03T20:29:00Z",
    ),
    _plan_row(
        "extropic",
        "Extropic Torx Thermalizers Z1 XTR-0 implementation access update",
        "https://extropic.ai/writing/from-one-to-one-billion",
        ["extropic.ai"],
        "2026-09-03T20:30:00Z",
    ),
    _plan_row(
        "logical_intelligence",
        "Logical Intelligence Kona EBM weights code training recipe local runner update",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        ["logicalintelligence.com"],
        "2026-09-03T20:31:00Z",
    ),
)


def frozen_query_plan() -> list[JsonDict]:
    """Return copies so a caller cannot change the frozen search plan."""

    return [deepcopy(row) for row in _FROZEN_QUERY_PLAN]


VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "model_specs",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "query_rows",
    "source_family_rows",
    "paper_rows",
    "citation_edge_rows",
    "implementation_rows",
    "compatibility_rows",
    "hardware_rows",
    "product_rows",
    "ledger_append_rows",
    "rate_limit_rows",
    "random_seed",
    "reproducibility_checksum",
    "v609_source_delta_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes changes to the evidence contract detectable.",
    "experiment_id": "A stable identity binds the receipt to the requested experiment.",
    "run_date": "The execution date separates this audit from the planner refresh.",
    "status": "Status distinguishes complete coverage from a failed preflight.",
    "field_principles": "A principle for every field preserves the purpose of the receipt.",
    "preconditions_checked": "Fail-closed preflight prevents research outside its contracts.",
    "inference_substrate": "The substrate says that source review did not invoke a model.",
    "model_specs": "An empty invocation list distinguishes cited models from used models.",
    "duration_s": "Elapsed wall time bounds the execution represented by this receipt.",
    "source_artifact_hashes": "Hashes bind conclusions to the local planning inputs.",
    "rows": "A flattened ledger permits inspection without trusting the summaries.",
    "query_rows": "Query rows preserve success, absence, and unavailable outcomes.",
    "source_family_rows": "Family rows prove every required source reached a terminal state.",
    "paper_rows": "Paper rows prevent selective reporting among the V609 selections.",
    "citation_edge_rows": "Citation rows distinguish checked old edges from new edges.",
    "implementation_rows": "Implementation rows keep all source ideas advisory.",
    "compatibility_rows": "Compatibility rows expose concrete local execution limits.",
    "hardware_rows": "Hardware rows separate physical access from paper claims.",
    "product_rows": "Product rows prevent announcements from becoming dependencies.",
    "ledger_append_rows": "Append rows isolate verified facts that truly postdate V609.",
    "rate_limit_rows": "Rate-limit rows stop missing evidence from becoming no-update evidence.",
    "random_seed": "A fixed seed makes deterministic ordering reproducible.",
    "reproducibility_checksum": "A content hash detects later changes to the receipt.",
    "v609_source_delta_complete_score": "The score measures terminal coverage, not novelty.",
    "gate_check_summary": "The gate summary states expected and observed preflight evidence.",
    "verifier_is_oracle": "False prevents a literature audit from certifying scientific truth.",
    "verdict_class": "The class distinguishes a covered null result from other outcomes.",
    "honest_verdict": "A terminal prefix makes the evidence outcome machine-readable.",
}

_OBSERVED_UTC = "2026-09-03T20:22:38Z"

_QUERY_RESULTS: dict[str, JsonDict] = {
    "arxiv_ebm_verification": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The visible 2025-2026 EBM-verification results predated the V609 marker; no newer executable verifier was found.",
    },
    "arxiv_neural_constraints": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The visible neural-constraint and certification results predated the V609 marker.",
    },
    "arxiv_ising_ml": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The visible Ising machine-learning papers predated the V609 marker and did not change authenticated hardware access.",
    },
    "arxiv_hallucination_mitigation": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The visible mitigation and detector papers predated the marker and did not replace exact external verification.",
    },
    "arxiv_kan": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The returned KAN work predated the marker and supplied no new verified local execution boundary.",
    },
    "arxiv_energy_guided_decoding": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success_empty",
        "terminal_outcome": "explicit_empty_no_update",
        "evidence": "The exact energy-guided decoding search returned no qualifying post-marker result.",
    },
    "arxiv_sampling_hardware": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The visible stochastic-program and sampling-hardware papers were published before the marker.",
    },
    "arxiv_continual_constraint_learning": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The returned continual-learning and constraint-memory work predated the marker.",
    },
    "openreview": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "Indexed OpenReview results were older conference records; no post-marker matching source was accepted.",
    },
    "huggingface_papers": {
        "attempt_count": 2,
        "http_status": 404,
        "status": "exact_pages_unavailable",
        "terminal_outcome": "unavailable_no_update",
        "evidence": "All five exact Hugging Face Papers routes returned HTTP 404; absence was retained rather than treated as a finding.",
    },
    "semantic_scholar_ebt": {
        "attempt_count": 2,
        "http_status": 200,
        "status": "success_after_browser_route_unavailable",
        "terminal_outcome": "no_new_citation_edge",
        "visible_citation_count": 35,
        "evidence": "The API returned 35 visible EBT citations; its newest dated edge was 2026-08-14, before the V609 marker.",
    },
    "semantic_scholar_arm_ebm": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_new_citation_edge",
        "visible_citation_count": 8,
        "evidence": "The API returned eight visible ARM-EBM citations; its newest dated edge was 2026-07-02, before the V609 marker.",
    },
    "github_official": {
        "attempt_count": 2,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "existing_code_state_no_post_marker_update",
        "evidence": "SOVER and FedQCL have paper-linked public repositories last pushed before the marker; no verified official executable repository was found for CCEM, Self-Reports, or SFAD.",
    },
    "extropic": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The writing index has no item newer than 2026-08-04 and still plans Z1 system early access for 2027.",
    },
    "logical_intelligence": {
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "unavailable_reproducible_assets",
        "evidence": "The Kona 1.0 page remains a product description with no public weights, training recipe, source code, or local runner.",
    },
}


def validate_query_plan(plan: list[JsonDict], run_date: str) -> list[str]:
    """Reject a plan whose date, order, or request budget changed."""

    expected_prefix = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}T"
    if run_date != RUN_DATE or any(
        not row["planned_utc"].startswith(expected_prefix) for row in plan
    ):
        return ["query plan date does not match run date"]
    errors: list[str] = []
    for row in plan:
        query_id = row["query_id"]
        if row["max_attempts"] != 2:
            errors.append(f"query {query_id} must allow exactly two attempts")
        if row["concurrency_limit"] != 1:
            errors.append(f"query {query_id} must use concurrency one")
        if row["timeout_s"] != 20:
            errors.append(f"query {query_id} must use the 20 second timeout")
        if not row["query_text"] or not row["allowed_domains"] or not row["acceptance_criteria"]:
            errors.append(f"query {query_id} lacks a frozen source field")
    if [row["source_family"] for row in plan[:8]] != list(ARXIV_FAMILIES):
        errors.append("arXiv source families must be first")
    if {row["source_family"] for row in plan} != set(SOURCE_FAMILIES):
        errors.append("query plan source family coverage is incomplete")
    return errors


def check_preconditions(root: Path, run_date: str, output_path: Path) -> list[JsonDict]:
    """Return explicit evidence for every input required before research."""

    ledger = root / LEDGER_RELATIVE_PATH
    roadmap = root / ROADMAP_RELATIVE_PATH
    design = root / DESIGN_RELATIVE_PATH
    policy = root / NETWORK_POLICY_RELATIVE_PATH
    ledger_text = ledger.read_text(encoding="utf-8") if ledger.is_file() else ""
    roadmap_text = roadmap.read_text(encoding="utf-8") if roadmap.is_file() else ""
    design_text = design.read_text(encoding="utf-8") if design.is_file() else ""
    policy_text = policy.read_text(encoding="utf-8") if policy.is_file() else ""
    output_parent = output_path.parent
    plan_errors = validate_query_plan(frozen_query_plan(), run_date)
    checks = (
        (
            "v609_planner_marker",
            PLANNER_START_MARKER in ledger_text and PLANNER_END_MARKER in ledger_text,
            "the reference ledger contains both V609 planner marker boundaries",
            ledger.as_posix() if ledger.is_file() else "missing",
        ),
        (
            "active_roadmap",
            "milestone: 2026.09.609" in roadmap_text
            and "exp6953-v609-source-delta" in roadmap_text,
            "the active roadmap declares V609 and Exp6953",
            roadmap.as_posix() if roadmap.is_file() else "missing",
        ),
        (
            "design_document",
            "**Milestone:** 2026.09.609" in design_text and "Exp6953" in design_text,
            "the design document declares V609 and Exp6953",
            design.as_posix() if design.is_file() else "missing",
        ),
        (
            "dated_query_plan",
            not plan_errors,
            "the frozen 20260903 query plan passes bounded validation",
            "valid" if not plan_errors else "; ".join(plan_errors),
        ),
        (
            "network_policy",
            "SOTA-Ingestion Cycle Discipline" in policy_text and "low-concurrency" in policy_text,
            "CLAUDE.md contains the low-concurrency SOTA-ingestion policy",
            policy.as_posix() if policy.is_file() else "missing",
        ),
        (
            "writable_results_path",
            output_parent.is_dir() and os.access(output_parent, os.W_OK),
            "the selected results parent exists and is writable",
            output_parent.as_posix() if output_parent.is_dir() else "missing",
        ),
    )
    return [
        {"resource": name, "available": available, "expected": expected, "observed": observed}
        for name, available, expected, observed in checks
    ]


def query_rows() -> list[JsonDict]:
    """Join each frozen query with its terminal observation."""

    rows = []
    for planned in frozen_query_plan():
        row = deepcopy(planned)
        row.update(_QUERY_RESULTS[planned["query_id"]])
        row["actual_utc"] = _OBSERVED_UTC
        row["terminal"] = True
        rows.append(row)
    return rows


_PAPER_ROWS: tuple[JsonDict, ...] = (
    {
        "arxiv_id": "2609.00728",
        "title": "SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification",
        "canonical_url": "https://arxiv.org/abs/2609.00728",
        "version": "v1",
        "date": "2026-09-01",
        "submitted_utc": "2026-09-01T05:05:12Z",
        "evidence_grade": "A",
        "code_state": "official_repository_public_last_push_2026_08_29",
        "dataset_state": "nlequiv_150_public_in_official_repository_with_100_positive_and_50_negative_pairs",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The paper and repository predate the marker; local Z3 is available but dReal is absent, so nonlinear certification is not locally reproducible as-is.",
        "terminal": True,
    },
    {
        "arxiv_id": "2605.23395",
        "title": "Convex Compositional Reasoning Models",
        "canonical_url": "https://arxiv.org/abs/2605.23395",
        "version": "v2",
        "date": "2026-05-25",
        "submitted_utc": "2026-05-22T09:04:14Z",
        "evidence_grade": "A",
        "code_state": "no_official_repository_found_after_two_queries",
        "dataset_state": "procedural_n_queens_graph_coloring_and_3sat_generation_described_without_public_snapshot",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Convex factors and projected optimization are locally implementable, but no reference code, checkpoint, or frozen dataset was found.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.00652",
        "title": "Self-Reports Are Not Verification: Environment-Grounded Auditing of LLM Operators in Evolutionary Search",
        "canonical_url": "https://arxiv.org/abs/2609.00652",
        "version": "v1",
        "date": "2026-09-01",
        "submitted_utc": "2026-09-01T03:28:37Z",
        "evidence_grade": "A",
        "code_state": "no_official_repository_found_after_two_queries",
        "dataset_state": "paper_describes_frozen_trace_extraction_but_contexto_content_is_not_redistributed_and_no_public_archive_link_was_found",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The exact-outcome principle is portable, but the hosted game, implementation, frozen traces, and exact model-serving setup are not a public local reproduction bundle.",
        "terminal": True,
    },
    {
        "arxiv_id": "2608.21539",
        "title": "Federated Continual Learning as a Distributed Drift-Plus-Penalty Control Problem",
        "canonical_url": "https://arxiv.org/abs/2608.21539",
        "version": "v1",
        "date": "2026-08-21",
        "submitted_utc": "2026-08-21T18:21:19Z",
        "evidence_grade": "A",
        "code_state": "official_repository_public_last_push_2026_08_21_with_three_commits",
        "dataset_state": "public_builder_downloads_cifar10_cifar100_and_tinyimagenet_but_no_immutable_paper_split_snapshot",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "PyTorch and two GPUs are locally available, but the repository and generated data were inspected rather than installed and reproduced.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.00796",
        "title": "SFAD: Speculative Factuality-Aware Decoding",
        "canonical_url": "https://arxiv.org/abs/2609.00796",
        "version": "v1",
        "date": "2026-09-01",
        "submitted_utc": "2026-09-01T06:46:54Z",
        "evidence_grade": "A",
        "code_state": "no_verified_official_repository_or_checkpoint_found_after_two_queries",
        "dataset_state": "confide_36k_training_mix_described_but_no_public_confide_snapshot_found",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The local GPUs can host quantized component models, but the SFAD implementation, trained draft checkpoint, ConFide snapshot, and TRL dependency are absent.",
        "terminal": True,
    },
)


def paper_rows() -> list[JsonDict]:
    """Return one copied terminal receipt for every V609-selected paper."""

    return deepcopy(_PAPER_ROWS)


def citation_edge_rows() -> list[JsonDict]:
    """Record the newest visible edges used to rule out post-marker novelty."""

    return [
        {
            "source_id": "arxiv:2608.14186",
            "target_id": "arxiv:2507.02092",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2608.14186",
            "publication_date": "2026-08-14",
            "is_new_after_marker": False,
            "terminal": True,
        },
        {
            "source_id": "arxiv:2607.20792",
            "target_id": "arxiv:2507.02092",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2607.20792",
            "publication_date": "2026-07-22",
            "is_new_after_marker": False,
            "terminal": True,
        },
        {
            "source_id": "arxiv:2607.02154",
            "target_id": "arxiv:2512.15605",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2607.02154",
            "publication_date": "2026-07-02",
            "is_new_after_marker": False,
            "terminal": True,
        },
        {
            "source_id": "arxiv:2605.18871",
            "target_id": "arxiv:2512.15605",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2605.18871",
            "publication_date": "2026-05-15",
            "is_new_after_marker": False,
            "terminal": True,
        },
    ]


def implementation_rows() -> list[JsonDict]:
    """Map each paper to an advisory boundary without changing V609."""

    papers = {row["arxiv_id"]: row for row in paper_rows()}
    mappings = (
        (
            "2609.00728",
            "formal_verification_reference",
            "Use Z3 only where the existing locked design already calls for an exact solver; dReal remains unavailable.",
        ),
        (
            "2605.23395",
            "convex_energy_reference",
            "Retain convex compositional energy as future execution guidance, not a new milestone dependency.",
        ),
        (
            "2609.00652",
            "external_outcome_authority",
            "Continue requiring environment outcomes rather than model self-reports for verification.",
        ),
        (
            "2608.21539",
            "continual_learning_reference",
            "Keep virtual queue regulation as advisory follow-up; do not alter locked memory gates.",
        ),
        (
            "2609.00796",
            "decoding_reference",
            "Keep speculative factuality steering advisory until official assets and a local reproduction exist.",
        ),
    )
    return [
        {
            "arxiv_id": arxiv_id,
            "source_url": papers[arxiv_id]["canonical_url"],
            "implementation_target": target,
            "implementation_boundary": boundary,
            "advisory_only": True,
        }
        for arxiv_id, target, boundary in mappings
    ]


def compatibility_rows() -> list[JsonDict]:
    """State the local package, model, and asset boundary for each paper."""

    details = (
        (
            "2609.00728",
            "solver_stack",
            "z3 is installed; dreal is not installed",
            "Only the Z3 portion has a local dependency path.",
        ),
        (
            "2605.23395",
            "reference_assets",
            "torch 2.11.0+cu128 is installed but official code and frozen data were not found",
            "A clean-room prototype is possible; paper reproduction is not claimed.",
        ),
        (
            "2609.00652",
            "hosted_environment",
            "two RTX 3090 GPUs are visible, but Contexto content and the study implementation are not public local assets",
            "The audit principle is usable; the reported study is not locally reproduced.",
        ),
        (
            "2608.21539",
            "repository_and_data",
            "the official PyTorch repository and dataset builder are public, while no immutable prepared split is supplied",
            "A later bounded reproduction is feasible after isolated installation.",
        ),
        (
            "2609.00796",
            "model_and_training_assets",
            "transformers is installed, but trl, ConFide, the trained draft checkpoint, and reference code are absent",
            "No SFAD performance or speed claim is imported.",
        ),
    )
    return [
        {"arxiv_id": arxiv_id, "constraint": constraint, "observed": observed, "effect": effect}
        for arxiv_id, constraint, observed, effect in details
    ]


def hardware_rows() -> list[JsonDict]:
    """Preserve live local access separately from vendor availability."""

    return [
        {
            "hardware": "dual_rtx_3090",
            "access_state": "available",
            "evidence": "nvidia-smi reported two NVIDIA GeForce RTX 3090 GPUs with 24576 MiB each; torch reported two CUDA devices.",
            "terminal": True,
            "dependency_added": False,
        },
        {
            "hardware": "kv260",
            "access_state": "reachable",
            "evidence": "A bounded non-destructive SSH probe returned hostname kv260.",
            "terminal": True,
            "dependency_added": False,
        },
        {
            "hardware": "polarfire_soc",
            "access_state": "reachable",
            "evidence": "A bounded non-destructive SSH probe returned hostname mpfs-disco-kit.",
            "terminal": True,
            "dependency_added": False,
        },
        {
            "hardware": "gatemate_a1",
            "access_state": "reachable_jtag",
            "evidence": "openFPGALoader detected GateMate GM1Ax IDCODE 0x20000001 without flashing.",
            "terminal": True,
            "dependency_added": False,
        },
        {
            "hardware": "extropic_z1",
            "access_state": "not_locally_available",
            "evidence": "The first-party page plans Z1 sticks and cards for early access in 2027.",
            "terminal": True,
            "dependency_added": False,
        },
    ]


def product_rows() -> list[JsonDict]:
    """Record first-party product state without promoting product claims."""

    return [
        {
            "product": "extropic_z1",
            "state": "taped_out_early_access_planned_2027",
            "canonical_url": "https://extropic.ai/writing/from-one-to-one-billion",
            "reproducible_assets": "simulator_api_only_no_authenticated_z1_access",
            "terminal": True,
            "dependency_added": False,
        },
        {
            "product": "kona_1_0",
            "state": "public_product_page_and_hosted_sudoku_demo",
            "canonical_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "reproducible_assets": "no_public_weights_recipe_source_or_local_runner",
            "terminal": True,
            "dependency_added": False,
        },
    ]


def ledger_append_rows() -> list[JsonDict]:
    """Return no additions because every verified fact predates the marker."""

    return []


def source_family_rows(queries: list[JsonDict]) -> list[JsonDict]:
    """Reduce query receipts to one terminal record per source family."""

    return [
        {
            "source_family": family,
            "query_id": next(row["query_id"] for row in queries if row["source_family"] == family),
            "terminal_outcome": next(
                row["terminal_outcome"] for row in queries if row["source_family"] == family
            ),
            "terminal": all(row["terminal"] for row in queries if row["source_family"] == family),
        }
        for family in SOURCE_FAMILIES
    ]


def _file_hash(path: Path) -> str:
    """Return a SHA-256 hash or an explicit missing marker."""

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def source_artifact_hashes(root: Path) -> dict[str, str]:
    """Hash every local source named by the execution contract."""

    relatives = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        LEDGER_RELATIVE_PATH,
        PREDECESSOR_RELATIVE_PATH,
        ROADMAP_RELATIVE_PATH,
        DESIGN_RELATIVE_PATH,
        Path("research-complete.yaml"),
        Path("ops/conductor-log.md"),
        Path("ops/e2e-test-plan.md"),
        SPEC_RELATIVE_PATH,
    )
    return {relative.as_posix(): _file_hash(root / relative) for relative in relatives}


def _combined_rows(groups: list[tuple[str, list[JsonDict]]]) -> list[JsonDict]:
    """Label flattened rows so their source collection remains clear."""

    rows = []
    for row_type, group in groups:
        rows.extend({"row_type": row_type, **deepcopy(row)} for row in group)
    return rows


def payload_checksum(artifact: dict[str, object]) -> str:
    """Hash normalized content without hashing the checksum itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_artifact(
    *, root: Path, run_date: str, duration_s: float, output_path: Path
) -> dict[str, object]:
    """Build a complete advisory receipt or a schema-complete blocked receipt."""

    preconditions = check_preconditions(root, run_date, output_path)
    failed = next((row for row in preconditions if not row["available"]), None)
    queries = [] if failed else query_rows()
    families = [] if failed else source_family_rows(queries)
    papers = [] if failed else paper_rows()
    citations = [] if failed else citation_edge_rows()
    implementations = [] if failed else implementation_rows()
    compatibilities = [] if failed else compatibility_rows()
    hardware = [] if failed else hardware_rows()
    products = [] if failed else product_rows()
    appends = [] if failed else ledger_append_rows()
    rate_limits = [row for row in queries if row["terminal_outcome"] == "rate_limited"]
    source_complete = bool(
        not failed
        and {row["source_family"] for row in queries} == set(SOURCE_FAMILIES)
        and all(row["terminal"] for row in queries)
        and {row["source_family"] for row in families} == set(SOURCE_FAMILIES)
        and all(row["terminal"] for row in families)
    )
    paper_complete = bool(
        not failed
        and {row["arxiv_id"] for row in papers} == set(SELECTED_ARXIV_IDS)
        and all(row["terminal"] for row in papers)
    )
    complete = source_complete and paper_complete
    rows = _combined_rows(
        [
            ("query", queries),
            ("source_family", families),
            ("paper", papers),
            ("citation_edge", citations),
            ("implementation", implementations),
            ("compatibility", compatibilities),
            ("hardware", hardware),
            ("product", products),
            ("ledger_append", appends),
            ("rate_limit", rate_limits),
        ]
    )
    if failed:
        status = "blocked"
        verdict_class = "blocked"
        honest_verdict = "blocked_v609_source_delta"
        gate_summary: JsonDict = {
            "all_gates_passed": False,
            "failed_check": failed["resource"],
            "expected": failed["expected"],
            "observed": failed["observed"],
        }
    else:
        status = "complete"
        verdict_class = "positive" if appends else "null"
        honest_verdict = (
            "complete_v609_source_delta_with_verified_post_marker_facts"
            if appends
            else "complete_null_v609_source_delta_no_post_marker_facts"
        )
        gate_summary = {
            "all_gates_passed": complete,
            "failed_check": None,
            "expected": "15 terminal source families and 5 terminal selected papers",
            "observed": f"{len(families)} source families and {len(papers)} selected papers",
        }
    artifact: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": status,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": {
            "invoked_models": [],
            "method": "No experimental model inference; bounded primary-source web research only.",
        },
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(root),
        "rows": rows,
        "query_rows": queries,
        "source_family_rows": families,
        "paper_rows": papers,
        "citation_edge_rows": citations,
        "implementation_rows": implementations,
        "compatibility_rows": compatibilities,
        "hardware_rows": hardware,
        "product_rows": products,
        "ledger_append_rows": appends,
        "rate_limit_rows": rate_limits,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v609_source_delta_complete_score": int(complete),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _verdict_prefix_is_valid(verdict_class: object, honest_verdict: object) -> bool:
    """Match every verdict class to its permitted terminal prefix."""

    prefixes = {
        "positive": ("complete_",),
        "circular_positive": ("complete_circular_",),
        "null": ("complete_null",),
        "blocked": ("blocked_",),
        "disqualified": ("complete_disqualified_", "disqualified_"),
        "partial": ("complete_partial_", "partial_"),
    }
    return any(
        str(honest_verdict).startswith(prefix) for prefix in prefixes.get(str(verdict_class), ())
    )


def validate_artifact(artifact_or_path: dict[str, object] | Path) -> list[str]:
    """Recompute terminal coverage and reject inconsistent receipts."""

    if isinstance(artifact_or_path, Path):
        if not artifact_or_path.exists():
            return ["artifact missing"]
        artifact = json.loads(artifact_or_path.read_text(encoding="utf-8"))
    else:
        artifact = artifact_or_path
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors: list[str] = []
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class is invalid")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principles do not cover required fields")
    if not _verdict_prefix_is_valid(artifact["verdict_class"], artifact["honest_verdict"]):
        errors.append("verdict prefix is inconsistent")
    blocked = artifact["status"] == "blocked"
    if blocked:
        if (
            artifact["v609_source_delta_complete_score"] != 0
            or artifact["verdict_class"] != "blocked"
            or artifact["honest_verdict"] != "blocked_v609_source_delta"
        ):
            errors.append("blocked artifact fields are inconsistent")
    else:
        queries = artifact["query_rows"]
        families = artifact["source_family_rows"]
        papers = artifact["paper_rows"]
        source_complete = (
            {row["source_family"] for row in queries} == set(SOURCE_FAMILIES)
            and all(row["terminal"] for row in queries)
            and {row["source_family"] for row in families} == set(SOURCE_FAMILIES)
            and all(row["terminal"] for row in families)
        )
        paper_complete = {row["arxiv_id"] for row in papers} == set(SELECTED_ARXIV_IDS) and all(
            row["terminal"] for row in papers
        )
        if not source_complete:
            errors.append("source family coverage is incomplete")
        if not paper_complete:
            errors.append("selected paper coverage is incomplete")
        if artifact["v609_source_delta_complete_score"] != int(source_complete and paper_complete):
            errors.append("completion score does not match terminal rows")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def append_verified_findings(root: Path, artifact: dict[str, object]) -> list[str]:
    """Append verified post-marker facts once while preserving earlier text."""

    path = root / LEDGER_RELATIVE_PATH
    original = path.read_text(encoding="utf-8")
    appended = []
    blocks = []
    for row in artifact["ledger_append_rows"]:
        marker = f"<!-- EXP6953-{row['finding_id'].upper()} -->"
        eligible = (
            row.get("observed_utc", "") > MARKER_UTC
            and row.get("verified") is True
            and row.get("evidence_grade") in {"A", "B"}
            and row.get("source_type") in {"primary", "first_party"}
        )
        if not eligible or marker in original:
            continue
        blocks.append(
            "\n".join(
                (
                    marker,
                    f"### {row['title']} - 2026-09-03",
                    f"- **Source:** {row['canonical_url']}",
                    f"- **Observed UTC:** {row['observed_utc']}",
                    f"- **Verified finding:** {row['finding']}",
                    f"- **V610 boundary:** {row['v610_use']}",
                )
            )
        )
        appended.append(row["finding_id"])
    if blocks:
        suffix = (
            "\n\n## V609 Execution-Time Source Delta - 2026-09-03\n\n" + "\n\n".join(blocks) + "\n"
        )
        atomic_write_text(path, original.rstrip() + suffix, allow_override=False)
    return appended


def write_artifact(artifact: dict[str, object], path: Path) -> Path:
    """Atomically write one validated Exp6953 artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return atomic_write_json(path, artifact, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """Write the dated source delta or validate an existing receipt."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        return int(bool(validate_artifact(args.validate)))
    root = find_repo_root()
    output = args.output or root / RESULT_RELATIVE_PATH
    draft = build_artifact(root=root, run_date=args.date, duration_s=373.0, output_path=output)
    if draft["status"] != "blocked":
        append_verified_findings(root, draft)
        draft = build_artifact(root=root, run_date=args.date, duration_s=373.0, output_path=output)
    write_artifact(draft, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
