"""Build the bounded V608 post-marker source-delta artifact.

Spec refs: REQ-REPORT-6941 and SCENARIO-REPORT-6941-*.

The module stores the dated query plan in source before any request runs. This
makes the scope auditable and prevents later search results from changing the
questions that produced the receipt.
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

EXPERIMENT_ID = "experiment_6941_v608_source_delta"
RUN_DATE = "20260903"
RANDOM_SEED = 6941
MARKER_UTC = "2026-09-03T16:23:00Z"
INFERENCE_SUBSTRATE = "bounded_primary_source_web_research_no_model_inference"
SCHEMA_VERSION = "carnot.experiment_6941.v608_source_delta.v1"

MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6941_v608_source_delta.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6941_v608_source_delta.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6941_v608_source_delta.json")
LEDGER_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
DESIGN_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PREDECESSOR_RELATIVE_PATH = Path("results/experiment_6927_v607_literature_delta.json")
NETWORK_POLICY_RELATIVE_PATH = Path("CLAUDE.md")

PLANNER_START_MARKER = "<!-- V608-PLANNER-REFRESH-20260903-START -->"
PLANNER_END_MARKER = "<!-- V608-PLANNER-REFRESH-20260903-END -->"

ARXIV_FAMILIES = (
    "arxiv_ebm_verification",
    "arxiv_neural_constraints",
    "arxiv_ising_ml",
    "arxiv_hallucination_detection",
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
    "2609.02438",
    "2609.02885",
    "2609.02702",
    "2609.02859",
    "2609.02417",
    "2609.02817",
    "2609.02783",
)

VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

_ACCEPTANCE = (
    "Accept a ledger fact only when a primary paper or first-party source has a stable "
    "identity, was observed after 2026-09-03T16:23:00Z, is absent from the V608 planner "
    "marker, and changes a material local execution boundary. Keep unchanged sources, "
    "duplicates, access failures, announcements, non-public models, unverified repositories, "
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


# Frozen at 2026-09-03T16:42:21Z, before the first network request. Planned
# times specify request order. They are not claims about response timing.
_FROZEN_QUERY_PLAN: tuple[JsonDict, ...] = (
    _plan_row(
        "arxiv_ebm_verification",
        '("energy-based model" OR "energy-based transformer") AND (verification OR verifier OR reasoning) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:44:00Z",
    ),
    _plan_row(
        "arxiv_neural_constraints",
        '("neural constraint" OR "constraint reasoning" OR neuro-symbolic) AND (verification OR language) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:45:00Z",
    ),
    _plan_row(
        "arxiv_ising_ml",
        '(Ising AND ("machine learning" OR sampling OR "probabilistic computing")) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:46:00Z",
    ),
    _plan_row(
        "arxiv_hallucination_detection",
        '((hallucination AND (verification OR detection)) OR "hidden-state verifier") AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:47:00Z",
    ),
    _plan_row(
        "arxiv_kan",
        '("Kolmogorov-Arnold Network" OR KAN) AND (verification OR symbolic OR constraint OR FPGA) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:48:00Z",
    ),
    _plan_row(
        "arxiv_energy_guided_decoding",
        '("energy-guided decoding" OR "reward-guided decoding" OR "energy-based decoding") AND (LLM OR language) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:49:00Z",
    ),
    _plan_row(
        "arxiv_sampling_hardware",
        "((thermodynamic OR Ising OR stochastic) AND (hardware OR accelerator) AND sampling) AND submittedDate:[202501010000 TO 202612312359]",
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:50:00Z",
    ),
    _plan_row(
        "arxiv_continual_constraint_learning",
        '(("continual learning" OR "strategy memory" OR "episodic memory") AND (constraint OR verifier OR LLM OR agent)) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T16:51:00Z",
    ),
    _plan_row(
        "openreview",
        'site:openreview.net ("energy based" OR constraint OR verifier OR continual memory) ICLR NeurIPS 2025 2026',
        "https://openreview.net/search",
        ["openreview.net"],
        "2026-09-03T16:52:00Z",
    ),
    _plan_row(
        "huggingface_papers",
        'site:huggingface.co/papers ("2609.02438" OR "2609.02885" OR "2609.02702" OR "2609.02859" OR "2609.02417" OR "2609.02817" OR "2609.02783")',
        "https://huggingface.co/papers",
        ["huggingface.co"],
        "2026-09-03T16:53:00Z",
    ),
    _plan_row(
        "semantic_scholar_ebt",
        "ARXIV:2507.02092 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations",
        ["api.semanticscholar.org", "semanticscholar.org"],
        "2026-09-03T16:54:00Z",
    ),
    _plan_row(
        "semantic_scholar_arm_ebm",
        "ARXIV:2512.15605 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations",
        ["api.semanticscholar.org", "semanticscholar.org"],
        "2026-09-03T16:55:00Z",
    ),
    _plan_row(
        "github_official",
        "official repositories for arXiv 2609.02438 2609.02885 2609.02702 2609.02859 2609.02417 2609.02817 2609.02783",
        "https://github.com/search",
        ["github.com", "api.github.com"],
        "2026-09-03T16:56:00Z",
    ),
    _plan_row(
        "extropic",
        "Extropic Torx Thermalizers Z1 XTR-0 implementation access update",
        "https://extropic.ai/writing/from-one-to-one-billion",
        ["extropic.ai"],
        "2026-09-03T16:57:00Z",
    ),
    _plan_row(
        "logical_intelligence",
        "Logical Intelligence Kona EBM weights code training recipe local runner update",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        ["logicalintelligence.com"],
        "2026-09-03T16:58:00Z",
    ),
)


def frozen_query_plan() -> list[JsonDict]:
    """Return copies so a caller cannot change the frozen search plan."""

    return [deepcopy(row) for row in _FROZEN_QUERY_PLAN]


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
    "candidate_rows",
    "accepted_finding_rows",
    "rejected_finding_rows",
    "citation_edge_rows",
    "implementation_rows",
    "compatibility_rows",
    "ledger_append_rows",
    "v609_candidate_rows",
    "rate_limit_rows",
    "random_seed",
    "reproducibility_checksum",
    "v608_source_delta_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes changes to the evidence contract detectable.",
    "experiment_id": "A stable identity binds the receipt to the requested experiment.",
    "run_date": "The date separates execution evidence from the earlier planner pass.",
    "status": "Status distinguishes completed research from a failed preflight.",
    "field_principles": "A principle for each field preserves its scientific purpose.",
    "preconditions_checked": "Preflight evidence prevents research against missing contracts.",
    "inference_substrate": "The substrate prevents source review from implying model inference.",
    "model_specs": "An empty invocation record distinguishes cited models from runtime models.",
    "duration_s": "Elapsed time exposes whether the bounded research run was plausible.",
    "source_artifact_hashes": "Hashes bind the result to the local inputs that defined its scope.",
    "rows": "A combined ledger lets a reader inspect evidence without trusting summaries.",
    "query_rows": "Query receipts preserve updates, failures, and explicit no-update results.",
    "source_family_rows": "Family rows prove that every required source reached an outcome.",
    "candidate_rows": "Selected-paper rows prevent favorable papers from being reported alone.",
    "accepted_finding_rows": "Accepted rows retain verified evidence even when it is not new.",
    "rejected_finding_rows": "Rejected rows preserve negative decisions against later promotion.",
    "citation_edge_rows": "Citation rows distinguish checked existing edges from new edges.",
    "implementation_rows": "Implementation rows keep source ideas advisory to the locked design.",
    "compatibility_rows": "Compatibility rows expose missing code, data, and local interfaces.",
    "ledger_append_rows": "Append rows isolate verified facts that truly postdate the marker.",
    "v609_candidate_rows": "Future rows keep follow-up work outside the locked milestone.",
    "rate_limit_rows": "Rate-limit rows stop missing evidence from becoming a false no-update.",
    "random_seed": "A fixed seed makes deterministic row order reproducible.",
    "reproducibility_checksum": "A content hash detects later changes to the complete receipt.",
    "v608_source_delta_complete_score": "The score is one only after every required check terminates.",
    "gate_check_summary": "The summary states expected and observed gate evidence.",
    "verifier_is_oracle": "False prevents a literature audit from certifying scientific truth.",
    "verdict_class": "The class separates a covered null result from positive or blocked work.",
    "honest_verdict": "A terminal prefix makes the outcome machine-readable.",
}


_QUERY_RESULTS: dict[str, JsonDict] = {
    "arxiv_ebm_verification": {
        "actual_utc": "2026-09-03T16:43:00Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The selected verification paper remained arXiv v1, and the topic search found no newer executable matching-base verifier.",
    },
    "arxiv_neural_constraints": {
        "actual_utc": "2026-09-03T16:43:05Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The search returned earlier symbolic-integration work and no post-marker exact-authority implementation.",
    },
    "arxiv_ising_ml": {
        "actual_utc": "2026-09-03T16:43:10Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "No dated Ising result changed the local software or authenticated-hardware boundary recorded by the planner.",
    },
    "arxiv_hallucination_detection": {
        "actual_utc": "2026-09-03T16:43:15Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The visible hidden-state detectors predated the marker and did not replace exact outcome authority.",
    },
    "arxiv_kan": {
        "actual_utc": "2026-09-03T16:45:05Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The visible KAN hardware and geometry papers predated the marker and supplied no verified V608 energy-ordering gain.",
    },
    "arxiv_energy_guided_decoding": {
        "actual_utc": "2026-09-03T16:45:10Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The returned reward-guided decoding papers predated the marker and did not alter the fixed-budget complete-candidate boundary.",
    },
    "arxiv_sampling_hardware": {
        "actual_utc": "2026-09-03T16:45:15Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "No result established a new authenticated local thermodynamic or Ising sampling route.",
    },
    "arxiv_continual_constraint_learning": {
        "actual_utc": "2026-09-03T16:45:20Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The returned memory systems predated the marker and did not supersede exact external write authority.",
    },
    "openreview": {
        "actual_utc": "2026-09-03T16:44:35Z",
        "attempt_count": 2,
        "http_status": 403,
        "status": "browser_challenge",
        "terminal_outcome": "unavailable_no_update",
        "evidence": "Both named forum pages redirected to OpenReview browser verification, so no post-marker metadata claim was accepted.",
    },
    "huggingface_papers": {
        "actual_utc": "2026-09-03T16:44:45Z",
        "attempt_count": 2,
        "http_status": None,
        "status": "search_empty_direct_open_unavailable",
        "terminal_outcome": "unavailable_no_update",
        "evidence": "The exact-ID search returned no page, and the direct paper route was unavailable; no update was inferred from absence.",
    },
    "semantic_scholar_ebt": {
        "actual_utc": "2026-09-03T16:45:45Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_new_citation_edge",
        "evidence": "The API returned 35 visible EBT citation rows. None was newer than the planner marker or a new local dependency.",
        "visible_citation_count": 35,
    },
    "semantic_scholar_arm_ebm": {
        "actual_utc": "2026-09-03T16:45:55Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_new_citation_edge",
        "evidence": "The API returned the same eight visible ARM-EBM citation rows recorded by the planner.",
        "visible_citation_count": 8,
    },
    "github_official": {
        "actual_utc": "2026-09-03T16:46:20Z",
        "attempt_count": 2,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "existing_code_state_no_post_marker_update",
        "evidence": "EarlyEval links public code, but its latest push was 2026-06-28 and required data and artifacts are absent. DWM still says code and data are coming soon; no other official repository was found.",
    },
    "extropic": {
        "actual_utc": "2026-09-03T16:46:25Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_post_marker_update",
        "evidence": "The page still describes Thermalizers as upcoming and Z1 systems as planned for 2027 early access.",
    },
    "logical_intelligence": {
        "actual_utc": "2026-09-03T16:46:30Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "unavailable_reproducible_assets",
        "evidence": "The Kona product page still provides no public weights, training recipe, source code, or local runner.",
    },
}


def validate_query_plan(plan: list[JsonDict], run_date: str) -> list[str]:
    """Reject a query plan that exceeds the limits frozen before research."""

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
    """Return one fail-closed receipt for each required local contract."""

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
            "v608_planner_marker",
            PLANNER_START_MARKER in ledger_text and PLANNER_END_MARKER in ledger_text,
            "the reference ledger contains both V608 planner marker boundaries",
            ledger.as_posix() if ledger.is_file() else "missing",
        ),
        (
            "active_roadmap",
            "milestone: 2026.09.608" in roadmap_text
            and "exp6941-v608-source-delta" in roadmap_text,
            "the active roadmap declares V608 and Exp6941",
            roadmap.as_posix() if roadmap.is_file() else "missing",
        ),
        (
            "design_document",
            "**Milestone:** 2026.09.608" in design_text and "Exp6941" in design_text,
            "the design document declares V608 and Exp6941",
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
    """Join each frozen query to its explicit terminal research receipt."""

    rows = []
    for planned in frozen_query_plan():
        row = deepcopy(planned)
        row.update(_QUERY_RESULTS[planned["query_id"]])
        row["terminal"] = True
        rows.append(row)
    return rows


_CANDIDATE_ROWS: tuple[JsonDict, ...] = (
    {
        "arxiv_id": "2609.02438",
        "title": "When Decodability Is Not Enough: Logical Validity Representations, Behavioral Dissociation, and Causal Tests in Language Models",
        "canonical_url": "https://arxiv.org/abs/2609.02438",
        "version": "v1",
        "date": "2026-09-02",
        "submitted_utc": "2026-09-02T11:01:09Z",
        "evidence_grade": "A",
        "source_type": "primary_paper",
        "code_available": False,
        "code_state": "no_official_repository_found_after_two_queries",
        "code_state_evidence_url": "https://arxiv.org/abs/2609.02438",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The causal-control design is usable, but no first-party code exists and local GGUF layer intervention remains unproven.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.02885",
        "title": "Discriminative World Models for Web Agents",
        "canonical_url": "https://arxiv.org/abs/2609.02885",
        "version": "v1",
        "date": "2026-09-02",
        "submitted_utc": "2026-09-02T17:59:40Z",
        "evidence_grade": "A",
        "source_type": "primary_paper_and_first_party_project_page",
        "code_available": False,
        "code_state": "project_page_says_code_and_data_coming_soon",
        "code_state_evidence_url": "https://dhruvpendharkar.github.io/dwm/",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Existing ARC branch receipts support a local analogue, but the reported pipeline and data are not publicly executable.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.02702",
        "title": "Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers",
        "canonical_url": "https://arxiv.org/abs/2609.02702",
        "version": "v1",
        "date": "2026-09-02",
        "submitted_utc": "2026-09-02T15:06:46Z",
        "evidence_grade": "A",
        "source_type": "primary_paper",
        "code_available": False,
        "code_state": "no_official_repository_found_after_two_queries",
        "code_state_evidence_url": "https://arxiv.org/abs/2609.02702",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The prompt-order control is locally implementable, but the paper provides no reference code or published run assets.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.02859",
        "title": "User Feedback Provides a Unique Signal that LLMs Can not Detect",
        "canonical_url": "https://arxiv.org/abs/2609.02859",
        "version": "v1",
        "date": "2026-09-02",
        "submitted_utc": "2026-09-02T17:42:44Z",
        "evidence_grade": "A",
        "source_type": "primary_paper",
        "code_available": False,
        "code_state": "no_official_repository_found_after_two_queries",
        "code_state_evidence_url": "https://arxiv.org/abs/2609.02859",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Exact external outcomes fit the local policy, but the study has no public implementation or local model recipe.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.02417",
        "title": "Coverage, Not Targeting: A Structural Regime in Multi-Turn Agent Credit Assignment",
        "canonical_url": "https://arxiv.org/abs/2609.02417",
        "version": "v1",
        "date": "2026-09-02",
        "submitted_utc": "2026-09-02T10:37:12Z",
        "evidence_grade": "A",
        "source_type": "primary_paper",
        "code_available": False,
        "code_state": "no_official_repository_found_after_two_queries",
        "code_state_evidence_url": "https://arxiv.org/abs/2609.02417",
        "local_compatibility": "compatible",
        "disposition": "accepted_existing",
        "exclusion_reason": "The density metric and matched controls need no external model, but the paper's training result is not a local reproduced dependency.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.02817",
        "title": "Cliff: Learning Process Rewards from the First Mistake",
        "canonical_url": "https://arxiv.org/abs/2609.02817",
        "version": "v1",
        "date": "2026-09-02",
        "submitted_utc": "2026-09-02T17:03:42Z",
        "evidence_grade": "A",
        "source_type": "primary_paper",
        "code_available": False,
        "code_state": "no_official_repository_found_after_two_queries",
        "code_state_evidence_url": "https://arxiv.org/abs/2609.02817",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The credit shape is usable only after an exact first-error label replaces the paper's LLM teacher.",
        "terminal": True,
    },
    {
        "arxiv_id": "2609.02783",
        "title": "EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction",
        "canonical_url": "https://arxiv.org/abs/2609.02783",
        "version": "v1",
        "date": "2026-09-02",
        "submitted_utc": "2026-09-02T16:15:18Z",
        "evidence_grade": "A",
        "source_type": "primary_paper_and_first_party_repository",
        "code_available": True,
        "code_state": "public_code_only_missing_required_data_models_and_run_artifacts",
        "code_state_evidence_url": "https://github.com/inphotoo/earlyeval",
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The Python code is inspectable, but full reproduction needs unpublished trajectory tables, trained models, predictions, and caches.",
        "terminal": True,
    },
)


def candidate_rows() -> list[JsonDict]:
    """Return one copied terminal receipt for every planner-selected paper."""

    return deepcopy(_CANDIDATE_ROWS)


def accepted_finding_rows() -> list[JsonDict]:
    """Return verified selected evidence, including its execution limits."""

    return [row for row in candidate_rows() if row["disposition"].startswith("accepted")]


def rejected_finding_rows() -> list[JsonDict]:
    """Return selected papers rejected from the advisory evidence set."""

    return [row for row in candidate_rows() if row["disposition"] == "rejected"]


def citation_edge_rows() -> list[JsonDict]:
    """Record representative citation edges checked for post-marker change."""

    return [
        {
            "source_id": "arxiv:2607.20792",
            "target_id": "arxiv:2507.02092",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2607.20792",
            "is_new_after_marker": False,
            "evidence": "Memoir remained an existing visible EBT citation.",
            "terminal": True,
        },
        {
            "source_id": "arxiv:2607.17047",
            "target_id": "arxiv:2507.02092",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2607.17047",
            "is_new_after_marker": False,
            "evidence": "Solver-Hard remained an existing visible EBT citation.",
            "terminal": True,
        },
        {
            "source_id": "arxiv:2605.18871",
            "target_id": "arxiv:2512.15605",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2605.18871",
            "is_new_after_marker": False,
            "evidence": "Distributional EBM remained an existing visible ARM-EBM citation.",
            "terminal": True,
        },
        {
            "source_id": "arxiv:2602.02991",
            "target_id": "arxiv:2512.15605",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2602.02991",
            "is_new_after_marker": False,
            "evidence": "False First Steps remained an existing visible ARM-EBM citation.",
            "terminal": True,
        },
    ]


def implementation_rows() -> list[JsonDict]:
    """Map checked ideas to the locked design without changing dependencies."""

    candidates = {row["arxiv_id"]: row for row in candidate_rows()}
    mappings = (
        (
            "2609.02438",
            "Exp6946/Exp6947",
            "Require matched random-direction and score-only controls for any causal hidden-state claim.",
        ),
        (
            "2609.02885",
            "Exp6948/Exp6949",
            "Use authentic alternative-action successor states for the offline branch-energy comparison.",
        ),
        (
            "2609.02702",
            "Exp6950/Exp6951",
            "Compare trace-before-context, append-after, and no-memory conditions under one budget.",
        ),
        (
            "2609.02859",
            "Exp6950/Exp6951",
            "Let exact external outcomes authorize writes; a model judge cannot certify itself.",
        ),
        (
            "2609.02417",
            "Exp6943/Exp6945",
            "Measure verifier information density before comparing targeted and uniform credit.",
        ),
        (
            "2609.02817",
            "Exp6943/Exp6945",
            "Use first-error credit only where the fixture supplies an exact first-invalid-step label.",
        ),
        (
            "2609.02783",
            "advisory_only",
            "Keep early stopping disabled; a later shadow study must charge every false stop.",
        ),
    )
    return [
        {
            "arxiv_id": arxiv_id,
            "source_url": candidates[arxiv_id]["canonical_url"],
            "v608_experiment": experiment,
            "implementation_boundary": boundary,
            "advisory_only": True,
        }
        for arxiv_id, experiment, boundary in mappings
    ]


def compatibility_rows() -> list[JsonDict]:
    """State the concrete local execution boundary for each selected paper."""

    rows = (
        (
            "2609.02438",
            "hidden_state_intervention",
            "No official code was found, and the local GGUF runner has not proved layer intervention.",
            "Retain the causal-control design but add no executable dependency.",
        ),
        (
            "2609.02885",
            "repository_state",
            "The first-party project page says code and data are coming soon.",
            "Use existing ARC receipts for a local analogue; do not claim paper reproduction.",
        ),
        (
            "2609.02702",
            "reference_assets",
            "The prompt method is portable, but no official code or run assets were found.",
            "Implement only the locked matched prompt-order control.",
        ),
        (
            "2609.02859",
            "evaluation_authority",
            "The paper has no public code, and model judges can miss feedback-specific corrections.",
            "Use exact outcomes for memory admission and keep judges non-authoritative.",
        ),
        (
            "2609.02417",
            "metric_portability",
            "Verifier density and matched credit geometry are deterministic from exact fixture rows.",
            "Compute the metric locally without importing a model or result claim.",
        ),
        (
            "2609.02817",
            "teacher_authority",
            "The paper uses an LLM teacher to locate the first mistake.",
            "Replace the teacher with exact first-invalid-step labels before using the shape.",
        ),
        (
            "2609.02783",
            "missing_reproduction_inputs",
            "The official repository omits raw trajectories, processed tables, trained models, predictions, and caches.",
            "Keep the code as an advisory reference and do not enable stopping.",
        ),
    )
    return [
        {
            "arxiv_id": arxiv_id,
            "constraint": constraint,
            "observed": observed,
            "effect": effect,
        }
        for arxiv_id, constraint, observed, effect in rows
    ]


def ledger_append_rows() -> list[JsonDict]:
    """Return no additions because every verified fact predates the marker."""

    return []


def v609_candidate_rows() -> list[JsonDict]:
    """Keep release checks and stopping work outside the locked milestone."""

    return [
        {
            "candidate_id": "dwm_public_release_recheck",
            "target_milestone": "V609+",
            "proposal": "Recheck branch-world-model reproduction after official code and data become public.",
            "prerequisite": "A first-party executable repository and downloadable branching data.",
            "canonical_url": "https://dhruvpendharkar.github.io/dwm/",
        },
        {
            "candidate_id": "earlyeval_artifact_recheck",
            "target_milestone": "V609+",
            "proposal": "Run a bounded shadow-only early-outcome check after required inputs are public.",
            "prerequisite": "Published trajectory tables and a false-stop gate with exact outcomes.",
            "canonical_url": "https://github.com/inphotoo/earlyeval",
        },
    ]


def source_family_rows(queries: list[JsonDict]) -> list[JsonDict]:
    """Reduce queries to one explicit terminal row for each source family."""

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
    """Hash every local input named by the task contract."""

    relatives = (
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        LEDGER_RELATIVE_PATH,
        PREDECESSOR_RELATIVE_PATH,
        ROADMAP_RELATIVE_PATH,
        DESIGN_RELATIVE_PATH,
        Path("research-complete.yaml"),
        Path("ops/conductor-log.md"),
        SPEC_RELATIVE_PATH,
    )
    return {relative.as_posix(): _file_hash(root / relative) for relative in relatives}


def _combined_rows(groups: list[tuple[str, list[JsonDict]]]) -> list[JsonDict]:
    """Label each row so the flattened evidence ledger stays self-describing."""

    rows = []
    for row_type, group in groups:
        rows.extend({"row_type": row_type, **deepcopy(row)} for row in group)
    return rows


def payload_checksum(artifact: dict[str, object]) -> str:
    """Hash normalized content without the self-referential checksum field."""

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
    candidates = [] if failed else candidate_rows()
    families = [] if failed else source_family_rows(queries)
    accepted = [] if failed else accepted_finding_rows()
    rejected = [] if failed else rejected_finding_rows()
    citations = [] if failed else citation_edge_rows()
    implementations = [] if failed else implementation_rows()
    compatibilities = [] if failed else compatibility_rows()
    appends = [] if failed else ledger_append_rows()
    future = [] if failed else v609_candidate_rows()
    rate_limits = [row for row in queries if row["terminal_outcome"] == "rate_limited"]
    source_complete = bool(
        not failed
        and {row["source_family"] for row in queries} == set(SOURCE_FAMILIES)
        and all(row["terminal"] for row in queries)
    )
    candidate_complete = bool(
        not failed
        and {row["arxiv_id"] for row in candidates} == set(SELECTED_ARXIV_IDS)
        and all(row["terminal"] for row in candidates)
    )
    complete = source_complete and candidate_complete
    rows = _combined_rows(
        [
            ("query", queries),
            ("source_family", families),
            ("candidate", candidates),
            ("citation_edge", citations),
            ("implementation", implementations),
            ("compatibility", compatibilities),
            ("ledger_append", appends),
            ("v609_candidate", future),
            ("rate_limit", rate_limits),
        ]
    )
    if failed:
        status = "blocked"
        verdict_class = "blocked"
        honest_verdict = "blocked_v608_source_delta"
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
            "complete_v608_source_delta_with_verified_post_marker_facts"
            if appends
            else "complete_null_v608_source_delta_no_post_marker_facts"
        )
        gate_summary = {
            "all_gates_passed": complete,
            "failed_check": None,
            "expected": "15 terminal source families and 7 terminal selected papers",
            "observed": f"{len(families)} source families and {len(candidates)} selected papers",
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
        "candidate_rows": candidates,
        "accepted_finding_rows": accepted,
        "rejected_finding_rows": rejected,
        "citation_edge_rows": citations,
        "implementation_rows": implementations,
        "compatibility_rows": compatibilities,
        "ledger_append_rows": appends,
        "v609_candidate_rows": future,
        "rate_limit_rows": rate_limits,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v608_source_delta_complete_score": int(complete),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _verdict_prefix_is_valid(verdict_class: object, honest_verdict: object) -> bool:
    """Match each verdict class to a stable terminal prefix."""

    verdict = str(honest_verdict)
    prefixes = {
        "positive": ("complete_",),
        "circular_positive": ("complete_circular_",),
        "null": ("complete_null",),
        "blocked": ("blocked_",),
        "disqualified": ("complete_disqualified_", "disqualified_"),
        "partial": ("complete_partial_", "partial_"),
    }
    return any(verdict.startswith(prefix) for prefix in prefixes.get(str(verdict_class), ()))


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
            artifact["v608_source_delta_complete_score"] != 0
            or artifact["verdict_class"] != "blocked"
            or artifact["honest_verdict"] != "blocked_v608_source_delta"
        ):
            errors.append("blocked artifact fields are inconsistent")
    else:
        queries = artifact["query_rows"]
        candidates = artifact["candidate_rows"]
        source_complete = {row["source_family"] for row in queries} == set(SOURCE_FAMILIES) and all(
            row["terminal"] for row in queries
        )
        candidate_complete = {row["arxiv_id"] for row in candidates} == set(
            SELECTED_ARXIV_IDS
        ) and all(row["terminal"] for row in candidates)
        if not source_complete:
            errors.append("source family coverage is incomplete")
        if not candidate_complete:
            errors.append("selected paper coverage is incomplete")
        if artifact["v608_source_delta_complete_score"] != int(
            source_complete and candidate_complete
        ):
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
        marker = f"<!-- EXP6941-{row['finding_id'].upper()} -->"
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
                    f"- **V609 boundary:** {row['v609_use']}",
                )
            )
        )
        appended.append(row["finding_id"])
    if blocks:
        suffix = (
            "\n\n## V608 Execution-Time Source Delta - 2026-09-03\n\n" + "\n\n".join(blocks) + "\n"
        )
        atomic_write_text(path, original.rstrip() + suffix, allow_override=False)
    return appended


def write_artifact(artifact: dict[str, object], path: Path) -> Path:
    """Atomically write one validated Exp6941 artifact."""

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
    draft = build_artifact(root=root, run_date=args.date, duration_s=300.0, output_path=output)
    if draft["status"] != "blocked":
        append_verified_findings(root, draft)
        draft = build_artifact(
            root=root,
            run_date=args.date,
            duration_s=300.0,
            output_path=output,
        )
    write_artifact(draft, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
