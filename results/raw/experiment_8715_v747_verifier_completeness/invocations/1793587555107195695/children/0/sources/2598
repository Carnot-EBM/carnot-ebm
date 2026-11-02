"""Build the bounded V607 execution-time literature-delta artifact.

Spec refs: REQ-REPORT-6927 and SCENARIO-REPORT-6927-*.

The module keeps the dated query plan in source. This makes the search scope
auditable and stops later results from changing the questions that produced
them.
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

EXPERIMENT_ID = "experiment_6927_v607_literature_delta"
RUN_DATE = "20260903"
RANDOM_SEED = 6927
INFERENCE_SUBSTRATE = "bounded_primary_source_web_research_no_model_inference"
SCHEMA_VERSION = "carnot.experiment_6927.v607_literature_delta.v1"

MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6927_v607_literature_delta.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6927_v607_literature_delta.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6927_v607_literature_delta.json")
LEDGER_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
DESIGN_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PREDECESSOR_RELATIVE_PATH = Path("results/experiment_6925_v606_sota_ingestion.json")
NETWORK_POLICY_RELATIVE_PATH = Path("CLAUDE.md")

ARXIV_FAMILIES = (
    "arxiv_ebm_verification",
    "arxiv_neural_constraints",
    "arxiv_ising_ml",
    "arxiv_hallucination_detection",
    "arxiv_kan",
    "arxiv_energy_guided_decoding",
    "arxiv_hardware_sampling",
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

NAMED_CANDIDATES = (
    "hsrm",
    "sparse_reward_subsystem",
    "nsvif",
    "sampling_for_quality",
    "ism",
    "memoir",
    "solver_hard",
    "kan_verification",
    "torx",
    "thermalizers",
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
    "Accept a fact only when a primary paper or first-party source has a stable identity, "
    "a verifiable date, clear V607 relevance, and a material local execution consequence. "
    "Keep duplicates, unchanged sources, access failures, unavailable code, hosted-only "
    "paths, and incompatible hardware as terminal non-findings."
)


def _plan_row(
    query_id: str,
    query_text: str,
    canonical_url: str,
    allowed_domains: list[str],
    planned_utc: str,
) -> JsonDict:
    """Create one bounded query row with the same network limits."""

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


# Frozen at 2026-09-03T12:28:10Z, before the first network request. The
# planned times define request order. They do not claim exact response times.
_FROZEN_QUERY_PLAN: tuple[JsonDict, ...] = (
    _plan_row(
        "arxiv_ebm_verification",
        '("energy-based model" OR "energy-based transformer") AND (verification OR verifier OR reasoning) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:29:00Z",
    ),
    _plan_row(
        "arxiv_neural_constraints",
        '("neural constraint" OR "constraint reasoning" OR neuro-symbolic) AND (verification OR language) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:30:00Z",
    ),
    _plan_row(
        "arxiv_ising_ml",
        '(Ising AND ("machine learning" OR sampling OR "probabilistic computing")) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:31:00Z",
    ),
    _plan_row(
        "arxiv_hallucination_detection",
        '((hallucination AND (verification OR detection)) OR "hidden-state verifier") AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:32:00Z",
    ),
    _plan_row(
        "arxiv_kan",
        '("Kolmogorov-Arnold Network" OR KAN) AND (verification OR symbolic OR constraint OR FPGA) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:33:00Z",
    ),
    _plan_row(
        "arxiv_energy_guided_decoding",
        '("energy-guided decoding" OR "reward-guided decoding" OR "energy-based decoding") AND (LLM OR language) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:34:00Z",
    ),
    _plan_row(
        "arxiv_hardware_sampling",
        "((thermodynamic OR Ising OR stochastic) AND (hardware OR accelerator) AND sampling) AND submittedDate:[202501010000 TO 202612312359]",
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:35:00Z",
    ),
    _plan_row(
        "arxiv_continual_constraint_learning",
        '(("continual learning" OR "strategy memory" OR "episodic memory") AND (constraint OR verifier OR LLM OR agent)) AND submittedDate:[202501010000 TO 202612312359]',
        "https://export.arxiv.org/api/query",
        ["arxiv.org", "export.arxiv.org"],
        "2026-09-03T12:36:00Z",
    ),
    _plan_row(
        "openreview",
        'site:openreview.net ("energy based" OR constraint OR verifier OR continual memory) ICLR NeurIPS 2025 2026',
        "https://openreview.net/search",
        ["openreview.net"],
        "2026-09-03T12:37:00Z",
    ),
    _plan_row(
        "huggingface_papers",
        'site:huggingface.co/papers (HSRM OR NSVIF OR ISM OR verifier OR "energy-guided decoding")',
        "https://huggingface.co/papers",
        ["huggingface.co"],
        "2026-09-03T12:38:00Z",
    ),
    _plan_row(
        "semantic_scholar_ebt",
        "ARXIV:2507.02092 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations",
        ["api.semanticscholar.org", "semanticscholar.org"],
        "2026-09-03T12:39:00Z",
    ),
    _plan_row(
        "semantic_scholar_arm_ebm",
        "ARXIV:2512.15605 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations",
        ["api.semanticscholar.org", "semanticscholar.org"],
        "2026-09-03T12:40:00Z",
    ),
    _plan_row(
        "github_official",
        "official repositories HSRM ISM NSVIF Sampling for Quality KAN Torx Thermalizers",
        "https://github.com/search",
        ["github.com", "api.github.com"],
        "2026-09-03T12:41:00Z",
    ),
    _plan_row(
        "extropic",
        "Extropic Torx Thermalizers Z1 XTR-0 implementation access update",
        "https://extropic.ai/writing/from-one-to-one-billion",
        ["extropic.ai"],
        "2026-09-03T12:42:00Z",
    ),
    _plan_row(
        "logical_intelligence",
        "Logical Intelligence Kona EBM weights code training recipe local runner update",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        ["logicalintelligence.com"],
        "2026-09-03T12:43:00Z",
    ),
)


def frozen_query_plan() -> list[JsonDict]:
    """Return a copy so callers cannot mutate the frozen source plan."""

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
    "v608_candidate_rows",
    "rate_limit_rows",
    "random_seed",
    "reproducibility_checksum",
    "v607_literature_delta_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes contract drift detectable.",
    "experiment_id": "A stable identity binds the receipt to Exp6927.",
    "run_date": "A date separates this execution check from planning evidence.",
    "status": "Status distinguishes complete research from blocked preflight.",
    "field_principles": "A principle for each field explains why it is retained.",
    "preconditions_checked": "Preflight evidence prevents research against missing inputs.",
    "inference_substrate": "The substrate prevents source review from implying model inference.",
    "model_specs": "An empty invocation receipt distinguishes cited models from runtime models.",
    "duration_s": "Elapsed time makes the bounded execution cost auditable.",
    "source_artifact_hashes": "Hashes bind conclusions to their local task inputs.",
    "rows": "A combined ledger exposes all evidence without trusting summaries.",
    "query_rows": "Query receipts preserve updates, no-updates, and access limits.",
    "source_family_rows": "Family summaries prove all required surfaces terminated.",
    "candidate_rows": "Named candidate rows prevent selective reporting.",
    "accepted_finding_rows": "Accepted rows retain relevant verified methods and boundaries.",
    "rejected_finding_rows": "Rejected rows preserve why weak dependencies were excluded.",
    "citation_edge_rows": "Citation edges record the checked relationship between sources.",
    "implementation_rows": "Implementation rows map methods to locked tasks without promoting them.",
    "compatibility_rows": "Compatibility rows expose local-GGUF and hardware limits.",
    "ledger_append_rows": "Append rows isolate facts absent from the prior ledger.",
    "v608_candidate_rows": "Future ideas stay advisory and cannot rewrite V607.",
    "rate_limit_rows": "Rate-limit rows stop missing data becoming a false no-update.",
    "random_seed": "A fixed seed makes deterministic row order reproducible.",
    "reproducibility_checksum": "A payload checksum detects later receipt mutation.",
    "v607_literature_delta_complete_score": "The score is one only after every required source and candidate terminates.",
    "gate_check_summary": "The summary reports expected and observed gate evidence.",
    "verifier_is_oracle": "False prevents literature review from becoming a correctness oracle.",
    "verdict_class": "The class separates an advisory update from null or blocked research.",
    "honest_verdict": "A terminal prefix makes the outcome machine-readable.",
}

_QUERY_RESULTS: dict[str, JsonDict] = {
    "arxiv_ebm_verification": {
        "actual_utc": "2026-09-03T12:28:20Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The relevant EBM verifier results were already present; no local matching-base verifier superseded HSRM.",
    },
    "arxiv_neural_constraints": {
        "actual_utc": "2026-09-03T12:28:25Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "NSVIF remained the relevant post-hoc constraint lead and no executable exact authority appeared.",
    },
    "arxiv_ising_ml": {
        "actual_utc": "2026-09-03T12:28:30Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The returned p-bit and Ising work did not change V607's local-GGUF or unavailable-hardware boundary.",
    },
    "arxiv_hallucination_detection": {
        "actual_utc": "2026-09-03T12:28:35Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "new_incompatible_watch_item",
        "evidence": "IntroConformal (2609.01375) is new hidden-state conformal work for multimodal models, but has no verified local text-GGUF path.",
    },
    "arxiv_kan": {
        "actual_utc": "2026-09-03T12:28:40Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "metadata_and_code_state_correction",
        "evidence": "The official 2602.06737 record names optimized piecewise-affine abstractions and links no public implementation.",
    },
    "arxiv_energy_guided_decoding": {
        "actual_utc": "2026-09-03T12:28:45Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "No result replaced Sampling for Quality or removed the complete-candidate budget boundary.",
    },
    "arxiv_hardware_sampling": {
        "actual_utc": "2026-09-03T12:28:50Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "Thermalizers remained paper-and-preview evidence with no authenticated local thermodynamic device route.",
    },
    "arxiv_continual_constraint_learning": {
        "actual_utc": "2026-09-03T12:28:55Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "No result changed the delayed-write, exact-admission, or fresh-process audit boundaries.",
    },
    "openreview": {
        "actual_utc": "2026-09-03T12:29:05Z",
        "attempt_count": 2,
        "http_status": 403,
        "status": "fallback_to_primary_arxiv",
        "terminal_outcome": "unavailable_no_update",
        "evidence": "The search surface challenged automated access; NSVIF metadata was checked through its primary arXiv record instead.",
    },
    "huggingface_papers": {
        "actual_utc": "2026-09-03T12:29:15Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The paper pages added no public implementation or stronger exact-authority route.",
    },
    "semantic_scholar_ebt": {
        "actual_utc": "2026-09-03T12:30:06Z",
        "attempt_count": 2,
        "http_status": 200,
        "status": "success_after_browser_rejection",
        "terminal_outcome": "no_update",
        "evidence": "The direct API returned 35 visible citation rows; the newest relevant rows were already ledgered.",
    },
    "semantic_scholar_arm_ebm": {
        "actual_utc": "2026-09-03T12:30:15Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The direct API returned eight visible citation rows and no new V607 execution boundary.",
    },
    "github_official": {
        "actual_utc": "2026-09-03T12:30:20Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "HSRM remained README-only, ISM remained hosted-service-bound, Torx remained public JAX code, and no Thermalizers repository appeared.",
    },
    "extropic": {
        "actual_utc": "2026-09-03T12:30:25Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The first-party page still releases Torx, previews Thermalizers, and describes future Z1 access rather than a local device route.",
    },
    "logical_intelligence": {
        "actual_utc": "2026-09-03T12:30:30Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "unavailable_reproducible_assets",
        "evidence": "The Kona page remained a product description without public weights, recipe, or local runner.",
    },
}


def validate_query_plan(plan: list[JsonDict], run_date: str) -> list[str]:
    """Reject a dated query plan that exceeds its frozen network bounds."""

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
    """Return one fail-closed row for every required local precondition."""

    ledger = root / LEDGER_RELATIVE_PATH
    roadmap = root / ROADMAP_RELATIVE_PATH
    policy = root / NETWORK_POLICY_RELATIVE_PATH
    plan_errors = validate_query_plan(frozen_query_plan(), run_date)
    output_parent = output_path.parent
    checks = (
        (
            "reference_ledger",
            ledger.is_file(),
            "research-references.md exists and is readable",
            ledger.as_posix() if ledger.is_file() else "missing",
        ),
        (
            "active_roadmap",
            roadmap.is_file() and "milestone: 2026.09.607" in roadmap.read_text(encoding="utf-8"),
            "active roadmap declares milestone 2026.09.607",
            roadmap.as_posix() if roadmap.is_file() else "missing",
        ),
        (
            "dated_query_plan",
            not plan_errors,
            "frozen 20260903 query plan passes bounded validation",
            "valid" if not plan_errors else "; ".join(plan_errors),
        ),
        (
            "network_policy",
            policy.is_file()
            and "SOTA-Ingestion Cycle Discipline" in policy.read_text(encoding="utf-8")
            and "low-concurrency" in policy.read_text(encoding="utf-8"),
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
    """Join the frozen plan to one terminal receipt per source family."""

    rows = []
    for planned in frozen_query_plan():
        row = deepcopy(planned)
        row.update(_QUERY_RESULTS[planned["query_id"]])
        row["terminal"] = True
        rows.append(row)
    return rows


_CANDIDATE_ROWS: tuple[JsonDict, ...] = (
    {
        "candidate_id": "hsrm",
        "title": "HSRM: Hidden-State Reward Models for Test-Time Verification",
        "canonical_url": "https://arxiv.org/abs/2608.30841",
        "date": "2026-08-31",
        "source_type": "paper_and_first_party_repository",
        "relevance": "Step-boundary hidden states motivate Exp6937 reachability and a tie-safe compact ranker.",
        "evidence_grade": "A",
        "code_available": False,
        "code_state": "readme_only_repository",
        "local_gguf_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The repository has no executable files and GGUF intermediate-state capture is unproven.",
        "terminal": True,
    },
    {
        "candidate_id": "sparse_reward_subsystem",
        "title": "Sparse Reward Subsystem in Large Language Models",
        "canonical_url": "https://arxiv.org/abs/2602.00986",
        "date": "2026-02-01",
        "source_type": "paper",
        "relevance": "Sparse coordinates motivate a canary paired with a full-state control.",
        "evidence_grade": "A",
        "code_available": False,
        "code_state": "paper_only_no_linked_code",
        "local_gguf_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "No code is public and transfer across Qwen and Gemma cannot be assumed.",
        "terminal": True,
    },
    {
        "candidate_id": "nsvif",
        "title": "Neuro-Symbolic Verification on Instruction Following of LLMs",
        "canonical_url": "https://arxiv.org/abs/2601.17789",
        "date": "2026-01-25",
        "source_type": "paper",
        "relevance": "Typed logical and semantic constraints reinforce span-first acquisition and independent checking.",
        "evidence_grade": "A",
        "code_available": False,
        "code_state": "paper_only_no_official_repository_found",
        "local_gguf_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Its neural semantic solver is not reproducible locally and cannot replace exact authority.",
        "terminal": True,
    },
    {
        "candidate_id": "sampling_for_quality",
        "title": "Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo",
        "canonical_url": "https://arxiv.org/abs/2604.16453",
        "date": "2026-04-07",
        "source_type": "paper",
        "relevance": "Soft weights, ESS resampling, and resample-move steps map to Exp6936.",
        "evidence_grade": "A",
        "code_available": False,
        "code_state": "paper_only_no_linked_code",
        "local_gguf_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "V607 scores complete candidates and cannot claim parity with prefix-potential decoding.",
        "terminal": True,
    },
    {
        "candidate_id": "ism",
        "title": "ISM: Self-Improving Strategy Memory for Continual Mathematical Reasoning",
        "canonical_url": "https://github.com/pdx97/ISM",
        "date": "2026-06-30",
        "source_type": "first_party_repository",
        "relevance": "Hard resets, strategy records, and delayed outcomes inform the episodic-memory arm.",
        "evidence_grade": "B",
        "code_available": True,
        "code_state": "executable_hosted_service_bound",
        "local_gguf_compatibility": "incompatible",
        "disposition": "accepted_existing",
        "exclusion_reason": "The public runner pins hosted OpenAI solver, synthesis, classification, and embedding services.",
        "terminal": True,
    },
    {
        "candidate_id": "memoir",
        "title": "Memoir: Should a Model Write to Its Memory While It Thinks?",
        "canonical_url": "https://arxiv.org/abs/2607.20792",
        "date": "2026-07-22",
        "source_type": "paper_and_first_party_repository",
        "relevance": "Its fixed-budget result supports post-episode writes and an unchanged read-only control.",
        "evidence_grade": "A",
        "code_available": True,
        "code_state": "public_reference_code",
        "local_gguf_compatibility": "compatible",
        "disposition": "accepted_existing",
        "exclusion_reason": "Exclude coupled within-episode reads and writes from the locked V607 design.",
        "terminal": True,
    },
    {
        "candidate_id": "solver_hard",
        "title": "Solver-Hard Is Not Model-Hard",
        "canonical_url": "https://arxiv.org/abs/2607.17047",
        "date": "2026-07-19",
        "source_type": "paper_and_first_party_repository",
        "relevance": "Measured model headroom must replace solver conflicts as the admission signal.",
        "evidence_grade": "A",
        "code_available": True,
        "code_state": "public_code_and_aggregate_data",
        "local_gguf_compatibility": "compatible",
        "disposition": "accepted_existing",
        "exclusion_reason": "Solver runtime and conflict counts remain diagnostics, not model-hardness labels.",
        "terminal": True,
    },
    {
        "candidate_id": "kan_verification",
        "title": "Optimized Piecewise Affine Abstractions of Neural Networks with Learnable Activation Functions",
        "canonical_url": "https://arxiv.org/abs/2602.06737",
        "date": "2026-08-02",
        "source_type": "paper",
        "relevance": "PWA abstractions give a paper-level MILP and SMT design for learnable activation networks.",
        "evidence_grade": "A",
        "code_available": False,
        "code_state": "paper_only_no_linked_code",
        "local_gguf_compatibility": "partial",
        "disposition": "accepted_new",
        "exclusion_reason": "The official record links no implementation, so it is not an executable V607 dependency.",
        "terminal": True,
    },
    {
        "candidate_id": "torx",
        "title": "A Framework for Stochastic Differentiable Programming",
        "canonical_url": "https://arxiv.org/abs/2608.01612",
        "date": "2026-08-13",
        "source_type": "paper_and_first_party_repository",
        "relevance": "Typed stochastic circuits remain a useful serializable sampler-interface comparator.",
        "evidence_grade": "A",
        "code_available": True,
        "code_state": "public_jax_package",
        "local_gguf_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "CPU software availability does not authorize XTR-0, Z1, speed, power, or hardware claims.",
        "terminal": True,
    },
    {
        "candidate_id": "thermalizers",
        "title": "Thermalizing Stochastic Programs",
        "canonical_url": "https://arxiv.org/abs/2608.01615",
        "date": "2026-08-13",
        "source_type": "paper_and_first_party_preview",
        "relevance": "Factor-level and accumulated error would be useful future portability checks.",
        "evidence_grade": "A",
        "code_available": False,
        "code_state": "preview_no_public_compiler",
        "local_gguf_compatibility": "incompatible",
        "disposition": "rejected",
        "exclusion_reason": "No public Thermalizers implementation or authenticated thermodynamic hardware route exists.",
        "terminal": True,
    },
)


def candidate_rows() -> list[JsonDict]:
    """Return a copy of every terminal named-candidate receipt."""

    return deepcopy(_CANDIDATE_ROWS)


def accepted_finding_rows() -> list[JsonDict]:
    """Return relevant candidates, including an incompatible method boundary."""

    return [row for row in candidate_rows() if row["disposition"].startswith("accepted")]


def rejected_finding_rows() -> list[JsonDict]:
    """Return candidates excluded from V607's dependency graph."""

    return [row for row in candidate_rows() if row["disposition"] == "rejected"]


def citation_edge_rows() -> list[JsonDict]:
    """Record citation relations checked during the dated source pass."""

    return [
        {
            "source_id": "arxiv:2608.30841",
            "target_id": "arxiv:2505.14999",
            "relation": "cites_comparator",
            "canonical_url": "https://arxiv.org/abs/2505.14999",
            "is_new": True,
            "evidence": "HSRM identifies the 55M outcome energy reward model as its text-only comparator.",
        },
        {
            "source_id": "arxiv:2608.30841",
            "target_id": "arxiv:2505.12225",
            "relation": "cites_hidden_state_verifier",
            "canonical_url": "https://arxiv.org/abs/2505.12225",
            "is_new": True,
            "evidence": "HSRM cites SWIFT in its hidden-state verification neighborhood.",
        },
        {
            "source_id": "arxiv:2608.30841",
            "target_id": "arxiv:2511.06209",
            "relation": "cites_hidden_state_verifier",
            "canonical_url": "https://arxiv.org/abs/2511.06209",
            "is_new": True,
            "evidence": "HSRM cites ReProbe in its hidden-state verification neighborhood.",
        },
        {
            "source_id": "arxiv:2607.20792",
            "target_id": "arxiv:2507.02092",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2507.02092",
            "is_new": False,
            "evidence": "The EBT Semantic Scholar citation receipt still includes Memoir.",
        },
        {
            "source_id": "arxiv:2607.17047",
            "target_id": "arxiv:2507.02092",
            "relation": "cites",
            "canonical_url": "https://arxiv.org/abs/2507.02092",
            "is_new": False,
            "evidence": "The EBT Semantic Scholar citation receipt still includes Solver-Hard.",
        },
    ]


def implementation_rows() -> list[JsonDict]:
    """Map reusable ideas to locked V607 tasks without creating gates."""

    candidates = {row["candidate_id"]: row for row in candidate_rows()}
    rows = [
        (
            "nsvif",
            "Exp6929/Exp6930",
            "Keep byte spans and typed relations; exact independent checks remain final authority.",
        ),
        (
            "ism",
            "Exp6931/Exp6932",
            "Reuse schema-bank and reset shapes only; make no hosted OpenAI call.",
        ),
        (
            "memoir",
            "Exp6932/Exp6933",
            "Write only after an episode and retain an unchanged read-only control.",
        ),
        (
            "solver_hard",
            "Exp6934/Exp6935",
            "Measure model headroom directly under a sealed proposal bank.",
        ),
        (
            "sampling_for_quality",
            "Exp6936",
            "Use complete-candidate energy under matched proposal and verifier budgets.",
        ),
        (
            "hsrm",
            "Exp6937/Exp6938",
            "Gate training on GGUF layer capture and compare against full-state and score-only controls.",
        ),
        (
            "sparse_reward_subsystem",
            "Exp6937",
            "Treat sparse coordinates as a canary and test transfer separately per model family.",
        ),
        (
            "kan_verification",
            "advisory_only",
            "Retain the PWA abstraction as a design source until executable code is available.",
        ),
        (
            "torx",
            "advisory_only",
            "Use software interface semantics only and make no thermodynamic-device claim.",
        ),
        (
            "thermalizers",
            "advisory_only",
            "Do not promote a preview compiler or unavailable hardware.",
        ),
    ]
    return [
        {
            "candidate_id": candidate_id,
            "source_url": candidates[candidate_id]["canonical_url"],
            "v607_experiment": experiment,
            "implementation_boundary": boundary,
        }
        for candidate_id, experiment, boundary in rows
    ]


def compatibility_rows() -> list[JsonDict]:
    """State the concrete local execution constraint for each deep check."""

    return [
        {
            "candidate_id": "hsrm",
            "constraint": "repository_state",
            "observed": "JXL884/HSRM contains a README but no executable implementation.",
            "effect": "Exp6937 must prove GGUF intermediate-state capture before any HSRM-style ranker.",
        },
        {
            "candidate_id": "sparse_reward_subsystem",
            "constraint": "hidden_state_access",
            "observed": "The paper exposes no code and reports transfer only across tested related variants.",
            "effect": "Do not assume sparse coordinate transfer across Qwen and Gemma.",
        },
        {
            "candidate_id": "nsvif",
            "constraint": "unavailable_semantic_solver",
            "observed": "No official public implementation was found.",
            "effect": "Reuse typed constraints but retain exact independent authority.",
        },
        {
            "candidate_id": "sampling_for_quality",
            "constraint": "runner_interface",
            "observed": "The paper links no local-GGUF reference implementation.",
            "effect": "Use the algorithm shape only with complete candidates and matched budgets.",
        },
        {
            "candidate_id": "ism",
            "constraint": "hosted_services",
            "observed": "The runner calls gpt-4.1-mini, gpt-4o, gpt-4o-mini, and text-embedding-3-small.",
            "effect": "ISM cannot be promoted as a local-GGUF dependency.",
        },
        {
            "candidate_id": "memoir",
            "constraint": "causal_budget",
            "observed": "Coupled memory writes can be slower under a fixed budget.",
            "effect": "Retain delayed writes and the read-only control.",
        },
        {
            "candidate_id": "solver_hard",
            "constraint": "proxy_validity",
            "observed": "Solver hardness does not reliably measure model difficulty.",
            "effect": "Use measured local success headroom, not solver conflicts.",
        },
        {
            "candidate_id": "kan_verification",
            "constraint": "implementation_availability",
            "observed": "The official paper record links no code or repository.",
            "effect": "Keep PWA verification as paper-level guidance only.",
        },
        {
            "candidate_id": "torx",
            "constraint": "hardware_access",
            "observed": "Public JAX code does not establish authenticated XTR-0 or Z1 execution.",
            "effect": "Software semantics are reusable; device claims are not.",
        },
        {
            "candidate_id": "thermalizers",
            "constraint": "hardware_access",
            "observed": "Thermalizers remains a preview with no public compiler or authenticated device route.",
            "effect": "Defer reproduction and do not promote the dependency.",
        },
    ]


_LEDGER_APPEND_ROWS: tuple[JsonDict, ...] = (
    {
        "candidate_id": "kan_verification_metadata_code_state",
        "title": "KAN verification metadata and implementation correction",
        "canonical_url": "https://arxiv.org/abs/2602.06737",
        "date": "2026-09-03",
        "finding": "The official v2 title is 'Optimized Piecewise Affine Abstractions of Neural Networks with Learnable Activation Functions'; the record links no public code or repository.",
        "v608_use": "Treat it as a paper-level PWA verification design, not an executable dependency, unless a first-party implementation appears.",
    },
)


def ledger_append_rows() -> list[JsonDict]:
    """Return the one verified correction absent from the prior ledger."""

    return deepcopy(_LEDGER_APPEND_ROWS)


def v608_candidate_rows() -> list[JsonDict]:
    """Keep new or unavailable ideas outside the locked V607 roadmap."""

    return [
        {
            "candidate_id": "introconformal_hidden_state_canary",
            "target_milestone": "V608+",
            "proposal": "Recheck conformal hidden-state risk signals on text GGUF outputs after local layer capture exists.",
            "prerequisite": "A public implementation or separately specified text-only calibration protocol.",
            "canonical_url": "https://arxiv.org/abs/2609.01375",
        },
        {
            "candidate_id": "kan_pwa_reproduction",
            "target_milestone": "V608+",
            "proposal": "Reproduce a bounded PWA abstraction against an existing exact solver.",
            "prerequisite": "First-party code or a separately specified clean-room implementation.",
            "canonical_url": "https://arxiv.org/abs/2602.06737",
        },
        {
            "candidate_id": "thermalizers_recheck",
            "target_milestone": "V608+",
            "proposal": "Recheck compilation error only after a public compiler and accessible substrate exist.",
            "prerequisite": "Public Thermalizers code and an authenticated compatible runtime.",
            "canonical_url": "https://arxiv.org/abs/2608.01615",
        },
    ]


def source_family_rows(queries: list[JsonDict]) -> list[JsonDict]:
    """Reduce query receipts to one explicit terminal family row."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def source_artifact_hashes(root: Path) -> dict[str, str]:
    """Hash every local source named by the task contract."""

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
    """Build either a complete advisory receipt or a schema-complete block."""

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
    future = [] if failed else v608_candidate_rows()
    rate_limits = [row for row in queries if row["terminal_outcome"] == "rate_limited"]
    complete = bool(
        not failed
        and {row["source_family"] for row in queries} == set(SOURCE_FAMILIES)
        and all(row["terminal"] for row in queries)
        and {row["candidate_id"] for row in candidates} == set(NAMED_CANDIDATES)
        and all(row["terminal"] for row in candidates)
    )
    rows = _combined_rows(
        [
            ("query", queries),
            ("source_family", families),
            ("candidate", candidates),
            ("citation_edge", citations),
            ("implementation", implementations),
            ("compatibility", compatibilities),
            ("ledger_append", appends),
            ("v608_candidate", future),
            ("rate_limit", rate_limits),
        ]
    )
    if failed:
        status = "blocked"
        verdict_class = "blocked"
        honest_verdict = "blocked_v607_literature_delta"
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
            "complete_v607_literature_delta_with_one_verified_correction"
            if appends
            else "complete_null_v607_literature_delta_no_new_findings"
        )
        gate_summary = {
            "all_gates_passed": complete,
            "failed_check": None,
            "expected": "15 terminal source families and 10 terminal named candidates",
            "observed": f"{len(families)} source families and {len(candidates)} candidates",
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
        "v608_candidate_rows": future,
        "rate_limit_rows": rate_limits,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v607_literature_delta_complete_score": int(complete),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact_or_path: dict[str, object] | Path) -> list[str]:
    """Recompute completion and reject inconsistent terminal receipts."""

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
    blocked = artifact["status"] == "blocked"
    if blocked:
        if (
            artifact["v607_literature_delta_complete_score"] != 0
            or artifact["verdict_class"] != "blocked"
            or artifact["honest_verdict"] != "blocked_v607_literature_delta"
        ):
            errors.append("blocked artifact fields are inconsistent")
    else:
        queries = artifact["query_rows"]
        candidates = artifact["candidate_rows"]
        source_complete = {row["source_family"] for row in queries} == set(SOURCE_FAMILIES) and all(
            row["terminal"] for row in queries
        )
        candidate_complete = {row["candidate_id"] for row in candidates} == set(
            NAMED_CANDIDATES
        ) and all(row["terminal"] for row in candidates)
        if not source_complete:
            errors.append("source family coverage is incomplete")
        if not candidate_complete:
            errors.append("candidate coverage is incomplete")
        if artifact["v607_literature_delta_complete_score"] != int(
            source_complete and candidate_complete
        ):
            errors.append("completion score does not match terminal rows")
        if not str(artifact["honest_verdict"]).startswith("complete_"):
            errors.append("complete artifact honest verdict lacks complete_ prefix")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def append_verified_findings(root: Path, artifact: dict[str, object]) -> list[str]:
    """Append marked verified facts once while retaining every prior byte."""

    path = root / LEDGER_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    appended = []
    blocks = []
    for row in artifact["ledger_append_rows"]:
        marker = f"<!-- EXP6927-{row['candidate_id'].upper()} -->"
        if marker in text:
            continue
        blocks.append(
            "\n".join(
                (
                    marker,
                    f"### {row['title']} - {row['date']}",
                    f"- **Source:** {row['canonical_url']}",
                    f"- **Verified finding:** {row['finding']}",
                    f"- **V608 boundary:** {row['v608_use']}",
                )
            )
        )
        appended.append(row["candidate_id"])
    if blocks:
        suffix = (
            "\n\n## V607 Execution-Time Literature Delta - 2026-09-03\n\n"
            + "\n\n".join(blocks)
            + "\n"
        )
        atomic_write_text(path, text.rstrip() + suffix, allow_override=False)
    return appended


def write_artifact(artifact: dict[str, object], path: Path) -> Path:
    """Atomically write one validated Exp6927 artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return atomic_write_json(path, artifact, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """Write the dated literature delta or validate an existing receipt."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        return int(bool(validate_artifact(args.validate)))
    root = find_repo_root()
    output = args.output or root / RESULT_RELATIVE_PATH
    draft = build_artifact(root=root, run_date=args.date, duration_s=720.0, output_path=output)
    if draft["status"] != "blocked":
        append_verified_findings(root, draft)
        draft = build_artifact(root=root, run_date=args.date, duration_s=720.0, output_path=output)
    write_artifact(draft, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
