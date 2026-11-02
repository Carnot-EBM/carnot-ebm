"""Build the bounded V606 execution-time SOTA ingestion artifact.

Spec refs: REQ-REPORT-6925 and SCENARIO-REPORT-6925-*.

The module records a small source recheck. The query plan is fixed in source
before access. This prevents a later result from changing the search terms
that produced it.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_6925_v606_sota_ingestion"
RUN_DATE = "20260903"
RANDOM_SEED = 6925
INFERENCE_SUBSTRATE = "bounded_primary_source_web_research_no_model_inference"
SCHEMA_VERSION = "carnot.experiment_6925.v606_sota_ingestion.v1"

MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6925_v606_sota_ingestion.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6925_v606_sota_ingestion.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6925_v606_sota_ingestion.json")
LEDGER_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
DESIGN_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
NETWORK_POLICY_RELATIVE_PATH = Path("CLAUDE.md")

ARXIV_FAMILIES = (
    "arxiv_ebm",
    "arxiv_neural_constraints",
    "arxiv_ising_ml",
    "arxiv_hallucination_verification",
    "arxiv_kan",
    "arxiv_constrained_generation",
    "arxiv_accelerated_sampling",
    "arxiv_continual_learning",
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
    "vera",
    "solver_hard",
    "sampling_for_quality",
    "twisted_smc",
    "syntactic_semantic_smc",
    "ism",
    "memoir",
    "cl_bench",
    "grounded_in_context",
    "span_level_hallucination",
    "torx",
    "thermalizers",
    "smc_sd",
    "kanele",
    "symbolic_kan",
)

_ACCEPTANCE = (
    "Accept only a primary paper or first-party source with stable identity, a verifiable date, "
    "clear V606 relevance, and an executable local consequence. Record duplicates, no-update "
    "results, access failures, and incompatible implementations as terminal non-findings."
)

# This is the dated query plan. It was frozen at 2026-09-03T10:20:00Z before
# the first network request. Planned times order the sequential requests; they
# are not claims about response time.
_FROZEN_QUERY_PLAN: tuple[JsonDict, ...] = (
    {
        "query_id": "arxiv_ebm",
        "source_family": "arxiv_ebm",
        "query_text": '("energy-based model" OR "energy-based transformer") AND (reasoning OR verification) AND submittedDate:[202501010000 TO 202612312359]',
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:30:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "arxiv_neural_constraints",
        "source_family": "arxiv_neural_constraints",
        "query_text": '("neural constraint" OR "constraint reasoning" OR neuro-symbolic) AND submittedDate:[202501010000 TO 202612312359]',
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:31:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "arxiv_ising_ml",
        "source_family": "arxiv_ising_ml",
        "query_text": '(Ising AND ("machine learning" OR sampling OR "probabilistic computing")) AND submittedDate:[202501010000 TO 202612312359]',
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:32:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "arxiv_hallucination_verification",
        "source_family": "arxiv_hallucination_verification",
        "query_text": '((hallucination AND (verification OR detection)) OR "span grounding") AND submittedDate:[202501010000 TO 202612312359]',
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:33:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "arxiv_kan",
        "source_family": "arxiv_kan",
        "query_text": '("Kolmogorov-Arnold Network" OR KAN) AND (symbolic OR constraint OR FPGA OR LUT) AND submittedDate:[202501010000 TO 202612312359]',
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:34:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "arxiv_constrained_generation",
        "source_family": "arxiv_constrained_generation",
        "query_text": '("constrained generation" OR "controlled generation") AND (SMC OR "sequential Monte Carlo" OR verifier) AND submittedDate:[202501010000 TO 202612312359]',
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:35:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "arxiv_accelerated_sampling",
        "source_family": "arxiv_accelerated_sampling",
        "query_text": "((accelerated OR speculative OR reward-guided) AND sampling AND (LLM OR EBM OR Ising)) AND submittedDate:[202501010000 TO 202612312359]",
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:36:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "arxiv_continual_learning",
        "source_family": "arxiv_continual_learning",
        "query_text": '(("continual learning" OR "strategy memory" OR "episodic memory") AND (LLM OR agent)) AND submittedDate:[202501010000 TO 202612312359]',
        "canonical_url": "https://export.arxiv.org/api/query",
        "allowed_domains": ["arxiv.org", "export.arxiv.org"],
        "planned_utc": "2026-09-03T10:37:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "openreview",
        "source_family": "openreview",
        "query_text": 'site:openreview.net ("energy based" OR constraint OR verifier OR continual memory) ICLR NeurIPS 2025 2026',
        "canonical_url": "https://openreview.net/search",
        "allowed_domains": ["openreview.net"],
        "planned_utc": "2026-09-03T10:38:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "huggingface_papers",
        "source_family": "huggingface_papers",
        "query_text": 'site:huggingface.co/papers (VeRA OR ISM OR CL-Bench OR verification OR "continual learning")',
        "canonical_url": "https://huggingface.co/papers",
        "allowed_domains": ["huggingface.co"],
        "planned_utc": "2026-09-03T10:39:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "semantic_scholar_ebt",
        "source_family": "semantic_scholar_ebt",
        "query_text": "ARXIV:2507.02092 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "canonical_url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations",
        "allowed_domains": ["api.semanticscholar.org", "semanticscholar.org"],
        "planned_utc": "2026-09-03T10:40:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "semantic_scholar_arm_ebm",
        "source_family": "semantic_scholar_arm_ebm",
        "query_text": "ARXIV:2512.15605 citations fields=title,year,url,externalIds,publicationDate limit=100",
        "canonical_url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations",
        "allowed_domains": ["api.semanticscholar.org", "semanticscholar.org"],
        "planned_utc": "2026-09-03T10:41:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "github_official",
        "source_family": "github_official",
        "query_text": "official repositories ISM SMC-SD KANELE EBT Torx updated 2025 2026",
        "canonical_url": "https://github.com/search",
        "allowed_domains": ["github.com", "api.github.com"],
        "planned_utc": "2026-09-03T10:42:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "extropic",
        "source_family": "extropic",
        "query_text": "Extropic Torx Thermalizers Z1 XTR-0 implementation update",
        "canonical_url": "https://extropic.ai/writing/from-one-to-one-billion",
        "allowed_domains": ["extropic.ai"],
        "planned_utc": "2026-09-03T10:43:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
    {
        "query_id": "logical_intelligence",
        "source_family": "logical_intelligence",
        "query_text": "Logical Intelligence Kona EBM weights code training recipe runner update",
        "canonical_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "allowed_domains": ["logicalintelligence.com"],
        "planned_utc": "2026-09-03T10:44:00Z",
        "max_attempts": 2,
        "concurrency_limit": 1,
        "timeout_s": 20,
        "acceptance_criteria": _ACCEPTANCE,
    },
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
    "v607_candidate_rows",
    "rate_limit_rows",
    "random_seed",
    "reproducibility_checksum",
    "v606_sota_ingestion_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema lets later validators detect a changed artifact contract.",
    "experiment_id": "A stable identity binds the research receipt to the scheduled V606 task.",
    "run_date": "The date separates this execution-time check from the earlier planning sweep.",
    "status": "A terminal status distinguishes a complete source pass from a blocked preflight.",
    "field_principles": "A principle for each field explains why the evidence is needed.",
    "preconditions_checked": "Preflight rows prove the roadmap, ledger, policy, and dated plan existed before research.",
    "inference_substrate": "The substrate states that this task used source research and no model inference.",
    "model_specs": "An explicit empty invocation receipt prevents cited model names from implying that inference ran.",
    "duration_s": "Elapsed research time makes an implausibly short source claim visible.",
    "source_artifact_hashes": "Hashes bind the result to the exact local sources that defined V606.",
    "rows": "A combined row ledger lets a checker inspect evidence without trusting summaries.",
    "query_rows": "One terminal row per query preserves no-update and access-failure outcomes.",
    "source_family_rows": "Family summaries prove that every required source surface terminated.",
    "candidate_rows": "One row per named method prevents selective reporting of rechecks.",
    "accepted_finding_rows": "Accepted rows show which checked methods remain relevant or changed.",
    "rejected_finding_rows": "Rejected rows retain exclusions and stop weak evidence from disappearing.",
    "citation_edge_rows": "Citation edges bind each local method claim to a checked canonical source.",
    "implementation_rows": "Implementation rows map source methods to exact V606 experiments and boundaries.",
    "compatibility_rows": "Compatibility rows expose dependencies that the local stack cannot reuse directly.",
    "ledger_append_rows": "Append rows isolate genuinely new verified facts from repeated ledger content.",
    "v607_candidate_rows": "Future ideas stay advisory and cannot silently rewrite locked V606 work.",
    "rate_limit_rows": "Rate-limit rows keep missing responses from becoming false no-update claims.",
    "random_seed": "A fixed seed makes deterministic row ordering reproducible.",
    "reproducibility_checksum": "The payload hash detects any later change to the research receipt.",
    "v606_sota_ingestion_complete_score": "The score becomes one only when every required source and method has a terminal row.",
    "gate_check_summary": "The gate summary names exact expected and observed values when preflight blocks.",
    "verifier_is_oracle": "False states that source review is not an executable correctness oracle.",
    "verdict_class": "The class separates an advisory update from a null or blocked research outcome.",
    "honest_verdict": "A terminal complete_ prefix lets the conductor classify the outcome without guessing.",
}

_QUERY_RESULTS: dict[str, JsonDict] = {
    "arxiv_ebm": {
        "actual_utc": "2026-09-03T10:20:42Z",
        "attempt_count": 2,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The results returned EBT and Distributional EBM. Both sources were already in the ledger.",
    },
    "arxiv_neural_constraints": {
        "actual_utc": "2026-09-03T10:21:02Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The relevant symbolic-integration position paper was already selected for V606.",
    },
    "arxiv_ising_ml": {
        "actual_utc": "2026-09-03T10:21:18Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "No returned 2025-2026 Ising result changed the V606 no-hardware boundary.",
    },
    "arxiv_hallucination_verification": {
        "actual_utc": "2026-09-03T10:21:32Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "Span-level work, including arXiv:2607.00895, was already present in the ledger.",
    },
    "arxiv_kan": {
        "actual_utc": "2026-09-03T10:21:47Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "KANELÉ and Symbolic-KAN remained the relevant results and were already ledgered.",
    },
    "arxiv_constrained_generation": {
        "actual_utc": "2026-09-03T10:22:03Z",
        "attempt_count": 2,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The second bounded attempt returned the already-ledgered self-distilled twisted-SMC paper.",
    },
    "arxiv_accelerated_sampling": {
        "actual_utc": "2026-09-03T10:22:19Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "No result superseded Sampling for Quality or changed the SMC-SD compatibility boundary.",
    },
    "arxiv_continual_learning": {
        "actual_utc": "2026-09-03T10:22:35Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "ISM, AdaMEM, AgentCL, and related strategy-memory work were already present in the ledger.",
    },
    "openreview": {
        "actual_utc": "2026-09-03T10:22:52Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "Current ICLR and NeurIPS records added no reproducible method that changes V606.",
    },
    "huggingface_papers": {
        "actual_utc": "2026-09-03T10:23:08Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "CL-Bench, AgentCL, and SkillLearnBench pages repeated already-ledgered sources.",
    },
    "semantic_scholar_ebt": {
        "actual_utc": "2026-09-03T10:23:32Z",
        "attempt_count": 2,
        "http_status": 200,
        "status": "success_after_tool_rejection",
        "terminal_outcome": "no_update",
        "evidence": "The direct API returned 35 rows. The newest visible paper was arXiv:2608.14186, already ledgered.",
    },
    "semantic_scholar_arm_ebm": {
        "actual_utc": "2026-09-03T10:23:38Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The direct API returned eight rows. No row post-dated the prior V606 sweep.",
    },
    "github_official": {
        "actual_utc": "2026-09-03T10:24:01Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "implementation_update",
        "evidence": "ISM, SMC-SD, and KANELÉ repositories were reachable. ISM disclosed hosted-model dependencies not recorded in the V606 section.",
    },
    "extropic": {
        "actual_utc": "2026-09-03T10:24:18Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "no_update",
        "evidence": "The August update still exposes Torx, previews Thermalizers, and targets Z1 access in 2027.",
    },
    "logical_intelligence": {
        "actual_utc": "2026-09-03T10:24:32Z",
        "attempt_count": 1,
        "http_status": 200,
        "status": "success",
        "terminal_outcome": "unavailable_reproducible_assets",
        "evidence": "The Kona page remained a product description with no public weights, training recipe, or local runner.",
    },
}


def validate_query_plan(plan: list[JsonDict], run_date: str) -> list[str]:
    """Check the fixed plan before any caller uses its source routes."""

    expected_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}T"
    if run_date != RUN_DATE or any(
        not row["planned_utc"].startswith(expected_date) for row in plan
    ):
        return ["query plan date does not match run date"]
    errors: list[str] = []
    for row in plan:
        if row["max_attempts"] != 2:
            errors.append(f"query {row['query_id']} must allow exactly two attempts")
        if row["concurrency_limit"] != 1:
            errors.append(f"query {row['query_id']} must use concurrency one")
        if row["timeout_s"] != 20:
            errors.append(f"query {row['query_id']} must use the 20 second timeout")
        if not row["query_text"] or not row["allowed_domains"] or not row["acceptance_criteria"]:
            errors.append(f"query {row['query_id']} lacks a frozen source field")
    if [row["source_family"] for row in plan[:8]] != list(ARXIV_FAMILIES):
        errors.append("arXiv source families must be first")
    if {row["source_family"] for row in plan} != set(SOURCE_FAMILIES):
        errors.append("query plan source family coverage is incomplete")
    return errors


def check_preconditions(root: Path, run_date: str) -> list[JsonDict]:
    """Return fail-closed checks for every required local research input."""

    roadmap = root / ROADMAP_RELATIVE_PATH
    ledger = root / LEDGER_RELATIVE_PATH
    policy = root / NETWORK_POLICY_RELATIVE_PATH
    plan_errors = validate_query_plan(frozen_query_plan(), run_date)
    checks = (
        (
            "current_roadmap",
            roadmap.exists() and "milestone: 2026.09.606" in roadmap.read_text(encoding="utf-8"),
            "active roadmap exists and declares 2026.09.606",
            roadmap.as_posix() if roadmap.exists() else "missing",
        ),
        (
            "reference_ledger",
            ledger.exists(),
            "research-references.md exists",
            ledger.as_posix() if ledger.exists() else "missing",
        ),
        (
            "network_policy",
            policy.exists()
            and "SOTA-Ingestion Cycle Discipline" in policy.read_text(encoding="utf-8")
            and "low-concurrency" in policy.read_text(encoding="utf-8"),
            "CLAUDE.md contains the low-concurrency SOTA-ingestion policy",
            policy.as_posix() if policy.exists() else "missing",
        ),
        (
            "dated_query_plan",
            not plan_errors,
            "frozen query plan dated 20260903 passes validation",
            "valid" if not plan_errors else "; ".join(plan_errors),
        ),
    )
    return [
        {"resource": name, "available": available, "expected": expected, "observed": observed}
        for name, available, expected, observed in checks
    ]


def query_rows() -> list[JsonDict]:
    """Join the frozen query plan to the retained terminal receipts."""

    rows = []
    for planned in frozen_query_plan():
        row = deepcopy(planned)
        row.update(_QUERY_RESULTS[planned["query_id"]])
        row["terminal"] = True
        rows.append(row)
    return rows


_CANDIDATE_ROWS: tuple[JsonDict, ...] = (
    {
        "candidate_id": "vera",
        "title": "VeRA: Verified Reasoning Data Augmentation at Scale",
        "canonical_url": "https://arxiv.org/abs/2602.13217",
        "date": "2026-01-23",
        "source_type": "paper",
        "relevance": "Executable generators create fresh verified variants for the V606 headroom fixture.",
        "evidence_grade": "A",
        "code_available": True,
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Reuse the generator-and-checker pattern only; do not import unknown benchmark assumptions.",
        "terminal": True,
    },
    {
        "candidate_id": "solver_hard",
        "title": "Solver-Hard Is Not Model-Hard",
        "canonical_url": "https://arxiv.org/abs/2607.17047",
        "date": "2026-07-19",
        "source_type": "paper",
        "relevance": "Measured model headroom must replace solver conflict counts as the admission signal.",
        "evidence_grade": "A",
        "code_available": True,
        "local_compatibility": "compatible",
        "disposition": "accepted_existing",
        "exclusion_reason": "Exclude solver runtime and conflict counts from model-hardness labels.",
        "terminal": True,
    },
    {
        "candidate_id": "sampling_for_quality",
        "title": "Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo",
        "canonical_url": "https://arxiv.org/abs/2604.16453",
        "date": "2026-04-07",
        "source_type": "paper",
        "relevance": "Soft weights, ESS resampling, and resample-move updates map to Exp6934.",
        "evidence_grade": "A",
        "code_available": False,
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Use complete-candidate energy; do not claim parity with the paper's prefix reward model.",
        "terminal": True,
    },
    {
        "candidate_id": "twisted_smc",
        "title": "Improving Constrained Language Generation via Self-Distilled Twisted Sequential Monte Carlo",
        "canonical_url": "https://arxiv.org/abs/2507.02315",
        "date": "2025-07-03",
        "source_type": "paper",
        "relevance": "Twisted SMC supports particle allocation under sparse constraints.",
        "evidence_grade": "A",
        "code_available": False,
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "V606 forbids self-distillation and weight changes; only the inference primitive remains relevant.",
        "terminal": True,
    },
    {
        "candidate_id": "syntactic_semantic_smc",
        "title": "Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo",
        "canonical_url": "https://arxiv.org/abs/2504.13139",
        "date": "2025-04-18",
        "source_type": "paper",
        "relevance": "The paper supports adaptive particle allocation for controlled generation.",
        "evidence_grade": "A",
        "code_available": True,
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Its probabilistic-program stack is not the local llama.cpp complete-candidate interface.",
        "terminal": True,
    },
    {
        "candidate_id": "ism",
        "title": "ISM: Self-Improving Strategy Memory for Continual Mathematical Reasoning",
        "canonical_url": "https://github.com/pdx97/ISM",
        "date": "2026-06-30",
        "source_type": "first_party_repository",
        "relevance": "The repository exposes the strategy bank, episodic reset, and active maintenance implementation.",
        "evidence_grade": "B",
        "code_available": True,
        "local_compatibility": "partial",
        "disposition": "accepted_new",
        "exclusion_reason": "The reference runner requires hosted OpenAI solvers, synthesizers, and embeddings; V606 can reuse only the data and reset pattern.",
        "terminal": True,
    },
    {
        "candidate_id": "memoir",
        "title": "Memoir: Should a Model Write to Its Memory While It Thinks?",
        "canonical_url": "https://arxiv.org/abs/2607.20792",
        "date": "2026-07-22",
        "source_type": "paper",
        "relevance": "Its fixed-budget result supports delayed post-episode memory writes.",
        "evidence_grade": "A",
        "code_available": True,
        "local_compatibility": "compatible",
        "disposition": "accepted_existing",
        "exclusion_reason": "Exclude within-episode coupled reads and writes from V606.",
        "terminal": True,
    },
    {
        "candidate_id": "cl_bench",
        "title": "Continual Learning Bench: Evaluating Frontier AI Systems in Real-World Stateful Environments",
        "canonical_url": "https://arxiv.org/abs/2606.05661",
        "date": "2026-06-04",
        "source_type": "paper",
        "relevance": "Gain over prior ability and forgetting metrics map to the four-arm V606 audit.",
        "evidence_grade": "A",
        "code_available": False,
        "local_compatibility": "compatible",
        "disposition": "accepted_existing",
        "exclusion_reason": "Do not treat a memory-write count as a learning result.",
        "terminal": True,
    },
    {
        "candidate_id": "grounded_in_context",
        "title": "ORION Grounded in Context: Retrieval-Based Method for Hallucination Detection",
        "canonical_url": "https://arxiv.org/abs/2504.15771",
        "date": "2025-05-22",
        "source_type": "paper",
        "relevance": "Small claim units aligned to retrieved evidence support span-first acquisition.",
        "evidence_grade": "A",
        "code_available": False,
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Do not use its learned NLI score as final relation authority.",
        "terminal": True,
    },
    {
        "candidate_id": "span_level_hallucination",
        "title": "Span-Level Hallucination Detection for LLM-Generated Answers",
        "canonical_url": "https://arxiv.org/abs/2504.18639",
        "date": "2025-04-25",
        "source_type": "paper",
        "relevance": "Semantic-role decomposition reinforces exact evidence-span capture before classification.",
        "evidence_grade": "A",
        "code_available": False,
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "Exclude GPT and NLI adjudication from the final exact V606 qualifier.",
        "terminal": True,
    },
    {
        "candidate_id": "torx",
        "title": "A Framework for Stochastic Differentiable Programming",
        "canonical_url": "https://arxiv.org/abs/2608.01612",
        "date": "2026-08-13",
        "source_type": "paper",
        "relevance": "Typed stochastic circuits motivate a serializable sampler interface.",
        "evidence_grade": "A",
        "code_available": True,
        "local_compatibility": "partial",
        "disposition": "accepted_existing",
        "exclusion_reason": "The local extro-torx 0.0.1 CPU path does not authorize an XTR-0 or Z1 claim.",
        "terminal": True,
    },
    {
        "candidate_id": "thermalizers",
        "title": "Thermalizing Stochastic Programs",
        "canonical_url": "https://arxiv.org/abs/2608.01615",
        "date": "2026-08-13",
        "source_type": "paper",
        "relevance": "Per-factor and accumulated compilation error define a future portability check.",
        "evidence_grade": "A",
        "code_available": False,
        "local_compatibility": "incompatible",
        "disposition": "rejected",
        "exclusion_reason": "Thermalizers remains a preview and no authenticated thermodynamic device route exists.",
        "terminal": True,
    },
    {
        "candidate_id": "smc_sd",
        "title": "SMC Speculative Decoding",
        "canonical_url": "https://github.com/abdelfattah-lab/smcsd",
        "date": "2026-04-17",
        "source_type": "first_party_repository",
        "relevance": "Its ESS-driven particle reweighting is a useful implementation comparison.",
        "evidence_grade": "B",
        "code_available": True,
        "local_compatibility": "incompatible",
        "disposition": "rejected",
        "exclusion_reason": "The current repository requires patched SGLang; V606 uses llama.cpp and targets quality rather than speculative throughput.",
        "terminal": True,
    },
    {
        "candidate_id": "kanele",
        "title": "KANELÉ: Kolmogorov-Arnold Networks for Efficient LUT-based Evaluation",
        "canonical_url": "https://github.com/Duchstf/KANELE",
        "date": "2025-12-14",
        "source_type": "first_party_repository",
        "relevance": "The repository provides a concrete future LUT-based KAN hardware path.",
        "evidence_grade": "B",
        "code_available": True,
        "local_compatibility": "incompatible",
        "disposition": "rejected",
        "exclusion_reason": "It targets Vivado 2024.1 and Xilinx devices; Vivado is absent and V606 has no FPGA task.",
        "terminal": True,
    },
    {
        "candidate_id": "symbolic_kan",
        "title": "Symbolic-KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning",
        "canonical_url": "https://arxiv.org/abs/2603.23854",
        "date": "2026-03-25",
        "source_type": "paper",
        "relevance": "Discrete primitive selection could support future interpretable constraint energy.",
        "evidence_grade": "A",
        "code_available": False,
        "local_compatibility": "incompatible",
        "disposition": "rejected",
        "exclusion_reason": "No public implementation was found and V606 excludes adaptive KAN work from its blocking graph.",
        "terminal": True,
    },
)


def candidate_rows() -> list[JsonDict]:
    """Return one terminal evidence row for every named recheck."""

    return deepcopy(_CANDIDATE_ROWS)


def implementation_rows() -> list[JsonDict]:
    """Map the five requested method families to locked V606 experiments."""

    return [
        {
            "method_family": "grounded_acquisition",
            "candidate_id": "grounded_in_context",
            "source_url": "https://arxiv.org/abs/2504.15771",
            "v606_experiment": "Exp6926",
            "reusable_method": "Copy exact evidence spans before assigning a semantic relation.",
            "implementation_boundary": "Deterministic offsets and ASP effects remain final authority; no learned NLI judge.",
        },
        {
            "method_family": "continual_memory",
            "candidate_id": "ism",
            "source_url": "https://github.com/pdx97/ISM",
            "v606_experiment": "Exp6930",
            "reusable_method": "Use hard episode resets and a compact success-and-failure strategy bank.",
            "implementation_boundary": "Use local frozen-model rows and exact delayed outcomes, not hosted OpenAI calls.",
        },
        {
            "method_family": "verified_variant",
            "candidate_id": "vera",
            "source_url": "https://arxiv.org/abs/2602.13217",
            "v606_experiment": "Exp6932",
            "reusable_method": "Generate fresh equivalent and hardened instances from executable specifications.",
            "implementation_boundary": "Keep generation and final checking independent and seal held-out seeds.",
        },
        {
            "method_family": "sequential_monte_carlo",
            "candidate_id": "sampling_for_quality",
            "source_url": "https://arxiv.org/abs/2604.16453",
            "v606_experiment": "Exp6934",
            "reusable_method": "Use soft Boltzmann weights, ESS resampling, and bounded resample-move steps.",
            "implementation_boundary": "Particles are complete candidates and budgets match both controls.",
        },
        {
            "method_family": "hardware_sampling",
            "candidate_id": "torx",
            "source_url": "https://arxiv.org/abs/2608.01612",
            "v606_experiment": "Exp6934",
            "reusable_method": "Keep particle and factor rows serializable as typed stochastic-program inputs.",
            "implementation_boundary": "V606 runs no thermodynamic hardware and makes no speed, power, or device claim.",
        },
    ]


def compatibility_rows() -> list[JsonDict]:
    """Record concrete source-to-local implementation mismatches."""

    return [
        {
            "candidate_id": "ism",
            "constraint": "hosted_models",
            "observed": "gpt-4.1-mini, gpt-4o, gpt-4o-mini, and text-embedding-3-small",
            "effect": "Reuse the schema format only; implement V606 with local frozen-model evidence and exact outcomes.",
        },
        {
            "candidate_id": "twisted_smc",
            "constraint": "weight_mutation",
            "observed": "self-distillation refines the base model",
            "effect": "V606 may reuse only the training-free particle concept.",
        },
        {
            "candidate_id": "syntactic_semantic_smc",
            "constraint": "runner_interface",
            "observed": "language-model probabilistic-program stack",
            "effect": "Build a bounded llama.cpp complete-candidate sampler instead of importing the stack.",
        },
        {
            "candidate_id": "smc_sd",
            "constraint": "backend",
            "observed": "patched SGLang is required and is not installed",
            "effect": "Do not claim SMC-SD throughput or direct compatibility.",
        },
        {
            "candidate_id": "torx",
            "constraint": "substrate",
            "observed": "extro-torx 0.0.1 is importable on the local CPU path",
            "effect": "Serialization experiments are possible, but no thermodynamic hardware claim is allowed.",
        },
        {
            "candidate_id": "thermalizers",
            "constraint": "availability",
            "observed": "paper and preview only",
            "effect": "Defer compiler parity and accumulated-error reproduction.",
        },
        {
            "candidate_id": "kanele",
            "constraint": "toolchain",
            "observed": "Vivado 2024.1 and named Xilinx devices are required; Vivado is absent",
            "effect": "Defer FPGA work beyond V606.",
        },
        {
            "candidate_id": "symbolic_kan",
            "constraint": "implementation",
            "observed": "no public code found",
            "effect": "Keep as a V607 candidate, not an executable V606 dependency.",
        },
        {
            "candidate_id": "logical_intelligence",
            "constraint": "assets",
            "observed": "no public Kona weights, recipe, or local runner",
            "effect": "Keep Kona as an architecture comparator only.",
        },
    ]


def citation_edge_rows() -> list[JsonDict]:
    """Bind every candidate disposition to the source that supports it."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "canonical_url": row["canonical_url"],
            "supports": row["relevance"],
            "evidence_grade": row["evidence_grade"],
        }
        for row in candidate_rows()
    ]


def v607_candidate_rows() -> list[JsonDict]:
    """Keep roadmap-changing ideas outside the locked V606 contract."""

    return [
        {
            "candidate_id": "ism_local_adapter",
            "target_milestone": "V607+",
            "proposal": "Test a local typed strategy-bank adapter against the unchanged V606 read-only control.",
            "prerequisite": "V606 Exp6931 terminal cold audit.",
        },
        {
            "candidate_id": "smc_sd_backend_canary",
            "target_milestone": "V607+",
            "proposal": "Recheck a pinned SMC-SD backend only if it supports the local runner and matched quality budgets.",
            "prerequisite": "An installable SGLang or llama.cpp-compatible reference path.",
        },
        {
            "candidate_id": "kanele_simulation",
            "target_milestone": "V607+",
            "proposal": "Run a device-free KANELÉ conversion and simulation canary before any board task.",
            "prerequisite": "A reproducible compatible Vivado toolchain or open simulator path.",
        },
        {
            "candidate_id": "symbolic_kan_energy",
            "target_milestone": "V607+",
            "proposal": "Compare discrete primitive expressions with the existing exact constraint-energy baseline.",
            "prerequisite": "Public code or a separately specified clean-room implementation.",
        },
    ]


_LEDGER_APPEND_ROWS: tuple[JsonDict, ...] = (
    {
        "candidate_id": "ism_reference_implementation_compatibility",
        "title": "ISM reference implementation compatibility update",
        "canonical_url": "https://github.com/pdx97/ISM",
        "date": "2026-09-03",
        "finding": "The public runner pins hosted OpenAI solver, synthesis, classification, and embedding services. It is not a direct local-GGUF implementation.",
        "v607_use": "Reuse only the typed schema-bank, hard-reset, and post-outcome update patterns unless a local adapter is tested separately.",
    },
)


def ledger_append_rows() -> list[JsonDict]:
    """Return the single verified fact absent from the prior V606 block."""

    return deepcopy(_LEDGER_APPEND_ROWS)


def source_family_rows(queries: list[JsonDict]) -> list[JsonDict]:
    """Reduce terminal query receipts to one row per required family."""

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
    """Hash all task-defining local inputs named in the V606 prompt."""

    relatives = (
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        LEDGER_RELATIVE_PATH,
        ROADMAP_RELATIVE_PATH,
        DESIGN_RELATIVE_PATH,
        Path("ops/conductor-log.md"),
        Path("research-complete.yaml"),
        SPEC_RELATIVE_PATH,
    )
    return {relative.as_posix(): _file_hash(root / relative) for relative in relatives}


def _combined_rows(groups: list[tuple[str, list[JsonDict]]]) -> list[JsonDict]:
    rows = []
    for row_type, group in groups:
        rows.extend({"row_type": row_type, **deepcopy(row)} for row in group)
    return rows


def payload_checksum(artifact: dict[str, object]) -> str:
    """Hash a normalized payload while excluding its self-referential field."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_artifact(
    *, root: Path, run_date: str, duration_s: float, output_root: Path | None = None
) -> dict[str, object]:
    """Build a complete artifact or a schema-complete blocked preflight."""

    del output_root
    preconditions = check_preconditions(root, run_date)
    failed = next((row for row in preconditions if not row["available"]), None)
    queries = [] if failed else query_rows()
    candidates = [] if failed else candidate_rows()
    implementations = [] if failed else implementation_rows()
    compatibilities = [] if failed else compatibility_rows()
    citations = [] if failed else citation_edge_rows()
    future = [] if failed else v607_candidate_rows()
    appends = [] if failed else ledger_append_rows()
    families = [] if failed else source_family_rows(queries)
    accepted = [row for row in candidates if row["disposition"].startswith("accepted")]
    rejected = [row for row in candidates if row["disposition"] == "rejected"]
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
            ("v607_candidate", future),
        ]
    )
    if failed:
        gate_summary: JsonDict = {
            "all_gates_passed": False,
            "failed_check": failed["resource"],
            "expected": failed["expected"],
            "observed": failed["observed"],
        }
        status = "blocked"
        verdict_class = "blocked"
        verdict = "complete_blocked_v606_sota_ingestion"
    else:
        gate_summary = {
            "all_gates_passed": complete,
            "failed_check": None,
            "expected": "15 terminal source families and 15 terminal named candidates",
            "observed": f"{len(families)} source families and {len(candidates)} candidates",
        }
        status = "complete"
        verdict_class = "positive" if appends else "null"
        verdict = (
            "complete_v606_sota_ingestion_with_one_new_compatibility_finding"
            if appends
            else "complete_null_v606_sota_ingestion_no_new_findings"
        )
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
            "method": "No experimental model inference; bounded primary-source retrieval only.",
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
        "v607_candidate_rows": future,
        "rate_limit_rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v606_sota_ingestion_complete_score": int(complete),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact_or_path: dict[str, object] | Path) -> list[str]:
    """Validate row coverage, terminal states, and the payload checksum."""

    if isinstance(artifact_or_path, Path):
        if not artifact_or_path.exists():
            return ["artifact missing"]
        artifact = json.loads(artifact_or_path.read_text(encoding="utf-8"))
    else:
        artifact = artifact_or_path
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors = []
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principles do not cover required fields")
    blocked = artifact["status"] == "blocked"
    if not blocked:
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
        if artifact["v606_sota_ingestion_complete_score"] != int(
            source_complete and candidate_complete
        ):
            errors.append("completion score does not match terminal rows")
    elif (
        artifact["v606_sota_ingestion_complete_score"] != 0
        or artifact["honest_verdict"] != "complete_blocked_v606_sota_ingestion"
    ):
        errors.append("blocked artifact fields are inconsistent")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def append_verified_findings(root: Path, artifact: dict[str, object]) -> list[str]:
    """Append marked verified findings and never replace earlier ledger text."""

    path = root / LEDGER_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    appended = []
    blocks = []
    for row in artifact["ledger_append_rows"]:
        marker = f"<!-- EXP6925-{row['candidate_id'].upper()} -->"
        if marker in text:
            continue
        blocks.append(
            "\n".join(
                (
                    marker,
                    f"### {row['title']} - {row['date']}",
                    f"- **Source:** {row['canonical_url']}",
                    f"- **Verified finding:** {row['finding']}",
                    f"- **V607 boundary:** {row['v607_use']}",
                )
            )
        )
        appended.append(row["candidate_id"])
    if blocks:
        suffix = (
            "\n\n## V606 Execution-Time SOTA Ingestion - 2026-09-03\n\n"
            + "\n\n".join(blocks)
            + "\n"
        )
        atomic_write_text(path, text.rstrip() + suffix, allow_override=False)
    return appended


def write_artifact(artifact: dict[str, object], path: Path) -> Path:
    """Atomically write one validated experiment JSON file."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return atomic_write_json(path, artifact, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """Write or validate the V606 SOTA-ingestion result."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        return int(bool(validate_artifact(args.validate)))
    root = find_repo_root()
    draft = build_artifact(root=root, run_date=args.date, duration_s=315.0)
    if draft["status"] != "blocked":
        append_verified_findings(root, draft)
        draft = build_artifact(root=root, run_date=args.date, duration_s=315.0)
    output = args.output or root / RESULT_RELATIVE_PATH
    write_artifact(draft, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
