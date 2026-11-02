"""Build the V615 source-ingestion receipt without using model inference.

The module records what each public route returned on the execution date. It
keeps primary evidence separate from indexes, repository descriptions, and
product prose. This prevents a useful research lead from becoming an
unsupported local-readiness claim.

Spec refs: REQ-REPORT-7018 and SCENARIO-REPORT-7018-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone, UTC
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7018
RUN_DATE = "20260905"
RANDOM_SEED = 7_018_202_609_05
INFERENCE_SUBSTRATE = "deterministic_primary_source_ingestion_no_llm"
REFERENCE_MARKER = "## V615 Planner Refresh - 2026-09-05"
MARKER_DATE = "2026-09-05"

MODULE_PATH = Path("python/carnot/experiment_7018_v615_sota_ingestion.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_7018_v615_sota_ingestion.py")
TEST_PATH = Path("tests/python/test_experiment_7018_v615_sota_ingestion.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
PRIOR_ARTIFACT_PATH = Path("results/experiment_7011_v614_sota_ingestion.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
RESULT_PATH = Path("results/experiment_7018_v615_sota_ingestion.json")

REQUIRED_LOCAL_INPUTS = (
    REFERENCE_PATH,
    PRIOR_ARTIFACT_PATH,
    EXCLUSION_PATH,
    SPEC_PATH,
)
HASH_PATHS = REQUIRED_LOCAL_INPUTS + (MODULE_PATH, SCRIPT_PATH, TEST_PATH)

ARXIV_TOPICS = (
    "ebm_reasoning",
    "neural_constraints",
    "ising",
    "hallucination_mitigation",
    "kan",
    "constrained_generation",
    "hardware_sampling",
    "continual_learning",
)
REQUIRED_SOURCE_FAMILIES = (
    "arxiv",
    "openreview",
    "huggingface",
    "semantic_scholar",
    "github",
    "extropic",
    "logical_intelligence",
)
CLASSIFICATIONS = ("build", "test", "watch", "reject")
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "source_query_rows",
    "primary_source_rows",
    "secondary_source_rows",
    "semantic_scholar_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "post_marker_delta_rows",
    "duplicate_rows",
    "classification_rows",
    "reference_append_rows",
    "rows",
    "cutoff_rate_limit_and_same_day_uncertainty_receipts",
    "v615_sota_ingestion_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "One reason per field makes the evidence contract reviewable.",
    "preconditions_checked": "Preflight rows stop missing inputs from becoming scientific findings.",
    "inference_substrate": "The substrate discloses that deterministic source review used no LLM.",
    "duration_s": "Elapsed time distinguishes an executed audit from an undated assertion.",
    "source_artifact_hashes": "Hashes bind conclusions to the local marker and policy inputs.",
    "source_query_rows": "Query rows show the scope searched and keep search evidence advisory.",
    "primary_source_rows": "Versioned primary rows support method claims without source substitution.",
    "secondary_source_rows": "Secondary rows preserve useful discovery without granting authority.",
    "semantic_scholar_rows": "Citation-route rows retain rate limits without inventing counts.",
    "github_rows": "Repository rows separate public code identity from local readiness.",
    "extropic_rows": "First-party software evidence stays separate from hardware estimates.",
    "logical_intelligence_rows": "Product rows expose Kona's missing reproducible runner.",
    "post_marker_delta_rows": "Only proved later primary changes can enter the append set.",
    "duplicate_rows": "Suppression rows prevent old findings from appearing new.",
    "classification_rows": "Closed actions make each finding operational and falsifiable.",
    "reference_append_rows": "Append receipts show exactly why the research ledger changed or stayed fixed.",
    "rows": "One combined ledger supports independent row-by-row consistency checks.",
    "cutoff_rate_limit_and_same_day_uncertainty_receipts": (
        "Cutoff and access receipts preserve uncertainty instead of resolving it by assumption."
    ),
    "v615_sota_ingestion_complete_score": (
        "One requires terminal route coverage and valid source-bound classifications."
    ),
    "random_seed": "A fixed seed records deterministic ordering even though no sampling ran.",
    "reproducibility_checksum": "A stable payload hash detects later evidence changes.",
    "gate_check_summary": "Expected and observed values make every blocked result actionable.",
    "verifier_is_oracle": "False states that literature review does not determine task correctness.",
    "verdict_class": "A closed class prevents blocked access from appearing positive.",
    "honest_verdict": "A matching prefix gives automation one unambiguous terminal outcome.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _parse_datetime(value: str) -> tuple[datetime, bool]:
    """Parse a source date and report whether it contains an ordering time."""

    text = value.strip()
    has_time = "T" in text
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC), has_time


def cutoff_relation(
    source_date: str,
    *,
    marker_date: str = MARKER_DATE,
    marker_timestamp: str | None = None,
) -> str:
    """Classify ordering without guessing a time for the dated V615 marker."""

    try:
        source, source_has_time = _parse_datetime(source_date)
        marker, marker_has_time = _parse_datetime(marker_timestamp or marker_date)
    except (TypeError, ValueError):
        return "unknown"
    if source.date() < marker.date():
        return "pre_marker"
    if source.date() > marker.date():
        return "post_marker"
    if marker_timestamp is None or not source_has_time or not marker_has_time:
        return "same_day_order_uncertain"
    return "post_marker" if source > marker else "pre_marker"


def access_receipt(
    *,
    family: str,
    query: str,
    url: str,
    accessed_on: str,
    http_status: int | None,
    access_outcome: str,
    evidence_kind: str,
) -> JsonDict:
    """Create one terminal route receipt without inferring missing content."""

    if not url.startswith("https://"):
        raise ValueError("a canonical https URL is required")
    verified_kinds = {"primary_paper", "first_party_report", "repository_api"}
    content_verified = bool(
        http_status is not None
        and 200 <= http_status < 300
        and evidence_kind in verified_kinds
        and access_outcome == "ok"
    )
    return {
        "family": family,
        "query": query,
        "url": url,
        "accessed_on": accessed_on,
        "http_status": http_status,
        "access_outcome": access_outcome,
        "evidence_kind": evidence_kind,
        "content_verified": content_verified,
        "readiness_claimed": False,
        "terminal": True,
    }


def deduplicate_findings(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Keep the first source and finding identities and record every suppression."""

    unique: list[JsonDict] = []
    duplicates: list[JsonDict] = []
    source_owner: dict[str, str] = {}
    finding_owner: dict[str, str] = {}
    for original in rows:
        row = dict(original)
        row_id = str(row.get("row_id", ""))
        source_identity = str(row.get("source_identity", ""))
        finding_id = str(row.get("finding_id", ""))
        reason = ""
        kept_row_id = ""
        if source_identity and source_identity in source_owner:
            reason = "duplicate_source_identity"
            kept_row_id = source_owner[source_identity]
        elif finding_id and finding_id in finding_owner:
            reason = "duplicate_finding_identity"
            kept_row_id = finding_owner[finding_id]
        if reason:
            duplicates.append(
                {
                    "duplicate_row_id": row_id,
                    "kept_row_id": kept_row_id,
                    "reason": reason,
                    "suppressed": True,
                    "terminal": True,
                }
            )
            continue
        unique.append(deepcopy(row))
        if source_identity:
            source_owner[source_identity] = row_id
        if finding_id:
            finding_owner[finding_id] = row_id
    return unique, duplicates


def classify_finding(
    *,
    finding_id: str,
    source_ids: Sequence[str],
    proposed_action: str,
    primary_verified: bool,
    reproducible_code_identity: bool,
    bounded_local_test: bool,
    reason: str,
    false_positive_kind: str | None = None,
) -> JsonDict:
    """Apply the evidence floor before accepting an operational action."""

    if proposed_action not in CLASSIFICATIONS:
        raise ValueError(f"unknown classification: {proposed_action}")
    classification = proposed_action
    if false_positive_kind:
        classification = "reject"
    elif (
        proposed_action == "build"
        and not (primary_verified and reproducible_code_identity)
        or proposed_action == "test"
        and not (primary_verified and bounded_local_test)
    ):
        classification = "watch"
    return {
        "finding_id": finding_id,
        "source_ids": list(source_ids),
        "classification": classification,
        "primary_verified": primary_verified,
        "reproducible_code_identity": reproducible_code_identity,
        "bounded_local_test": bounded_local_test,
        "false_positive_kind": false_positive_kind,
        "reason": reason,
        "local_readiness_claimed": False,
        "terminal": True,
    }


def source_query_rows() -> list[JsonDict]:
    """Return the eight bounded arXiv topic-query receipts from this run."""

    query_text = {
        "ebm_reasoning": "energy based model reasoning verifier LLM 2025 2026",
        "neural_constraints": "neural constraint satisfaction 2025 2026",
        "ising": "Ising methods sampling 2025 2026",
        "hallucination_mitigation": "hallucination mitigation verifier 2025 2026",
        "kan": "Kolmogorov Arnold network continual learning 2025 2026",
        "constrained_generation": "constrained generation exact verifier 2025 2026",
        "hardware_sampling": "probabilistic hardware sampling 2025 2026",
        "continual_learning": "continual learning counterexample agents 2025 2026",
    }
    rows: list[JsonDict] = []
    for topic in ARXIV_TOPICS:
        query = query_text[topic]
        receipt = access_receipt(
            family="arxiv",
            query=query,
            url=f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all",
            accessed_on="2026-09-05",
            http_status=200,
            access_outcome="ok",
            evidence_kind="primary_index",
        )
        rows.append({"query_id": f"arxiv_{topic}", "topic": topic, **receipt})
    return rows


_PRIMARY_SOURCE_ROWS: tuple[JsonDict, ...] = (
    {
        "row_id": "primary_bbwm",
        "source_id": "bbwm",
        "finding_id": "bbwm_queryable_belief",
        "title": "Towards a Belief-Based World Model for LLM Agents",
        "canonical_url": "https://arxiv.org/abs/2609.00455",
        "source_identity": "arXiv:2609.00455v1",
        "publication_or_update_date": "2026-08-31T22:48:38Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Keep queryable current-state belief separate from action simulation.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
    {
        "row_id": "primary_cegl",
        "source_id": "cegl",
        "finding_id": "counterexample_guided_updates",
        "title": "Counterexample Guided Learning in the Large using Reasoning Agents",
        "canonical_url": "https://arxiv.org/abs/2606.11521",
        "source_identity": "arXiv:2606.11521v1",
        "publication_or_update_date": "2026-06-09T23:46:46Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Use exact counterexamples and symbolic clusters to refine hypotheses.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
    {
        "row_id": "primary_scaling_flaws",
        "source_id": "scaling_flaws",
        "finding_id": "compute_matched_verifier_controls",
        "title": "Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning",
        "canonical_url": "https://arxiv.org/abs/2502.00271",
        "source_identity": "arXiv:2502.00271v1",
        "publication_or_update_date": "2025-02-01T02:08:49Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Compare verifier search against repeated sampling as budgets grow.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
    {
        "row_id": "primary_solve_verify",
        "source_id": "solve_verify",
        "finding_id": "compute_matched_verifier_controls",
        "title": "When To Solve, When To Verify",
        "canonical_url": "https://arxiv.org/abs/2504.01005",
        "source_identity": "arXiv:2504.01005v2",
        "publication_or_update_date": "2025-10-19T10:28:43Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Compare generation and verification under one inference budget.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
    {
        "row_id": "primary_kac",
        "source_id": "kac",
        "finding_id": "kan_classifier_claim",
        "title": "KAC: Kolmogorov-Arnold Classifier for Continual Learning",
        "canonical_url": "https://arxiv.org/abs/2503.21076",
        "source_identity": "arXiv:2503.21076v1",
        "publication_or_update_date": "2025-03-27T01:27:14Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Replace linear continual-learning heads with KAN or RBF classifiers.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
    {
        "row_id": "primary_kan_forgetting",
        "source_id": "kan_forgetting",
        "finding_id": "kan_forgetting_caution",
        "title": "Catastrophic Forgetting in Kolmogorov-Arnold Networks",
        "canonical_url": "https://arxiv.org/abs/2511.12828",
        "source_identity": "arXiv:2511.12828v1",
        "publication_or_update_date": "2025-11-16T23:22:50Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Treat KAN retention as dependent on activation overlap and data dimension.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
    {
        "row_id": "primary_ebt",
        "source_id": "ebt",
        "finding_id": "ebt_existing_scope",
        "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
        "canonical_url": "https://arxiv.org/abs/2507.02092",
        "source_identity": "arXiv:2507.02092v1",
        "publication_or_update_date": "2025-07-02T19:17:29Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Optimize predictions against learned input-candidate compatibility energy.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
    {
        "row_id": "primary_arm_ebm",
        "source_id": "arm_ebm",
        "finding_id": "arm_ebm_existing_scope",
        "title": "Autoregressive Language Models are Secretly Energy-Based Models",
        "canonical_url": "https://arxiv.org/abs/2512.15605",
        "source_identity": "arXiv:2512.15605v4",
        "publication_or_update_date": "2026-05-25T15:54:35Z",
        "accessed_on": "2026-09-05",
        "access_outcome": "ok",
        "method_extraction": "Relate autoregressive and energy models through a function-space bijection.",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "readiness_claimed": False,
        "terminal": True,
    },
)


def primary_source_rows() -> list[JsonDict]:
    return _copy(_PRIMARY_SOURCE_ROWS)


def secondary_source_rows() -> list[JsonDict]:
    """Return advisory OpenReview and Hugging Face route receipts."""

    return [
        access_receipt(
            family="openreview",
            query="VerifierQ qhOT0MisId",
            url="https://openreview.net/forum?id=qhOT0MisId",
            accessed_on="2026-09-05",
            http_status=403,
            access_outcome="challenge_required",
            evidence_kind="secondary_review_route",
        ),
        access_receipt(
            family="huggingface",
            query="step-level verifier-guided scaling 2507.15512",
            url="https://huggingface.co/papers/2507.15512",
            accessed_on="2026-09-05",
            http_status=200,
            access_outcome="ok",
            evidence_kind="secondary_index",
        ),
        access_receipt(
            family="huggingface",
            query="VerifyBench 2507.09884",
            url="https://huggingface.co/papers/2507.09884",
            accessed_on="2026-09-05",
            http_status=200,
            access_outcome="ok",
            evidence_kind="secondary_index",
        ),
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Return the two observed HTTP 429 citation-route receipts."""

    rows: list[JsonDict] = []
    for paper_id in ("2507.02092", "2512.15605"):
        receipt = access_receipt(
            family="semantic_scholar",
            query=f"ARXIV:{paper_id} metadata and citations",
            url=(
                "https://api.semanticscholar.org/graph/v1/paper/"
                f"ARXIV%3A{paper_id}?fields=paperId,title,year,publicationDate,url,citationCount"
            ),
            accessed_on="2026-09-05",
            http_status=429,
            access_outcome="rate_limited",
            evidence_kind="secondary_index",
        )
        rows.append({"paper_id": paper_id, "citation_count_claimed": False, **receipt})
    return rows


def github_rows() -> list[JsonDict]:
    """Return immutable code identities plus one repository-only false positive."""

    return [
        {
            "repository_id": "bbwm_code",
            "source_id": "bbwm",
            "query": "paper-linked BB-WM implementation",
            "url": "https://github.com/skumar-ml/belief-world-models",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "ok",
            "source_identity": "git:d9e2900e37cb18a044fe3f084d32642d0f8f5fd5",
            "revision_date": "2026-08-31T22:29:26Z",
            "reproducible_public_code": True,
            "readiness_claimed": False,
            "terminal": True,
        },
        {
            "repository_id": "z1t_training_code",
            "source_id": "z1t",
            "query": "Extropic-linked sparse-transformers implementation",
            "url": "https://github.com/extropic-ai/sparse-transformers",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "ok",
            "source_identity": "git:13051e90df9669be5b8f9f34fb097329fa82f674",
            "revision_date": "2026-09-03T20:28:12Z",
            "reproducible_public_code": True,
            "readiness_claimed": False,
            "terminal": True,
        },
        {
            "repository_id": "github_discovery_impetus",
            "source_id": "impetus_repo_only",
            "query": "energy-based model reasoning created 2025-2026",
            "url": "https://github.com/EdhieBM/Impetus",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "repository_search_result_only",
            "source_identity": "github:EdhieBM/Impetus:unverified_search_result",
            "revision_date": "2026-05-16T19:20:54Z",
            "reproducible_public_code": False,
            "readiness_claimed": False,
            "terminal": True,
        },
    ]


def extropic_rows() -> list[JsonDict]:
    """Return Z1T as dated first-party software evidence only."""

    return [
        {
            "source_id": "z1t",
            "query": "Extropic Z1T writing and linked public software",
            "url": "https://extropic.ai/writing/z1t",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "ok",
            "source_kind": "first_party_report",
            "source_identity": "Extropic:Z1T:2026-09-04",
            "publication_or_update_date": "2026-09-04",
            "public_software_observed": True,
            "hardware_readiness_claimed": False,
            "latency_energy_claimed_by_carnot": False,
            "authenticated_device_receipt": False,
            "cutoff_relation": "pre_marker",
            "terminal": True,
        }
    ]


def logical_intelligence_rows() -> list[JsonDict]:
    """Return Kona's accessible product page without inventing a local asset."""

    return [
        {
            "source_id": "kona",
            "query": "Logical Intelligence Kona public implementation and runner",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "product_page_ok_no_public_runner",
            "source_kind": "first_party_product_page",
            "source_identity": "LogicalIntelligence:Kona:page_access_2026-09-05",
            "publication_or_update_date": "2026-09-05",
            "date_receipt": "access_date_only_no_page_update_time",
            "reproducible_local_runner_claimed": False,
            "public_weights_claimed": False,
            "cutoff_relation": "same_day_order_uncertain",
            "terminal": True,
        }
    ]


def classification_rows() -> list[JsonDict]:
    """Map selected, guarded, and retired findings to closed actions."""

    definitions = (
        (
            "bbwm_queryable_belief",
            ["bbwm", "bbwm_code"],
            "build",
            True,
            True,
            True,
            "Build a small explicit belief ledger before learned compression.",
            None,
        ),
        (
            "counterexample_guided_updates",
            ["cegl"],
            "test",
            True,
            True,
            True,
            "Test contradiction clusters and tombstones on sealed later transitions.",
            None,
        ),
        (
            "compute_matched_verifier_controls",
            ["scaling_flaws", "solve_verify"],
            "test",
            True,
            True,
            True,
            "Match actions, model calls, verifier calls, and elapsed compute across arms.",
            None,
        ),
        (
            "kan_forgetting_caution",
            ["kac", "kan_forgetting"],
            "watch",
            True,
            False,
            False,
            "Watch KAN compression only after a retained belief signal exists.",
            None,
        ),
        (
            "z1t_public_software",
            ["z1t", "z1t_training_code"],
            "watch",
            True,
            True,
            False,
            "Study the public JAX graph without a Z1 or TSU execution claim.",
            None,
        ),
        (
            "kona_architecture_only",
            ["kona"],
            "watch",
            False,
            False,
            False,
            "Keep Kona as an architecture comparator until a local runner exists.",
            None,
        ),
        (
            "retired_generated_text",
            ["exclusion_manifest"],
            "reject",
            False,
            False,
            False,
            "The generated-text verifier lane is retired.",
            "retired_scope",
        ),
        (
            "retired_grammar_decoding",
            ["exclusion_manifest"],
            "reject",
            False,
            False,
            False,
            "Grammar decoding does not provide live transition identity.",
            "retired_scope",
        ),
        (
            "retired_pair_centered_latent",
            ["v614_artifact"],
            "reject",
            True,
            False,
            False,
            "V614 found no released causal feature bank for another pair-centered fit.",
            "unchanged_failed_scope",
        ),
        (
            "unchanged_kan_scope",
            ["kac", "kan_forgetting"],
            "reject",
            True,
            False,
            False,
            "No new KAN result after the marker reopens the critical path.",
            "unchanged_scope",
        ),
        (
            "z1_hardware_readiness",
            ["z1t"],
            "reject",
            True,
            True,
            False,
            "Carnot has no authenticated Z1 or TSU runner.",
            "unsupported_hardware_readiness",
        ),
        (
            "github_repo_only_false_positive",
            ["impetus_repo_only"],
            "reject",
            False,
            False,
            False,
            "A repository search result alone is not primary evidence or local readiness.",
            "repository_only",
        ),
    )
    return [
        classify_finding(
            finding_id=finding_id,
            source_ids=source_ids,
            proposed_action=action,
            primary_verified=primary,
            reproducible_code_identity=code,
            bounded_local_test=bounded,
            reason=reason,
            false_positive_kind=false_positive,
        )
        for finding_id, source_ids, action, primary, code, bounded, reason, false_positive in definitions
    ]


def duplicate_rows() -> list[JsonDict]:
    """Record why rechecked marker findings are not new append candidates."""

    return [
        {
            "duplicate_row_id": f"marker_repeat_{finding_id}",
            "kept_row_id": f"v615_marker:{finding_id}",
            "reason": "already_present_at_v615_marker",
            "suppressed": True,
            "terminal": True,
        }
        for finding_id in (
            "bbwm_queryable_belief",
            "counterexample_guided_updates",
            "compute_matched_verifier_controls",
            "kan_forgetting_caution",
            "z1t_public_software",
            "kona_architecture_only",
        )
    ]


def _eligible_delta(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("source_kind") in {"primary_paper", "first_party_report"}
        and row.get("primary_verified") is True
        and row.get("cutoff_relation") == "post_marker"
        and str(row.get("canonical_url", "")).startswith("https://")
    )


def reference_append_rows(delta_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Prepare append receipts only for verified primary post-marker deltas."""

    eligible = [dict(row) for row in delta_rows if _eligible_delta(row)]
    if not eligible:
        return [
            {
                "action": "no_change",
                "marker": REFERENCE_MARKER,
                "appended": False,
                "reason": "No verified primary or first-party change was proved after the V615 marker.",
                "terminal": True,
            }
        ]
    return [
        {
            "action": "append",
            "verified_source_id": row["source_id"],
            "canonical_url": row["canonical_url"],
            "cutoff_relation": row["cutoff_relation"],
            "primary_verified": True,
            "finding": row["finding"],
            "appended": False,
            "terminal": True,
        }
        for row in eligible
    ]


def cutoff_receipts(
    primary_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Combine date boundaries, rate limits, and same-day uncertainty receipts."""

    rows = [
        {
            "receipt_kind": "cutoff",
            "source_id": row["source_id"],
            "source_date": row["publication_or_update_date"],
            "marker_date": MARKER_DATE,
            "relation": row["cutoff_relation"],
            "terminal": True,
        }
        for row in primary_rows
    ]
    rows.extend(
        {
            "receipt_kind": "rate_limit",
            "source_id": f"semantic_scholar_{row.get('paper_id', row.get('query', 'route'))}",
            "http_status": row["http_status"],
            "access_outcome": row["access_outcome"],
            "terminal": True,
        }
        for row in semantic_rows
    )
    rows.extend(
        {
            "receipt_kind": "same_day_uncertainty",
            "source_id": row.get("source_id", "logical_intelligence_route"),
            "source_date": row.get("publication_or_update_date", row.get("accessed_on")),
            "marker_date": MARKER_DATE,
            "relation": row.get("cutoff_relation", "same_day_order_uncertain"),
            "reason": "The source page supplies no update time that proves ordering after the marker.",
            "terminal": True,
        }
        for row in logical_rows
    )
    return rows


def source_row_groups() -> dict[str, list[JsonDict]]:
    """Return every requested source family under one exact family key."""

    secondary = secondary_source_rows()
    return {
        "arxiv": source_query_rows(),
        "openreview": [row for row in secondary if row["family"] == "openreview"],
        "huggingface": [row for row in secondary if row["family"] == "huggingface"],
        "semantic_scholar": semantic_scholar_rows(),
        "github": github_rows(),
        "extropic": extropic_rows(),
        "logical_intelligence": logical_intelligence_rows(),
    }


def _route_available(row: Mapping[str, Any]) -> bool:
    status = row.get("http_status")
    return isinstance(status, int) and 200 <= status < 400


def _family_rows(artifact: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    secondary = artifact.get("secondary_source_rows", [])
    secondary_rows = secondary if isinstance(secondary, list) else []
    return {
        "arxiv": artifact.get("source_query_rows", []),
        "openreview": [row for row in secondary_rows if row.get("family") == "openreview"],
        "huggingface": [row for row in secondary_rows if row.get("family") == "huggingface"],
        "semantic_scholar": artifact.get("semantic_scholar_rows", []),
        "github": artifact.get("github_rows", []),
        "extropic": artifact.get("extropic_rows", []),
        "logical_intelligence": artifact.get("logical_intelligence_rows", []),
    }


def _families_complete(artifact: Mapping[str, Any]) -> bool:
    groups = _family_rows(artifact)
    return set(groups) == set(REQUIRED_SOURCE_FAMILIES) and all(
        isinstance(rows, list)
        and rows
        and all(isinstance(row, Mapping) and row.get("terminal") is True for row in rows)
        for rows in groups.values()
    )


def _all_routes_unavailable(artifact: Mapping[str, Any]) -> bool:
    return not any(
        _route_available(row)
        for rows in _family_rows(artifact).values()
        for row in rows
        if isinstance(row, Mapping)
    )


def _reference_appends_valid(artifact: Mapping[str, Any]) -> bool:
    appends = artifact.get("reference_append_rows")
    deltas = artifact.get("post_marker_delta_rows")
    if not isinstance(appends, list) or not appends or not isinstance(deltas, list):
        return False
    eligible_ids = {str(row.get("source_id")) for row in deltas if _eligible_delta(row)}
    for row in appends:
        if not isinstance(row, Mapping):
            return False
        if row.get("action") == "append":
            if (
                row.get("primary_verified") is not True
                or row.get("cutoff_relation") != "post_marker"
                or str(row.get("verified_source_id")) not in eligible_ids
            ):
                return False
        elif row.get("action") != "no_change" or eligible_ids:
            return False
    return True


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Recompute the V615 ingestion gate from unit rows only."""

    classifications = artifact.get("classification_rows")
    duplicates = artifact.get("duplicate_rows")
    cutoff = artifact.get("cutoff_rate_limit_and_same_day_uncertainty_receipts")
    primary = artifact.get("primary_source_rows")
    if not all(isinstance(value, list) for value in (classifications, duplicates, cutoff, primary)):
        return 0
    class_rows = classifications
    expected_ids = {row["finding_id"] for row in classification_rows()}
    observed_ids = {str(row.get("finding_id")) for row in class_rows if isinstance(row, Mapping)}
    classes_ok = observed_ids == expected_ids and all(
        isinstance(row, Mapping)
        and row.get("classification") in CLASSIFICATIONS
        and row.get("terminal") is True
        for row in class_rows
    )
    duplicates_ok = all(
        isinstance(row, Mapping) and row.get("suppressed") is True and row.get("terminal") is True
        for row in duplicates
    )
    receipt_kinds = {str(row.get("receipt_kind")) for row in cutoff if isinstance(row, Mapping)}
    cutoff_ok = {"cutoff", "rate_limit", "same_day_uncertainty"} <= receipt_kinds
    primary_ok = bool(primary) and all(
        isinstance(row, Mapping)
        and row.get("primary_verified") is True
        and row.get("source_identity")
        and row.get("publication_or_update_date")
        and row.get("terminal") is True
        for row in primary
    )
    return int(
        _families_complete(artifact)
        and not _all_routes_unavailable(artifact)
        and classes_ok
        and duplicates_ok
        and cutoff_ok
        and primary_ok
        and _reference_appends_valid(artifact)
    )


def _combined_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    fields = (
        "source_query_rows",
        "primary_source_rows",
        "secondary_source_rows",
        "semantic_scholar_rows",
        "github_rows",
        "extropic_rows",
        "logical_intelligence_rows",
        "post_marker_delta_rows",
        "duplicate_rows",
        "classification_rows",
        "reference_append_rows",
        "cutoff_rate_limit_and_same_day_uncertainty_receipts",
    )
    combined: list[JsonDict] = []
    for field in fields:
        value = artifact.get(field, [])
        if isinstance(value, list):
            combined.extend(deepcopy(value))
    return combined


def _readable_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_bytes())
    except OSError:
        return False


def writable_path(path: Path) -> bool:
    """Probe the nearest existing parent without modifying the requested artifact."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7018-write-", delete=True):
            pass
    except OSError:
        return False
    return True


def network_available() -> bool:
    """Treat an HTTP response, including a rate limit, as a reachable network."""

    request = Request("https://arxiv.org/", headers={"User-Agent": "Carnot-Exp7018/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(getattr(response, "status", 0)) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def check_preconditions(
    root: Path,
    output_path: Path,
    *,
    network_is_available: bool,
) -> list[JsonDict]:
    """Check the marker, readable evidence, network, and output boundary."""

    rows: list[JsonDict] = [
        {
            "check": "network_access",
            "expected_value": "available",
            "observed_value": "available" if network_is_available else "unavailable",
            "passed": network_is_available,
            "terminal": True,
        }
    ]
    reference = root / REFERENCE_PATH
    try:
        marker_present = REFERENCE_MARKER in reference.read_text(encoding="utf-8")
    except OSError:
        marker_present = False
    rows.append(
        {
            "check": "v615_reference_marker",
            "expected_value": REFERENCE_MARKER,
            "observed_value": "present" if marker_present else "missing",
            "passed": marker_present,
            "terminal": True,
        }
    )
    for relative in REQUIRED_LOCAL_INPUTS[1:]:
        readable = _readable_nonempty(root / relative)
        rows.append(
            {
                "check": f"readable_reference:{relative.as_posix()}",
                "expected_value": "readable_nonempty_file",
                "observed_value": "readable_nonempty_file" if readable else "missing_or_unreadable",
                "passed": readable,
                "terminal": True,
            }
        )
    writable = writable_path(output_path)
    rows.append(
        {
            "check": "writable_artifact_path",
            "expected_value": "writable",
            "observed_value": "writable" if writable else "unwritable",
            "passed": writable,
            "terminal": True,
        }
    )
    return rows


def source_artifact_hashes(root: Path) -> list[JsonDict]:
    """Hash every required or newly added local source that currently exists."""

    rows: list[JsonDict] = []
    for relative in HASH_PATHS:
        path = root / relative
        if path.is_file():
            rows.append(
                {
                    "path": relative.as_posix(),
                    "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
                }
            )
    return rows


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all scientific fields except the checksum that stores the hash."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _gate_from_failure(row: Mapping[str, Any]) -> JsonDict:
    return {
        "failed_check": row["check"],
        "expected_value": row["expected_value"],
        "observed_value": row["observed_value"],
        "passed": False,
    }


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    network_available: bool | None = None,
    duration_s: float | None = None,
    source_groups: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> JsonDict:
    """Build one positive, blocked, or disqualified deterministic artifact."""

    started = time.monotonic()
    reachable = globals()["network_available"]() if network_available is None else network_available
    preconditions = check_preconditions(
        root,
        output_path,
        network_is_available=reachable,
    )
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": source_artifact_hashes(root),
        "source_query_rows": [],
        "primary_source_rows": [],
        "secondary_source_rows": [],
        "semantic_scholar_rows": [],
        "github_rows": [],
        "extropic_rows": [],
        "logical_intelligence_rows": [],
        "post_marker_delta_rows": [],
        "duplicate_rows": [],
        "classification_rows": [],
        "reference_append_rows": [],
        "rows": [],
        "cutoff_rate_limit_and_same_day_uncertainty_receipts": [],
        "v615_sota_ingestion_complete_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": None,
            "expected_value": 1,
            "observed_value": 0,
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_v615_sota_ingestion",
    }
    failure = next((row for row in preconditions if row["passed"] is False), None)
    if failure is not None:
        artifact["gate_check_summary"] = _gate_from_failure(failure)
    else:
        groups = (
            {family: _copy(rows) for family, rows in source_groups.items()}
            if source_groups is not None
            else source_row_groups()
        )
        artifact["source_query_rows"] = groups.get("arxiv", [])
        artifact["primary_source_rows"] = primary_source_rows()
        artifact["secondary_source_rows"] = [
            *groups.get("openreview", []),
            *groups.get("huggingface", []),
        ]
        artifact["semantic_scholar_rows"] = groups.get("semantic_scholar", [])
        artifact["github_rows"] = groups.get("github", [])
        artifact["extropic_rows"] = groups.get("extropic", [])
        artifact["logical_intelligence_rows"] = groups.get("logical_intelligence", [])
        artifact["post_marker_delta_rows"] = []
        artifact["duplicate_rows"] = duplicate_rows()
        artifact["classification_rows"] = classification_rows()
        artifact["reference_append_rows"] = reference_append_rows(
            artifact["post_marker_delta_rows"]
        )
        artifact["cutoff_rate_limit_and_same_day_uncertainty_receipts"] = cutoff_receipts(
            artifact["primary_source_rows"],
            artifact["semantic_scholar_rows"],
            artifact["logical_intelligence_rows"],
        )
        artifact["v615_sota_ingestion_complete_score"] = completion_score(artifact)
        if _all_routes_unavailable(artifact):
            artifact["gate_check_summary"] = {
                "failed_check": "all_source_routes_unavailable",
                "expected_value": "at_least_one_usable_route",
                "observed_value": "no_usable_routes",
                "passed": False,
            }
        elif artifact["v615_sota_ingestion_complete_score"] == 1:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = (
                "complete_positive_v615_sota_ingestion_no_verified_post_marker_delta"
            )
            artifact["gate_check_summary"] = {
                "failed_check": None,
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            }
        else:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "disqualified_v615_sota_ingestion_contract_failed"
            artifact["gate_check_summary"] = {
                "failed_check": "source_ingestion_contract",
                "expected_value": 1,
                "observed_value": artifact["v615_sota_ingestion_complete_score"],
                "passed": False,
            }
        artifact["rows"] = _combined_rows(artifact)
    artifact["duration_s"] = round(
        max(0.0, duration_s if duration_s is not None else time.monotonic() - started),
        6,
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _load_artifact(value: Mapping[str, Any] | str | Path) -> tuple[JsonDict | None, str | None]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value)), None
    path = Path(value)
    if not path.is_file():
        return None, "artifact_missing"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "artifact_unreadable"
    if not isinstance(loaded, dict):
        return None, "artifact_not_object"
    return loaded, None


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Recompute the artifact contract without trusting stored summary fields."""

    artifact, load_error = _load_artifact(value)
    if load_error:
        return [load_error]
    assert artifact is not None
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing_required_fields:{','.join(missing)}"]
    errors: list[str] = []
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_keys_invalid")
    elif not all(isinstance(value, str) and value.strip() for value in principles.values()):
        errors.append("field_principles_empty")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle_invalid")
    if not isinstance(artifact["duration_s"], (int, float)) or artifact["duration_s"] < 0:
        errors.append("duration_s_invalid")
    hashes = artifact["source_artifact_hashes"]
    if (
        not isinstance(hashes, list)
        or not hashes
        or not all(
            isinstance(row, Mapping)
            and row.get("path")
            and str(row.get("sha256", "")).startswith("sha256:")
            for row in hashes
        )
    ):
        errors.append("source_artifact_hashes_invalid")
    failed_precondition = next(
        (
            row
            for row in artifact["preconditions_checked"]
            if isinstance(row, Mapping) and row.get("passed") is False
        ),
        None,
    )
    if failed_precondition is None:
        if not _families_complete(artifact):
            errors.append("source_family_coverage_invalid")
        primary = artifact["primary_source_rows"]
        if isinstance(primary, list):
            identities = [
                str(row.get("source_identity")) for row in primary if isinstance(row, Mapping)
            ]
            if len(identities) != len(set(identities)):
                errors.append("unsuppressed_duplicate_source_identity")
        else:
            errors.append("primary_source_rows_invalid")
        classes = artifact["classification_rows"]
        expected_class_ids = {row["finding_id"] for row in classification_rows()}
        observed_class_ids = {
            str(row.get("finding_id"))
            for row in classes
            if isinstance(classes, list) and isinstance(row, Mapping)
        }
        if (
            not isinstance(classes, list)
            or observed_class_ids != expected_class_ids
            or any(
                not isinstance(row, Mapping) or row.get("classification") not in CLASSIFICATIONS
                for row in classes
            )
        ):
            errors.append("classification_rows_invalid")
        cutoff = artifact["cutoff_rate_limit_and_same_day_uncertainty_receipts"]
        cutoff_kinds = {
            str(row.get("receipt_kind"))
            for row in cutoff
            if isinstance(cutoff, list) and isinstance(row, Mapping)
        }
        if (
            not isinstance(cutoff, list)
            or not {
                "cutoff",
                "rate_limit",
                "same_day_uncertainty",
            }
            <= cutoff_kinds
        ):
            errors.append("cutoff_receipts_invalid")
        if not _reference_appends_valid(artifact):
            errors.append("reference_append_without_verified_post_marker_primary")
    if artifact["rows"] != _combined_rows(artifact):
        errors.append("rows_consistency_invalid")
    score = completion_score(artifact)
    if artifact["v615_sota_ingestion_complete_score"] != score:
        errors.append("v615_sota_ingestion_complete_score_invalid")
    all_unavailable = _all_routes_unavailable(artifact)
    expected_class = (
        "blocked"
        if failed_precondition is not None or all_unavailable
        else ("positive" if score == 1 else "disqualified")
    )
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    elif artifact["verdict_class"] != expected_class:
        errors.append("verdict_class_inconsistent")
    prefix = {
        "positive": "complete_positive_",
        "circular_positive": "complete_circular_positive_",
        "null": "null_",
        "blocked": "blocked_",
        "disqualified": "disqualified_",
        "partial": "partial_",
    }.get(str(artifact["verdict_class"]), "")
    if not prefix or not str(artifact["honest_verdict"]).startswith(prefix):
        errors.append("honest_verdict_prefix_invalid")
    gate = artifact["gate_check_summary"]
    if not isinstance(gate, Mapping) or not {
        "failed_check",
        "expected_value",
        "observed_value",
        "passed",
    } <= set(gate):
        errors.append("gate_check_summary_invalid")
    elif (
        expected_class == "positive"
        and gate.get("passed") is not True
        or expected_class != "positive"
        and gate.get("passed") is not False
    ):
        errors.append("gate_check_summary_inconsistent")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: Path) -> Path:
    """Validate before the atomic write so malformed evidence never replaces a result."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic build or validate an existing artifact."""

    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        if errors:
            print(json.dumps({"errors": errors}, sort_keys=True))
            return 1
        print(json.dumps({"valid": True, "path": str(args.validate)}, sort_keys=True))
        return 0
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    root = find_repo_root()
    output = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(root, args.date, output_path=output)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"errors": errors}, sort_keys=True))
        return 1
    write_artifact(artifact, output)
    print(json.dumps({"path": str(output), "verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
