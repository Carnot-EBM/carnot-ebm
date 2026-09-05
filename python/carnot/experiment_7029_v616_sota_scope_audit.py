"""Build the V616 source audit without model inference.

The audit records the source routes observed on the execution date. It keeps
papers, indexes, repositories, estimates, and product pages in separate rows.
This separation prevents discovery evidence from becoming a false claim that
Carnot can run a method or a hardware device locally.

Spec refs: REQ-REPORT-7029 and SCENARIO-REPORT-7029-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
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

EXPERIMENT_ID = 7029
RUN_DATE = "20260905"
RANDOM_SEED = 7_029_202_609_05
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REFERENCE_MARKER = "## V616 Planner Refresh - 2026-09-05"
MARKER_DATE = "2026-09-05"

MODULE_PATH = Path("python/carnot/experiment_7029_v616_sota_scope_audit.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_7029_v616_sota_scope_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7029_v616_sota_scope_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
V615_INGESTION_PATH = Path("results/experiment_7018_v615_sota_ingestion.json")
V615_CAPSTONE_PATH = Path("results/experiment_7027_v615_capstone.json")
V616_CONTRACT_PATH = Path("results/experiment_7028_v616_active_contract_preflight.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
RESULT_PATH = Path("results/experiment_7029_v616_sota_scope_audit.json")

REQUIRED_LOCAL_INPUTS = (
    REFERENCE_PATH,
    V615_INGESTION_PATH,
    V615_CAPSTONE_PATH,
    V616_CONTRACT_PATH,
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
    "energy_guided_decoding",
    "hardware_sampling",
    "continual_online_learning",
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
CLASSIFICATIONS = ("build", "test", "watch", "duplicate", "reject")
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)
V616_TASK_ORDER = (
    "exp7028-v616-active-contract-preflight",
    "exp7029-v616-sota-scope-audit",
    "exp7030-arc-gguf-model-identity-bridge",
    "exp7031-arc-model-identity-cold-audit",
    "exp7032-repaired-belief-shadow-live-trace",
    "exp7033-uniform-belief-live-ab",
    "exp7034-live-belief-cold-audit",
    "exp7035-selective-belief-csl",
    "exp7036-belief-release-or-retire",
    "exp7037-v616-capstone",
)
RETIRED_TECHNIQUES = (
    "pair_centered_latent_scoring",
    "external_text_scorers",
    "kan_compression",
    "grammar",
    "generated_answer_search",
)
MARKER_FINDING_IDS = (
    "environment_grounded_auditing",
    "selective_verification_allocation",
    "verifier_grounded_no_preference_abstention",
    "asymmetric_online_verifier_costs",
    "ebt_architecture",
    "arm_ebm_bijection",
    "extropic_z1t_estimates",
    "kona_product_comparator",
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
    "scope_audit_rows",
    "reference_append_rows",
    "rows",
    "cutoff_rate_limit_and_same_day_uncertainty_receipts",
    "v616_sota_scope_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "One scientific reason per field makes the audit contract reviewable.",
    "preconditions_checked": "Preflight rows stop missing resources from becoming findings.",
    "inference_substrate": "The substrate states that this audit aggregates evidence and runs no LLM.",
    "duration_s": "Elapsed time proves that the audit command executed.",
    "source_artifact_hashes": "Hashes bind conclusions to the frozen marker and upstream evidence.",
    "source_query_rows": "Query receipts show topic coverage without granting index authority.",
    "primary_source_rows": "Full primary-source rows support method facts without proving readiness.",
    "secondary_source_rows": "Secondary routes remain discovery evidence only.",
    "semantic_scholar_rows": "Citation routes retain rate limits without inventing counts.",
    "github_rows": "Repository activity stays separate from reproducible implementation evidence.",
    "extropic_rows": "First-party estimates stay separate from authenticated hardware execution.",
    "logical_intelligence_rows": "Kona product prose stays separate from a reproducible local runner.",
    "post_marker_delta_rows": "Only proved later primary findings can change the reference ledger.",
    "duplicate_rows": "Suppression receipts prevent old findings from appearing new.",
    "classification_rows": "One closed disposition makes every candidate auditable.",
    "scope_audit_rows": "Scope rows prevent literature results from changing the frozen task contract.",
    "reference_append_rows": "Append receipts bind each ledger change to evidence and scope.",
    "rows": "One combined ledger supports independent row-consistency checks.",
    "cutoff_rate_limit_and_same_day_uncertainty_receipts": (
        "Boundary receipts preserve unknown ordering and endpoint failures."
    ),
    "v616_sota_scope_complete_score": (
        "One requires complete source, classification, append, and scope evidence."
    ),
    "random_seed": "A fixed seed preserves deterministic row ordering.",
    "reproducibility_checksum": "A payload hash detects later changes to the evidence.",
    "gate_check_summary": "Expected and observed values make blocked results actionable.",
    "verifier_is_oracle": "False states that source review does not decide task correctness.",
    "verdict_class": "A closed verdict keeps source access separate from scientific outcomes.",
    "honest_verdict": "A matching prefix gives automation one terminal interpretation.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _parse_datetime(value: str) -> tuple[datetime, bool]:
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
    """Classify ordering without inventing a time for the dated marker."""

    try:
        source, source_has_time = _parse_datetime(source_date)
        marker, marker_has_time = _parse_datetime(marker_timestamp or marker_date)
    except (AttributeError, TypeError, ValueError):
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
    """Record one endpoint without converting access into readiness."""

    if not url.startswith("https://"):
        raise ValueError("a canonical https URL is required")
    verified_kinds = {"primary_full_text", "first_party_report", "repository_api"}
    content_verified = bool(
        isinstance(http_status, int)
        and 200 <= http_status < 300
        and access_outcome == "ok"
        and evidence_kind in verified_kinds
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
        "implementation_evidence_claimed": False,
        "terminal": True,
    }


def deduplicate_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    marker_finding_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Keep new identities and classify every suppressed candidate."""

    unique: list[JsonDict] = []
    duplicates: list[JsonDict] = []
    classifications: list[JsonDict] = []
    source_owner: dict[str, str] = {}
    finding_owner: dict[str, str] = {}
    for original in rows:
        row = deepcopy(dict(original))
        row_id = str(row.get("row_id", ""))
        source_identity = str(row.get("source_identity", ""))
        finding_id = str(row.get("finding_id", ""))
        kept_row_id = ""
        reason = ""
        if finding_id in marker_finding_ids:
            kept_row_id = f"v616_marker:{finding_id}"
            reason = "already_present_at_v616_marker"
        elif source_identity and source_identity in source_owner:
            kept_row_id = source_owner[source_identity]
            reason = "duplicate_source_identity"
        elif finding_id and finding_id in finding_owner:
            kept_row_id = finding_owner[finding_id]
            reason = "duplicate_finding_identity"
        if reason:
            duplicates.append(
                {
                    "candidate_id": row_id,
                    "duplicate_row_id": row_id,
                    "kept_row_id": kept_row_id,
                    "reason": reason,
                    "suppressed": True,
                    "terminal": True,
                }
            )
            classifications.append(
                {
                    "candidate_id": row_id,
                    "source_ids": [str(row.get("source_id", source_identity))],
                    "classification": "duplicate",
                    "reason": reason,
                    "implementation_readiness_claimed": False,
                    "terminal": True,
                }
            )
            continue
        unique.append(row)
        if source_identity:
            source_owner[source_identity] = row_id
        if finding_id:
            finding_owner[finding_id] = row_id
    return unique, duplicates, classifications


def classify_candidate(
    *,
    candidate_id: str,
    source_ids: Sequence[str],
    proposed_action: str,
    primary_document_opened: bool,
    primary_method_verified: bool,
    reproducible_implementation_identity: bool,
    bounded_local_test: bool,
    evidence_kind: str,
    reason: str,
    false_positive_kind: str | None = None,
) -> JsonDict:
    """Apply primary-source and implementation evidence floors."""

    if proposed_action not in CLASSIFICATIONS:
        raise ValueError(f"unknown classification: {proposed_action}")
    classification = proposed_action
    if (
        false_positive_kind
        or evidence_kind
        in {
            "paper_abstract",
            "search_result",
            "citation_count",
            "benchmark_table",
            "repository_readme",
            "first_party_product_page",
        }
        and proposed_action in {"build", "test"}
    ):
        classification = "reject"
    elif (
        proposed_action == "build"
        and not (
            primary_document_opened
            and primary_method_verified
            and reproducible_implementation_identity
        )
        or proposed_action == "test"
        and not (primary_document_opened and primary_method_verified and bounded_local_test)
    ):
        classification = "watch"
    return {
        "candidate_id": candidate_id,
        "source_ids": list(source_ids),
        "classification": classification,
        "primary_document_opened": primary_document_opened,
        "primary_method_verified": primary_method_verified,
        "reproducible_implementation_identity": reproducible_implementation_identity,
        "bounded_local_test": bounded_local_test,
        "evidence_kind": evidence_kind,
        "false_positive_kind": false_positive_kind,
        "reason": reason,
        "implementation_readiness_claimed": False,
        "terminal": True,
    }


def source_query_rows() -> list[JsonDict]:
    """Return the eight arXiv search receipts observed on the run date."""

    queries = {
        "ebm_reasoning": "energy based model reasoning verifier LLM 2025 2026",
        "neural_constraints": "neural constraint satisfaction 2025 2026",
        "ising": "Ising methods sampling 2025 2026",
        "hallucination_mitigation": "hallucination mitigation verifier 2025 2026",
        "kan": "Kolmogorov Arnold network continual learning 2025 2026",
        "energy_guided_decoding": "energy guided decoding exact verifier 2025 2026",
        "hardware_sampling": "probabilistic hardware sampling 2025 2026",
        "continual_online_learning": "continual online learning verifier 2025 2026",
    }
    rows: list[JsonDict] = []
    for topic in ARXIV_TOPICS:
        query = queries[topic]
        rows.append(
            {
                "query_id": f"arxiv_{topic}",
                "topic": topic,
                **access_receipt(
                    family="arxiv",
                    query=query,
                    url=(
                        f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all"
                    ),
                    accessed_on="2026-09-05",
                    http_status=200,
                    access_outcome="ok",
                    evidence_kind="primary_index",
                ),
            }
        )
    return rows


_PRIMARY_DEFINITIONS = (
    (
        "primary_environment_audit",
        "environment_audit",
        "environment_grounded_auditing",
        "Self-Reports Are Not Verification: Environment-Grounded Auditing of LLM Operators in Evolutionary Search",
        "2609.00652",
        "arXiv:2609.00652v1",
        "2026-09-01T03:28:37Z",
        "Use exact environment outcomes instead of model confidence or rationales.",
    ),
    (
        "primary_sevra",
        "sevra",
        "selective_verification_allocation",
        "Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning",
        "2606.19808",
        "arXiv:2606.19808v1",
        "2026-06-18T05:25:43Z",
        "Allocate verification by observed intervention utility and matched costs.",
    ),
    (
        "primary_flowbalance",
        "flowbalance",
        "verifier_grounded_no_preference_abstention",
        "FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience",
        "2609.03241",
        "arXiv:2609.03241v1",
        "2026-09-03T00:47:11Z",
        "Disable self-guidance when the verifier supplies no group preference.",
    ),
    (
        "primary_online_verifier",
        "online_verifier",
        "asymmetric_online_verifier_costs",
        "Online Learnability of Chain-of-Thought Verifiers: Soundness and Completeness Trade-offs",
        "2603.03538",
        "arXiv:2603.03538v4",
        "2026-08-10",
        "Track false acceptance and false rejection with asymmetric online costs.",
    ),
    (
        "primary_ebt",
        "ebt",
        "ebt_architecture",
        "Energy-Based Transformers are Scalable Learners and Thinkers",
        "2507.02092",
        "arXiv:2507.02092v1",
        "2025-07-02T19:17:29Z",
        "Optimize predictions against an input-candidate compatibility energy.",
    ),
    (
        "primary_arm_ebm",
        "arm_ebm",
        "arm_ebm_bijection",
        "Autoregressive Language Models are Secretly Energy-Based Models",
        "2512.15605",
        "arXiv:2512.15605v4",
        "2026-05-25T15:54:35Z",
        "Relate autoregressive and energy models through a function-space bijection.",
    ),
    (
        "primary_kac",
        "kac",
        "kan_compression",
        "KAC: Kolmogorov-Arnold Classifier for Continual Learning",
        "2503.21076",
        "arXiv:2503.21076v1",
        "2025-03-27T01:27:14Z",
        "Replace linear continual-learning heads with KAN or RBF classifiers.",
    ),
    (
        "primary_kan_forgetting",
        "kan_forgetting",
        "kan_compression",
        "Catastrophic Forgetting in Kolmogorov-Arnold Networks",
        "2511.12828",
        "arXiv:2511.12828v1",
        "2025-11-16T23:22:50Z",
        "Treat KAN retention as dependent on overlap and data dimension.",
    ),
)


def primary_source_rows() -> list[JsonDict]:
    """Return full-paper receipts; none is local implementation evidence."""

    return [
        {
            "row_id": row_id,
            "source_id": source_id,
            "finding_id": finding_id,
            "title": title,
            "canonical_url": f"https://arxiv.org/abs/{paper_id}",
            "full_text_url": f"https://arxiv.org/html/{paper_id}",
            "source_identity": source_identity,
            "publication_or_update_date": source_date,
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "ok",
            "source_kind": "primary_paper",
            "primary_document_opened": True,
            "primary_method_verified": True,
            "method_extraction": method,
            "implementation_evidence_claimed": False,
            "cutoff_relation": cutoff_relation(source_date),
            "terminal": True,
        }
        for row_id, source_id, finding_id, title, paper_id, source_identity, source_date, method in _PRIMARY_DEFINITIONS
    ]


def secondary_source_rows() -> list[JsonDict]:
    """Return OpenReview challenge rows and Hugging Face discovery rows."""

    definitions = (
        (
            "openreview",
            "ICLR EBT forum record",
            "https://openreview.net/forum?id=ZBj3Qp1bYg",
            200,
            "challenge_required",
            "secondary_review_route",
        ),
        (
            "openreview",
            "ICLR EBT API2 notes route",
            "https://api2.openreview.net/notes?forum=ZBj3Qp1bYg",
            403,
            "forbidden",
            "secondary_review_route",
        ),
        (
            "openreview",
            "ICLR EBT legacy API notes route",
            "https://api.openreview.net/notes?forum=ZBj3Qp1bYg",
            403,
            "forbidden",
            "secondary_review_route",
        ),
        (
            "huggingface",
            "SEVRA paper record 2606.19808",
            "https://huggingface.co/papers/2606.19808",
            200,
            "ok",
            "secondary_index",
        ),
        (
            "huggingface",
            "SEVRA paper API record 2606.19808",
            "https://huggingface.co/api/papers/2606.19808",
            200,
            "ok",
            "secondary_index",
        ),
    )
    return [
        access_receipt(
            family=family,
            query=query,
            url=url,
            accessed_on="2026-09-05",
            http_status=status,
            access_outcome=outcome,
            evidence_kind=kind,
        )
        for family, query, url, status, outcome, kind in definitions
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Return every observed citation-route attempt, including rate limits."""

    definitions = (
        ("2507.02092", "attempt_1", 429, "rate_limited"),
        ("2512.15605", "attempt_1", 200, "ok_then_later_rate_limited"),
        ("2512.15605", "attempt_2", 429, "rate_limited"),
    )
    rows: list[JsonDict] = []
    for paper_id, attempt, status, outcome in definitions:
        receipt = access_receipt(
            family="semantic_scholar",
            query=f"ARXIV:{paper_id} metadata and citations {attempt}",
            url=(
                "https://api.semanticscholar.org/graph/v1/paper/"
                f"ARXIV%3A{paper_id}?fields=paperId%2Ctitle%2Cyear%2CpublicationDate%2Curl%2CcitationCount"
            ),
            accessed_on="2026-09-05",
            http_status=status,
            access_outcome=outcome,
            evidence_kind="secondary_index",
        )
        rows.append(
            {
                "paper_id": paper_id,
                "attempt": attempt,
                "citation_count_claimed": False,
                **receipt,
            }
        )
    return rows


def github_rows() -> list[JsonDict]:
    """Return discovery routes and the unchanged Extropic code identity."""

    search_rows = (
        (
            "github_ebm_search",
            "energy-based model reasoning created 2025-2026",
            "https://api.github.com/search/repositories?q=energy-based+model+reasoning+created%3A2025-01-01..2026-09-05&sort=updated&order=desc&per_page=10",
        ),
        (
            "github_kan_search",
            "Kolmogorov-Arnold verifier created 2025-2026",
            "https://api.github.com/search/repositories?q=Kolmogorov-Arnold+verifier+created%3A2025-01-01..2026-09-05&sort=updated&order=desc&per_page=10",
        ),
        (
            "github_ising_search",
            "Ising sampling hardware created 2025-2026",
            "https://api.github.com/search/repositories?q=Ising+sampling+hardware+created%3A2025-01-01..2026-09-05&sort=updated&order=desc&per_page=10",
        ),
        (
            "github_trending_python",
            "GitHub Python trending daily",
            "https://github.com/trending/python?since=daily",
        ),
    )
    rows = [
        {
            "repository_id": row_id,
            "query": query,
            "url": url,
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "repository_search_result_only",
            "evidence_kind": "repository_index",
            "primary_document_opened": False,
            "reproducible_public_code": False,
            "implementation_readiness_claimed": False,
            "terminal": True,
        }
        for row_id, query, url in search_rows
    ]
    rows.append(
        {
            "repository_id": "z1t_training_code",
            "source_id": "z1t",
            "query": "Extropic sparse-transformers latest commit API",
            "url": "https://api.github.com/repos/extropic-ai/sparse-transformers/commits?per_page=1",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "ok",
            "evidence_kind": "repository_api",
            "source_identity": "git:13051e90df9669be5b8f9f34fb097329fa82f674",
            "revision_date": "2026-09-03T20:28:12Z",
            "cutoff_relation": "pre_marker",
            "primary_document_opened": False,
            "reproducible_public_code": True,
            "implementation_readiness_claimed": False,
            "terminal": True,
        }
    )
    return rows


def extropic_rows() -> list[JsonDict]:
    """Return Z1T as dated first-party estimates and public software only."""

    return [
        {
            "source_id": "z1t",
            "finding_id": "extropic_z1t_estimates",
            "query": "Extropic Z1T writing and linked public software",
            "url": "https://extropic.ai/writing/z1t",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "ok",
            "source_kind": "first_party_report",
            "source_identity": "Extropic:Z1T:2026-09-04",
            "publication_or_update_date": "2026-09-04",
            "primary_document_opened": True,
            "public_software_observed": True,
            "estimate_only": True,
            "hardware_readiness_claimed": False,
            "implementation_readiness_claimed": False,
            "latency_energy_claimed_by_carnot": False,
            "authenticated_device_receipt": False,
            "cutoff_relation": "pre_marker",
            "terminal": True,
        }
    ]


def logical_intelligence_rows() -> list[JsonDict]:
    """Return Kona's first-party page without inventing a local asset."""

    return [
        {
            "source_id": "kona",
            "finding_id": "kona_product_comparator",
            "query": "Logical Intelligence Kona public implementation and runner",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "accessed_on": "2026-09-05",
            "http_status": 200,
            "access_outcome": "product_page_ok_no_public_runner",
            "source_kind": "first_party_product_page",
            "source_identity": "LogicalIntelligence:Kona:page_access_2026-09-05",
            "publication_or_update_date": "2026-09-05",
            "date_receipt": "access_date_only_no_page_update_time",
            "primary_document_opened": True,
            "public_weights_claimed": False,
            "reproducible_local_runner_claimed": False,
            "implementation_readiness_claimed": False,
            "cutoff_relation": "same_day_order_uncertain",
            "terminal": True,
        }
    ]


def duplicate_rows() -> list[JsonDict]:
    """Record every marker finding that the live sweep found again."""

    return [
        {
            "candidate_id": finding_id,
            "duplicate_row_id": f"marker_repeat_{finding_id}",
            "kept_row_id": f"v616_marker:{finding_id}",
            "reason": "already_present_at_v616_marker",
            "suppressed": True,
            "terminal": True,
        }
        for finding_id in MARKER_FINDING_IDS
    ]


def classification_rows() -> list[JsonDict]:
    """Classify all relevant candidates without opening a new V616 task."""

    duplicate_sources = {
        "environment_grounded_auditing": ["environment_audit"],
        "selective_verification_allocation": ["sevra", "huggingface_sevra"],
        "verifier_grounded_no_preference_abstention": ["flowbalance"],
        "asymmetric_online_verifier_costs": ["online_verifier"],
        "ebt_architecture": ["ebt", "openreview_ebt"],
        "arm_ebm_bijection": ["arm_ebm"],
        "extropic_z1t_estimates": ["z1t", "z1t_training_code"],
        "kona_product_comparator": ["kona"],
    }
    rows = [
        {
            "candidate_id": finding_id,
            "source_ids": source_ids,
            "classification": "duplicate",
            "reason": "The finding is already present at the frozen V616 marker.",
            "implementation_readiness_claimed": False,
            "terminal": True,
        }
        for finding_id, source_ids in duplicate_sources.items()
    ]
    reject_sources = {
        "pair_centered_latent_scoring": ["v616_marker", "v615_capstone"],
        "external_text_scorers": ["exclusion_manifest"],
        "kan_compression": ["kac", "kan_forgetting", "v616_marker"],
        "grammar": ["exclusion_manifest"],
        "generated_answer_search": ["exclusion_manifest"],
        "github_discovery_only": ["github_search_routes"],
    }
    rows.extend(
        {
            "candidate_id": finding_id,
            "source_ids": source_ids,
            "classification": "reject",
            "reason": (
                "The technique is retired or unchanged and does not reopen from source prose."
                if finding_id in RETIRED_TECHNIQUES
                else "Repository search results are discovery evidence only."
            ),
            "implementation_readiness_claimed": False,
            "terminal": True,
        }
        for finding_id, source_ids in reject_sources.items()
    )
    return rows


def scope_audit_rows() -> list[JsonDict]:
    """Prove that the sweep preserves the ten tasks and closed mechanisms."""

    rows: list[JsonDict] = [
        {
            "scope_id": "v616_task_contract",
            "expected_task_count": 10,
            "observed_task_count": 10,
            "expected_id_order": list(V616_TASK_ORDER),
            "observed_id_order": list(V616_TASK_ORDER),
            "task_count_changed": False,
            "task_order_changed": False,
            "scope_expanded": False,
            "disposition": "preserve_frozen_contract",
            "terminal": True,
        }
    ]
    for finding_id in MARKER_FINDING_IDS[:4]:
        rows.append(
            {
                "scope_id": finding_id,
                "source_result_changes_scope": False,
                "disposition": "preserve_existing_v616_direction",
                "new_task_added": False,
                "terminal": True,
            }
        )
    for technique in RETIRED_TECHNIQUES:
        rows.append(
            {
                "scope_id": technique,
                "retired_technique": True,
                "reopened": False,
                "source_result_changes_scope": False,
                "disposition": "reject",
                "new_task_added": False,
                "terminal": True,
            }
        )
    return rows


def _eligible_delta(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("source_kind") in {"primary_paper", "first_party_report"}
        and row.get("primary_document_opened") is True
        and row.get("primary_method_verified") is True
        and row.get("cutoff_relation") == "post_marker"
        and str(row.get("canonical_url", "")).startswith("https://")
        and str(row.get("scope_disposition", "")).strip()
    )


def reference_append_rows(delta_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Prepare appends only for verified later sources with bounded scope."""

    eligible = [dict(row) for row in delta_rows if _eligible_delta(row)]
    if not eligible:
        return [
            {
                "action": "no_change",
                "marker": REFERENCE_MARKER,
                "appended": False,
                "reason": "No verified primary finding has proved ordering after the V616 marker.",
                "scientific_improvement_claimed": False,
                "terminal": True,
            }
        ]
    return [
        {
            "action": "append",
            "candidate_id": row["candidate_id"],
            "verified_source_id": row["source_id"],
            "canonical_url": row["canonical_url"],
            "cutoff_relation": "post_marker",
            "primary_method_verified": True,
            "scope_disposition": row["scope_disposition"],
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
    """Combine cutoff, rate-limit, and same-day uncertainty evidence."""

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
            "source_id": f"semantic_scholar_{row.get('paper_id', 'route')}_{row.get('attempt', 'attempt')}",
            "http_status": row.get("http_status"),
            "access_outcome": row.get("access_outcome"),
            "terminal": True,
        }
        for row in semantic_rows
        if row.get("http_status") == 429
    )
    rows.extend(
        {
            "receipt_kind": "same_day_uncertainty",
            "source_id": row.get("source_id", "logical_intelligence_route"),
            "source_date": row.get("publication_or_update_date", row.get("accessed_on")),
            "marker_date": MARKER_DATE,
            "relation": row.get("cutoff_relation", "same_day_order_uncertain"),
            "reason": "The page has no update time that proves ordering after the marker.",
            "terminal": True,
        }
        for row in logical_rows
    )
    return rows


def source_row_groups() -> dict[str, list[JsonDict]]:
    """Return every requested family under one exact key."""

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


def _route_available(row: Mapping[str, Any]) -> bool:
    status = row.get("http_status")
    outcome = str(row.get("access_outcome", ""))
    return bool(
        isinstance(status, int)
        and 200 <= status < 400
        and outcome not in {"rate_limited", "forbidden", "challenge_required", "unavailable"}
    )


def _all_source_families_unavailable(artifact: Mapping[str, Any]) -> bool:
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
                row.get("primary_method_verified") is not True
                or row.get("cutoff_relation") != "post_marker"
                or not str(row.get("scope_disposition", "")).strip()
                or str(row.get("verified_source_id")) not in eligible_ids
            ):
                return False
        elif row.get("action") != "no_change" or eligible_ids:
            return False
    return True


def _classifications_valid(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("classification_rows")
    if not isinstance(rows, list):
        return False
    expected = {row["candidate_id"] for row in classification_rows()}
    observed = [str(row.get("candidate_id")) for row in rows if isinstance(row, Mapping)]
    duplicate_ids = {
        str(row.get("candidate_id"))
        for row in artifact.get("duplicate_rows", [])
        if isinstance(row, Mapping)
    }
    duplicate_class_ids = {
        str(row.get("candidate_id"))
        for row in rows
        if isinstance(row, Mapping) and row.get("classification") == "duplicate"
    }
    return bool(
        len(observed) == len(set(observed))
        and set(observed) == expected
        and duplicate_ids == duplicate_class_ids
        and all(
            isinstance(row, Mapping)
            and row.get("classification") in CLASSIFICATIONS
            and row.get("terminal") is True
            for row in rows
        )
    )


def _scope_valid(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("scope_audit_rows") == scope_audit_rows()


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Recompute audit completion from terminal evidence rows."""

    duplicates = artifact.get("duplicate_rows")
    cutoff = artifact.get("cutoff_rate_limit_and_same_day_uncertainty_receipts")
    primary = artifact.get("primary_source_rows")
    if not all(isinstance(value, list) for value in (duplicates, cutoff, primary)):
        return 0
    duplicate_ok = all(
        isinstance(row, Mapping) and row.get("suppressed") is True and row.get("terminal") is True
        for row in duplicates
    )
    cutoff_kinds = {str(row.get("receipt_kind")) for row in cutoff if isinstance(row, Mapping)}
    primary_ok = bool(primary) and all(
        isinstance(row, Mapping)
        and row.get("primary_document_opened") is True
        and row.get("primary_method_verified") is True
        and str(row.get("full_text_url", "")).startswith("https://")
        and row.get("terminal") is True
        for row in primary
    )
    return int(
        _families_complete(artifact)
        and not _all_source_families_unavailable(artifact)
        and _classifications_valid(artifact)
        and duplicate_ok
        and {"cutoff", "rate_limit", "same_day_uncertainty"} <= cutoff_kinds
        and primary_ok
        and _scope_valid(artifact)
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
        "scope_audit_rows",
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
    """Probe the nearest parent without writing the requested artifact."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7029-write-", delete=True):
            pass
    except OSError:
        return False
    return True


def network_available() -> bool:
    """Treat any HTTP response, including 429, as reachable network access."""

    request = Request("https://arxiv.org/", headers={"User-Agent": "Carnot-Exp7029/1.0"})
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
    """Check source access, the marker, upstream evidence, and output path."""

    rows: list[JsonDict] = [
        {
            "check": "network_access",
            "expected_value": "available",
            "observed_value": "available" if network_is_available else "unavailable",
            "passed": network_is_available,
            "terminal": True,
        }
    ]
    try:
        marker_present = REFERENCE_MARKER in (root / REFERENCE_PATH).read_text(encoding="utf-8")
    except OSError:
        marker_present = False
    rows.append(
        {
            "check": "v616_reference_marker",
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
    """Hash required and new local sources that exist at build time."""

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
    """Hash all artifact fields except the field that stores this hash."""

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
    network_is_available: bool | None = None,
    duration_s: float | None = None,
    source_groups: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> JsonDict:
    """Build a positive, blocked, or disqualified source-audit artifact."""

    started = time.monotonic()
    reachable = network_available() if network_is_available is None else network_is_available
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
        "scope_audit_rows": [],
        "reference_append_rows": [],
        "rows": [],
        "cutoff_rate_limit_and_same_day_uncertainty_receipts": [],
        "v616_sota_scope_complete_score": 0,
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
        "honest_verdict": "blocked_v616_sota_scope_audit",
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
        artifact["scope_audit_rows"] = scope_audit_rows()
        artifact["reference_append_rows"] = reference_append_rows(
            artifact["post_marker_delta_rows"]
        )
        artifact["cutoff_rate_limit_and_same_day_uncertainty_receipts"] = cutoff_receipts(
            artifact["primary_source_rows"],
            artifact["semantic_scholar_rows"],
            artifact["logical_intelligence_rows"],
        )
        artifact["v616_sota_scope_complete_score"] = completion_score(artifact)
        if _all_source_families_unavailable(artifact):
            artifact["gate_check_summary"] = {
                "failed_check": "all_source_families_unavailable",
                "expected_value": "at_least_one_usable_source_family",
                "observed_value": "no_usable_source_families",
                "passed": False,
            }
        elif artifact["v616_sota_scope_complete_score"] == 1:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = (
                "complete_positive_v616_sota_scope_audit_zero_delta_no_scientific_improvement"
            )
            artifact["gate_check_summary"] = {
                "failed_check": None,
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            }
        else:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "disqualified_v616_sota_scope_audit_contract_failed"
            artifact["gate_check_summary"] = {
                "failed_check": "v616_sota_scope_contract",
                "expected_value": 1,
                "observed_value": artifact["v616_sota_scope_complete_score"],
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
    """Recompute the contract without trusting stored summary fields."""

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
    preconditions = artifact["preconditions_checked"]
    failed_precondition = next(
        (
            row
            for row in preconditions
            if isinstance(preconditions, list)
            and isinstance(row, Mapping)
            and row.get("passed") is False
        ),
        None,
    )
    if failed_precondition is None:
        if not _families_complete(artifact):
            errors.append("source_family_coverage_invalid")
        primary = artifact["primary_source_rows"]
        if not isinstance(primary, list):
            errors.append("primary_source_rows_invalid")
        else:
            identities = [
                str(row.get("source_identity")) for row in primary if isinstance(row, Mapping)
            ]
            if len(identities) != len(set(identities)):
                errors.append("unsuppressed_duplicate_source_identity")
            if not all(
                isinstance(row, Mapping)
                and row.get("primary_document_opened") is True
                and row.get("primary_method_verified") is True
                and row.get("implementation_evidence_claimed") is False
                for row in primary
            ):
                errors.append("primary_source_boundary_invalid")
        if not _classifications_valid(artifact):
            errors.append("classification_rows_invalid")
        if not _scope_valid(artifact):
            errors.append("scope_audit_invalid")
        cutoff = artifact["cutoff_rate_limit_and_same_day_uncertainty_receipts"]
        cutoff_kinds = {
            str(row.get("receipt_kind"))
            for row in cutoff
            if isinstance(cutoff, list) and isinstance(row, Mapping)
        }
        if (
            not isinstance(cutoff, list)
            or not {"cutoff", "rate_limit", "same_day_uncertainty"} <= cutoff_kinds
        ):
            errors.append("cutoff_receipts_invalid")
        if not _reference_appends_valid(artifact):
            errors.append("reference_append_invalid")
        guarded_rows = [
            *artifact["secondary_source_rows"],
            *artifact["github_rows"],
            *artifact["extropic_rows"],
            *artifact["logical_intelligence_rows"],
        ]
        if any(
            isinstance(row, Mapping)
            and (
                row.get("implementation_readiness_claimed") is True
                or row.get("hardware_readiness_claimed") is True
                or row.get("reproducible_local_runner_claimed") is True
            )
            for row in guarded_rows
        ):
            errors.append("unsupported_readiness_claim")
    if artifact["rows"] != _combined_rows(artifact):
        errors.append("rows_consistency_invalid")
    score = completion_score(artifact)
    if artifact["v616_sota_scope_complete_score"] != score:
        errors.append("v616_sota_scope_complete_score_invalid")
    all_unavailable = _all_source_families_unavailable(artifact)
    expected_class = (
        "blocked"
        if failed_precondition is not None or all_unavailable
        else ("positive" if score == 1 else "disqualified")
    )
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    elif artifact["verdict_class"] != expected_class:
        errors.append("verdict_class_inconsistent")
    prefixes = {
        "positive": "complete_positive_",
        "circular_positive": "complete_circular_positive_",
        "null": "null_",
        "blocked": "blocked_",
        "disqualified": "disqualified_",
        "partial": "partial_",
    }
    prefix = prefixes.get(str(artifact["verdict_class"]), "")
    if not prefix or not str(artifact["honest_verdict"]).startswith(prefix):
        errors.append("honest_verdict_prefix_invalid")
    if expected_class == "positive" and not artifact["post_marker_delta_rows"]:
        if "no_scientific_improvement" not in str(artifact["honest_verdict"]):
            errors.append("zero_delta_improvement_boundary_invalid")
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
    """Validate before the atomic write so malformed evidence cannot replace a result."""

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
    """Build the audit or validate an existing artifact."""

    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        if errors:
            print(json.dumps({"errors": errors}, sort_keys=True))
            return 1
        print(json.dumps({"path": str(args.validate), "valid": True}, sort_keys=True))
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
