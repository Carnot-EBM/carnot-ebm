"""Build the bounded V622 source-ingestion and claim-boundary receipt.

The web sweep is evidence collection, not model inference. This module freezes
the observed receipts, validates their boundaries, and appends only a verified
post-planner control or adoption. Spec refs: REQ-REPORT-7092 and
SCENARIO-REPORT-7092-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260907"
SOURCE_CAPTURE_UTC = "2026-09-07T00:34:50Z"
RANDOM_SEED = 7_092_202_609_07
INFERENCE_SUBSTRATE = (
    "low_concurrency_primary_source_ingestion_no_experimental_llm: "
    "bounded network literature and repository audit"
)
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
PLANNER_HEADING = "## V622 planner refresh - 2026-09-06"
REFERENCE_START_MARKER = "<!-- V622-EXECUTION-DELTA-20260907-START -->"
REFERENCE_END_MARKER = "<!-- V622-EXECUTION-DELTA-20260907-END -->"
VENDOR_BOUNDARY = "vendor_claim_not_independent_evidence"

REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7092_v622_sota_ingestion.json")
SOURCE_ARTIFACT_PATHS = (
    REFERENCE_PATH,
    Path("research-program.md"),
    Path("research-complete.yaml"),
    Path("research-roadmap.yaml"),
    Path("ops/status.md"),
    Path("ops/known-issues.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/experiment_7092_v622_sota_ingestion.py"),
    Path("scripts/experiments/experiment_7092_v622_sota_ingestion.py"),
    Path("tests/python/test_experiment_7092_v622_sota_ingestion.py"),
)

QUERY_FAMILIES = (
    "ebm_verification_reasoning",
    "neural_constraints",
    "ising_machine_learning",
    "hallucination_control",
    "kolmogorov_arnold_networks",
    "energy_guided_constrained_generation",
    "probabilistic_hardware",
    "continual_learning",
)
SOURCE_CLASSES = (
    "arxiv",
    "openreview",
    "semantic_scholar",
    "huggingface_papers",
    "github",
    "extropic",
    "logical_intelligence",
)
SECONDARY_ROUTES = ("openreview", "semantic_scholar", "huggingface_papers")
CLASSIFICATIONS = ("adopt", "control", "watch", "reject", "duplicate")
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)
ALLOWED_EXPERIMENT_HOOKS = {
    "exp7093-recovered-entrance-bank-sufficiency-audit",
    "exp7094-matched-hardness-entrance-diagnostic",
    "exp7095-entrance-energy-matched-controls",
    "exp7096-cold-entrance-energy-abstention-audit",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "search_window",
    "query_rows",
    "source_class_rows",
    "arxiv_rows",
    "openreview_rows",
    "semantic_scholar_rows",
    "huggingface_papers_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "candidate_rows",
    "deduplication_rows",
    "primary_source_receipts",
    "adoption_rows",
    "claim_boundary_rows",
    "references_append_path",
    "references_append_hash",
    "rows",
    "per_game_results",
    "v622_sota_ingestion_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

ROW_COLLECTION_FIELDS = (
    "query_rows",
    "source_class_rows",
    "arxiv_rows",
    "openreview_rows",
    "semantic_scholar_rows",
    "huggingface_papers_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "candidate_rows",
    "deduplication_rows",
    "primary_source_receipts",
    "adoption_rows",
    "claim_boundary_rows",
    "per_game_results",
)

FIELD_PRINCIPLES = {
    "field_principles": "One principle per field makes the scientific contract reviewable.",
    "preconditions_checked": "Explicit gates prevent unavailable sources from becoming fabricated results.",
    "inference_substrate": "The declared substrate limits the work to a bounded network audit.",
    "inference_substrate_class": "The compute class states that the sweep loads no model.",
    "execution_venue": "The venue separates host execution from unavailable hardware claims.",
    "duration_s": "Measured duration distinguishes an executed audit from a template.",
    "source_artifact_hashes": "Hashes bind the result to the local planning evidence that was read.",
    "search_window": "An explicit window makes freshness and exclusion decisions falsifiable.",
    "query_rows": "Exact queries disclose coverage without converting search rank into evidence.",
    "source_class_rows": "Every requested source class ends in an honest dated receipt.",
    "arxiv_rows": "Primary-paper rows preserve canonical identity, date, and bounded claims.",
    "openreview_rows": "Challenge responses remain access receipts rather than inferred content.",
    "semantic_scholar_rows": "Visible citation rows are index observations, not scientific authority.",
    "huggingface_papers_rows": "Paper-feed rows remain discovery signals below primary papers.",
    "github_rows": "Repository activity and licenses identify code but do not prove quality.",
    "extropic_rows": "Vendor hardware projections stay separate from attached-hardware evidence.",
    "logical_intelligence_rows": "Kona remains a product comparator without public reproducibility.",
    "candidate_rows": "A closed disposition prevents an interesting lead from silently becoming work.",
    "deduplication_rows": "Canonical suppression stops rediscovery from appearing novel.",
    "primary_source_receipts": "Each candidate traces to a primary paper or official project page.",
    "adoption_rows": "Every promoted method names one bounded existing experiment hook.",
    "claim_boundary_rows": "Each item states what its source cannot establish for Carnot.",
    "references_append_path": "The declared ledger path makes the only permitted documentation write explicit.",
    "references_append_hash": "The final ledger hash proves byte stability for an empty delta.",
    "rows": "A combined ledger permits independent collection-consistency checks.",
    "per_game_results": "An empty game ledger states that literature review measured no game outcome.",
    "v622_sota_ingestion_complete_score": "One means source coverage and every promotion boundary passed.",
    "random_seed": "A fixed seed records deterministic ordering even though no sampling occurs.",
    "reproducibility_checksum": "A stable payload hash detects any evidence or verdict change.",
    "gate_check_summary": "Exact expected and observed values make a blocked run actionable.",
    "verifier_is_oracle": "False states that source review cannot certify Carnot correctness.",
    "verdict_class": "A closed terminal class prevents outages or invalid rows from reading positive.",
    "honest_verdict": "A class-matched terminal prefix gives automation one unambiguous outcome.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash a readable file while preserving absence as an explicit null."""

    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def search_window(run_date: str) -> JsonDict:
    """Return the broad review window and the strict execution-delta boundary."""

    execution_date = datetime.strptime(run_date, "%Y%m%d").date().isoformat()
    return {
        "literature_start_date": "2025-01-01",
        "literature_end_date": execution_date,
        "planner_cutoff_date": "2026-09-06",
        "delta_rule": "publication_or_revision_date > planner_cutoff_date and <= literature_end_date",
        "post_cutoff_arxiv_result_count": 0,
    }


def in_execution_delta(value: str, window: Mapping[str, Any]) -> bool:
    """Return true only for a real source date after planning and by execution."""

    try:
        observed = date.fromisoformat(value)
        lower = date.fromisoformat(str(window["planner_cutoff_date"]))
        upper = date.fromisoformat(str(window["literature_end_date"]))
    except (KeyError, TypeError, ValueError):
        return False
    return lower < observed <= upper


def identifier_matches_url(source_id: str, source_url: str) -> bool:
    """Check canonical identifiers against the route that claims to own them."""

    try:
        parsed = urlparse(source_url)
    except (TypeError, ValueError):
        return False
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    if source_id.startswith("arxiv:"):
        identifier = source_id.removeprefix("arxiv:")
        return bool(re.fullmatch(r"\d{4}\.\d{5}", identifier)) and (
            parsed.netloc == "arxiv.org" and parsed.path.rstrip("/") == f"/abs/{identifier}"
        )
    if source_id.startswith("openreview:"):
        identifier = source_id.removeprefix("openreview:")
        return parsed.netloc.endswith("openreview.net") and parse_qs(parsed.query).get("id") == [
            identifier
        ]
    if source_id.startswith("github:"):
        identifier = source_id.removeprefix("github:")
        return parsed.netloc == "github.com" and parsed.path.strip("/") == identifier
    if source_id.startswith("hf-paper:"):
        identifier = source_id.removeprefix("hf-paper:")
        return parsed.netloc == "huggingface.co" and parsed.path.rstrip("/") == (
            f"/papers/{identifier}"
        )
    if source_id == "extropic:z1t-2026-09-04":
        return parsed.netloc == "extropic.ai" and parsed.path.rstrip("/") == "/writing/z1t"
    if source_id == "logical-intelligence:kona-1.0":
        return parsed.netloc == "logicalintelligence.com" and parsed.path.rstrip("/") == (
            "/kona-ebms-energy-based-models"
        )
    return False


def _receipt_hash(row: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in row.items() if key != "metadata_receipt_sha256"}
    return _sha256_json(stable)


def primary_source_receipts() -> list[JsonDict]:
    """Return primary-paper and official-project identities checked in the sweep."""

    records: list[JsonDict] = [
        {
            "receipt_id": "receipt-arxiv-2608.28128",
            "source_id": "arxiv:2608.28128",
            "title": "VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning",
            "source_url": "https://arxiv.org/abs/2608.28128",
            "publication_or_revision_date": "2026-08-28",
            "version_or_revision": "v1",
            "source_role": "primary_paper",
            "core_claim": "Verifier evidence can be traced to actions through dependency-valid proof edges while preserving terminal reward.",
            "content_verified": True,
        },
        {
            "receipt_id": "receipt-arxiv-2606.02461",
            "source_id": "arxiv:2606.02461",
            "title": "AgentCL: Toward Rigorous Evaluation of Continual Learning in Language Agents",
            "source_url": "https://arxiv.org/abs/2606.02461",
            "publication_or_revision_date": "2026-06-02",
            "version_or_revision": "v2",
            "source_role": "primary_paper",
            "core_claim": "Controlled reusable and naive task streams separate transfer from incidental context effects.",
            "content_verified": True,
        },
        {
            "receipt_id": "receipt-arxiv-2608.03874",
            "source_id": "arxiv:2608.03874",
            "title": "ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?",
            "source_url": "https://arxiv.org/abs/2608.03874",
            "publication_or_revision_date": "2026-08-04",
            "version_or_revision": "v1",
            "source_role": "primary_paper",
            "core_claim": "Explicit skill memory is not uniformly better than equal-context adaptation.",
            "content_verified": True,
        },
        {
            "receipt_id": "receipt-arxiv-2607.17047",
            "source_id": "arxiv:2607.17047",
            "title": "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
            "source_url": "https://arxiv.org/abs/2607.17047",
            "publication_or_revision_date": "2026-07-19",
            "version_or_revision": "v1",
            "source_role": "primary_paper",
            "core_claim": "Matched SAT instances can separate solver hardness from language-model accuracy.",
            "content_verified": True,
        },
        {
            "receipt_id": "receipt-arxiv-2608.00220",
            "source_id": "arxiv:2608.00220",
            "title": "Verifier-Induced Support Reshaping in On-Policy Optimization",
            "source_url": "https://arxiv.org/abs/2608.00220",
            "publication_or_revision_date": "2026-07-31",
            "version_or_revision": "v1",
            "source_role": "primary_paper",
            "core_claim": "A verifier-scored policy can improve pass-at-one while reducing bounded best-of-k support.",
            "content_verified": True,
        },
        {
            "receipt_id": "receipt-extropic-z1t",
            "source_id": "extropic:z1t-2026-09-04",
            "title": "Z1T: Sparse Transformer-Like Models for Probabilistic Hardware",
            "source_url": "https://extropic.ai/writing/z1t",
            "publication_or_revision_date": "2026-09-04",
            "version_or_revision": "captured-2026-09-07",
            "source_role": "official_vendor_project_page",
            "core_claim": "The public design maps sparse sampled layers to a degree-16 graph and assigns other operations to an FPGA.",
            "content_verified": True,
        },
    ]
    for row in records:
        row.update({"accessed_at": SOURCE_CAPTURE_UTC, "terminal": True})
        row["metadata_receipt_sha256"] = _receipt_hash(row)
    return records


def query_rows(run_date: str) -> list[JsonDict]:
    """Return exact topic and requested-route queries with bounded outcomes."""

    window = search_window(run_date)
    rows = [
        {
            "query_id": f"arxiv-{family}",
            "query_family": family,
            "route": "arxiv",
            "exact_query": (
                f'submittedDate:[202501010000 TO {run_date}2359] AND all:"{family.replace("_", " ")}"'
            ),
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "completed_bounded_search",
            "search_rank_used_as_evidence": False,
            "terminal": True,
        }
        for family in QUERY_FAMILIES
    ]
    rows.extend(
        [
            {
                "query_id": "arxiv-post-v622-cutoff",
                "query_family": "execution_delta",
                "route": "arxiv",
                "exact_query": (
                    "submittedDate:[202609060000 TO 202609072359] AND "
                    "(energy-based OR constraint reasoning OR Ising OR hallucination control "
                    "OR KAN OR constrained generation OR probabilistic hardware OR continual learning)"
                ),
                "result_count_observed": window["post_cutoff_arxiv_result_count"],
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_complete",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
            {
                "query_id": "openreview-ebm-submissions",
                "query_family": "requested_source_route",
                "route": "openreview",
                "exact_query": "OpenReview EBM submissions and forum MtKSNKnNzN",
                "http_status": 403,
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "challenge_verification_required",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
            {
                "query_id": "semantic-scholar-ebt-citations",
                "query_family": "requested_source_route",
                "route": "semantic_scholar",
                "exact_query": "citations for ARXIV:2507.02092 limit 100",
                "result_count_observed": 35,
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_complete",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
            {
                "query_id": "semantic-scholar-arm-ebm-citations",
                "query_family": "requested_source_route",
                "route": "semantic_scholar",
                "exact_query": "citations for ARXIV:2512.15605 limit 100",
                "result_count_observed": 8,
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_complete",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
            {
                "query_id": "huggingface-daily-papers",
                "query_family": "requested_source_route",
                "route": "huggingface_papers",
                "exact_query": "GET /api/daily_papers?limit=100 and relevant paper pages",
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_complete",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
            {
                "query_id": "github-targeted-repositories",
                "query_family": "requested_source_route",
                "route": "github",
                "exact_query": "official EBT, Z1T, EB-JEPA, Enso, and support-reshaping repositories",
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_complete",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
            {
                "query_id": "extropic-writing-z1t",
                "query_family": "requested_source_route",
                "route": "extropic",
                "exact_query": "GET https://extropic.ai/writing/z1t",
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_complete",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
            {
                "query_id": "logical-intelligence-kona",
                "query_family": "requested_source_route",
                "route": "logical_intelligence",
                "exact_query": "GET https://logicalintelligence.com/kona-ebms-energy-based-models",
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_complete",
                "search_rank_used_as_evidence": False,
                "terminal": True,
            },
        ]
    )
    return rows


def arxiv_rows() -> list[JsonDict]:
    """Return canonical paper observations, including the empty delta query."""

    rows = [
        {
            "row_id": "arxiv-post-cutoff-query",
            "source_url": "https://export.arxiv.org/api/query",
            "search_start_date": "2026-09-06",
            "search_end_date": "2026-09-07",
            "matching_result_count": 0,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]
    rows.extend(
        {
            "row_id": receipt["source_id"],
            "source_id": receipt["source_id"],
            "title": receipt["title"],
            "source_url": receipt["source_url"],
            "publication_or_revision_date": receipt["publication_or_revision_date"],
            "version_or_revision": receipt["version_or_revision"],
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_verified_primary",
            "terminal": True,
        }
        for receipt in primary_source_receipts()
        if str(receipt["source_id"]).startswith("arxiv:")
    )
    return rows


def openreview_rows() -> list[JsonDict]:
    """Preserve the observed anti-bot challenge without inferring page content."""

    return [
        {
            "source_id": "openreview:MtKSNKnNzN",
            "title_hint": "The Energy to Say No",
            "source_url": "https://openreview.net/forum?id=MtKSNKnNzN",
            "http_status": 403,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "challenge_verification_required",
            "content_verified_in_this_sweep": False,
            "claim_promoted": False,
            "terminal": True,
        }
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Return complete visible citation-list sizes without claiming authority."""

    return [
        {
            "source_id": "semantic-scholar:ARXIV:2507.02092",
            "paper_id": "ARXIV:2507.02092",
            "source_url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations",
            "visible_response_rows": 35,
            "newest_visible_publication_date": "2026-08-14",
            "citation_count_claimed_as_authoritative": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        },
        {
            "source_id": "semantic-scholar:ARXIV:2512.15605",
            "paper_id": "ARXIV:2512.15605",
            "source_url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations",
            "visible_response_rows": 8,
            "newest_visible_publication_date": "2026-07-02",
            "citation_count_claimed_as_authoritative": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        },
    ]


def huggingface_papers_rows() -> list[JsonDict]:
    """Return paper-index observations while excluding generated summaries."""

    return [
        {
            "source_id": "hf-paper:2608.28128",
            "source_url": "https://huggingface.co/papers/2608.28128",
            "daily_feed_url": "https://huggingface.co/api/daily_papers?limit=100",
            "latest_relevant_feed_submission_date": "2026-09-04",
            "evidence_role": "secondary_discovery_index_only",
            "generated_summary_used_as_evidence": False,
            "upvotes_used_as_evidence": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]


def github_rows() -> list[JsonDict]:
    """Return exact repository identities observed through the public API."""

    records = (
        (
            "alexiglad/EBT",
            "19420cbeae655bbf11930219a675ade6897019e8",
            "2026-04-21T00:53:47Z",
            "Apache-2.0",
        ),
        (
            "extropic-ai/sparse-transformers",
            "13051e90df9669be5b8f9f34fb097329fa82f674",
            "2026-09-03T20:28:22Z",
            "Apache-2.0",
        ),
        (
            "facebookresearch/eb_jepa",
            "966e61e9285b3a876f49b9774e9720d9a99a7925",
            "2026-07-17T22:33:32Z",
            "Apache-2.0",
        ),
        ("MVPandey/Enso", "8013c8f95e9e4402faaa91c99835c4ccedc54ef9", "2026-03-10T05:27:24Z", None),
        (
            "sylvain-wei/verifier-induced-support-reshaping",
            "48a1d56927253a913ad22e89ec960fad98a204d5",
            "2026-08-12T06:25:46Z",
            "Apache-2.0",
        ),
    )
    return [
        {
            "source_id": f"github:{name}",
            "source_url": f"https://github.com/{name}",
            "default_branch_revision": revision,
            "pushed_at": pushed_at,
            "license": license_name,
            "stars_used_as_evidence": False,
            "implementation_promoted": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
        for name, revision, pushed_at, license_name in records
    ]


def extropic_rows() -> list[JsonDict]:
    """Return Z1T identity while making every hardware limitation explicit."""

    return [
        {
            "source_id": "extropic:z1t-2026-09-04",
            "source_url": "https://extropic.ai/writing/z1t",
            "source_role": "official_vendor_project_page",
            "claim_boundary_label": VENDOR_BOUNDARY,
            "verified_public_design_scope": "degree-16 sparse graph and heterogeneous Z1/FPGA design",
            "projection_exclusions_observed": [
                "final_dense_logit_readout",
                "interchip_data_movement",
                "required_chip_count",
            ],
            "scientific_evidence_promoted": False,
            "attached_hardware_claimed": False,
            "runtime_claimed": False,
            "power_claimed": False,
            "availability_claimed": False,
            "speed_claimed": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]


def logical_intelligence_rows() -> list[JsonDict]:
    """Keep Kona as proprietary product context, not reproducible evidence."""

    return [
        {
            "source_id": "logical-intelligence:kona-1.0",
            "source_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "source_role": "official_vendor_project_page",
            "claim_boundary_label": VENDOR_BOUNDARY,
            "public_weights_observed": False,
            "training_recipe_observed": False,
            "local_runner_observed": False,
            "scientific_evidence_promoted": False,
            "attached_hardware_claimed": False,
            "runtime_claimed": False,
            "power_claimed": False,
            "availability_claimed": False,
            "speed_claimed": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]


def source_class_rows() -> list[JsonDict]:
    """Return one terminal reachability or access receipt for every route."""

    outcomes = {
        "arxiv": ("http_200_complete", True),
        "openreview": ("challenge_verification_required", False),
        "semantic_scholar": ("http_200_complete", True),
        "huggingface_papers": ("http_200_complete", True),
        "github": ("http_200_complete", True),
        "extropic": ("http_200_complete", True),
        "logical_intelligence": ("http_200_complete", True),
    }
    return [
        {
            "source_class": source_class,
            "access_outcome": outcomes[source_class][0],
            "content_available": outcomes[source_class][1],
            "honest_receipt": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for source_class in SOURCE_CLASSES
    ]


def candidate_rows() -> list[JsonDict]:
    """Classify the bounded candidates without inventing a post-cutoff delta."""

    records = (
        (
            "vict-credit-tracing",
            "receipt-arxiv-2608.28128",
            "watch",
            "The verifier-to-action proof-edge method is relevant, but it predates the execution-delta window and has no active V622 task hook.",
            "The reported ALFWorld and WebShop results do not establish Carnot ARC credit or authorize a weight update.",
        ),
        (
            "agentcl-controlled-streams",
            "receipt-arxiv-2606.02461",
            "duplicate",
            "The V622 planner refresh already records AgentCL and its controlled-stream hook.",
            "AgentCL motivates controls; exact later Carnot outcomes remain the admission authority.",
        ),
        (
            "continual-skill-bench",
            "receipt-arxiv-2608.03874",
            "duplicate",
            "The V622 planner refresh already records equal-context and no-memory controls.",
            "Benchmark averages do not prove that a Carnot memory rule transfers.",
        ),
        (
            "solver-hard-diagnostic",
            "receipt-arxiv-2607.17047",
            "duplicate",
            "The V622 planner refresh already records matched-hardness source-group controls.",
            "Solver statistics do not stand in for Carnot model headroom or exact reachability.",
        ),
        (
            "verifier-support-reshaping",
            "receipt-arxiv-2608.00220",
            "duplicate",
            "The V621 activation delta already records support and coverage as separate controls.",
            "Published RLVR behavior does not establish a Carnot selector benefit.",
        ),
        (
            "extropic-z1t",
            "receipt-extropic-z1t",
            "duplicate",
            "The V622 planner refresh already records the degree-16 software receipt and exclusions.",
            "Vendor projections cannot establish Carnot runtime, power, speed, availability, or Z1 execution.",
        ),
    )
    receipts = {row["receipt_id"]: row for row in primary_source_receipts()}
    rows = []
    for candidate_id, receipt_id, classification, reason, boundary in records:
        receipt = receipts[receipt_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_id": receipt["source_id"],
                "title": receipt["title"],
                "source_url": receipt["source_url"],
                "publication_or_revision_date": receipt["publication_or_revision_date"],
                "source_receipt_id": receipt_id,
                "source_role": receipt["source_role"],
                "identity_verified": True,
                "core_claim_verified": True,
                "classification": classification,
                "decision_relevant": False,
                "experiment_hook": None,
                "claim_boundary": boundary,
                "classification_reason": reason,
                "terminal": True,
            }
        )
    return rows


def deduplication_rows(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return the exact duplicate set and its existing source-ledger boundary."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "canonical_source_id": row["source_id"],
            "existing_reference_marker": PLANNER_HEADING,
            "suppressed": True,
            "terminal": True,
        }
        for row in candidates
        if isinstance(row, Mapping) and row.get("classification") == "duplicate"
    ]


def adoption_rows(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return only verified control or adoption decisions that change V622 work."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "source_id": row["source_id"],
            "classification": row["classification"],
            "experiment_hook": row["experiment_hook"],
            "source_receipt_id": row["source_receipt_id"],
            "claim_boundary": row["claim_boundary"],
            "terminal": True,
        }
        for row in candidates
        if isinstance(row, Mapping)
        and row.get("classification") in {"adopt", "control"}
        and row.get("decision_relevant") is True
    ]


def claim_boundary_rows(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Make every candidate's prohibited inference independently enumerable."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "source_id": row["source_id"],
            "claim_boundary": row["claim_boundary"],
            "claim_boundary_present": bool(row.get("claim_boundary")),
            "terminal": True,
        }
        for row in candidates
        if isinstance(row, Mapping)
    ]


def combined_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Combine typed evidence rows while preserving collection provenance."""

    rows: list[JsonDict] = []
    for field in ROW_COLLECTION_FIELDS:
        value = artifact.get(field, [])
        if isinstance(value, list):
            rows.extend(
                {"collection": field, **deepcopy(row)} for row in value if isinstance(row, dict)
            )
    return rows


def http_reachable(url: str) -> bool:
    """Treat any HTTP response as route reachability, including explicit denial."""

    request = Request(url, headers={"User-Agent": "carnot-v622-source-audit/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def probe_routes() -> dict[str, bool]:
    """Probe only the routes named by the minimum network precondition."""

    urls = {
        "arxiv": "https://arxiv.org/abs/2608.28128",
        "openreview": "https://openreview.net/forum?id=MtKSNKnNzN",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092",
        "huggingface_papers": "https://huggingface.co/papers/2608.28128",
    }
    return {route: http_reachable(url) for route, url in urls.items()}


def _readable_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_bytes())
    except OSError:
        return False


def _writable_target(path: Path) -> bool:
    parent = path.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7092-", delete=True):
            return True
    except OSError:
        return False


def check_preconditions(
    root: Path,
    output_path: Path,
    *,
    route_reachability: Mapping[str, bool] | None = None,
) -> list[JsonDict]:
    """Evaluate all network, local-read, and destination-write gates."""

    routes = dict(probe_routes() if route_reachability is None else route_reachability)
    arxiv_ok = routes.get("arxiv") is True
    secondary_ok = any(routes.get(route) is True for route in SECONDARY_ROUTES)
    references = root / REFERENCE_PATH
    readable = _readable_nonempty(references)
    marker_present = False
    if readable:
        try:
            marker_present = PLANNER_HEADING in references.read_text(encoding="utf-8")
        except OSError:
            marker_present = False
    reference_writable = _writable_target(references)
    artifact_writable = _writable_target(output_path)
    return [
        {
            "check": "arxiv_network_reachability",
            "expected_value": True,
            "observed_value": arxiv_ok,
            "passed": arxiv_ok,
        },
        {
            "check": "secondary_index_reachability",
            "expected_value": "at_least_one_of_openreview_semantic_scholar_huggingface_papers",
            "observed_value": sorted(route for route in SECONDARY_ROUTES if routes.get(route)),
            "passed": secondary_ok,
        },
        {
            "check": "readable_current_references",
            "expected_value": "readable_nonempty_file",
            "observed_value": "readable_nonempty_file" if readable else "missing_or_unreadable",
            "passed": readable,
        },
        {
            "check": "v622_planner_marker",
            "expected_value": PLANNER_HEADING,
            "observed_value": PLANNER_HEADING if marker_present else "missing",
            "passed": marker_present,
        },
        {
            "check": "writable_references_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if reference_writable else "not_writable",
            "passed": reference_writable,
        },
        {
            "check": "writable_artifact_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if artifact_writable else "not_writable",
            "passed": artifact_writable,
        },
    ]


def _appendable_candidates(
    candidates: Sequence[Mapping[str, Any]], run_date: str
) -> list[Mapping[str, Any]]:
    window = search_window(run_date)
    return [
        row
        for row in candidates
        if row.get("classification") in {"adopt", "control"}
        and row.get("decision_relevant") is True
        and row.get("identity_verified") is True
        and row.get("core_claim_verified") is True
        and row.get("source_role") == "primary_paper"
        and row.get("experiment_hook") in ALLOWED_EXPERIMENT_HOOKS
        and identifier_matches_url(str(row.get("source_id", "")), str(row.get("source_url", "")))
        and in_execution_delta(str(row.get("publication_or_revision_date", "")), window)
    ]


def append_references_delta(
    path: Path, candidates: Sequence[Mapping[str, Any]], run_date: str
) -> bool:
    """Append one marker-delimited verified delta, or leave the ledger unchanged."""

    accepted = _appendable_candidates(candidates, run_date)
    if not accepted:
        return False
    text = path.read_text(encoding="utf-8")
    if REFERENCE_START_MARKER in text or REFERENCE_END_MARKER in text:
        return False
    lines = [
        "",
        f"## V622 execution delta - {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}",
        "",
        REFERENCE_START_MARKER,
        "",
        "Only verified execution-window controls or adoptions are listed below.",
        "",
    ]
    for row in accepted:
        lines.append(
            f"- **{row['title']}** - {row['source_id']}, {row['source_url']}. "
            f"Carnot hook: `{row['experiment_hook']}`. Boundary: {row['claim_boundary']}"
        )
    lines.extend(["", REFERENCE_END_MARKER, ""])
    atomic_write_text(path, text.rstrip() + "\n" + "\n".join(lines), allow_override=False)
    return True


def _source_hashes(root: Path) -> list[JsonDict]:
    rows = []
    for relative in SOURCE_ARTIFACT_PATHS:
        digest = file_sha256(root / relative)
        if digest is not None:
            rows.append({"path": relative.as_posix(), "sha256": digest, "terminal": True})
    return rows


def _coverage_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    classes = artifact.get("source_class_rows", [])
    observed_classes = {row.get("source_class") for row in classes if isinstance(row, dict)}
    if observed_classes != set(SOURCE_CLASSES) or any(
        not isinstance(row, dict)
        or row.get("honest_receipt") is not True
        or row.get("terminal") is not True
        or row.get("accessed_at") != SOURCE_CAPTURE_UTC
        for row in classes
    ):
        errors.append("requested source class receipt coverage mismatch")

    queries = artifact.get("query_rows", [])
    query_families = {
        row.get("query_family")
        for row in queries
        if isinstance(row, dict)
        and row.get("route") == "arxiv"
        and row.get("query_family") in QUERY_FAMILIES
    }
    if query_families != set(QUERY_FAMILIES):
        errors.append("query family coverage mismatch")
    query_routes = {row.get("route") for row in queries if isinstance(row, dict)}
    if query_routes != set(SOURCE_CLASSES):
        errors.append("requested source route query coverage mismatch")

    required_nonempty = (
        "arxiv_rows",
        "openreview_rows",
        "semantic_scholar_rows",
        "huggingface_papers_rows",
        "github_rows",
        "extropic_rows",
        "logical_intelligence_rows",
        "candidate_rows",
        "primary_source_receipts",
        "claim_boundary_rows",
    )
    for field in required_nonempty:
        rows = artifact.get(field)
        if not isinstance(rows, list) or not rows:
            errors.append(f"required row collection missing: {field}")
        elif any(not isinstance(row, dict) or row.get("terminal") is not True for row in rows):
            errors.append(f"nonterminal row in {field}")

    receipts = artifact.get("primary_source_receipts", [])
    receipt_by_id = {row.get("receipt_id"): row for row in receipts if isinstance(row, dict)}
    if len(receipt_by_id) != len(receipts):
        errors.append("duplicate primary source receipt id")
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        if not identifier_matches_url(
            str(receipt.get("source_id", "")), str(receipt.get("source_url", ""))
        ):
            errors.append("primary source identifier or URL mismatch")
        if receipt.get("metadata_receipt_sha256") != _receipt_hash(receipt):
            errors.append("primary source metadata receipt mismatch")
        if not receipt.get("title") or not receipt.get("publication_or_revision_date"):
            errors.append("primary source title or date missing")

    candidates = artifact.get("candidate_rows", [])
    candidate_ids = [row.get("candidate_id") for row in candidates if isinstance(row, dict)]
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("duplicate candidate id")
    window = artifact.get("search_window", {})
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            errors.append("candidate classification outside closed enum")
        receipt = receipt_by_id.get(candidate.get("source_receipt_id"))
        if receipt is None:
            errors.append("candidate primary source receipt missing")
        elif any(
            candidate.get(field) != receipt.get(field)
            for field in (
                "source_id",
                "title",
                "source_url",
                "publication_or_revision_date",
                "source_role",
            )
        ):
            errors.append("candidate and primary source receipt mismatch")
        if not identifier_matches_url(
            str(candidate.get("source_id", "")), str(candidate.get("source_url", ""))
        ):
            errors.append("candidate identifier or URL mismatch")
        if not candidate.get("claim_boundary"):
            errors.append("candidate claim boundary missing")
        if classification in {"adopt", "control"}:
            if candidate.get("source_role") != "primary_paper":
                errors.append("promotion requires a primary source")
            if (
                candidate.get("identity_verified") is not True
                or candidate.get("core_claim_verified") is not True
            ):
                errors.append("promotion requires verified title identity and core claim")
            if not in_execution_delta(
                str(candidate.get("publication_or_revision_date", "")), window
            ):
                errors.append("promotion is outside the execution delta date window")
            if candidate.get("experiment_hook") not in ALLOWED_EXPERIMENT_HOOKS:
                errors.append("promotion lacks an existing bounded V622 experiment hook")
            if candidate.get("decision_relevant") is not True:
                errors.append("promotion is not decision relevant")

    expected_duplicates = deduplication_rows(candidates)
    if artifact.get("deduplication_rows") != expected_duplicates:
        errors.append("duplicate suppression ledger mismatch")
    if artifact.get("adoption_rows") != adoption_rows(candidates):
        errors.append("adoption row mismatch")
    if artifact.get("claim_boundary_rows") != claim_boundary_rows(candidates):
        errors.append("claim boundary row mismatch")

    for field in ("extropic_rows", "logical_intelligence_rows"):
        for row in artifact.get(field, []):
            if not isinstance(row, dict):
                continue
            if row.get("claim_boundary_label") != VENDOR_BOUNDARY:
                errors.append("vendor boundary label missing")
            prohibited = (
                row.get("scientific_evidence_promoted") is True
                or row.get("attached_hardware_claimed") is True
                or any(
                    row.get(f"{claim}_claimed") is True
                    for claim in ("runtime", "power", "availability", "speed")
                )
            )
            if prohibited:
                errors.append("vendor-only claim promoted")
    for row in artifact.get("github_rows", []):
        if isinstance(row, dict) and (
            row.get("stars_used_as_evidence") is not False
            or row.get("implementation_promoted") is not False
        ):
            errors.append("GitHub discovery signal promoted")
    for row in artifact.get("huggingface_papers_rows", []):
        if isinstance(row, dict) and (
            row.get("generated_summary_used_as_evidence") is not False
            or row.get("upvotes_used_as_evidence") is not False
        ):
            errors.append("Hugging Face discovery signal promoted")

    append_hash = artifact.get("references_append_hash")
    if not isinstance(append_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", append_hash):
        errors.append("references append hash missing or invalid")
    if artifact.get("references_append_path") != REFERENCE_PATH.as_posix():
        errors.append("references append path mismatch")
    if artifact.get("per_game_results") != []:
        errors.append("literature audit must not claim per-game results")
    return list(dict.fromkeys(errors))


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Return one only when all source, candidate, and boundary checks pass."""

    return int(not _coverage_errors(artifact))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding elapsed time and the hash itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return _sha256_json(payload)


def _base_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any],
    reference_hash: str | None,
) -> JsonDict:
    return {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": _copy(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": [],
        "search_window": deepcopy(dict(window)),
        "query_rows": [],
        "source_class_rows": [],
        "arxiv_rows": [],
        "openreview_rows": [],
        "semantic_scholar_rows": [],
        "huggingface_papers_rows": [],
        "github_rows": [],
        "extropic_rows": [],
        "logical_intelligence_rows": [],
        "candidate_rows": [],
        "deduplication_rows": [],
        "primary_source_receipts": [],
        "adoption_rows": [],
        "claim_boundary_rows": [],
        "references_append_path": REFERENCE_PATH.as_posix(),
        "references_append_hash": reference_hash,
        "rows": [],
        "per_game_results": [],
        "v622_sota_ingestion_complete_score": 0,
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
        "honest_verdict": "blocked_v622_sota_ingestion",
    }


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
    route_reachability: Mapping[str, bool] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float | None = None,
    update_references: bool = False,
) -> JsonDict:
    """Build one schema-complete positive, disqualified, or blocked receipt."""

    started = time.monotonic()
    root = Path(root)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    references = root / REFERENCE_PATH
    preconditions = check_preconditions(root, output_path, route_reachability=route_reachability)
    artifact = _base_artifact(preconditions, search_window(run_date), file_sha256(references))
    artifact["source_artifact_hashes"] = _source_hashes(root)
    failure = next((row for row in preconditions if row["passed"] is False), None)
    if failure is not None:
        artifact["gate_check_summary"] = {
            "failed_check": failure["check"],
            "expected_value": failure["expected_value"],
            "observed_value": failure["observed_value"],
            "passed": False,
        }
    else:
        candidate_data = _copy(candidate_rows() if candidates is None else candidates)
        receipt_data = _copy(primary_source_receipts() if receipts is None else receipts)
        if update_references:
            append_references_delta(references, candidate_data, run_date)
        artifact.update(
            {
                "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
                "source_artifact_hashes": _source_hashes(root),
                "query_rows": query_rows(run_date),
                "source_class_rows": source_class_rows(),
                "arxiv_rows": arxiv_rows(),
                "openreview_rows": openreview_rows(),
                "semantic_scholar_rows": semantic_scholar_rows(),
                "huggingface_papers_rows": huggingface_papers_rows(),
                "github_rows": github_rows(),
                "extropic_rows": extropic_rows(),
                "logical_intelligence_rows": logical_intelligence_rows(),
                "candidate_rows": candidate_data,
                "deduplication_rows": deduplication_rows(candidate_data),
                "primary_source_receipts": receipt_data,
                "adoption_rows": adoption_rows(candidate_data),
                "claim_boundary_rows": claim_boundary_rows(candidate_data),
                "references_append_hash": file_sha256(references),
            }
        )
        artifact["rows"] = combined_rows(artifact)
        score = completion_score(artifact)
        artifact["v622_sota_ingestion_complete_score"] = score
        if score == 1:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = "complete_positive_v622_sota_ingestion_empty_delta"
            artifact["gate_check_summary"] = {
                "failed_check": None,
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            }
        else:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "disqualified_v622_sota_ingestion_contract"
            artifact["gate_check_summary"] = {
                "failed_check": "source_ingestion_contract",
                "expected_value": 1,
                "observed_value": 0,
                "passed": False,
            }
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact["duration_s"] = round(max(0.0, float(elapsed)), 6)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _terminal_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or not preconditions:
        return ["preconditions_checked missing"]
    failure = next(
        (row for row in preconditions if isinstance(row, dict) and row.get("passed") is False),
        None,
    )
    if failure is not None:
        expected_gate = {
            "failed_check": failure.get("check"),
            "expected_value": failure.get("expected_value"),
            "observed_value": failure.get("observed_value"),
            "passed": False,
        }
        if artifact.get("gate_check_summary") != expected_gate:
            errors.append("blocked gate_check_summary mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked inference_substrate_class mismatch")
        if artifact.get("v622_sota_ingestion_complete_score") != 0:
            errors.append("blocked completion score must be zero")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked honest_verdict prefix mismatch")
        return errors

    coverage_errors = _coverage_errors(artifact)
    expected_score = int(not coverage_errors)
    if coverage_errors and (
        artifact.get("v622_sota_ingestion_complete_score") == 1
        or artifact.get("verdict_class") == "positive"
    ):
        errors.extend(coverage_errors)
    if artifact.get("v622_sota_ingestion_complete_score") != expected_score:
        errors.append("v622_sota_ingestion_complete_score mismatch")
    expected_class = "positive" if expected_score == 1 else "disqualified"
    expected_prefix = "complete_positive_" if expected_score == 1 else "disqualified_"
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class inconsistent with completion")
    if not str(artifact.get("honest_verdict", "")).startswith(expected_prefix):
        errors.append("honest_verdict prefix inconsistent with completion")
    expected_gate = {
        "failed_check": None if expected_score == 1 else "source_ingestion_contract",
        "expected_value": 1,
        "observed_value": expected_score,
        "passed": expected_score == 1,
    }
    if artifact.get("gate_check_summary") != expected_gate:
        errors.append("gate_check_summary inconsistent with completion")
    return errors


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Independently validate shape, rows, boundaries, terminal state, and hash."""

    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return ["artifact_missing"]
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
        if not isinstance(loaded, dict):
            return ["artifact_not_object"]
        artifact = loaded
    elif isinstance(value, Mapping):
        artifact = deepcopy(dict(value))
    else:
        return ["artifact_not_object"]

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing or extra:
        return [f"artifact fields mismatch missing={missing} extra={extra}"]

    errors: list[str] = []
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact["inference_substrate_class"] not in {
        INFERENCE_SUBSTRATE_CLASS,
        "blocked_no_run",
    }:
        errors.append("inference_substrate_class mismatch")
    if artifact["execution_venue"] != EXECUTION_VENUE:
        errors.append("execution_venue mismatch")
    if not isinstance(artifact["duration_s"], (int, float)) or artifact["duration_s"] < 0:
        errors.append("duration_s invalid")
    if artifact["random_seed"] != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact["search_window"] != search_window(RUN_DATE):
        errors.append("search_window mismatch")
    if artifact["rows"] != combined_rows(artifact):
        errors.append("rows consistency mismatch")
    errors.extend(_terminal_errors(artifact))
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(artifact: Mapping[str, Any], path: Path) -> Path:
    """Validate before atomically replacing the requested JSON artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    return atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source audit or independently validate an existing artifact."""

    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(json.dumps({"errors": errors} if errors else {"valid": True}, sort_keys=True))
        return int(bool(errors))
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    root = find_repo_root()
    output = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(
        root,
        args.date,
        output_path=output,
        route_reachability=probe_routes(),
        update_references=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"errors": errors}, sort_keys=True))
        return 1
    write_artifact(artifact, output)
    print(json.dumps({"path": str(output), "verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
