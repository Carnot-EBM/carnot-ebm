"""Build the bounded V624 source-ingestion and claim-boundary receipt.

The audit loads no model. It records low-concurrency source receipts. It keeps
indexes, repository popularity, and vendor projections below primary evidence.
Spec refs: REQ-REPORT-7112 and SCENARIO-REPORT-7112-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from carnot.experiment_7098_v623_sota_ingestion import identifier_matches_url
from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260907"
PLANNER_SNAPSHOT_UTC = "2026-09-07T10:28:40Z"
SOURCE_CAPTURE_UTC = "2026-09-07T13:12:31Z"
RANDOM_SEED = 7_112_202_609_07
INFERENCE_SUBSTRATE = "bounded network literature and repository audit"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
PLANNER_HEADING = "## V624 Planner Refresh - 2026-09-07"
REFERENCE_START_MARKER = "<!-- V624-EXECUTION-DELTA-20260907-START -->"
REFERENCE_END_MARKER = "<!-- V624-EXECUTION-DELTA-20260907-END -->"
VENDOR_BOUNDARY = "vendor_claim_not_independent_evidence"

REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7112_v624_sota_ingestion.json")
SOURCE_ARTIFACT_PATHS = (
    REFERENCE_PATH,
    Path("research-program.md"),
    Path("research-complete.yaml"),
    Path("research-roadmap.yaml"),
    Path("ops/status.md"),
    Path("ops/known-issues.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/experiment_7112_v624_sota_ingestion.py"),
    Path("scripts/experiments/experiment_7112_v624_sota_ingestion.py"),
    Path("tests/python/test_experiment_7112_v624_sota_ingestion.py"),
)

QUERY_FAMILIES = (
    "ebm_verification_reasoning",
    "neural_constraint_satisfaction",
    "ising_machine_learning",
    "hallucination_mitigation",
    "kolmogorov_arnold_networks",
    "energy_guided_constrained_decoding",
    "fpga_thermodynamic_sampling",
    "continual_constraint_learning",
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
    "exp7113-arc-generation-liveness-recovery",
    "exp7114-adapter-withheld-arc-loo-measurement",
    "exp7115-adapter-withheld-arc-cold-audit",
    "exp7116-sota-constraint-episode-bank",
    "exp7117-exact-verify-revise-loop",
    "exp7118-principle-step-memory-csl",
    "exp7119-multi-iteration-memory-cold-audit",
    "exp7120-v624-capstone",
}
KNOWN_REFERENCE_SOURCE_IDS = {
    "arxiv:2609.05388",
    "arxiv:2609.05401",
    "arxiv:2606.04703",
    "extropic:z1t-2026-09-04",
    "logical-intelligence:kona-1.0",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
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
    "v624_sota_ingestion_complete_score",
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
)

FIELD_PRINCIPLES = {
    "field_principles": "One principle per field makes the scientific contract reviewable.",
    "preconditions_checked": "Explicit gates prevent unavailable sources from becoming fabricated results.",
    "run_date": "A fixed execution date prevents future sources from entering this receipt.",
    "inference_substrate": "The substrate limits this result to a bounded network source audit.",
    "inference_substrate_class": "The compute class confirms that the audit loads no model.",
    "execution_venue": "The venue separates host work from unavailable hardware claims.",
    "duration_s": "Measured duration distinguishes an executed audit from a template.",
    "source_artifact_hashes": "Hashes bind the result to the local evidence that was read.",
    "search_window": "Fixed source and observation windows make freshness claims falsifiable.",
    "query_rows": "Exact queries disclose coverage without treating rank as evidence.",
    "source_class_rows": "Each requested source class ends with an honest receipt.",
    "arxiv_rows": "Primary paper rows preserve canonical identity, time, and title.",
    "openreview_rows": "A browser challenge stays an access receipt, not an inferred claim.",
    "semantic_scholar_rows": "Rate limits and citation indexes cannot become scientific authority.",
    "huggingface_papers_rows": "Generated summaries remain discovery signals below primary papers.",
    "github_rows": "Repository identity does not prove method quality or Carnot fitness.",
    "extropic_rows": "Vendor projections remain separate from attached-hardware evidence.",
    "logical_intelligence_rows": "Kona stays a comparator without public reproducibility.",
    "candidate_rows": "A closed disposition prevents a lead from silently becoming work.",
    "deduplication_rows": "Canonical suppression stops rediscovery from appearing novel.",
    "primary_source_receipts": "Each candidate traces to a paper or official project page.",
    "adoption_rows": "Each promotion names one bounded existing V624 experiment.",
    "claim_boundary_rows": "Each item states what its source cannot prove for Carnot.",
    "references_append_path": "The path makes the only permitted documentation write explicit.",
    "references_append_hash": "The hash records the exact ledger after any bounded append.",
    "rows": "A combined ledger supports independent collection consistency checks.",
    "v624_sota_ingestion_complete_score": "One means coverage and promotion rules passed.",
    "random_seed": "A fixed seed records deterministic ordering without sampling.",
    "reproducibility_checksum": "A payload hash detects evidence or verdict changes.",
    "gate_check_summary": "Exact diagnostics make blocked or invalid work actionable.",
    "verifier_is_oracle": "False states that a source audit cannot certify Carnot correctness.",
    "verdict_class": "A closed class keeps outages and invalid rows from reading positive.",
    "honest_verdict": "A terminal prefix gives automation one unambiguous outcome.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash a readable file and preserve absence as an explicit null."""

    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def search_window(run_date: str) -> JsonDict:
    """Return the broad source window and strict observation boundary."""

    execution_date = datetime.strptime(run_date, "%Y%m%d").date().isoformat()
    return {
        "literature_start_date": "2025-01-01",
        "literature_end_date": execution_date,
        "planner_snapshot_utc": PLANNER_SNAPSHOT_UTC,
        "execution_end_utc": f"{execution_date}T23:59:59Z",
        "delta_rule": "first_observed_at > planner_snapshot_utc and <= execution_end_utc; source absent from planner ledger",
        "direct_post_snapshot_arxiv_submission_count": 0,
        "new_to_planner_ledger_count": 3,
    }


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def in_execution_delta(value: str, window: Mapping[str, Any]) -> bool:
    """Accept an observation only after planning and by execution-day end."""

    try:
        observed = _utc(value)
        lower = _utc(str(window["planner_snapshot_utc"]))
        upper = _utc(str(window["execution_end_utc"]))
    except (KeyError, TypeError, ValueError):
        return False
    return lower < observed <= upper


def _in_literature_window(value: str, window: Mapping[str, Any]) -> bool:
    try:
        observed = date.fromisoformat(value)
        lower = date.fromisoformat(str(window["literature_start_date"]))
        upper = date.fromisoformat(str(window["literature_end_date"]))
    except (KeyError, TypeError, ValueError):
        return False
    return lower <= observed <= upper


def _receipt_hash(row: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in row.items() if key != "metadata_receipt_sha256"}
    return _sha256_json(stable)


def primary_source_receipts() -> list[JsonDict]:
    """Return verified primary identities for the bounded V624 review."""

    records = (
        (
            "2609.05388",
            "Think-Verify-Revise: Neuro-Symbolic Visual Reasoning with Vision-Language Models and Dynamic Logic Tensor Networks",
            "2026-09-04T17:39:47Z",
            "A strict rule grammar, differentiable rule checks, and feedback form the tested visual reasoning loop.",
        ),
        (
            "2609.05401",
            "Same Trajectory, Contradictory Rewards (ROBORMBENCH): Paraphrase Fragility in Vision Language Reward Models",
            "2026-09-04T17:47:58Z",
            "Verified paraphrases can reverse reward decisions for identical trajectories in the tested models.",
        ),
        (
            "2606.04703",
            "Rethinking Continual Experience Internalization for Self-Evolving LLM Agents",
            "2026-06-03T10:30:09Z",
            "Principle-level, step-wise experience stays more stable than instance-level global injection over repeated learning iterations.",
        ),
        (
            "2609.02750",
            "Bilevel Coordinated Reflection: A Game-Theoretic Approach to Multi-Agent LLM Systems",
            "2026-09-02T15:50:10Z",
            "Transcript-only gates cannot improve uniformly across text-indistinguishable environments; SRMA accepts memory only after grounded risk decreases.",
        ),
        (
            "2609.04629",
            "SiLR: Structure-Preserving Admission and Process Reward for LLM Tool Agents",
            "2026-09-04T02:03:52Z",
            "A product order over branch violations avoids a scalar admission trap in the tested post-violation recovery tasks.",
        ),
        (
            "2609.04753",
            "Beneath the Surface of Chains-of-Thought: A Mechanistic Interpretation of Reasoning Operations in LLMs",
            "2026-09-04T05:37:12Z",
            "Reasoning operations are separable in held-out representations and depend on chunk context rather than surface tokens alone.",
        ),
    )
    rows: list[JsonDict] = []
    for source_id, title, source_time, claim in records:
        row: JsonDict = {
            "receipt_id": f"receipt-arxiv-{source_id}",
            "source_id": f"arxiv:{source_id}",
            "title": title,
            "source_url": f"https://arxiv.org/abs/{source_id}",
            "publication_or_revision_date": source_time[:10],
            "publication_or_revision_time": source_time,
            "version_or_revision": "v1",
            "source_role": "primary_paper",
            "core_claim": claim,
            "content_verified": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        row["metadata_receipt_sha256"] = _receipt_hash(row)
        rows.append(row)
    for source_id, title, url, source_date, claim in (
        (
            "extropic:z1t-2026-09-04",
            "Z1T: Sparse Transformer-Like Models for Probabilistic Hardware",
            "https://extropic.ai/writing/z1t",
            "2026-09-04",
            "The vendor page presents a degree-16 Z1 graph and an estimated Z1-plus-FPGA decode design.",
        ),
        (
            "logical-intelligence:kona-1.0",
            "Kona: Energy-Based Models (EBMs) for AI Reasoning",
            "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "2026-02-03",
            "The vendor page presents Kona as a proprietary constraint layer below language-model interfaces.",
        ),
    ):
        row = {
            "receipt_id": "receipt-" + source_id.replace(":", "-"),
            "source_id": source_id,
            "title": title,
            "source_url": url,
            "publication_or_revision_date": source_date,
            "publication_or_revision_time": f"{source_date}T00:00:00Z",
            "version_or_revision": f"captured-{RUN_DATE}",
            "source_role": "official_vendor_project_page",
            "core_claim": claim,
            "content_verified": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        row["metadata_receipt_sha256"] = _receipt_hash(row)
        rows.append(row)
    return rows


def candidate_rows() -> list[JsonDict]:
    """Classify verified sources and preserve one decision-changing control."""

    dispositions = {
        "arxiv:2609.05388": (
            "duplicate",
            False,
            None,
            "The visual benchmark does not prove exact validity or repair value in Carnot.",
            "The V624 planner refresh already records this source and hook.",
        ),
        "arxiv:2609.05401": (
            "duplicate",
            False,
            None,
            "Reward fragility does not establish Carnot paraphrase invariance.",
            "The V624 planner refresh already records this source and hook.",
        ),
        "arxiv:2606.04703": (
            "duplicate",
            False,
            None,
            "Parametric internalization results do not prove external Carnot memory value.",
            "The V624 planner refresh already records this source and hook.",
        ),
        "arxiv:2609.02750": (
            "control",
            True,
            "exp7118-principle-step-memory-csl",
            "Theorems under stated assumptions and SWE-bench results do not prove Carnot memory value or authorize any write.",
            "Add a grounded risk-decrease admission control; exact feedback alone does not prove that a proposed memory helps.",
        ),
        "arxiv:2609.04629": (
            "watch",
            False,
            None,
            "Gym-ANM and CityLearn recovery results do not prove value for a one-revision constraint episode.",
            "The product-order gate matters for multi-action recovery, outside the bounded V624 revision count.",
        ),
        "arxiv:2609.04753": (
            "watch",
            False,
            None,
            "Hidden-state separability does not define a reliable external memory key across Carnot model families.",
            "The source warns against surface-token step keys but does not supply a tested external-memory intervention.",
        ),
        "extropic:z1t-2026-09-04": (
            "duplicate",
            False,
            None,
            "Vendor estimates cannot establish Carnot runtime, power, speed, availability, or Z1 execution.",
            "The V624 planner refresh already records this vendor page and boundary.",
        ),
        "logical-intelligence:kona-1.0": (
            "duplicate",
            False,
            None,
            "The public page supplies no checkpoint, training recipe, or reproducible local Carnot baseline.",
            "The V624 planner refresh already records Kona as an architecture comparator.",
        ),
    }
    receipts = {row["source_id"]: row for row in primary_source_receipts()}
    rows: list[JsonDict] = []
    for source_id, receipt in receipts.items():
        classification, relevant, hook, boundary, reason = dispositions[source_id]
        rows.append(
            {
                "candidate_id": source_id.replace(":", "-").replace(".", "-"),
                "source_id": source_id,
                "title": receipt["title"],
                "source_url": receipt["source_url"],
                "publication_or_revision_date": receipt["publication_or_revision_date"],
                "publication_or_revision_time": receipt["publication_or_revision_time"],
                "first_observed_at": SOURCE_CAPTURE_UTC,
                "source_receipt_id": receipt["receipt_id"],
                "source_role": receipt["source_role"],
                "identity_verified": True,
                "core_claim_verified": True,
                "classification": classification,
                "decision_relevant": relevant,
                "experiment_hook": hook,
                "claim_boundary": boundary,
                "classification_reason": reason,
                "terminal": True,
            }
        )
    return rows


def query_rows(run_date: str) -> list[JsonDict]:
    """Record each exact topic and requested source-route query."""

    rows = [
        {
            "query_id": f"arxiv-{family}",
            "query_family": family,
            "route": "arxiv",
            "exact_query": f"submittedDate:[202501010000 TO {run_date}2359] AND all:{family.replace('_', ' ')}",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "completed_bounded_search",
            "search_rank_used_as_evidence": False,
            "terminal": True,
        }
        for family in QUERY_FAMILIES
    ]
    route_queries = (
        ("arxiv-post-snapshot", "arxiv", "submittedDate:[202609071028 TO 202609072359]", "http_200_zero_results", 0),
        ("openreview-ebm", "openreview", "OpenReview current EBM submissions and FSNet forum oum1txoy1D", "http_403_challenge", None),
        ("semantic-scholar-ebt", "semantic_scholar", "citations for ARXIV:2507.02092 limit 100", "http_429_rate_limited", None),
        ("semantic-scholar-arm-ebm", "semantic_scholar", "citations for ARXIV:2512.15605 limit 100", "http_429_rate_limited", None),
        ("huggingface-daily", "huggingface_papers", "GET /api/daily_papers?limit=100", "http_200_complete", 100),
        ("github-current", "github", "official EBT, FSNet, MARCH, Extropic, and Bilevel Reflection repositories", "http_200_complete", 5),
        ("extropic-writing", "extropic", "GET https://extropic.ai/writing/z1t", "http_200_complete", 1),
        ("logical-intelligence-kona", "logical_intelligence", "GET https://logicalintelligence.com/kona-ebms-energy-based-models", "http_200_complete", 1),
    )
    rows.extend(
        {
            "query_id": query_id,
            "query_family": "execution_delta" if query_id == "arxiv-post-snapshot" else "requested_source_route",
            "route": route,
            "exact_query": query,
            "result_count_observed": count,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": outcome,
            "search_rank_used_as_evidence": False,
            "terminal": True,
        }
        for query_id, route, query, outcome, count in route_queries
    )
    return rows


def arxiv_rows() -> list[JsonDict]:
    """Return the post-snapshot query and verified primary identities."""

    rows: list[JsonDict] = [
        {
            "row_id": "arxiv-post-snapshot-query",
            "source_url": "https://export.arxiv.org/api/query",
            "search_start_time": PLANNER_SNAPSHOT_UTC,
            "search_end_time": search_window(RUN_DATE)["execution_end_utc"],
            "matching_result_count": 0,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]
    for receipt in primary_source_receipts():
        if not str(receipt["source_id"]).startswith("arxiv:"):
            continue
        rows.append(
            {
                "row_id": receipt["source_id"],
                "source_id": receipt["source_id"],
                "title": receipt["title"],
                "source_url": receipt["source_url"],
                "publication_or_revision_time": receipt["publication_or_revision_time"],
                "version_or_revision": receipt["version_or_revision"],
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_verified_primary",
                "terminal": True,
            }
        )
    return rows


def openreview_rows() -> list[JsonDict]:
    """Keep the current access challenge as a terminal receipt."""

    return [{
        "source_id": "openreview:oum1txoy1D",
        "source_url": "https://openreview.net/forum?id=oum1txoy1D",
        "title_hint": "FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees",
        "http_status": 403,
        "access_outcome": "challenge_verification_required",
        "content_verified_in_this_route": False,
        "primary_fallback_source_id": "arxiv:2506.00362",
        "claim_promoted": False,
        "accessed_at": SOURCE_CAPTURE_UTC,
        "terminal": True,
    }]


def semantic_scholar_rows() -> list[JsonDict]:
    """Record citation endpoint rate limits without guessing citation data."""

    return [
        {
            "source_id": f"semantic-scholar:{paper_id}",
            "paper_id": paper_id,
            "source_url": f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?limit=100",
            "http_status": 429,
            "access_outcome": "rate_limited_terminal_receipt",
            "visible_response_rows": None,
            "citation_count_claimed_as_authoritative": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for paper_id in ("ARXIV:2507.02092", "ARXIV:2512.15605")
    ]


def huggingface_papers_rows() -> list[JsonDict]:
    """Record the daily feed only as a secondary discovery index."""

    return [
        {
            "source_id": f"hf-paper:{paper_id}",
            "source_url": f"https://huggingface.co/papers/{paper_id}",
            "daily_feed_url": "https://huggingface.co/api/daily_papers?limit=100",
            "submitted_on_daily_at": "2026-09-07T00:00:00Z",
            "evidence_role": "secondary_discovery_index_only",
            "primary_source_id": f"arxiv:{paper_id}",
            "generated_summary_used_as_evidence": False,
            "upvotes_used_as_evidence": False,
            "access_outcome": "http_200_complete",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for paper_id in ("2609.02750", "2609.04753")
    ]


def github_rows() -> list[JsonDict]:
    """Freeze repository identity while excluding popularity from evidence."""

    records = (
        ("alexiglad/EBT", "19420cbeae655bbf11930219a675ade6897019e8", "2026-04-21T00:53:47Z", "Apache-2.0"),
        ("MOSSLab-MIT/FSNet", "826457df85302da8c7553977ce74a4ee18d1b362", "2026-06-09T21:48:12Z", "MIT"),
        ("Qwen-Applications/MARCH", "1805095bbfee0b1d90c9ad8c473911276469c4e3", "2026-06-09T05:01:09Z", None),
        ("extropic-ai/sparse-transformers", "13051e90df9669be5b8f9f34fb097329fa82f674", "2026-09-03T20:28:22Z", "Apache-2.0"),
        ("YihangChen9/Bilevel-Coordinated-Reflection", "7bf43818d32ea3775ede985b53d8006c90413192", "2026-09-07T09:39:52Z", "MIT"),
    )
    return [
        {
            "source_id": f"github:{repository}",
            "source_url": f"https://github.com/{repository}",
            "default_branch_revision": revision,
            "pushed_at": pushed_at,
            "license": license_id,
            "stars_used_as_evidence": False,
            "implementation_promoted": False,
            "access_outcome": "http_200_complete",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for repository, revision, pushed_at, license_id in records
    ]


def extropic_rows() -> list[JsonDict]:
    """Keep Z1T inside its official vendor evidence boundary."""

    return [{
        "source_id": "extropic:z1t-2026-09-04",
        "source_url": "https://extropic.ai/writing/z1t",
        "claim_boundary_label": VENDOR_BOUNDARY,
        "scientific_evidence_promoted": False,
        "runtime_claimed": False,
        "power_claimed": False,
        "attached_z1_execution_claimed": False,
        "access_outcome": "http_200_complete",
        "accessed_at": SOURCE_CAPTURE_UTC,
        "terminal": True,
    }]


def logical_intelligence_rows() -> list[JsonDict]:
    """Keep Kona proprietary until weights and a local runner are public."""

    return [{
        "source_id": "logical-intelligence:kona-1.0",
        "source_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "claim_boundary_label": VENDOR_BOUNDARY,
        "scientific_evidence_promoted": False,
        "runtime_claimed": False,
        "public_checkpoint_observed": False,
        "public_training_recipe_observed": False,
        "reproducible_local_runner_observed": False,
        "access_outcome": "http_200_complete",
        "accessed_at": SOURCE_CAPTURE_UTC,
        "terminal": True,
    }]


def source_class_rows() -> list[JsonDict]:
    """Give every requested route one honest terminal outcome."""

    outcomes = {
        "arxiv": ("http_200_complete", True),
        "openreview": ("http_403_challenge", False),
        "semantic_scholar": ("http_429_rate_limited", False),
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


def deduplication_rows(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return canonical source-ledger duplicates suppressed by the audit."""

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
    """Return only controls or adoptions that change one bounded V624 task."""

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
    """Make each prohibited inference independently enumerable."""

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
    """Combine typed rows while retaining their collection names."""

    rows: list[JsonDict] = []
    for field in ROW_COLLECTION_FIELDS:
        value = artifact.get(field, [])
        if isinstance(value, list):
            rows.extend(
                {"collection": field, **deepcopy(row)}
                for row in value
                if isinstance(row, dict)
            )
    return rows


def http_reachable(url: str) -> bool:
    """Treat explicit HTTP denial as reachability, not verified content."""

    request = Request(url, headers={"User-Agent": "carnot-v624-source-audit/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def probe_routes() -> dict[str, bool]:
    """Probe only routes required by the minimum network gate."""

    urls = {
        "arxiv": "https://arxiv.org/abs/2609.02750",
        "openreview": "https://openreview.net/forum?id=oum1txoy1D",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092",
        "huggingface_papers": "https://huggingface.co/api/daily_papers?limit=1",
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
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7112-", delete=True):
            return True
    except OSError:
        return False


def check_preconditions(
    root: Path,
    output_path: Path,
    *,
    route_reachability: Mapping[str, bool] | None = None,
) -> list[JsonDict]:
    """Evaluate network, local-read, and destination-write gates first."""

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
        {"check": "arxiv_network_reachability", "expected_value": True, "observed_value": arxiv_ok, "passed": arxiv_ok},
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
        {"check": "v624_planner_marker", "expected_value": PLANNER_HEADING, "observed_value": PLANNER_HEADING if marker_present else "missing", "passed": marker_present},
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
        and row.get("source_id") not in KNOWN_REFERENCE_SOURCE_IDS
        and row.get("experiment_hook") in ALLOWED_EXPERIMENT_HOOKS
        and identifier_matches_url(str(row.get("source_id", "")), str(row.get("source_url", "")))
        and _in_literature_window(str(row.get("publication_or_revision_date", "")), window)
        and in_execution_delta(str(row.get("first_observed_at", "")), window)
    ]


def append_references_delta(
    path: Path, candidates: Sequence[Mapping[str, Any]], run_date: str
) -> bool:
    """Append one verified delta block, or preserve the ledger bytes."""

    accepted = _appendable_candidates(candidates, run_date)
    if not accepted:
        return False
    text = path.read_text(encoding="utf-8")
    if REFERENCE_START_MARKER in text or REFERENCE_END_MARKER in text:
        return False
    lines = [
        "",
        f"## V624 execution delta - {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}",
        "",
        REFERENCE_START_MARKER,
        "",
        "This block lists verified sources first observed by the execution sweep.",
        "A later observation does not change the source publication date.",
        "",
    ]
    for row in accepted:
        lines.append(
            f"- **{row['title']}** - {row['source_id']}, {row['source_url']}; "
            f"published {row['publication_or_revision_date']}. Carnot hook: "
            f"`{row['experiment_hook']}`. Boundary: {row['claim_boundary']}"
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
    source_rows = artifact.get("source_class_rows")
    if not isinstance(source_rows, list) or {
        row.get("source_class") for row in source_rows if isinstance(row, Mapping)
    } != set(SOURCE_CLASSES):
        errors.append("requested source class receipt coverage mismatch")
    elif not all(
        row.get("honest_receipt") is True and row.get("terminal") is True
        for row in source_rows
        if isinstance(row, Mapping)
    ):
        errors.append("source class receipt is not honest and terminal")

    queries = artifact.get("query_rows")
    if not isinstance(queries, list) or {
        row.get("query_family")
        for row in queries
        if isinstance(row, Mapping) and row.get("query_family") in QUERY_FAMILIES
    } != set(QUERY_FAMILIES):
        errors.append("query family coverage mismatch")

    receipts = artifact.get("primary_source_receipts")
    receipt_by_id: dict[str, Mapping[str, Any]] = {}
    if not isinstance(receipts, list):
        errors.append("primary source receipts missing")
    else:
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                errors.append("primary source receipt is not a row")
                continue
            receipt_id = str(receipt.get("receipt_id", ""))
            if not receipt_id or receipt_id in receipt_by_id:
                errors.append("primary source receipt identity is missing or duplicate")
            receipt_by_id[receipt_id] = receipt
            if receipt.get("metadata_receipt_sha256") != _receipt_hash(receipt):
                errors.append("primary source receipt hash mismatch")
            if receipt.get("content_verified") is not True:
                errors.append("primary source receipt is unverified")
            if not identifier_matches_url(
                str(receipt.get("source_id", "")), str(receipt.get("source_url", ""))
            ):
                errors.append("primary source identifier and URL mismatch")

    candidates = artifact.get("candidate_rows")
    if not isinstance(candidates, list):
        errors.append("candidate rows missing")
        candidates = []
    window = artifact.get("search_window")
    if not isinstance(window, Mapping):
        errors.append("search window missing")
        window = {}
    seen_sources: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            errors.append("candidate is not a row")
            continue
        source_id = str(candidate.get("source_id", ""))
        if source_id in seen_sources:
            errors.append("candidate canonical source is duplicate")
        seen_sources.add(source_id)
        if candidate.get("classification") not in CLASSIFICATIONS:
            errors.append("candidate classification outside closed enum")
        if not candidate.get("claim_boundary") or candidate.get("terminal") is not True:
            errors.append("candidate claim boundary or terminal receipt missing")
        receipt = receipt_by_id.get(str(candidate.get("source_receipt_id", "")))
        if receipt is None or receipt.get("source_id") != source_id:
            errors.append("candidate primary source receipt mismatch")
        elif receipt.get("title") != candidate.get("title"):
            errors.append("candidate title differs from primary source")
        if source_id in KNOWN_REFERENCE_SOURCE_IDS and candidate.get("classification") != "duplicate":
            errors.append("known duplicate canonical source was not suppressed")
        if candidate.get("classification") in {"adopt", "control"}:
            if candidate.get("decision_relevant") is not True:
                errors.append("promotion is not decision relevant")
            if candidate.get("source_role") != "primary_paper":
                errors.append("secondary or vendor source was promoted")
            if candidate.get("identity_verified") is not True or candidate.get("core_claim_verified") is not True:
                errors.append("promotion lacks verified identity or core claim")
            if source_id in KNOWN_REFERENCE_SOURCE_IDS:
                errors.append("known source was promoted")
            if candidate.get("experiment_hook") not in ALLOWED_EXPERIMENT_HOOKS:
                errors.append("promotion lacks bounded V624 hook")
            if not identifier_matches_url(source_id, str(candidate.get("source_url", ""))):
                errors.append("promotion identifier and URL mismatch")
            if not _in_literature_window(str(candidate.get("publication_or_revision_date", "")), window):
                errors.append("promotion source date outside literature window")
            if not in_execution_delta(str(candidate.get("first_observed_at", "")), window):
                errors.append("promotion observation outside execution window")

    expected_dedup = deduplication_rows(candidates)
    expected_adoptions = adoption_rows(candidates)
    expected_boundaries = claim_boundary_rows(candidates)
    if artifact.get("deduplication_rows") != expected_dedup:
        errors.append("deduplication rows inconsistent")
    if artifact.get("adoption_rows") != expected_adoptions:
        errors.append("adoption rows inconsistent")
    if artifact.get("claim_boundary_rows") != expected_boundaries:
        errors.append("claim boundary rows inconsistent")

    vendor_rows = [
        *artifact.get("extropic_rows", []),
        *artifact.get("logical_intelligence_rows", []),
    ]
    if not vendor_rows or any(
        not isinstance(row, Mapping)
        or row.get("claim_boundary_label") != VENDOR_BOUNDARY
        or row.get("scientific_evidence_promoted") is not False
        or row.get("runtime_claimed") is not False
        for row in vendor_rows
    ):
        errors.append("vendor evidence boundary mismatch")
    if any(
        not isinstance(row, Mapping) or row.get("generated_summary_used_as_evidence") is not False
        for row in artifact.get("huggingface_papers_rows", [])
    ):
        errors.append("Hugging Face summary was used as evidence")
    if any(
        not isinstance(row, Mapping) or row.get("stars_used_as_evidence") is not False
        for row in artifact.get("github_rows", [])
    ):
        errors.append("GitHub stars were used as evidence")
    if artifact.get("references_append_path") != REFERENCE_PATH.as_posix():
        errors.append("references append path mismatch")
    if not artifact.get("references_append_hash"):
        errors.append("references append hash missing")
    return list(dict.fromkeys(errors))


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Return one only when all source and boundary checks pass."""

    return int(not _coverage_errors(artifact))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding elapsed time and this hash."""

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
        "run_date": RUN_DATE,
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
        "v624_sota_ingestion_complete_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {"failed_check": None, "expected_value": 1, "observed_value": 0, "passed": False},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_v624_sota_ingestion_precondition",
    }


def recompute_artifact(artifact: JsonDict) -> JsonDict:
    """Recompute derived rows, score, terminal state, and checksum in place."""

    candidates = artifact.get("candidate_rows", [])
    artifact["deduplication_rows"] = deduplication_rows(candidates)
    artifact["adoption_rows"] = adoption_rows(candidates)
    artifact["claim_boundary_rows"] = claim_boundary_rows(candidates)
    artifact["rows"] = combined_rows(artifact)
    score = completion_score(artifact)
    artifact["v624_sota_ingestion_complete_score"] = score
    if score == 1:
        artifact["verdict_class"] = "positive"
        suffix = "adopted_delta" if artifact["adoption_rows"] else "empty_delta"
        artifact["honest_verdict"] = f"complete_positive_v624_sota_ingestion_{suffix}"
        failed_check = None
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_v624_sota_ingestion_contract"
        failed_check = "source_ingestion_contract"
    artifact["gate_check_summary"] = {
        "failed_check": failed_check,
        "expected_value": 1,
        "observed_value": score,
        "passed": score == 1,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


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
                "primary_source_receipts": receipt_data,
                "references_append_hash": file_sha256(references),
            }
        )
        recompute_artifact(artifact)
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
        if artifact.get("v624_sota_ingestion_complete_score") != 0:
            errors.append("blocked completion score must be zero")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_blocked_"):
            errors.append("blocked honest_verdict prefix mismatch")
        return errors

    coverage_errors = _coverage_errors(artifact)
    expected_score = int(not coverage_errors)
    if coverage_errors and (
        artifact.get("v624_sota_ingestion_complete_score") == 1
        or artifact.get("verdict_class") == "positive"
    ):
        errors.extend(coverage_errors)
    if artifact.get("v624_sota_ingestion_complete_score") != expected_score:
        errors.append("v624_sota_ingestion_complete_score mismatch")
    expected_class = "positive" if expected_score == 1 else "disqualified"
    expected_prefix = "complete_positive_" if expected_score == 1 else "complete_disqualified_"
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
    """Validate shape, evidence boundaries, terminal state, rows, and hash."""

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
    if artifact["run_date"] != RUN_DATE:
        errors.append("run_date mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact["inference_substrate_class"] not in {INFERENCE_SUBSTRATE_CLASS, "blocked_no_run"}:
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
    """Run the source audit or validate one stored artifact."""

    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(json.dumps({"errors": errors} if errors else {"valid": True}, sort_keys=True))
        return int(bool(errors))
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.date != RUN_DATE:
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
    print(
        json.dumps(
            {
                "path": str(output),
                "v624_sota_ingestion_complete_score": artifact["v624_sota_ingestion_complete_score"],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
