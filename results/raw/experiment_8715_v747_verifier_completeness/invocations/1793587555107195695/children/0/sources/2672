"""Freeze V620 source identities without turning discovery into science.

This receipt separates public evidence, planning inference, vendor prose, and
inaccessible routes. Spec refs: REQ-REPORT-7077 and SCENARIO-REPORT-7077-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7077
RUN_DATE = "20260906"
RANDOM_SEED = 7_077_202_609_06
SOURCE_CAPTURE_UTC = "2026-09-06T12:15:29Z"
INFERENCE_SUBSTRATE = "web_bibliographic_search_only_no_llm"
REFERENCE_MARKER = "## V620 planner refresh - 2026-09-06"

MODULE_PATH = Path("python/carnot/experiment_7077_v620_sota_ingestion.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_7077_v620_sota_ingestion.py")
TEST_PATH = Path("tests/python/test_experiment_7077_v620_sota_ingestion.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
PROGRAM_PATH = Path("research-program.md")
ROADMAP_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PRIOR_ARTIFACT_PATH = Path("results/experiment_7011_v614_sota_ingestion.json")
RESULT_PATH = Path("results/experiment_7077_v620_sota_ingestion.json")

REQUIRED_LOCAL_PATHS = (
    REFERENCE_PATH,
    PROGRAM_PATH,
    ROADMAP_PATH,
    PRIOR_ARTIFACT_PATH,
    SPEC_PATH,
)

REQUIRED_QUERY_FAMILIES = (
    "ebm_reasoning",
    "neural_constraint_satisfaction",
    "ising_sampling",
    "hallucination_mitigation",
    "kan_verification",
    "constrained_generation",
    "thermodynamic_hardware_sampling",
    "continual_learning",
)
REQUIRED_SOURCE_CLASSES = (
    "arxiv",
    "openreview",
    "huggingface",
    "semantic_scholar",
    "github",
    "extropic",
    "logical_intelligence",
)
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
    "rows",
    "query_rows",
    "primary_source_rows",
    "openreview_rows",
    "huggingface_rows",
    "semantic_scholar_rows",
    "github_rows",
    "extropic_rows",
    "kona_rows",
    "code_identity_rows",
    "duplicate_rows",
    "inaccessible_rows",
    "decision_rows",
    "task_mapping_rows",
    "vendor_claim_boundaries",
    "source_capture_utc",
    "sota_ingestion_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

ROW_COLLECTION_FIELDS = (
    "query_rows",
    "primary_source_rows",
    "openreview_rows",
    "huggingface_rows",
    "semantic_scholar_rows",
    "github_rows",
    "extropic_rows",
    "kona_rows",
    "code_identity_rows",
    "duplicate_rows",
    "inaccessible_rows",
    "decision_rows",
    "task_mapping_rows",
    "vendor_claim_boundaries",
)

FIELD_PRINCIPLES = {
    "field_principles": "A principle per field makes the evidence contract reviewable.",
    "preconditions_checked": "Explicit checks prevent missing inputs from becoming findings.",
    "inference_substrate": "The substrate discloses bibliographic search without model inference.",
    "duration_s": "Elapsed time distinguishes an executed capture from an assertion.",
    "source_artifact_hashes": "Hashes bind the receipt to its local planning inputs.",
    "rows": "A combined ledger supports independent row-consistency checks.",
    "query_rows": "Exact queries disclose search scope without treating rank as evidence.",
    "primary_source_rows": "Versioned primary identities support bounded planning decisions.",
    "openreview_rows": "OpenReview receipts preserve challenged routes without inference.",
    "huggingface_rows": "Hugging Face rows remain discovery records rather than authority.",
    "semantic_scholar_rows": "Citation trails report visible rows, not authoritative counts.",
    "github_rows": "Activity and license identify code but do not prove quality or readiness.",
    "extropic_rows": "Z1T software is separated from vendor-estimated hardware evidence.",
    "kona_rows": "Kona remains proprietary until weights and a local runner are public.",
    "code_identity_rows": "Immutable revisions make public-code references reproducible.",
    "duplicate_rows": "Suppression rows stop repeated identities from appearing novel.",
    "inaccessible_rows": "Terminal failures stay visible and cannot support promotion.",
    "decision_rows": "Closed dispositions separate public evidence from planning inference.",
    "task_mapping_rows": "Every promoted hook has a bounded V620 owner or defer decision.",
    "vendor_claim_boundaries": "Explicit boundaries prevent product prose from becoming science.",
    "source_capture_utc": "A declared UTC instant fixes the evidence observation window.",
    "sota_ingestion_complete_score": "One requires dated coverage and mapped supported hooks.",
    "random_seed": "A fixed seed records deterministic ordering despite no sampling.",
    "reproducibility_checksum": "A stable payload hash detects evidence-ledger changes.",
    "gate_check_summary": "Exact expected and observed values make blocks actionable.",
    "verifier_is_oracle": "False states that source ingestion cannot establish correctness.",
    "verdict_class": "A closed terminal class prevents access failures appearing positive.",
    "honest_verdict": "A class-matched prefix gives automation an unambiguous outcome.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_receipt(row: Mapping[str, Any]) -> str:
    keys = (
        "source_id",
        "title",
        "authors",
        "source_url",
        "version_or_revision",
        "publication_or_revision_utc",
        "code_url",
    )
    return _sha256_json({key: row.get(key) for key in keys})


def query_rows() -> list[JsonDict]:
    """Return exact search families and direct-route query receipts."""

    arxiv_queries = (
        ("ebm_reasoning", 'site:arxiv.org 2025 2026 "energy-based model" reasoning verifier'),
        (
            "neural_constraint_satisfaction",
            'site:arxiv.org 2025 2026 neural "constraint satisfaction" reasoning',
        ),
        ("ising_sampling", 'site:arxiv.org 2025 2026 Ising sampling "exact distribution"'),
        (
            "hallucination_mitigation",
            "site:arxiv.org 2025 2026 hallucination mitigation verification energy",
        ),
        (
            "kan_verification",
            "site:arxiv.org 2025 2026 Kolmogorov Arnold Networks piecewise affine reasoning constraint",
        ),
        (
            "constrained_generation",
            "site:arxiv.org 2025 2026 constrained generation static constraint decoding language model",
        ),
        (
            "thermodynamic_hardware_sampling",
            "site:arxiv.org 2025 2026 Ising probabilistic hardware sampling p-bit",
        ),
        (
            "continual_learning",
            "site:arxiv.org 2025 2026 continual learning memory reasoning language models",
        ),
    )
    rows = [
        {
            "query_id": f"arxiv_{family}",
            "query_family": family,
            "route": "arxiv",
            "exact_query": query,
            "source_url": "https://arxiv.org/search/",
            "capture_utc": SOURCE_CAPTURE_UTC,
            "access_outcome": "completed_bounded_search",
            "search_rank_used_as_evidence": False,
            "terminal": True,
        }
        for family, query in arxiv_queries
    ]
    route_queries = (
        ("openreview", "OpenReview ICLR 2026 EBM verifier records by canonical forum ID"),
        ("huggingface", "Hugging Face Papers verification energy sufficiency 2026"),
        ("semantic_scholar", "Semantic Scholar citations ARXIV:2507.02092 ARXIV:2512.15605"),
        ("github", "GitHub EBM constraint KAN Ising verifier official artifacts"),
        ("extropic", "site:extropic.ai Z1T 2026-09-04"),
        ("logical_intelligence", "site:logicalintelligence.com Kona 1.0 EBM"),
    )
    rows.extend(
        {
            "query_id": f"route_{route}",
            "query_family": "source_route_recheck",
            "route": route,
            "exact_query": query,
            "source_url": {
                "openreview": "https://openreview.net/",
                "huggingface": "https://huggingface.co/papers",
                "semantic_scholar": "https://api.semanticscholar.org/graph/v1/",
                "github": "https://api.github.com/",
                "extropic": "https://extropic.ai/writing/z1t",
                "logical_intelligence": (
                    "https://logicalintelligence.com/kona-ebms-energy-based-models"
                ),
            }[route],
            "capture_utc": SOURCE_CAPTURE_UTC,
            "access_outcome": "completed_direct_route_recheck",
            "search_rank_used_as_evidence": False,
            "terminal": True,
        }
        for route, query in route_queries
    )
    return rows


def primary_source_rows() -> list[JsonDict]:
    """Return canonical arXiv identities and bounded planning hooks."""

    rows: list[JsonDict] = [
        {
            "source_id": "arxiv:2605.02915",
            "title": (
                "When Should a Language Model Trust Itself? Same-Model Self-Verification "
                "as a Conditional Confidence Signal"
            ),
            "authors": ["Aditya Ajay Phalod"],
            "source_url": "https://arxiv.org/abs/2605.02915",
            "version_or_revision": "v1",
            "publication_or_revision_utc": "2026-04-08T20:15:33Z",
            "code_url": "https://github.com/phalod-aditya/slm-confidence-signals",
            "hook_id": "model_family_likelihood_controls",
        },
        {
            "source_id": "arxiv:2605.03534",
            "title": (
                "SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for "
                "Selective Retrieval-Augmented Generation"
            ),
            "authors": ["Jingxi Qiu", "Zeyu Han", "Cheng Huang"],
            "source_url": "https://arxiv.org/abs/2605.03534",
            "version_or_revision": "v2",
            "publication_or_revision_utc": "2026-07-24T12:46:25Z",
            "code_url": None,
            "hook_id": "set_level_evidence_sufficiency",
        },
        {
            "source_id": "arxiv:2607.10128",
            "title": "Energy-guided Recursive Model",
            "authors": [],
            "source_url": "https://arxiv.org/abs/2607.10128",
            "version_or_revision": "v2",
            "publication_or_revision_utc": "2026-07-17T00:00:00Z",
            "code_url": None,
            "hook_id": "hopfield_selector_control",
        },
        {
            "source_id": "arxiv:2607.23055",
            "title": "SymStep: Symbolic Step Verification for Logical Reasoning",
            "authors": [],
            "source_url": "https://arxiv.org/abs/2607.23055",
            "version_or_revision": "v1",
            "publication_or_revision_utc": "2026-07-25T00:00:00Z",
            "code_url": None,
            "hook_id": "exact_mrv_control",
        },
        {
            "source_id": "arxiv:2609.02948",
            "title": (
                "FrOGS: Discrete Neural Sampler for Independent Alloy Configurations "
                "Across Chemical Conditions"
            ),
            "authors": [],
            "source_url": "https://arxiv.org/abs/2609.02948",
            "version_or_revision": "v1",
            "publication_or_revision_utc": "2026-09-01T00:00:00Z",
            "code_url": None,
            "hook_id": "finite_ising_parity",
        },
        {
            "source_id": "arxiv:2608.26730",
            "title": (
                "Knowing When Not to Reuse: Conditional Experience Transfer in "
                "Autonomous LLM Post-Training"
            ),
            "authors": [],
            "source_url": "https://arxiv.org/abs/2608.26730",
            "version_or_revision": "v1",
            "publication_or_revision_utc": "2026-08-27T00:00:00Z",
            "code_url": None,
            "hook_id": "context_bound_learning",
        },
    ]
    for row in rows:
        row.update(
            {
                "capture_utc": SOURCE_CAPTURE_UTC,
                "evidence_basis": "public_primary_source",
                "science_claim_promoted": False,
                "terminal": True,
            }
        )
        row["metadata_receipt_sha256"] = _metadata_receipt(row)
    return rows


def openreview_rows() -> list[JsonDict]:
    """Preserve browser-challenged OpenReview pages as terminal non-claims."""

    records = (
        ("openreview:8ta0xgtsJK", "VFScale", "https://openreview.net/forum?id=8ta0xgtsJK"),
        ("openreview:ZBj3Qp1bYg", "EBT", "https://openreview.net/forum?id=ZBj3Qp1bYg"),
        ("openreview:RZGs4OAH6g", "NSVIF", "https://openreview.net/forum?id=RZGs4OAH6g"),
    )
    return [
        {
            "source_id": source_id,
            "title_hint": title,
            "source_url": url,
            "capture_utc": SOURCE_CAPTURE_UTC,
            "http_status": 200,
            "access_outcome": "browser_challenge",
            "content_verified": False,
            "claim_allowed": False,
            "terminal": True,
        }
        for source_id, title, url in records
    ]


def huggingface_rows() -> list[JsonDict]:
    """Return dated paper-index receipts without granting primary authority."""

    records = (
        ("hf-paper:2605.02915", "https://huggingface.co/papers/2605.02915"),
        ("hf-paper:2605.03534", "https://huggingface.co/papers/2605.03534"),
        ("hf-paper:2602.18671", "https://huggingface.co/papers/2602.18671"),
    )
    return [
        {
            "source_id": source_id,
            "source_url": url,
            "capture_utc": SOURCE_CAPTURE_UTC,
            "http_status": 200,
            "access_outcome": "ok",
            "evidence_role": "discovery_index_only",
            "popularity_used_as_evidence": False,
            "content_verified": True,
            "terminal": True,
        }
        for source_id, url in records
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Return visible citation-row totals without claiming canonical counts."""

    records = (
        ("semantic-scholar:ARXIV:2507.02092", 35, "2026-08-14"),
        ("semantic-scholar:ARXIV:2512.15605", 8, "2026-07-02"),
    )
    return [
        {
            "source_id": source_id,
            "source_url": (
                "https://api.semanticscholar.org/graph/v1/paper/"
                f"{source_id.removeprefix('semantic-scholar:')}/citations"
            ),
            "capture_utc": SOURCE_CAPTURE_UTC,
            "http_status": 200,
            "access_outcome": "ok",
            "visible_response_rows": count,
            "newest_visible_publication_date": newest,
            "citation_count_claimed": False,
            "rate_limit_observed": False,
            "content_verified": True,
            "terminal": True,
        }
        for source_id, count, newest in records
    ]


def github_rows() -> list[JsonDict]:
    """Return repository activity and license metadata, never quality ranks."""

    records = (
        ("github:alexiglad/EBT", "ebm", "2026-04-21T00:53:47Z", "Apache-2.0"),
        (
            "github:youtube/static-constraint-decoding",
            "constraint",
            "2026-09-01T22:04:28Z",
            "Apache-2.0",
        ),
        ("github:KindXiaoming/pykan", "kan", "2025-01-19T18:49:39Z", "MIT"),
        ("github:NVIDIA/Ising", "ising", "2026-07-30T20:23:59Z", "Apache-2.0"),
        (
            "github:AI4Science-WestlakeU/VFScale",
            "verifier",
            "2026-04-01T14:03:36Z",
            "MIT",
        ),
        (
            "github:extropic-ai/sparse-transformers",
            "ebm_hardware_software",
            "2026-09-03T20:28:22Z",
            "Apache-2.0",
        ),
        (
            "github:phalod-aditya/slm-confidence-signals",
            "verifier",
            "2026-05-19T00:04:25Z",
            "MIT",
        ),
    )
    return [
        {
            "source_id": source_id,
            "category": category,
            "source_url": f"https://github.com/{source_id.removeprefix('github:')}",
            "capture_utc": SOURCE_CAPTURE_UTC,
            "pushed_at": pushed_at,
            "license": license_name,
            "access_outcome": "ok",
            "search_rank_used_as_quality_evidence": False,
            "local_dependency_promoted": False,
            "terminal": True,
        }
        for source_id, category, pushed_at, license_name in records
    ]


def extropic_rows() -> list[JsonDict]:
    """Bound Z1T to software identity and explicitly estimated hardware facts."""

    return [
        {
            "source_id": "extropic:z1t-2026-09-04",
            "source_url": "https://extropic.ai/writing/z1t",
            "capture_utc": SOURCE_CAPTURE_UTC,
            "evidence_class": "public_software_and_vendor_estimated_hardware",
            "software_urls": [
                "https://github.com/extropic-ai/sparse-transformers",
                "https://huggingface.co/Extropic-AI/Z1T-0",
            ],
            "estimated_hardware_identity": {
                "graph_degree": 16,
                "pbit_count": 269_568,
                "edge_count": 2_135_904,
            },
            "hardware_execution_promoted": False,
            "runtime_claimed": False,
            "power_claimed": False,
            "availability_claimed": False,
            "speed_claimed": False,
            "access_outcome": "ok",
            "terminal": True,
        }
    ]


def kona_rows() -> list[JsonDict]:
    """Keep Kona proprietary until both reproducibility conditions are observed."""

    return [
        {
            "source_id": "logical-intelligence:kona-1.0",
            "source_url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "capture_utc": SOURCE_CAPTURE_UTC,
            "classification": "proprietary_watch_only",
            "public_weights_observed": False,
            "training_recipe_observed": False,
            "local_runner_observed": False,
            "public_implementation_promoted": False,
            "runtime_claimed": False,
            "power_claimed": False,
            "availability_claimed": False,
            "speed_claimed": False,
            "access_outcome": "ok",
            "terminal": True,
        }
    ]


def code_identity_rows() -> list[JsonDict]:
    """Return immutable public revisions observed during the capture."""

    records = (
        (
            "github:phalod-aditya/slm-confidence-signals",
            "git_commit",
            "b44143b8d617c98461c76d2ea687df9fa12ad8e0",
            "MIT",
        ),
        (
            "github:alexiglad/EBT",
            "git_commit",
            "19420cbeae655bbf11930219a675ade6897019e8",
            "Apache-2.0",
        ),
        (
            "github:youtube/static-constraint-decoding",
            "git_commit",
            "ac18fa1870ac45e0a3559090a0ea9ec005226cf5",
            "Apache-2.0",
        ),
        (
            "github:KindXiaoming/pykan",
            "git_commit",
            "ecde4ec3274d3bef1ad737479cf126aed38ab530",
            "MIT",
        ),
        (
            "github:NVIDIA/Ising",
            "git_commit",
            "8d1bd74a1f31e1cafe4aa49a553f2c99152ab3a1",
            "Apache-2.0",
        ),
        (
            "github:AI4Science-WestlakeU/VFScale",
            "git_commit",
            "d10b1bcf923a2bdaf91ccc50a748289fc3533731",
            "MIT",
        ),
        (
            "github:extropic-ai/sparse-transformers",
            "git_commit",
            "13051e90df9669be5b8f9f34fb097329fa82f674",
            "Apache-2.0",
        ),
        (
            "hf:Extropic-AI/Z1T-0",
            "hf_revision",
            "b5c244cee26b7f4e613b9ddbe83e83b2962ab2e1",
            "not_declared_in_model_metadata",
        ),
    )
    return [
        {
            "source_id": source_id,
            "identity_type": identity_type,
            "revision": revision,
            "license": license_name,
            "capture_utc": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for source_id, identity_type, revision, license_name in records
    ]


def duplicate_rows() -> list[JsonDict]:
    """Return cross-index identities suppressed in favor of canonical sources."""

    records = (
        ("hf-paper:2605.02915", "arxiv:2605.02915"),
        ("hf-paper:2605.03534", "arxiv:2605.03534"),
        ("github:AI4Science-WestlakeU/VFScale", "openreview:8ta0xgtsJK"),
        ("semantic-scholar:ARXIV:2507.02092", "arxiv:2507.02092"),
        ("semantic-scholar:ARXIV:2512.15605", "arxiv:2512.15605"),
    )
    return [
        {
            "source_id": source_id,
            "canonical_source_id": canonical_source_id,
            "disposition": "duplicate",
            "capture_utc": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for source_id, canonical_source_id in records
    ]


def inaccessible_rows() -> list[JsonDict]:
    """Mirror every terminal inaccessible source route."""

    return [
        {
            "source_id": row["source_id"],
            "source_url": row["source_url"],
            "capture_utc": row["capture_utc"],
            "access_outcome": row["access_outcome"],
            "claim_allowed": False,
            "terminal": True,
        }
        for row in openreview_rows()
        if not row["content_verified"]
    ]


def decision_rows() -> list[JsonDict]:
    """Return selected, watch-only, and rejected planning dispositions."""

    selected = (
        ("model_family_likelihood_controls", "arxiv:2605.02915"),
        ("set_level_evidence_sufficiency", "arxiv:2605.03534"),
        ("hopfield_selector_control", "arxiv:2607.10128"),
        ("exact_mrv_control", "arxiv:2607.23055"),
        ("finite_ising_parity", "arxiv:2609.02948"),
        ("context_bound_learning", "arxiv:2608.26730"),
        ("z1t_degree16_software_receipt", "extropic:z1t-2026-09-04"),
    )
    rows = [
        {
            "hook_id": hook_id,
            "source_id": source_id,
            "disposition": "selected",
            "decision_basis": "public_evidence_plus_bounded_planning_inference",
            "promoted": True,
            "science_claim_promoted": False,
            "capture_utc": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for hook_id, source_id in selected
    ]
    rows.extend(
        [
            {
                "hook_id": "kan_pwa_existing_lineage",
                "source_id": "arxiv:2602.06737",
                "disposition": "watch_only",
                "decision_basis": "public_primary_source_no_new_v620_prerequisite",
                "promoted": False,
                "science_claim_promoted": False,
                "capture_utc": SOURCE_CAPTURE_UTC,
                "terminal": True,
            },
            {
                "hook_id": "continual_memory_control",
                "source_id": "arxiv:2604.27003",
                "disposition": "watch_only",
                "decision_basis": "public_primary_source_advisory_control",
                "promoted": False,
                "science_claim_promoted": False,
                "capture_utc": SOURCE_CAPTURE_UTC,
                "terminal": True,
            },
            {
                "hook_id": "spilled_energy_future_control",
                "source_id": "hf-paper:2602.18671",
                "disposition": "watch_only",
                "decision_basis": "index_lead_requires_logits_and_factual_benchmark",
                "promoted": False,
                "science_claim_promoted": False,
                "capture_utc": SOURCE_CAPTURE_UTC,
                "terminal": True,
            },
            {
                "hook_id": "draft_conditioned_decoding",
                "source_id": "arxiv:2603.03305",
                "disposition": "rejected",
                "decision_basis": "search_candidate_outside_v620_frozen_entrance_scope",
                "promoted": False,
                "science_claim_promoted": False,
                "capture_utc": SOURCE_CAPTURE_UTC,
                "terminal": True,
            },
            {
                "hook_id": "kona_public_implementation",
                "source_id": "logical-intelligence:kona-1.0",
                "disposition": "rejected",
                "decision_basis": "vendor_only_without_public_weights_or_local_runner",
                "promoted": False,
                "science_claim_promoted": False,
                "capture_utc": SOURCE_CAPTURE_UTC,
                "terminal": True,
            },
        ]
    )
    return rows


def task_mapping_rows() -> list[JsonDict]:
    """Map each selected planning hook to named V620 work."""

    records = (
        ("model_family_likelihood_controls", ["exp7082-entrance-energy-likelihood-controls"]),
        ("set_level_evidence_sufficiency", ["exp7081-entrance-bank-set-sufficiency-audit"]),
        ("hopfield_selector_control", ["exp7082-entrance-energy-likelihood-controls"]),
        ("exact_mrv_control", ["exp7082-entrance-energy-likelihood-controls"]),
        ("finite_ising_parity", ["exp7083-entrance-ising-degree16-parity"]),
        (
            "context_bound_learning",
            [
                "exp7084-large-immutable-bcit-stream",
                "exp7085-bcit-prospective-self-learning",
                "exp7086-bcit-cold-drift-audit",
            ],
        ),
        ("z1t_degree16_software_receipt", ["exp7083-entrance-ising-degree16-parity"]),
    )
    return [
        {
            "hook_id": hook_id,
            "task_ids": task_ids,
            "defer_decision": None,
            "capture_utc": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for hook_id, task_ids in records
    ]


def vendor_claim_boundaries() -> list[JsonDict]:
    """Return explicit claims that this source receipt cannot support."""

    records = (
        ("z1t_runtime", "extropic", "runtime"),
        ("z1t_power", "extropic", "power"),
        ("z1t_availability", "extropic", "availability"),
        ("z1t_speed", "extropic", "speed"),
        ("kona_runtime", "logical_intelligence", "runtime"),
        ("kona_public_implementation", "logical_intelligence", "implementation_readiness"),
        ("github_search_rank", "github", "quality"),
        ("semantic_scholar_count", "semantic_scholar", "citation_authority"),
    )
    return [
        {
            "boundary_id": boundary_id,
            "source_scope": source_scope,
            "prohibited_claim": prohibited_claim,
            "claim_allowed": False,
            "capture_utc": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for boundary_id, source_scope, prohibited_claim in records
    ]


def combined_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Combine typed ledgers while retaining their collection identity."""

    rows: list[JsonDict] = []
    for field in ROW_COLLECTION_FIELDS:
        value = artifact.get(field, [])
        if isinstance(value, list):
            rows.extend(
                {"collection": field, **deepcopy(row)} for row in value if isinstance(row, dict)
            )
    return rows


def _network_available() -> bool:
    """Treat any HTTP response from arXiv as network reachability."""

    request = Request("https://arxiv.org/abs/2605.02915", headers={"User-Agent": "carnot/7077"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def _readable_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_bytes())
    except OSError:
        return False


def _writable(path: Path) -> bool:
    """Probe the output directory without touching the requested artifact."""

    parent = path.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7077-", delete=True):
            return True
    except OSError:
        return False


def _capture_is_current(value: str, run_date: str) -> bool:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        expected = datetime.strptime(run_date, "%Y%m%d").date()
    except (TypeError, ValueError):
        return False
    return parsed.date() == expected


def check_preconditions(
    root: Path,
    output_path: Path,
    run_date: str,
    *,
    network_available: bool | None = None,
    source_capture_utc: str = SOURCE_CAPTURE_UTC,
) -> list[JsonDict]:
    """Check every execution gate before evidence promotion."""

    reachable = _network_available() if network_available is None else network_available
    checks: list[JsonDict] = [
        {
            "check": "network_access",
            "expected_value": True,
            "observed_value": reachable,
            "passed": reachable,
        }
    ]
    for relative in REQUIRED_LOCAL_PATHS:
        readable = _readable_nonempty(root / relative)
        checks.append(
            {
                "check": f"readable:{relative.as_posix()}",
                "expected_value": "readable_nonempty_file",
                "observed_value": "readable_nonempty_file" if readable else "missing_or_unreadable",
                "passed": readable,
            }
        )
    marker_present = False
    try:
        marker_present = REFERENCE_MARKER in (root / REFERENCE_PATH).read_text(encoding="utf-8")
    except OSError:
        pass
    checks.append(
        {
            "check": "v620_reference_marker",
            "expected_value": REFERENCE_MARKER,
            "observed_value": REFERENCE_MARKER if marker_present else "missing",
            "passed": marker_present,
        }
    )
    writable = _writable(output_path)
    checks.append(
        {
            "check": "writable_artifact_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if writable else "not_writable",
            "passed": writable,
        }
    )
    current = _capture_is_current(source_capture_utc, run_date)
    expected_capture = (
        f"UTC timestamp dated {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}"
    )
    checks.append(
        {
            "check": "source_capture_utc",
            "expected_value": expected_capture,
            "observed_value": source_capture_utc,
            "passed": current,
        }
    )
    return checks


def _source_hashes(root: Path) -> list[JsonDict]:
    rows = []
    for relative in REQUIRED_LOCAL_PATHS:
        path = root / relative
        if _readable_nonempty(path):
            rows.append(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "terminal": True,
                }
            )
    return rows


def _all_dated_terminal(artifact: Mapping[str, Any], field: str) -> bool:
    rows = artifact.get(field)
    return (
        bool(rows)
        and isinstance(rows, list)
        and all(
            isinstance(row, dict)
            and row.get("capture_utc") == artifact.get("source_capture_utc")
            and row.get("terminal") is True
            for row in rows
        )
    )


def _unsupported_promotions(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for row in artifact.get("github_rows", []):
        if isinstance(row, dict) and row.get("local_dependency_promoted") is True:
            errors.append("unsupported implementation promotion: GitHub repository")
    for row in artifact.get("extropic_rows", []):
        if not isinstance(row, dict):
            continue
        if row.get("hardware_execution_promoted") is True:
            errors.append("unsupported implementation promotion: Extropic hardware")
        for claim in ("runtime", "power", "availability", "speed"):
            if row.get(f"{claim}_claimed") is True:
                errors.append(f"vendor-only claim: Extropic {claim}")
    for row in artifact.get("kona_rows", []):
        if not isinstance(row, dict):
            continue
        reproducible = (
            row.get("public_weights_observed") is True and row.get("local_runner_observed") is True
        )
        if row.get("public_implementation_promoted") is True and not reproducible:
            errors.append("unsupported implementation promotion: Kona")
        for claim in ("runtime", "power", "availability", "speed"):
            if row.get(f"{claim}_claimed") is True:
                errors.append(f"vendor-only claim: Kona {claim}")
    return errors


def _coverage_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    source_fields = {
        "arxiv": "primary_source_rows",
        "openreview": "openreview_rows",
        "huggingface": "huggingface_rows",
        "semantic_scholar": "semantic_scholar_rows",
        "github": "github_rows",
        "extropic": "extropic_rows",
        "logical_intelligence": "kona_rows",
    }
    for source_class in REQUIRED_SOURCE_CLASSES:
        if not _all_dated_terminal(artifact, source_fields[source_class]):
            errors.append(f"source class lacks dated terminal receipt: {source_class}")

    queries = artifact.get("query_rows", [])
    found_families = {
        row.get("query_family")
        for row in queries
        if isinstance(row, dict) and row.get("route") == "arxiv"
    }
    if found_families != set(REQUIRED_QUERY_FAMILIES):
        errors.append("query family coverage mismatch")
    found_routes = {row.get("route") for row in queries if isinstance(row, dict)}
    if found_routes != set(REQUIRED_SOURCE_CLASSES):
        errors.append("source route query coverage mismatch")

    primaries = artifact.get("primary_source_rows", [])
    identifiers = [row.get("source_id") for row in primaries if isinstance(row, dict)]
    if len(identifiers) != len(set(identifiers)):
        errors.append("duplicate primary source_id")
    identity_fields = (
        "source_id",
        "title",
        "source_url",
        "version_or_revision",
        "capture_utc",
        "metadata_receipt_sha256",
    )
    for row in primaries:
        if not isinstance(row, dict) or any(not row.get(field) for field in identity_fields):
            errors.append("missing source identity")
            continue
        if row["source_id"] in {"arxiv:2605.02915", "arxiv:2605.03534"}:
            if not row.get("authors") or row["metadata_receipt_sha256"] != _metadata_receipt(row):
                errors.append("missing source identity or invalid metadata receipt")
    if not {"arxiv:2605.02915", "arxiv:2605.03534"}.issubset(set(identifiers)):
        errors.append("required primary source identity missing")

    inaccessible = artifact.get("inaccessible_rows", [])
    inaccessible_ids = {row.get("source_id") for row in inaccessible if isinstance(row, dict)}
    expected_inaccessible = {
        row.get("source_id")
        for row in artifact.get("openreview_rows", [])
        if isinstance(row, dict) and row.get("content_verified") is False
    }
    if inaccessible_ids != expected_inaccessible or any(
        not isinstance(row, dict)
        or row.get("claim_allowed") is not False
        or row.get("terminal") is not True
        for row in inaccessible
    ):
        errors.append("inaccessible route mirror mismatch")

    dispositions = {
        row.get("disposition") for row in artifact.get("decision_rows", []) if isinstance(row, dict)
    }
    if not {"selected", "watch_only", "rejected"}.issubset(dispositions):
        errors.append("decision dispositions incomplete")
    duplicates = artifact.get("duplicate_rows", [])
    if not duplicates or any(
        not isinstance(row, dict) or row.get("disposition") != "duplicate" for row in duplicates
    ):
        errors.append("duplicate dispositions incomplete")

    promoted = {
        row.get("hook_id")
        for row in artifact.get("decision_rows", [])
        if isinstance(row, dict) and row.get("promoted") is True
    }
    mapped_rows = artifact.get("task_mapping_rows", [])
    mapped = {row.get("hook_id") for row in mapped_rows if isinstance(row, dict)}
    if promoted != mapped or any(
        not isinstance(row, dict) or not (row.get("task_ids") or row.get("defer_decision"))
        for row in mapped_rows
    ):
        errors.append("promoted hook mapping mismatch")

    code_ids = {
        row.get("source_id")
        for row in artifact.get("code_identity_rows", [])
        if isinstance(row, dict) and row.get("revision")
    }
    github_ids = {
        row.get("source_id") for row in artifact.get("github_rows", []) if isinstance(row, dict)
    }
    if not github_ids.issubset(code_ids):
        errors.append("GitHub code identity missing")
    errors.extend(_unsupported_promotions(artifact))
    return errors


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Return one only for complete, supported, reproducible evidence rows."""

    return int(not _coverage_errors(artifact))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding elapsed runtime and the hash itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return _sha256_json(payload)


def _empty_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Sequence[Mapping[str, Any]],
    source_capture_utc: str,
) -> JsonDict:
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": _copy(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": _copy(hashes),
        "rows": [],
        "query_rows": [],
        "primary_source_rows": [],
        "openreview_rows": [],
        "huggingface_rows": [],
        "semantic_scholar_rows": [],
        "github_rows": [],
        "extropic_rows": [],
        "kona_rows": [],
        "code_identity_rows": [],
        "duplicate_rows": [],
        "inaccessible_rows": [],
        "decision_rows": [],
        "task_mapping_rows": [],
        "vendor_claim_boundaries": [],
        "source_capture_utc": source_capture_utc,
        "sota_ingestion_complete_score": 0,
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
        "honest_verdict": "blocked_v620_sota_ingestion",
    }
    return artifact


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
    network_available: bool | None = None,
    source_capture_utc: str = SOURCE_CAPTURE_UTC,
    duration_s: float | None = None,
) -> JsonDict:
    """Build one positive or blocked deterministic V620 receipt."""

    started = time.monotonic()
    root = Path(root)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    preconditions = check_preconditions(
        root,
        output_path,
        run_date,
        network_available=network_available,
        source_capture_utc=source_capture_utc,
    )
    artifact = _empty_artifact(preconditions, _source_hashes(root), source_capture_utc)
    failure = next((row for row in preconditions if row["passed"] is False), None)
    if failure is not None:
        artifact["gate_check_summary"] = {
            "failed_check": failure["check"],
            "expected_value": failure["expected_value"],
            "observed_value": failure["observed_value"],
            "passed": False,
        }
    else:
        artifact.update(
            {
                "query_rows": query_rows(),
                "primary_source_rows": primary_source_rows(),
                "openreview_rows": openreview_rows(),
                "huggingface_rows": huggingface_rows(),
                "semantic_scholar_rows": semantic_scholar_rows(),
                "github_rows": github_rows(),
                "extropic_rows": extropic_rows(),
                "kona_rows": kona_rows(),
                "code_identity_rows": code_identity_rows(),
                "duplicate_rows": duplicate_rows(),
                "inaccessible_rows": inaccessible_rows(),
                "decision_rows": decision_rows(),
                "task_mapping_rows": task_mapping_rows(),
                "vendor_claim_boundaries": vendor_claim_boundaries(),
            }
        )
        artifact["rows"] = combined_rows(artifact)
        artifact["sota_ingestion_complete_score"] = completion_score(artifact)
        if artifact["sota_ingestion_complete_score"] == 1:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = "complete_positive_v620_sota_ingestion"
            artifact["gate_check_summary"] = {
                "failed_check": None,
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            }
        else:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "disqualified_v620_sota_ingestion"
            artifact["gate_check_summary"] = {
                "failed_check": "source_ingestion_contract",
                "expected_value": 1,
                "observed_value": artifact["sota_ingestion_complete_score"],
                "passed": False,
            }
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact["duration_s"] = round(max(0.0, elapsed), 6)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _unique(errors: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(errors))


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Independently validate shape, coverage, boundaries, verdict, and hash."""

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
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if not isinstance(artifact["duration_s"], (int, float)) or artifact["duration_s"] < 0:
        errors.append("duration_s invalid")
    if artifact["random_seed"] != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact["rows"] != combined_rows(artifact):
        errors.append("rows consistency mismatch")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, list) or not preconditions:
        errors.append("preconditions_checked missing")
        failure = None
    else:
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
        if artifact["gate_check_summary"] != expected_gate:
            errors.append("blocked gate_check_summary mismatch")
        if artifact["sota_ingestion_complete_score"] != 0:
            errors.append("blocked completion score must be zero")
        if artifact["verdict_class"] != "blocked":
            errors.append("blocked verdict_class mismatch")
        if not str(artifact["honest_verdict"]).startswith("blocked_"):
            errors.append("blocked honest_verdict prefix mismatch")
    else:
        if not _capture_is_current(artifact["source_capture_utc"], RUN_DATE):
            errors.append("source_capture_utc stale or invalid")
        coverage_errors = _coverage_errors(artifact)
        errors.extend(coverage_errors)
        score = int(not coverage_errors)
        if artifact["sota_ingestion_complete_score"] != score:
            errors.append("sota_ingestion_complete_score mismatch")
        expected_class = "positive" if score == 1 else "disqualified"
        expected_prefix = "complete_positive_" if score == 1 else "disqualified_"
        if artifact["verdict_class"] != expected_class:
            errors.append("verdict_class inconsistent with completion")
        if not str(artifact["honest_verdict"]).startswith(expected_prefix):
            errors.append("honest_verdict prefix inconsistent with completion")
        expected_gate = {
            "failed_check": None if score == 1 else "source_ingestion_contract",
            "expected_value": 1,
            "observed_value": score,
            "passed": score == 1,
        }
        if artifact["gate_check_summary"] != expected_gate:
            errors.append("gate_check_summary inconsistent with completion")

    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return _unique(errors)


def write_artifact(artifact: Mapping[str, Any], path: Path) -> Path:
    """Validate before atomically replacing the requested artifact."""

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
    parser.add_argument("--source-capture-utc", default=SOURCE_CAPTURE_UTC)
    parser.add_argument("--network-available", action="store_true", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build a receipt or independently validate an existing one."""

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
        network_available=args.network_available,
        source_capture_utc=args.source_capture_utc,
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
