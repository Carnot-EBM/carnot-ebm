"""Build the bounded V614 primary-source and public-code ingestion receipt.

The source rows are a dated execution receipt, not learner input. Exact labels,
source identities, and future ARC observations stay outside every proposed
learned mechanism. This keeps a useful software analogy from becoming an
unsupported benchmark, solve, or hardware claim.

Spec refs: REQ-REPORT-7011 and SCENARIO-REPORT-7011-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7011
RUN_DATE = "20260905"
RANDOM_SEED = 7_011_202_609_05
INFERENCE_SUBSTRATE = "deterministic_primary_source_ingestion_no_llm"
REFERENCE_MARKER = "## V614 Planner Refresh - 2026-09-04"
PLANNER_MARKER = "2026-09-05T03:46:23Z"

MODULE_PATH = Path("python/carnot/experiment_7011_v614_sota_ingestion.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_7011_v614_sota_ingestion.py")
TEST_PATH = Path("tests/python/test_experiment_7011_v614_sota_ingestion.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7011_v614_sota_ingestion.json")
REQUIRED_LOCAL_PATHS = (
    REFERENCE_PATH,
    Path("python/carnot/agentic/arc_pinductor.py"),
    Path("python/carnot/agentic/arc_invariant_memory.py"),
    Path("openspec/capabilities/constraint-verification/spec.md"),
    Path("openspec/capabilities/arc-world-model-trust-energy/spec.md"),
    SPEC_PATH,
)

SOURCE_IDS = ("bbwm", "hippo", "batchsum", "introconformal", "z1t")
ACCEPTED_METHOD_IDS = (
    "bbwm_queryable_belief",
    "hippo_pair_anchored_response",
    "batchsum_pair_centering",
    "z1t_sparse_software_graph",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_query_rows",
    "primary_source_rows",
    "repository_rows",
    "release_identity_rows",
    "method_rows",
    "method_to_module_rows",
    "leakage_boundary_rows",
    "control_rows",
    "falsification_rows",
    "unsupported_dependency_rows",
    "non_claim_rows",
    "secondary_check_rows",
    "reference_append_rows",
    "rows",
    "command_receipt_rows",
    "v614_sota_ingestion_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "One scientific reason per field makes the evidence contract reviewable.",
    "preconditions_checked": "Preflight evidence prevents a missing input from becoming a source conclusion.",
    "inference_substrate": "The substrate states that deterministic source review used no LLM inference.",
    "duration_s": "Measured time distinguishes an executed check from an undated static claim.",
    "source_query_rows": "Query receipts preserve direct access failures separately from source findings.",
    "primary_source_rows": "One terminal row per selected source prevents selective source reporting.",
    "repository_rows": "Repository rows separate public code presence from paper availability.",
    "release_identity_rows": "Immutable revisions make every available public-code observation replayable.",
    "method_rows": "Method rows distinguish direct reuse, analogy, watch items, and unsupported claims.",
    "method_to_module_rows": "Named targets constrain each accepted idea to the current Carnot stack.",
    "leakage_boundary_rows": "Explicit exclusions keep exact authority outside learned inputs.",
    "control_rows": "A control arm gives every accepted mechanism a fair local comparison.",
    "falsification_rows": "A failure condition makes each local adaptation scientifically testable.",
    "unsupported_dependency_rows": "Missing prerequisites stop public artifacts from implying local execution.",
    "non_claim_rows": "Prohibited claims keep paper and hardware results from drifting into Carnot claims.",
    "secondary_check_rows": "Advisory routes retain rate limits and missing assets without invented counts.",
    "reference_append_rows": "An explicit disposition proves whether the append-only ledger should change.",
    "rows": "A combined row ledger lets validators rebuild the bounded map from unit evidence.",
    "command_receipt_rows": "Terminal command receipts expose every required verification outcome.",
    "v614_sota_ingestion_complete_score": "One requires terminal sources and complete bounded maps for all accepted methods.",
    "random_seed": "A fixed seed documents deterministic ordering even though no stochastic model ran.",
    "reproducibility_checksum": "A stable payload hash detects later changes to scientific evidence.",
    "gate_check_summary": "Exact expected and observed values make blocked preconditions actionable.",
    "verifier_is_oracle": "False states that literature ingestion does not determine task correctness.",
    "verdict_class": "A closed verdict class prevents a blocked or incomplete run from appearing positive.",
    "honest_verdict": "A class-consistent terminal phrase gives automation an unambiguous outcome.",
}

VALIDATION_COMMAND_NAMES = (
    "focused_tests",
    "new_code_coverage_run",
    "new_code_coverage_report",
    "ruff_check",
    "ruff_format",
    "adversarial_verification",
    "row_consistency_lint",
    "openspec_coverage",
    "root_clutter_check",
    "full_python_tests",
    "artifact_validation",
)


_SOURCE_QUERY_ROWS: tuple[JsonDict, ...] = (
    {
        "query_id": "bbwm_arxiv",
        "source_id": "bbwm",
        "source_kind": "primary_paper",
        "url": "https://arxiv.org/abs/2609.00455",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "ok",
        "evidence_scope": "arXiv abstract and version history",
        "terminal": True,
    },
    {
        "query_id": "hippo_arxiv",
        "source_id": "hippo",
        "source_kind": "primary_paper",
        "url": "https://arxiv.org/abs/2606.29481",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "ok",
        "evidence_scope": "arXiv abstract, version history, and author code link",
        "terminal": True,
    },
    {
        "query_id": "batchsum_openreview",
        "source_id": "batchsum",
        "source_kind": "primary_review_record",
        "url": "https://openreview.net/forum?id=Tf4lRAOGkj",
        "accessed_on": "2026-09-05",
        "http_status": 403,
        "access_outcome": "challenge_required",
        "evidence_scope": "OpenReview route only; no content inferred from the challenge page",
        "terminal": True,
    },
    {
        "query_id": "batchsum_pmlr",
        "source_id": "batchsum",
        "source_kind": "primary_proceedings",
        "url": "https://proceedings.mlr.press/v267/hong25d.html",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "ok",
        "evidence_scope": "PMLR paper record and abstract",
        "terminal": True,
    },
    {
        "query_id": "introconformal_arxiv",
        "source_id": "introconformal",
        "source_kind": "primary_paper",
        "url": "https://arxiv.org/abs/2609.01375",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "ok",
        "evidence_scope": "arXiv abstract, version history, and author repository link",
        "terminal": True,
    },
    {
        "query_id": "z1t_first_party",
        "source_id": "z1t",
        "source_kind": "first_party_report",
        "url": "https://extropic.ai/writing/z1t",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "ok",
        "evidence_scope": "dated report, software links, graph description, and estimate boundary",
        "terminal": True,
    },
)

_PRIMARY_SOURCE_ROWS: tuple[JsonDict, ...] = (
    {
        "source_id": "bbwm",
        "title": "Towards a Belief-Based World Model for LLM Agents",
        "url": "https://arxiv.org/abs/2609.00455",
        "source_identity": "arXiv:2609.00455v1",
        "publication_or_update_date": "2026-08-31T22:48:38Z",
        "access_outcome": "ok",
        "method_extraction": "Expose a queryable belief about known and uncertain current state separately from action simulation.",
        "post_marker_change_proven": False,
        "terminal": True,
    },
    {
        "source_id": "hippo",
        "title": "To Reason or to Fabricate: Reasoning Without Shortcuts via Hint-Anchored Pairwise Aggregation",
        "url": "https://arxiv.org/abs/2606.29481",
        "source_identity": "arXiv:2606.29481v1",
        "publication_or_update_date": "2026-06-28T16:21:04Z",
        "access_outcome": "ok",
        "method_extraction": "Use hint-triggered traces as anchors for pairwise reward comparison instead of pointwise process scores.",
        "post_marker_change_proven": False,
        "terminal": True,
    },
    {
        "source_id": "batchsum",
        "title": "On the Robustness of Reward Models for Language Model Alignment",
        "url": "https://proceedings.mlr.press/v267/hong25d.html",
        "source_identity": "PMLR:267:hong25d; OpenReview:Tf4lRAOGkj",
        "publication_or_update_date": "2025-07-13",
        "access_outcome": "primary_proceedings_ok_openreview_challenged",
        "method_extraction": "Penalize each batch reward sum away from zero to limit abnormal reward magnitude and hidden-state norm dispersion.",
        "post_marker_change_proven": False,
        "terminal": True,
    },
    {
        "source_id": "introconformal",
        "title": "IntroConformal: Conformal Factuality Guarantees for Large Vision-Language Models via Introspective Signals",
        "url": "https://arxiv.org/abs/2609.01375",
        "source_identity": "arXiv:2609.01375v1",
        "publication_or_update_date": "2026-09-01T15:09:56Z",
        "access_outcome": "ok",
        "method_extraction": "Use layer-wise semantic stability or model self-verification probability as a conformal risk-control score.",
        "post_marker_change_proven": False,
        "terminal": True,
    },
    {
        "source_id": "z1t",
        "title": "Z1T: Sparse Transformer-Like Models for Probabilistic Hardware",
        "url": "https://extropic.ai/writing/z1t",
        "source_identity": "Extropic report 2026-09-04",
        "publication_or_update_date": "2026-09-04",
        "access_outcome": "ok",
        "method_extraction": "Train fixed-connectivity sparse JAX models whose local tanh-linear operations match a degree-16 probabilistic graph.",
        "post_marker_change_proven": False,
        "terminal": True,
    },
)

_REPOSITORY_ROWS: tuple[JsonDict, ...] = (
    {
        "repository_id": "bbwm_code",
        "source_id": "bbwm",
        "url": "https://github.com/skumar-ml/belief-world-models",
        "publication_or_update_date": "2026-08-31T22:29:26Z",
        "access_outcome": "ok",
        "available": True,
        "artifact_scope": "Python agent, environment, belief-world-model, and experiment code; paper trajectories absent",
        "terminal": True,
    },
    {
        "repository_id": "hippo_code",
        "source_id": "hippo",
        "url": "https://github.com/Infinite-set/HIPPO",
        "publication_or_update_date": "2026-06-28T08:34:12Z",
        "access_outcome": "readme_only_no_runnable_implementation",
        "available": True,
        "artifact_scope": "One README identifies the paper; no runnable method files are present",
        "terminal": True,
    },
    {
        "repository_id": "batchsum_code",
        "source_id": "batchsum",
        "url": "https://github.com/LinkedIn-XFACT/RM-Robustness",
        "publication_or_update_date": "2025-05-27T06:04:18Z",
        "access_outcome": "ok",
        "available": True,
        "artifact_scope": "Reward-model training and evaluation code with use_batch_sum configuration",
        "terminal": True,
    },
    {
        "repository_id": "introconformal_code",
        "source_id": "introconformal",
        "url": "https://github.com/Atabuzzaman/Introconformal",
        "publication_or_update_date": "2026-08-31T07:56:11Z",
        "access_outcome": "empty_repository_no_commit",
        "available": False,
        "artifact_scope": "The linked public Git repository exists but is empty",
        "terminal": True,
    },
    {
        "repository_id": "z1t_training_code",
        "source_id": "z1t",
        "url": "https://github.com/extropic-ai/sparse-transformers",
        "publication_or_update_date": "2026-09-03T20:28:12Z",
        "access_outcome": "ok",
        "available": True,
        "artifact_scope": "Apache-2.0 JAX sparse-transformer and Z1T research code",
        "terminal": True,
    },
    {
        "repository_id": "z1t_weights",
        "source_id": "z1t",
        "url": "https://huggingface.co/Extropic-AI/Z1T-0",
        "publication_or_update_date": "2026-09-04T20:43:34Z",
        "access_outcome": "ok",
        "available": True,
        "artifact_scope": "Public Equinox model revision with config and loader; not a GGUF release",
        "terminal": True,
    },
)

_RELEASE_IDENTITY_ROWS: tuple[JsonDict, ...] = (
    {
        "repository_id": "bbwm_code",
        "source_id": "bbwm",
        "identity_type": "git_commit",
        "identity": "d9e2900e37cb18a044fe3f084d32642d0f8f5fd5",
        "identity_date": "2026-08-31T22:29:26Z",
        "available": True,
        "terminal": True,
    },
    {
        "repository_id": "hippo_code",
        "source_id": "hippo",
        "identity_type": "git_commit",
        "identity": "fcb20525f6888b2d2aa18cd635a83d8e0b4c996c",
        "identity_date": "2026-06-28T08:34:12Z",
        "available": True,
        "terminal": True,
    },
    {
        "repository_id": "batchsum_code",
        "source_id": "batchsum",
        "identity_type": "git_commit",
        "identity": "3bafaabfb6e8c71234863103e9ba91d4e00c3498",
        "identity_date": "2025-05-27T06:04:18Z",
        "available": True,
        "terminal": True,
    },
    {
        "repository_id": "introconformal_code",
        "source_id": "introconformal",
        "identity_type": None,
        "identity": None,
        "identity_date": "2026-08-31T07:56:11Z",
        "available": False,
        "terminal": True,
    },
    {
        "repository_id": "z1t_training_code",
        "source_id": "z1t",
        "identity_type": "git_commit",
        "identity": "13051e90df9669be5b8f9f34fb097329fa82f674",
        "identity_date": "2026-09-03T20:28:12Z",
        "available": True,
        "terminal": True,
    },
    {
        "repository_id": "z1t_weights",
        "source_id": "z1t",
        "identity_type": "hf_revision",
        "identity": "b5c244cee26b7f4e613b9ddbe83e83b2962ab2e1",
        "identity_date": "2026-09-04T20:43:34Z",
        "available": True,
        "terminal": True,
    },
)

_METHOD_ROWS: tuple[JsonDict, ...] = (
    {
        "method_id": "bbwm_queryable_belief",
        "source_id": "bbwm",
        "classification": "architectural_analogy",
        "accepted": True,
        "mechanism": "Keep current-state belief queryable and separate from simulated action outcomes.",
        "local_scope": "Game-blind typed facts derived only from chronological ARC observations and actions.",
    },
    {
        "method_id": "hippo_pair_anchored_response",
        "source_id": "hippo",
        "classification": "direct_adaptation",
        "accepted": True,
        "mechanism": "Replace pointwise scoring with signed response differences inside matched intervention pairs.",
        "local_scope": "Exact minimal constraint violation-repair pairs over existing GGUF outputs.",
    },
    {
        "method_id": "batchsum_pair_centering",
        "source_id": "batchsum",
        "classification": "direct_adaptation",
        "accepted": True,
        "mechanism": "Center pair energies and penalize nonzero batch reward sums.",
        "local_scope": "Calibration-only PWA-KAN energy with source-group splits and shortcut controls.",
    },
    {
        "method_id": "introconformal_layerwise_crc",
        "source_id": "introconformal",
        "classification": "watch_only_dependency",
        "accepted": False,
        "mechanism": "Calibrate factuality risk from layer-wise semantic stability or self-verification probability.",
        "local_scope": "Deferred until a stable layer-wise state contract and independent calibration authority exist.",
    },
    {
        "method_id": "z1t_sparse_software_graph",
        "source_id": "z1t",
        "classification": "architectural_analogy",
        "accepted": True,
        "mechanism": "Represent sparse local operations and fixed graph degree in a typed, substrate-neutral graph receipt.",
        "local_scope": "Static software and serialization study over public JAX files; no Z1 execution.",
    },
    {
        "method_id": "z1t_hardware_efficiency",
        "source_id": "z1t",
        "classification": "unsupported_hardware_claim",
        "accepted": False,
        "mechanism": "Run sparse operations across Z1 chips and FPGA companion stages.",
        "local_scope": "Rejected locally because no authenticated Z1 or TSU runner is available.",
    },
)

_METHOD_TO_MODULE_ROWS: tuple[JsonDict, ...] = (
    {
        "method_id": "bbwm_queryable_belief",
        "target_modules": [
            "python/carnot/agentic/arc_pinductor.py",
            "python/carnot/agentic/arc_invariant_memory.py",
            "python/carnot/agentic/arc_competition_agent.py",
        ],
        "required_inputs": [
            "chronological observed ARC frames",
            "chosen ARC actions",
            "next observed frame admitted only after the frozen decision",
            "fixed game-blind fact schema",
        ],
        "adaptation": "Use Pinductor belief update structure and invariant-memory lifecycle without exposing game_id as a learner feature.",
    },
    {
        "method_id": "hippo_pair_anchored_response",
        "target_modules": [
            "python/carnot/experiment_6984_exact_contrast_fixture.py",
            "python/carnot/experiment_6999_blinded_feature_cold_audit.py",
        ],
        "required_inputs": [
            "matched exact minimal intervention pairs",
            "current GGUF response features",
            "signed within-pair response deltas",
        ],
        "adaptation": "Treat each exact intervention as an anchor and train only on within-pair response change.",
    },
    {
        "method_id": "batchsum_pair_centering",
        "target_modules": [
            "python/carnot/models/pwa_kan.py",
            "python/carnot/experiment_6977_certified_pwa_kan_energy.py",
        ],
        "required_inputs": [
            "signed pair-response vectors",
            "source-group split assignments",
            "calibration-only energy targets",
        ],
        "adaptation": "Add pair centering and a batch sum-to-zero term to calibration-only PWA-KAN fitting.",
    },
    {
        "method_id": "z1t_sparse_software_graph",
        "target_modules": [
            "python/carnot/experiment_6152_typed_stochastic_constraint_ir.py",
            "python/carnot/models/sparse_ising.py",
        ],
        "required_inputs": [
            "pinned public JAX graph and configuration",
            "typed sparse operation metadata",
            "fixed-degree adjacency description",
        ],
        "adaptation": "Serialize graph shape and supported operations as software evidence without selecting a hardware backend.",
    },
)

_LEAKAGE_BOUNDARY_ROWS: tuple[JsonDict, ...] = (
    {
        "method_id": "bbwm_queryable_belief",
        "learner_allowed_inputs": [
            "past frames",
            "past actions",
            "past observed outcomes",
            "typed fact values",
        ],
        "learner_excluded_inputs": [
            "game_id",
            "game source",
            "future frames",
            "exact level answer",
            "offline solver output",
        ],
        "exact_authority_external": True,
    },
    {
        "method_id": "hippo_pair_anchored_response",
        "learner_allowed_inputs": ["signed GGUF response-feature deltas inside one matched pair"],
        "learner_excluded_inputs": [
            "exact labels",
            "mutation IDs",
            "source IDs",
            "hint text",
            "provenance rows",
        ],
        "exact_authority_external": True,
    },
    {
        "method_id": "batchsum_pair_centering",
        "learner_allowed_inputs": [
            "pair-centered response features",
            "calibration targets after split freeze",
        ],
        "learner_excluded_inputs": [
            "held labels",
            "exact executor output",
            "source identity",
            "raw reward magnitude control labels",
        ],
        "exact_authority_external": True,
    },
    {
        "method_id": "z1t_sparse_software_graph",
        "learner_allowed_inputs": [
            "public operation type",
            "shape",
            "degree",
            "software configuration",
        ],
        "learner_excluded_inputs": [
            "Extropic performance estimates",
            "unobserved device telemetry",
            "Carnot exact labels",
        ],
        "exact_authority_external": True,
    },
)

_CONTROL_ROWS: tuple[JsonDict, ...] = (
    {
        "method_id": "bbwm_queryable_belief",
        "control_arm": "simulation-only, belief-only, and combined arms at matched action budgets",
    },
    {
        "method_id": "hippo_pair_anchored_response",
        "control_arm": "the frozen pointwise feature ranker plus pair-sign permutation",
    },
    {
        "method_id": "batchsum_pair_centering",
        "control_arm": "no regularizer, magnitude-only, and norm-only controls on source-group splits",
    },
    {
        "method_id": "z1t_sparse_software_graph",
        "control_arm": "the same typed graph serialized without Z1T-specific placement labels",
    },
)

_FALSIFICATION_ROWS: tuple[JsonDict, ...] = (
    {
        "method_id": "bbwm_queryable_belief",
        "failure_condition": "Belief-only and combined arms fail to beat matched simulation-only control on sealed future transition utility.",
    },
    {
        "method_id": "hippo_pair_anchored_response",
        "failure_condition": "Pair response signs fail under pair permutation or do not generalize across held source groups.",
    },
    {
        "method_id": "batchsum_pair_centering",
        "failure_condition": "The zero-sum term does not improve held-source ranking beyond norm-only and magnitude-only controls.",
    },
    {
        "method_id": "z1t_sparse_software_graph",
        "failure_condition": "The pinned public graph cannot round-trip through the typed sparse receipt without unsupported operations.",
    },
)

_UNSUPPORTED_DEPENDENCY_ROWS: tuple[JsonDict, ...] = (
    {
        "source_id": "bbwm",
        "dependency": "Full paper reproduction recommends a 40 GB vLLM GPU, Java 17, environment data, and gated-model access.",
        "local_effect": "Reuse only the belief-versus-simulation interface over existing ARC transitions and GGUF outputs.",
    },
    {
        "source_id": "hippo",
        "dependency": "The pinned public HIPPO repository contains only a README and no runnable implementation.",
        "local_effect": "Implement the pair unit from the primary method description and do not claim code reproduction.",
    },
    {
        "source_id": "batchsum",
        "dependency": "The reference training stack uses full reward-model training, Accelerate, FSDP, and external evaluation packages.",
        "local_effect": "Adapt only the batch sum-to-zero penalty to the local calibration-only PWA-KAN.",
    },
    {
        "source_id": "introconformal",
        "dependency": "Layer-wise LVLM hidden states and an independent factuality calibration set are unavailable on the current GGUF output contract.",
        "local_effect": "Keep IntroConformal watch-only and do not treat model self-judgment as exact authority.",
    },
    {
        "source_id": "z1t",
        "dependency": "The public model uses Equinox/JAX files, and Carnot has no authenticated Z1 or TSU device runner.",
        "local_effect": "Study graph and training software only; do not infer GGUF compatibility or device performance.",
    },
)

_NON_CLAIM_ROWS: tuple[JsonDict, ...] = (
    {
        "claim_id": "no_paper_reproduction_claim",
        "prohibited_claim": "Carnot reproduced any selected paper benchmark.",
        "reason": "This task verifies sources and maps mechanisms; it runs no paper training or evaluation.",
    },
    {
        "claim_id": "no_arc_solve_claim",
        "prohibited_claim": "A belief ledger solves an ARC game or level.",
        "reason": "No belief-state policy experiment runs in this ingestion task.",
    },
    {
        "claim_id": "no_oracle_distinct_claim",
        "prohibited_claim": "Pair centering or self-verification is an oracle-distinct verifier.",
        "reason": "Exact authorities remain external, and model self-signals do not establish correctness.",
    },
    {
        "claim_id": "no_introconformal_guarantee_claim",
        "prohibited_claim": "IntroConformal guarantees factuality for Carnot GGUF outputs.",
        "reason": "The required LVLM state and calibration contracts were not implemented or tested.",
    },
    {
        "claim_id": "no_z1_execution_claim",
        "prohibited_claim": "Carnot executed Z1T on Z1 or any TSU hardware.",
        "reason": "Only public software and model metadata were inspected.",
    },
    {
        "claim_id": "no_z1_efficiency_claim",
        "prohibited_claim": "Carnot verified Z1T energy, latency, or speedup.",
        "reason": "The first-party report includes estimates and Carnot has no authenticated device receipt.",
    },
)

_SECONDARY_CHECK_ROWS: tuple[JsonDict, ...] = (
    {
        "check_id": "ebt_citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?limit=100&fields=title,year,url,publicationDate",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "ok",
        "response_row_count": 35,
        "newest_visible_publication_date": "2026-08-14",
        "citation_count_claimed": False,
        "finding": "The returned page is an advisory route receipt, not an authoritative citation count or implementation release.",
        "terminal": True,
    },
    {
        "check_id": "arm_ebm_citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?limit=100&fields=title,year,url,publicationDate",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "ok",
        "response_row_count": 8,
        "newest_visible_publication_date": "2026-07-02",
        "citation_count_claimed": False,
        "finding": "The returned page is an advisory route receipt, not an authoritative citation count or implementation release.",
        "terminal": True,
    },
    {
        "check_id": "kona",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "accessed_on": "2026-09-05",
        "http_status": 200,
        "access_outcome": "product_page_ok_no_public_artifact_link_observed",
        "public_implementation_claimed": False,
        "finding": "The retrieved first-party page describes Kona 1.0 but did not expose public weights, a training recipe, or a reproducible local runner.",
        "terminal": True,
    },
)


def source_query_rows() -> list[JsonDict]:
    """Return direct access receipts without allowing callers to mutate evidence."""

    return [deepcopy(row) for row in _SOURCE_QUERY_ROWS]


def primary_source_rows() -> list[JsonDict]:
    """Return one terminal source-supported method row per selected family."""

    return [deepcopy(row) for row in _PRIMARY_SOURCE_ROWS]


def repository_rows() -> list[JsonDict]:
    """Return public-code availability separately from paper availability."""

    return [deepcopy(row) for row in _REPOSITORY_ROWS]


def release_identity_rows() -> list[JsonDict]:
    """Return immutable revisions, including an explicit empty-repository row."""

    return [deepcopy(row) for row in _RELEASE_IDENTITY_ROWS]


def method_rows() -> list[JsonDict]:
    """Return bounded method classifications for all selected mechanisms."""

    return [deepcopy(row) for row in _METHOD_ROWS]


def method_to_module_rows() -> list[JsonDict]:
    """Return only accepted software mechanisms and their local module targets."""

    return [deepcopy(row) for row in _METHOD_TO_MODULE_ROWS]


def leakage_boundary_rows() -> list[JsonDict]:
    """Return learner-visible and authority-only input boundaries."""

    return [deepcopy(row) for row in _LEAKAGE_BOUNDARY_ROWS]


def control_rows() -> list[JsonDict]:
    """Return one local comparison control for every accepted mechanism."""

    return [deepcopy(row) for row in _CONTROL_ROWS]


def falsification_rows() -> list[JsonDict]:
    """Return one explicit local failure condition for every accepted mechanism."""

    return [deepcopy(row) for row in _FALSIFICATION_ROWS]


def unsupported_dependency_rows() -> list[JsonDict]:
    """Return prerequisites that block direct reproduction or hardware claims."""

    return [deepcopy(row) for row in _UNSUPPORTED_DEPENDENCY_ROWS]


def non_claim_rows() -> list[JsonDict]:
    """Return claims that this source-ingestion task cannot support."""

    return [deepcopy(row) for row in _NON_CLAIM_ROWS]


def secondary_check_rows() -> list[JsonDict]:
    """Return advisory EBT, ARM-EBM, and Kona access receipts."""

    return [deepcopy(row) for row in _SECONDARY_CHECK_ROWS]


def reference_append_rows() -> list[JsonDict]:
    """Keep the ledger unchanged because no relevant post-marker delta was proved."""

    return [
        {
            "action": "no_change",
            "marker": PLANNER_MARKER,
            "appended": False,
            "reason": "No relevant primary or first-party artifact change was proved after the V614 marker.",
            "terminal": True,
        }
    ]


def _network_available() -> bool:
    """Probe one primary host; an HTTP response still proves network reachability."""

    request = Request("https://arxiv.org/abs/2609.00455", headers={"User-Agent": "Carnot/7011"})
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS precondition
            return int(getattr(response, "status", 0)) > 0
    except HTTPError:
        return True
    except (URLError, OSError):
        return False


def _writable(path: Path) -> bool:
    """Probe the output directory without touching the requested artifact path."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp7011-write-"):
            pass
        return True
    except OSError:
        return False


def _readable_nonempty(path: Path) -> bool:
    """Require an actual readable file so a directory cannot satisfy preflight."""

    try:
        return path.is_file() and bool(path.read_bytes())
    except OSError:
        return False


def check_preconditions(
    root: Path,
    output_path: Path,
    *,
    network_available: bool | None = None,
) -> list[JsonDict]:
    """Check the network, V614 marker, target sources, and output boundary."""

    network = _network_available() if network_available is None else bool(network_available)
    rows: list[JsonDict] = [
        {
            "check": "network_access",
            "expected_value": "available",
            "observed_value": "available" if network else "unavailable",
            "passed": network,
            "terminal": True,
        }
    ]
    reference = root / REFERENCE_PATH
    try:
        marker_present = reference.is_file() and REFERENCE_MARKER in reference.read_text(
            encoding="utf-8"
        )
    except OSError:
        marker_present = False
    rows.append(
        {
            "check": "v614_reference_marker",
            "expected_value": REFERENCE_MARKER,
            "observed_value": "present" if marker_present else "missing",
            "passed": marker_present,
            "terminal": True,
        }
    )
    for relative in REQUIRED_LOCAL_PATHS[1:]:
        readable = _readable_nonempty(root / relative)
        rows.append(
            {
                "check": f"readable_target:{relative.as_posix()}",
                "expected_value": "readable_nonempty_file",
                "observed_value": "readable_nonempty_file" if readable else "missing_or_unreadable",
                "passed": readable,
                "terminal": True,
            }
        )
    writable = _writable(output_path)
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


def _row_ledger(parts: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Combine evidence rows while retaining the source collection name."""

    return [{"row_kind": kind, **dict(row)} for kind, rows in parts.items() for row in rows]


def _method_coverage(artifact: Mapping[str, Any]) -> bool:
    """Require all accepted methods to have every bounded implementation row."""

    methods = artifact.get("method_rows", [])
    if not isinstance(methods, list):
        return False
    accepted = {
        row.get("method_id")
        for row in methods
        if isinstance(row, Mapping) and row.get("accepted") is True
    }
    if accepted != set(ACCEPTED_METHOD_IDS):
        return False
    for field in (
        "method_to_module_rows",
        "leakage_boundary_rows",
        "control_rows",
        "falsification_rows",
    ):
        rows = artifact.get(field, [])
        if not isinstance(rows, list):
            return False
        covered = {
            row.get("method_id")
            for row in rows
            if isinstance(row, Mapping) and row.get("method_id")
        }
        if covered != accepted:
            return False
    return True


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Recompute completion only from terminal unit rows and bounded mappings."""

    primaries = artifact.get("primary_source_rows", [])
    if not isinstance(primaries, list):
        return 0
    source_coverage = {
        row.get("source_id")
        for row in primaries
        if isinstance(row, Mapping)
        and row.get("terminal") is True
        and row.get("url")
        and row.get("publication_or_update_date")
        and row.get("access_outcome")
        and row.get("method_extraction")
    }
    repositories = artifact.get("repository_rows", [])
    identities = artifact.get("release_identity_rows", [])
    if not isinstance(repositories, list) or not isinstance(identities, list):
        return 0
    repository_ids = {
        row.get("repository_id")
        for row in repositories
        if isinstance(row, Mapping) and row.get("terminal") is True
    }
    identity_ids = {
        row.get("repository_id")
        for row in identities
        if isinstance(row, Mapping)
        and row.get("terminal") is True
        and (row.get("available") is False or bool(row.get("identity")))
    }
    complete = (
        source_coverage == set(SOURCE_IDS)
        and repository_ids == identity_ids
        and repository_ids == {row["repository_id"] for row in _REPOSITORY_ROWS}
        and _method_coverage(artifact)
    )
    return int(complete)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific rows while excluding runtime and command noise."""

    excluded = {"duration_s", "command_receipt_rows", "reproducibility_checksum"}
    payload = {key: value for key, value in artifact.items() if key not in excluded}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _command_state(artifact: Mapping[str, Any]) -> str:
    """Classify absent, passing, failed, and nonterminal validation receipts."""

    rows = artifact.get("command_receipt_rows", [])
    if rows == []:
        return "not_recorded"
    if not isinstance(rows, list):
        return "blocked"
    if [row.get("name") for row in rows if isinstance(row, Mapping)] != list(
        VALIDATION_COMMAND_NAMES
    ):
        return "blocked"
    if any(not isinstance(row, Mapping) or row.get("terminal") is not True for row in rows):
        return "blocked"
    if any(row.get("passed") is not True for row in rows):
        return "disqualified"
    return "pass"


def _empty_artifact(duration_s: float) -> JsonDict:
    """Create the complete blocked shape before preconditions are evaluated."""

    artifact: JsonDict = {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field
        not in {
            "field_principles",
            "inference_substrate",
            "duration_s",
            "v614_sota_ingestion_complete_score",
            "random_seed",
            "reproducibility_checksum",
            "gate_check_summary",
            "verifier_is_oracle",
            "verdict_class",
            "honest_verdict",
        }
    }
    artifact.update(
        {
            "field_principles": deepcopy(FIELD_PRINCIPLES),
            "inference_substrate": INFERENCE_SUBSTRATE,
            "duration_s": float(duration_s),
            "v614_sota_ingestion_complete_score": 0,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "gate_check_summary": {
                "failed_check": "preconditions",
                "expected_value": "all_pass",
                "observed_value": "not_checked",
                "passed": False,
            },
            "verifier_is_oracle": False,
            "verdict_class": "blocked",
            "honest_verdict": "blocked_v614_sota_ingestion",
        }
    )
    return artifact


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    network_available: bool | None = None,
    command_rows: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Build one positive or blocked deterministic V614 ingestion artifact."""

    started = time.monotonic()
    artifact = _empty_artifact(0.0)
    checks = check_preconditions(root, output_path, network_available=network_available)
    artifact["preconditions_checked"] = checks
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        artifact["gate_check_summary"] = {
            "failed_check": failed["check"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
            "passed": False,
        }
    else:
        parts = {
            "source_query": source_query_rows(),
            "primary_source": primary_source_rows(),
            "repository": repository_rows(),
            "release_identity": release_identity_rows(),
            "method": method_rows(),
            "method_to_module": method_to_module_rows(),
            "leakage_boundary": leakage_boundary_rows(),
            "control": control_rows(),
            "falsification": falsification_rows(),
            "unsupported_dependency": unsupported_dependency_rows(),
            "non_claim": non_claim_rows(),
            "secondary_check": secondary_check_rows(),
            "reference_append": reference_append_rows(),
        }
        for kind, rows in parts.items():
            artifact[f"{kind}_rows"] = rows
        artifact["rows"] = _row_ledger(parts)
        artifact["v614_sota_ingestion_complete_score"] = completion_score(artifact)
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete_positive_v614_sota_ingestion_mapped_no_post_marker_change"
        )
        artifact["gate_check_summary"] = {
            "failed_check": None,
            "expected_value": 1,
            "observed_value": artifact["v614_sota_ingestion_complete_score"],
            "passed": artifact["v614_sota_ingestion_complete_score"] == 1,
        }
    artifact["command_receipt_rows"] = (
        [dict(row) for row in command_rows] if command_rows is not None else []
    )
    if failed is None:
        command_state = _command_state(artifact)
        if command_state == "blocked":
            artifact["verdict_class"] = "blocked"
            artifact["honest_verdict"] = "blocked_v614_sota_ingestion"
            artifact["gate_check_summary"] = {
                "failed_check": "validation_commands",
                "expected_value": list(VALIDATION_COMMAND_NAMES),
                "observed_value": "nonterminal_or_missing",
                "passed": False,
            }
        elif command_state == "disqualified":
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "disqualified_v614_sota_ingestion_validation_failed"
            artifact["gate_check_summary"] = {
                "failed_check": "validation_commands",
                "expected_value": "all_pass",
                "observed_value": [
                    row.get("name")
                    for row in artifact["command_receipt_rows"]
                    if row.get("passed") is not True
                ],
                "passed": False,
            }
    artifact["duration_s"] = (
        float(duration_s) if duration_s is not None else round(time.monotonic() - started, 6)
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _verdict_errors(artifact: Mapping[str, Any], score: int) -> list[str]:
    """Require the class and phrase that follow from preconditions and score."""

    checks = artifact.get("preconditions_checked", [])
    blocked = not isinstance(checks, list) or any(
        not isinstance(row, Mapping) or row.get("passed") is not True for row in checks
    )
    command_state = _command_state(artifact)
    if blocked or command_state == "blocked":
        expected_class = "blocked"
    elif command_state == "disqualified" or score != 1:
        expected_class = "disqualified"
    else:
        expected_class = "positive"
    errors = []
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_inconsistent")
    verdict = str(artifact.get("honest_verdict", ""))
    prefix = {
        "blocked": "blocked_v614_sota_ingestion",
        "positive": "complete_positive_",
        "disqualified": "disqualified_v614_sota_ingestion",
    }[expected_class]
    if not verdict.startswith(prefix):
        errors.append("honest_verdict_prefix_inconsistent")
    return errors


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    """Independently validate source coverage, bounded maps, verdict, and hash."""

    if isinstance(value, Path):
        if not value.is_file():
            return ["artifact_missing"]
        try:
            loaded = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
        if not isinstance(loaded, Mapping):
            return ["artifact_not_object"]
        artifact: Mapping[str, Any] = loaded
    else:
        artifact = value
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]

    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_incomplete")
    elif any(not isinstance(value, str) or not value.strip() for value in principles.values()):
        errors.append("field_principles_empty")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")

    score = completion_score(artifact)
    if artifact.get("v614_sota_ingestion_complete_score") != score:
        errors.append("completion_score_mismatch")
    errors.extend(_verdict_errors(artifact, score))
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: Path) -> Path:
    """Validate before an atomic write so partial scientific rows never land."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return atomic_write_json(path, artifact, allow_override=False, sort_keys=True)


def _run_command(root: Path, name: str, argv: Sequence[str]) -> JsonDict:
    """Run one bounded verifier and preserve failures as terminal receipts."""

    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv),
            cwd=root,
            text=True,
            capture_output=True,
            timeout=7200,
            check=False,
        )
        terminal = True
        exit_code = result.returncode
        stdout = result.stdout[-4000:]
        stderr = result.stderr[-4000:]
        warning_only = False
        if name == "adversarial_verification" and exit_code == 1:
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError:
                report = {}
            reports = report.get("reports", []) if isinstance(report, Mapping) else []
            warning_only = bool(reports) and all(
                row.get("max_severity", 2) < 2 for row in reports if isinstance(row, Mapping)
            )
        if exit_code == 0:
            outcome = "pass"
        elif warning_only:
            outcome = "pass_with_warning"
        else:
            outcome = "contract_defect"
    except subprocess.TimeoutExpired as exc:
        terminal = True
        exit_code = None
        stdout = str(exc.stdout or "")[-4000:]
        stderr = str(exc.stderr or "")[-4000:]
        outcome = "tool_timeout"
    except OSError as exc:
        terminal = False
        exit_code = None
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        outcome = "tool_failure"
    return {
        "name": name,
        "command": " ".join(argv),
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_s": round(time.monotonic() - started, 6),
        "outcome": outcome,
        "terminal": terminal,
        "passed": exit_code == 0 or outcome == "pass_with_warning",
    }


def run_validation_commands(root: Path, artifact_path: Path) -> list[JsonDict]:
    """Run the focused, coverage, lint, artifact, and required full-suite checks."""

    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    coverage = str(root / ".venv/bin/coverage")
    ruff = str(root / ".venv/bin/ruff")
    module = MODULE_PATH.as_posix()
    test = TEST_PATH.as_posix()
    script = SCRIPT_PATH.as_posix()
    commands: tuple[tuple[str, list[str]], ...] = (
        ("focused_tests", [pytest, test, "-q", "--no-cov", "-n", "0"]),
        (
            "new_code_coverage_run",
            [
                coverage,
                "run",
                "--rcfile=/dev/null",
                f"--include={module}",
                "-m",
                "pytest",
                test,
                "-q",
                "--no-cov",
                "-n",
                "0",
            ],
        ),
        (
            "new_code_coverage_report",
            [
                coverage,
                "report",
                "--rcfile=/dev/null",
                f"--include={module}",
                "--fail-under=100",
            ],
        ),
        ("ruff_check", [ruff, "check", module, test, script]),
        ("ruff_format", [ruff, "format", "--check", module, test, script]),
        (
            "adversarial_verification",
            [python, "scripts/adversarial_verify.py", "--json", str(artifact_path)],
        ),
        (
            "row_consistency_lint",
            [python, "scripts/verdict_row_consistency_lint.py", str(artifact_path)],
        ),
        ("openspec_coverage", [python, "scripts/check_spec_coverage.py", test]),
        ("root_clutter_check", [python, "scripts/root_clutter_sweep.py"]),
        ("full_python_tests", [pytest, "tests/python", "-q"]),
    )
    return [_run_command(root, name, argv) for name, argv in commands]


def _date_argument(value: str) -> bool:
    """Accept only the exact compact calendar format used by experiment tasks."""

    try:
        return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """Write a preliminary receipt, run checks, then write the final receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        return int(bool(validate_artifact(args.validate)))
    if not _date_argument(args.date):
        return 2

    root = find_repo_root(start=__file__)
    output = args.output or (root / RESULT_PATH)
    started = time.monotonic()
    preliminary = build_artifact(root, args.date, output_path=output)
    write_artifact(preliminary, output)
    external = run_validation_commands(root, output)
    artifact_validation = {
        "name": "artifact_validation",
        "command": f"{root / '.venv/bin/python'} {SCRIPT_PATH} --validate {output}",
        "exit_code": int(bool(validate_artifact(preliminary))),
        "stdout": "artifact valid" if not validate_artifact(preliminary) else "artifact invalid",
        "stderr": "",
        "duration_s": 0.0,
        "outcome": "pass" if not validate_artifact(preliminary) else "contract_defect",
        "terminal": True,
        "passed": not validate_artifact(preliminary),
    }
    final = build_artifact(
        root,
        args.date,
        output_path=output,
        network_available=all(
            row.get("passed") is True
            for row in preliminary["preconditions_checked"]
            if row.get("check") == "network_access"
        ),
        command_rows=[*external, artifact_validation],
        duration_s=round(time.monotonic() - started, 6),
    )
    write_artifact(final, output)
    return int(bool(validate_artifact(final)) or final["verdict_class"] != "positive")


if __name__ == "__main__":
    raise SystemExit(main())
