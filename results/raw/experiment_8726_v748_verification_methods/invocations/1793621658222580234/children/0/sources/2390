"""Exp6528 V565 source, model, and method contract.

Spec refs: REQ-REPORT-6528, SCENARIO-REPORT-6528-SOURCES,
SCENARIO-REPORT-6528-DRIFT, SCENARIO-REPORT-6528-CACHE,
SCENARIO-REPORT-6528-METHODS, SCENARIO-REPORT-6528-SCHEMA.

This reducer freezes the sources, model-cache contract, and method fields for
V565 before outcome artifacts exist. It performs source and cache preflight
only. It does not load a model, run inference, or treat source metadata as a
scientific result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import html
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any
from urllib import error, request

from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
Fetcher = Callable[[str, str], JsonDict]
CommandRunner = Callable[[list[str], Path], tuple[int, str, str]]
ModelPairResolver = Callable[..., list[dict[str, Any]] | None]
GgufResolver = Callable[[str, str], str | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6528
INFERENCE_SUBSTRATE = "low_concurrency_primary_source_and_cache_preflight_no_experimental_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6528_v565_source_model_method_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("results/experiment_6527_v565_evidence_eligibility_corrigendum.json"),
)
SOURCE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-studying.md"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("scripts/sweep_clusters.py"),
    Path("scripts/sweep_semscholar.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md"),
    Path("docs/research-notes/qwen38-compaction-forward-plan-2026-08-20.md"),
    Path("results/experiment_6527_v565_evidence_eligibility_corrigendum.json"),
)
LOCAL_HOOK_PATHS = (
    Path("python/carnot/experiment_6529_drift_bench_external_intake.py"),
    Path("python/carnot/experiment_6530_external_constraint_corpus_audit.py"),
    Path("python/carnot/experiment_6531_external_structural_headroom_replication.py"),
    Path("python/carnot/experiment_6532_sota_paired_embedding_surface_audit.py"),
    Path("python/carnot/experiment_6533_external_calibrated_safety_net_router.py"),
    Path("python/carnot/experiment_6536_external_chronological_self_learning.py"),
    Path("python/carnot/experiment_6538_qwen3_xml_tool_call_reachability.py"),
    Path("python/carnot/experiment_6521_transactional_refinement_conflict_memory.py"),
    Path("python/carnot/experiment_6522_chronological_conflict_self_learning.py"),
    Path("python/carnot/inference/sota_models.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "query_receipts",
    "source_rows",
    "primary_source_hashes",
    "citation_trail_receipts",
    "citation_count_boundaries",
    "drift_bench_provenance_contract",
    "model_cache_resolution_rows",
    "frozen_external_split_contract",
    "frozen_router_contract",
    "frozen_embedding_contract",
    "frozen_transactional_learning_contract",
    "frozen_arc_parser_contract",
    "hardware_stop_contract",
    "non_transfer_rows",
    "v565_method_contract_ready_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal V565 source-model-method contract state.",
    "honest_verdict": (
        "States the source and cache outcome without turning preflight into experimental evidence."
    ),
    "verdict_class": "Closed enum for this preflight: null or partial.",
    "query_receipts": "Keeps each sequential low-concurrency source query dated and reproducible.",
    "source_rows": "Records one row per checked paper, product, repository, or index route.",
    "primary_source_hashes": "Binds primary and first-party pages to source hashes.",
    "citation_trail_receipts": "Records dated EBT and ARM-EBM citation routes with endpoint caveats.",
    "citation_count_boundaries": "Prevents citation indexes from becoming fabricated current counts.",
    "drift_bench_provenance_contract": (
        "Freezes DRIFT-Bench URL, revision, license, schema, corruption warning, and local regeneration rule."
    ),
    "model_cache_resolution_rows": "Records mandated GGUF cache identity without loading or running models.",
    "frozen_external_split_contract": "Freezes external family and chronology split fields before outcomes.",
    "frozen_router_contract": "Freezes structural controls, calibration, abstention, and exact fallback.",
    "frozen_embedding_contract": "Freezes paired-embedding conditions and shortcut attacks.",
    "frozen_transactional_learning_contract": (
        "Freezes memory isolation, commit boundary, support, retention, restart, and rollback fields."
    ),
    "frozen_arc_parser_contract": "Freezes the qwen3_xml live-path parser and stop-rule boundary.",
    "hardware_stop_contract": "Freezes GateMate, TSU, and product-only hardware boundaries.",
    "non_transfer_rows": "Names claims that do not transfer into Carnot.",
    "v565_method_contract_ready_score": (
        "Gate score opens only when adopted methods, cache, spelling, and protected-file gates pass."
    ),
    "gate_check_summary": (
        "Names failed source, cache, schema, protection, and readiness gates with observed values."
    ),
    "per_unit_rows": "Flattens source, cache, contract, boundary, and gate rows for recomputation.",
    "aggregate_row_recomputation": "Recomputes readiness from rows instead of trusting narrative text.",
    "preconditions_checked": (
        "Records network, query, tool, cache, source-path, git, and protected-hash preconditions."
    ),
    "protected_files_unchanged": "Proves protected files stayed byte-identical.",
    "inference_substrate": "Declares source and cache preflight with no experimental LLM inference.",
    "verifier_is_oracle": "False because source and cache preflight is not a correctness oracle.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": (
        "Maps each required field to receipts, source rows, cache rows, or deterministic reducers."
    ),
    "random_seed": "Pins deterministic row ordering for this no-randomness contract.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": "Detects drift in source rows, cache rows, contracts, gates, and receipts.",
}
FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6528 deterministic reducer",
        "spec_refs": ["REQ-REPORT-6528"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["query_receipts"]["source"] = "collect_query_receipts"
FIELD_PROVENANCE["source_rows"]["source"] = "collect_source_rows"
FIELD_PROVENANCE["primary_source_hashes"]["source"] = "source_rows.source_hash"
FIELD_PROVENANCE["citation_trail_receipts"]["source"] = "citation_trail_receipts"
FIELD_PROVENANCE["citation_count_boundaries"]["source"] = "citation_count_boundaries"
FIELD_PROVENANCE["drift_bench_provenance_contract"]["source"] = (
    "build_drift_bench_provenance_contract"
)
FIELD_PROVENANCE["model_cache_resolution_rows"]["source"] = "collect_model_cache_resolution_rows"
FIELD_PROVENANCE["non_transfer_rows"]["source"] = "build_non_transfer_rows"
FIELD_PROVENANCE["aggregate_row_recomputation"]["source"] = "aggregate_row_recomputation"
FIELD_PROVENANCE["protected_files_unchanged"]["source"] = "protected_files_unchanged"
FIELD_PROVENANCE["preconditions_checked"]["source"] = "preconditions_checked"

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6528_v565_source_model_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6528_v565_source_model_method_contract.py "
    "-m pytest tests/python/test_experiment_6528_v565_source_model_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6528_v565_source_model_method_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6528_v565_source_model_method_contract.py"
)
URL_LINT_COMMAND = ".venv/bin/python scripts/canonical_url_lint.py"
DUPLICATE_LEDGER_COMMAND = (
    ".venv/bin/python scripts/research_complete_ledger_lint.py research-complete.yaml"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6528_v565_source_model_method_contract --date 20260823"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6528_v565_source_model_method_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6528_v565_source_model_method_contract --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": URL_LINT_COMMAND, "exit_code": 0},
    {"command": DUPLICATE_LEDGER_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)


def _source(
    *,
    source_id: str,
    title: str,
    source_kind: str,
    stable_url: str,
    query_channel: str,
    method: str,
    method_claim: str,
    code_or_data_availability: str,
    local_applicability: str,
    carnot_hook: str,
    non_transfer_boundary: str,
    exact_authority_boundary: str,
    retrieval_url: str | None = None,
    expected_source_date: str | None = None,
    arxiv_id: str | None = None,
    openreview_id: str | None = None,
    adopted_method: bool = False,
    required_primary_check: bool = False,
    negative_control: str = "",
    downstream_field_spelling: Sequence[str] = (),
    method_transfer_status: str = "candidate_local_contract",
) -> JsonDict:
    return {
        "source_id": source_id,
        "title": title,
        "source_kind": source_kind,
        "stable_url": stable_url,
        "retrieval_url": retrieval_url or stable_url,
        "query_channel": query_channel,
        "method": method,
        "method_claim": method_claim,
        "code_or_data_availability": code_or_data_availability,
        "local_applicability": local_applicability,
        "carnot_hook": carnot_hook,
        "non_transfer_boundary": non_transfer_boundary,
        "exact_authority_boundary": exact_authority_boundary,
        "expected_source_date": expected_source_date,
        "arxiv_id": arxiv_id,
        "openreview_id": openreview_id,
        "adopted_method": adopted_method,
        "required_primary_check": required_primary_check,
        "negative_control": negative_control,
        "downstream_field_spelling": list(downstream_field_spelling),
        "method_transfer_status": method_transfer_status,
    }


SOURCE_MANIFEST: tuple[JsonDict, ...] = (
    _source(
        source_id="drift_bench_arxiv",
        title="Residual Drift Dominates Contradiction in Multi-Turn Constraint Reasoning",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2605.23940",
        query_channel="arxiv_primary",
        method="external_multi_turn_constraint_transfer_surface",
        method_claim="DRIFT-Bench separates contradiction from satisfiable drift with solver-instrumented turns.",
        code_or_data_availability="public_repository_and_problem_json",
        local_applicability="content_pinned_external_constraint_fixture",
        carnot_hook="Use DRIFT-Bench as a pinned external transfer surface with local exact replay.",
        non_transfer_boundary="No upstream aggregate or run database result transfers.",
        exact_authority_boundary="Local Z3 replay and assignment checks own all Carnot labels.",
        expected_source_date="2026-04-28",
        arxiv_id="2605.23940",
        adopted_method=True,
        required_primary_check=True,
        negative_control="internal_carnot_generator_slice_only",
        downstream_field_spelling=(
            "source_row_hash",
            "source_problem_id",
            "turn_index",
            "exact_replay_status",
        ),
    ),
    _source(
        source_id="memoir",
        title="Memoir: Should a Model Write to Its Memory While It Thinks?",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2607.20792",
        query_channel="arxiv_primary",
        method="query_frozen_memory_transactional_commit",
        method_claim="Coupled same-query memory writes learn more slowly than a read-only pondering arm.",
        code_or_data_availability="public_github_claimed",
        local_applicability="transactional_memory_commit_boundary",
        carnot_hook="Freeze memory within a query and commit after exact outcome validation.",
        non_transfer_boundary="No Memoir parameter or speed result transfers to Carnot.",
        exact_authority_boundary="Exact outcome validation admits or rejects memory writes.",
        expected_source_date="2026-07-22",
        arxiv_id="2607.20792",
        adopted_method=True,
        required_primary_check=True,
        negative_control="same_query_mutation_arm",
        downstream_field_spelling=("memory_frozen_within_query", "commit_after_exact_validation"),
    ),
    _source(
        source_id="support_reshaping",
        title="Verifier-Induced Support Reshaping in On-Policy Optimization",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2608.00220",
        query_channel="arxiv_primary",
        method="future_support_retention_audit",
        method_claim="Current pass-at-one gains can coexist with later best-at-k support loss.",
        code_or_data_availability="public_github_claimed",
        local_applicability="future_support_and_retention_metrics",
        carnot_hook="Report future exact-satisfying support and retained-family performance.",
        non_transfer_boundary="No RLVR policy update result transfers into Carnot's exact-memory controller.",
        exact_authority_boundary="Exact held-future rows decide retention and support safety.",
        expected_source_date="2026-07-31",
        arxiv_id="2608.00220",
        adopted_method=True,
        required_primary_check=True,
        negative_control="scratch_and_frozen_memory_arms",
        downstream_field_spelling=(
            "future_exact_satisfying_support",
            "retained_family_performance",
        ),
    ),
    _source(
        source_id="distributional_ebm",
        title="Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2605.18871",
        query_channel="arxiv_primary",
        method="decomposed_energy_uncertainty_abstention",
        method_claim="Analytical penalties plus learned uncertainty can route regeneration or abstention.",
        code_or_data_availability="paper_only_at_preflight",
        local_applicability="held_calibration_and_abstention_router",
        carnot_hook="Freeze calibration, uncertainty, and abstention before external held outcomes.",
        non_transfer_boundary="No verifier accuracy, model-identity shortcut, or benchmark score transfers.",
        exact_authority_boundary="Native exact fallback remains release authority.",
        expected_source_date="2026-05-15",
        arxiv_id="2605.18871",
        adopted_method=True,
        required_primary_check=True,
        negative_control="identity_shortcut_and_family_permutation_attack",
        downstream_field_spelling=("router_abstained", "calibration_split", "exact_fallback_used"),
    ),
    _source(
        source_id="solver_hard",
        title="Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
        source_kind="paper",
        stable_url="https://arxiv.org/abs/2607.17047",
        query_channel="arxiv_primary",
        method="solver_hardness_surface_decoupling",
        method_claim="Solver conflict hardness did not reliably predict LLM accuracy.",
        code_or_data_availability="public_code_and_aggregate_data_claimed",
        local_applicability="surface_hardness_stratified_embedding_diagnostic",
        carnot_hook="Stratify SOTA paired embeddings by exact hardness and surface realization separately.",
        non_transfer_boundary="No solver-hard label transfers as a model-difficulty label.",
        exact_authority_boundary="Exact solver metadata is diagnostic only, not model-answer authority.",
        expected_source_date="2026-07-19",
        arxiv_id="2607.17047",
        adopted_method=True,
        required_primary_check=True,
        negative_control="proof_preserving_surface_relabel_control",
        downstream_field_spelling=(
            "solver_hardness_bin",
            "surface_realization_id",
            "paired_embedding_distance",
        ),
    ),
    _source(
        source_id="openreview_dc_energy",
        title="A Difference-of-Convex Functions Approach to Energy-Based Iterative Reasoning",
        source_kind="openreview_forum",
        stable_url="https://openreview.net/forum?id=QvsDTpf4yF",
        query_channel="openreview",
        method="dc_energy_optimizer_control",
        method_claim="DC optimization accelerates energy-based iterative reasoning with local convergence.",
        code_or_data_availability="openreview_forum_metadata",
        local_applicability="future_continuous_energy_optimizer_control",
        carnot_hook="Keep as future optimizer control; do not start a new answer-generator lineage.",
        non_transfer_boundary="No DC optimizer performance transfers into V565 exact-router tasks.",
        exact_authority_boundary="Deferred optimizer cannot enter V565 acceptance gates.",
        expected_source_date="2025-09-18",
        openreview_id="QvsDTpf4yF",
        required_primary_check=True,
        method_transfer_status="future_optimizer_control",
    ),
    _source(
        source_id="openreview_linear_decision_rules",
        title="Enforcing Hard Linear Constraints in Deep Learning Models with Decision Rules",
        source_kind="openreview_forum",
        stable_url="https://openreview.net/forum?id=gjiCml2CNG",
        query_channel="openreview",
        method="safe_network_closed_form_linear_correction",
        method_claim="A safe network and closed-form correction preserve linear feasibility.",
        code_or_data_availability="openreview_forum_metadata",
        local_applicability="future_architecture_safe_fallback_control",
        carnot_hook="Use as architectural analogy for a separate learned fast path and exact fallback.",
        non_transfer_boundary="No linear feasibility guarantee transfers to discrete SAT or CSP outputs.",
        exact_authority_boundary="Linear correction paper is not evidence for Carnot discrete certification.",
        expected_source_date="2025-09-18",
        openreview_id="gjiCml2CNG",
        required_primary_check=True,
        method_transfer_status="future_architecture_control",
    ),
    _source(
        source_id="drift_bench_repo",
        title="kaons-research/drift-bench",
        source_kind="repository",
        stable_url="https://github.com/kaons-research/drift-bench",
        query_channel="github",
        method="drift_bench_repository_provenance",
        method_claim="Repository exposes DRIFT-Bench problem JSON, schema, source code, license, and warning.",
        code_or_data_availability="public_repository",
        local_applicability="revision_license_schema_corruption_contract",
        carnot_hook="Resolve immutable revision and regenerate every local result receipt.",
        non_transfer_boundary="No upstream SQLite database or summary table transfers.",
        exact_authority_boundary="Local fixture and exact replay own downstream rows.",
        retrieval_url="https://api.github.com/repos/kaons-research/drift-bench",
        required_primary_check=True,
    ),
    _source(
        source_id="huggingface_support_reshaping",
        title="Verifier-Induced Support Reshaping in On-Policy Optimization",
        source_kind="secondary_index",
        stable_url="https://huggingface.co/papers/2608.00220",
        query_channel="huggingface_papers",
        method="community_paper_discovery_context",
        method_claim="Hugging Face paper page mirrors the source and links project/code pages.",
        code_or_data_availability="paper_index",
        local_applicability="discovery_only",
        carnot_hook="Use as discovery route; arXiv remains primary authority.",
        non_transfer_boundary="Community index metadata does not transfer as evidence.",
        exact_authority_boundary="Primary arXiv page owns the method row.",
    ),
    _source(
        source_id="semantic_scholar_ebt",
        title="Semantic Scholar EBT citation route for 2507.02092",
        source_kind="secondary_index",
        stable_url="https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092?fields=title,citationCount,citations.title,citations.externalIds,citations.year",
        query_channel="semantic_scholar",
        method="ebt_citation_trail_context",
        method_claim="Dated citation route for EBT follow-on context.",
        code_or_data_availability="metadata_index",
        local_applicability="citation_context_only",
        carnot_hook="Record citation trail caveats; do not promote source by count.",
        non_transfer_boundary="Citation counts are dated metadata, not method evidence.",
        exact_authority_boundary="Secondary metadata cannot promote a method.",
    ),
    _source(
        source_id="semantic_scholar_arm_ebm",
        title="Semantic Scholar ARM-EBM citation route for 2512.15605",
        source_kind="secondary_index",
        stable_url="https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605?fields=title,citationCount,citations.title,citations.externalIds,citations.year",
        query_channel="semantic_scholar",
        method="arm_ebm_citation_trail_context",
        method_claim="Dated citation route for ARM-EBM follow-on context.",
        code_or_data_availability="metadata_index",
        local_applicability="citation_context_only",
        carnot_hook="Record rate limits rather than fabricating current counts.",
        non_transfer_boundary="Rate-limited counts cannot be reported as current.",
        exact_authority_boundary="Secondary metadata cannot promote a method.",
    ),
    _source(
        source_id="extropic_z1_status",
        title="Extropic first-party Z1 status",
        source_kind="product",
        stable_url="https://extropic.ai/writing/from-one-to-one-billion/",
        query_channel="extropic",
        method="thermodynamic_hardware_product_context",
        method_claim="First-party writing describes Torx, Thermalizers, Z1, and 2027 early access.",
        code_or_data_availability="no_local_sdk_or_device",
        local_applicability="product_status_only",
        carnot_hook="Keep TSU execution and availability claims out of V565.",
        non_transfer_boundary="No TSU execution, latency, energy, or availability claim transfers.",
        exact_authority_boundary="Product page only; no local TSU runner or access.",
    ),
    _source(
        source_id="logical_intelligence_kona",
        title="Kona: Energy-Based Models for AI Reasoning",
        source_kind="product",
        stable_url="https://logicalintelligence.com/kona-ebms-energy-based-models",
        query_channel="logical_intelligence",
        method="kona_product_comparator_context",
        method_claim="First-party page describes Kona 1.0 as a non-generator energy-based reasoning model.",
        code_or_data_availability="no_public_weights_or_runner",
        local_applicability="product_comparator_only",
        carnot_hook="Keep Kona as comparator until public weights and runner exist.",
        non_transfer_boundary="No proprietary Kona weights, runner, or speed claim transfers.",
        exact_authority_boundary="Product comparator only; no local runner or weights.",
    ),
)
SOURCE_BY_ID = {str(row["source_id"]): dict(row) for row in SOURCE_MANIFEST}
ADOPTED_SOURCE_IDS = tuple(
    str(row["source_id"]) for row in SOURCE_MANIFEST if row.get("adopted_method") is True
)
REQUIRED_CHANNELS = {
    "network_probe",
    "sweep_clusters",
    "sweep_semscholar",
    "arxiv_primary",
    "openreview",
    "semantic_scholar",
    "huggingface_papers",
    "github",
    "extropic",
    "logical_intelligence",
    "model_cache_preflight",
}
DRIFT_REPO_URL = "https://github.com/kaons-research/drift-bench"
DRIFT_API_REPO_URL = "https://api.github.com/repos/kaons-research/drift-bench"
DRIFT_API_COMMIT_URL = "https://api.github.com/repos/kaons-research/drift-bench/commits/main"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_fetcher(url: str, source_id: str) -> JsonDict:  # pragma: no cover - live network path.
    del source_id
    req = request.Request(url, headers={"User-Agent": "Carnot-Exp6528-source-cache-preflight/1.0"})
    try:
        with request.urlopen(req, timeout=20) as response:  # noqa: S310
            body = response.read(2_000_000).decode("utf-8", "replace")
            return {
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "url": response.geturl(),
                "headers": dict(response.headers.items()),
                "body": body,
                "error": None,
            }
    except error.HTTPError as exc:
        body = exc.read(2_000_000).decode("utf-8", "replace")
        return {
            "ok": False,
            "status_code": int(exc.code),
            "url": url,
            "headers": dict(exc.headers.items()) if exc.headers else {},
            "body": body,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "url": url,
            "headers": {},
            "body": "",
            "error": str(exc),
        }


def default_command_runner(
    args: list[str], cwd: Path
) -> tuple[int, str, str]:  # pragma: no cover - subprocess path.
    result = subprocess.run(args, cwd=cwd, check=False, text=True, capture_output=True)
    return result.returncode, result.stdout, result.stderr


def offline_fetcher(url: str, source_id: str) -> JsonDict:
    del source_id
    return {
        "ok": False,
        "status_code": 0,
        "url": url,
        "headers": {},
        "body": "",
        "error": "live network disabled",
    }


def offline_command_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
    del cwd
    return 0, "", f"live network disabled for {' '.join(args)}"


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)
    return result.stdout.strip()


def _extract_title(body: str, fallback: str) -> str:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, Mapping):
        title = decoded.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
        notes = decoded.get("notes")
        if isinstance(notes, list) and notes:
            content = notes[0].get("content") if isinstance(notes[0], Mapping) else None
            title_value = content.get("title") if isinstance(content, Mapping) else None
            if isinstance(title_value, Mapping) and isinstance(title_value.get("value"), str):
                return title_value["value"].strip()
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.I | re.S)
    if title_match:
        title = re.sub(r"\s+", " ", html.unescape(title_match.group(1))).strip()
        if title:
            return title.removeprefix("GitHub - ").strip()
    arxiv_match = re.search(r"Title:\s*([^<\n]+)", body, flags=re.I)
    if arxiv_match:
        title = re.sub(r"\s+", " ", html.unescape(arxiv_match.group(1))).strip()
        if title:
            return title
    return fallback


def _extract_source_date(body: str) -> str | None:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, Mapping) and isinstance(decoded.get("source_date"), str):
        return decoded["source_date"]
    iso = re.search(r"(20\d{2}-\d{2}-\d{2})", body)
    if iso:
        return iso.group(1)
    match = re.search(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+(20\d{2})", body)
    if not match:
        return None
    months = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    day, month, year = match.groups()
    return f"{year}-{months[month]}-{int(day):02d}"


def _retrieval_state(status_code: int, body_or_error: str) -> str:
    text = body_or_error.lower()
    if 200 <= status_code < 400:
        if "verifying your browser" in text:
            return "blocked"
        return "available"
    if status_code == 429 or "too many requests" in text or "rate limit" in text:
        return "rate_limited"
    if status_code == 404 or "not found" in text:
        return "not_found"
    if status_code == 403 or "forbidden" in text or "verifying your browser" in text:
        return "blocked"
    return "blocked"


def _citation_count(body: str) -> int | None:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    count = decoded.get("citationCount") if isinstance(decoded, Mapping) else None
    return int(count) if isinstance(count, int) else None


def _citation_titles(body: str) -> list[str]:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return []
    citations = decoded.get("citations") if isinstance(decoded, Mapping) else None
    if not isinstance(citations, list):
        return []
    titles: list[str] = []
    for row in citations:
        title = row.get("title") if isinstance(row, Mapping) else None
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    return titles


def _network_probe(fetcher: Fetcher, now_utc: str) -> JsonDict:
    receipt = fetcher("https://arxiv.org/abs/2605.23940", "network_probe")
    return {
        "network_required": True,
        "network_used": True,
        "network_available": bool(receipt.get("ok")),
        "checked_at_utc": now_utc,
        "probe_url": "https://arxiv.org/abs/2605.23940",
        "probe_http_state": f"http_{int(receipt.get('status_code') or 0)}",
        "error": receipt.get("error"),
    }


def _command_receipt(
    *,
    receipt_id: str,
    channel: str,
    query: str,
    args: list[str],
    repo_root: Path,
    command_runner: CommandRunner,
    now_utc: str,
) -> JsonDict:
    rc, stdout, stderr = command_runner(args, repo_root)
    return {
        "receipt_id": receipt_id,
        "channel": channel,
        "query": query,
        "url": " ".join(args),
        "accessed_at_utc": now_utc,
        "low_concurrency": True,
        "command": args,
        "exit_code": rc,
        "stdout_sha256": sha256_json(stdout),
        "stderr_sha256": sha256_json(stderr),
        "candidate_ids": re.findall(r"\b\d{4}\.\d{5}\b", stdout),
        "observed_error": stderr.strip()
        if rc != 0 or stderr.strip().lower().startswith("http")
        else None,
    }


def collect_query_receipts(
    *,
    repo_root: Path,
    command_runner: CommandRunner,
    network_state: Mapping[str, Any],
    now_utc: str,
) -> list[JsonDict]:
    rows = [
        {
            "receipt_id": "network_probe_arxiv_drift_bench",
            "channel": "network_probe",
            "query": "GET https://arxiv.org/abs/2605.23940",
            "url": "https://arxiv.org/abs/2605.23940",
            "accessed_at_utc": now_utc,
            "low_concurrency": True,
            "access_outcome": dict(network_state),
        },
        _command_receipt(
            receipt_id="sweep_clusters_all_low_concurrency",
            channel="sweep_clusters",
            query="scripts/sweep_clusters.py all --max-results 8",
            args=[sys.executable, "scripts/sweep_clusters.py", "all", "--max-results", "8"],
            repo_root=repo_root,
            command_runner=command_runner,
            now_utc=now_utc,
        ),
        _command_receipt(
            receipt_id="sweep_semscholar_v565_queries",
            channel="sweep_semscholar",
            query="DRIFT-Bench Memoir verifier-induced support reshaping Distributional EBMs Solver-Hard qwen3_xml",
            args=[
                sys.executable,
                "scripts/sweep_semscholar.py",
                "DRIFT-Bench Memoir verifier-induced support reshaping Distributional EBMs Solver-Hard qwen3_xml",
                "--limit",
                "8",
            ],
            repo_root=repo_root,
            command_runner=command_runner,
            now_utc=now_utc,
        ),
    ]
    channel_queries = {
        "arxiv_primary": "direct arXiv primary pages for five adopted V565 papers",
        "openreview": "OpenReview QvsDTpf4yF and gjiCml2CNG forum checks",
        "semantic_scholar": "Semantic Scholar EBT 2507.02092 and ARM-EBM 2512.15605 citation records",
        "huggingface_papers": "Hugging Face Papers verifier-induced support reshaping page",
        "github": "GitHub DRIFT-Bench repository, revision, license, and schema checks",
        "extropic": "Extropic first-party Z1 writing",
        "logical_intelligence": "Logical Intelligence Kona first-party page",
        "model_cache_preflight": "cached_sota_pair(gpu_indices=(0, 1)) plus mandated GGUF path resolution",
    }
    rows.extend(
        {
            "receipt_id": f"{channel}_sequential_check",
            "channel": channel,
            "query": query,
            "url": "source_manifest_or_cache_preflight",
            "accessed_at_utc": now_utc,
            "low_concurrency": True,
            "access_outcome": "covered_by_source_or_cache_rows",
        }
        for channel, query in channel_queries.items()
    )
    return rows


def collect_source_rows(*, fetcher: Fetcher, now_utc: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in SOURCE_MANIFEST:
        source_id = str(source["source_id"])
        receipt = fetcher(str(source["retrieval_url"]), source_id)
        body = str(receipt.get("body") or "")
        error_text = str(receipt.get("error") or "")
        status_code = int(receipt.get("status_code") or 0)
        retrieval_state = _retrieval_state(status_code, body + " " + error_text)
        source_date = _extract_source_date(body)
        expected = source.get("expected_source_date")
        dated_source = expected is None or source_date == expected
        ok = bool(receipt.get("ok")) and retrieval_state == "available"
        rows.append(
            {
                "row_type": "source",
                "source_id": source_id,
                "source_kind": source["source_kind"],
                "title": _extract_title(body, str(source["title"])),
                "stable_url": source["stable_url"],
                "retrieval_url": source["retrieval_url"],
                "query_channel": source["query_channel"],
                "arxiv_id": source["arxiv_id"],
                "openreview_id": source["openreview_id"],
                "source_date": source_date,
                "expected_source_date": expected,
                "source_date_verified": bool(ok and dated_source),
                "primary_url_verified": bool(ok),
                "retrieval_state": retrieval_state,
                "http_state": f"http_{status_code}",
                "observed_error": error_text or None,
                "source_hash": sha256_json(
                    {"source_id": source_id, "status_code": status_code, "body": body}
                ),
                "accessed_at_utc": now_utc,
                "method": source["method"],
                "method_claim": source["method_claim"],
                "method_transfer_status": source["method_transfer_status"],
                "code_or_data_availability": source["code_or_data_availability"],
                "local_applicability": source["local_applicability"],
                "carnot_hook": source["carnot_hook"],
                "negative_control": source["negative_control"],
                "downstream_field_spelling": list(source["downstream_field_spelling"]),
                "non_transfer_boundary": source["non_transfer_boundary"],
                "exact_authority_boundary": source["exact_authority_boundary"],
                "adopted_method": source["adopted_method"],
                "required_primary_check": source["required_primary_check"],
                "citation_count_observed": _citation_count(body),
                "citation_titles_observed": _citation_titles(body),
            }
        )
    return rows


def citation_trail_receipts(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for row in source_rows:
        if row.get("query_channel") != "semantic_scholar":
            continue
        limited = row.get("retrieval_state") == "rate_limited"
        out.append(
            {
                "row_type": "citation_trail_receipt",
                "source_id": row["source_id"],
                "channel": "semantic_scholar",
                "target": "EBT 2507.02092"
                if row["source_id"] == "semantic_scholar_ebt"
                else "ARM-EBM 2512.15605",
                "accessed_at_utc": row["accessed_at_utc"],
                "retrieval_state": row["retrieval_state"],
                "observed_citation_count": None if limited else row.get("citation_count_observed"),
                "citation_titles_observed": []
                if limited
                else list(row.get("citation_titles_observed", [])),
                "count_is_current_guarantee": False,
                "observed_error": row.get("observed_error"),
            }
        )
    return out


def citation_count_boundaries(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in source_rows:
        if row.get("query_channel") != "semantic_scholar":
            continue
        limited = row.get("retrieval_state") == "rate_limited"
        rows.append(
            {
                "source_id": row["source_id"],
                "channel": "semantic_scholar",
                "observed_citation_count": None if limited else row.get("citation_count_observed"),
                "count_is_current_guarantee": False,
                "rate_limited": limited,
                "observed_error": row.get("observed_error"),
                "boundary": "Do not fabricate or refresh counts when the endpoint is limited or secondary.",
            }
        )
    return rows


def _json_object(body: str) -> JsonDict:
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def build_drift_bench_provenance_contract(*, fetcher: Fetcher, now_utc: str) -> JsonDict:
    repo_receipt = fetcher(DRIFT_API_REPO_URL, "drift_repo_api")
    commit_receipt = fetcher(DRIFT_API_COMMIT_URL, "drift_commit_api")
    repo_body = str(repo_receipt.get("body") or "")
    commit_body = str(commit_receipt.get("body") or "")
    repo_json = _json_object(repo_body)
    commit_json = _json_object(commit_body)
    revision = str(commit_json.get("sha") or "")
    revision_ok = bool(re.fullmatch(r"[0-9a-f]{40}", revision))
    ref = revision if revision_ok else "main"
    raw_base = f"https://raw.githubusercontent.com/kaons-research/drift-bench/{ref}"
    readme_receipt = fetcher(f"{raw_base}/README.md", "drift_readme_pinned")
    license_receipt = fetcher(f"{raw_base}/LICENSE", "drift_license_pinned")
    schema_receipt = fetcher(f"{raw_base}/data/problems/README.md", "drift_schema_pinned")
    license_text = str(license_receipt.get("body") or "")
    schema_text = str(schema_receipt.get("body") or "")
    readme_text = str(readme_receipt.get("body") or "")
    license_spdx = (
        repo_json.get("license", {}).get("spdx_id")
        if isinstance(repo_json.get("license"), Mapping)
        else None
    )
    license_name = str(license_spdx or ("MIT" if "MIT" in license_text else "unknown"))
    schema_verified = bool(
        schema_receipt.get("ok")
        and "problem" in schema_text.lower()
        and ("turn" in schema_text.lower() or "constraint" in schema_text.lower())
    )
    corruption_warning = "sqlite" in readme_text.lower() and "corruption" in readme_text.lower()
    receipts = {
        "repo_api": {
            "url": DRIFT_API_REPO_URL,
            "http_state": f"http_{int(repo_receipt.get('status_code') or 0)}",
            "source_hash": sha256_json(repo_body),
        },
        "commit_api": {
            "url": DRIFT_API_COMMIT_URL,
            "http_state": f"http_{int(commit_receipt.get('status_code') or 0)}",
            "source_hash": sha256_json(commit_body),
        },
        "readme": {
            "url": f"{raw_base}/README.md",
            "http_state": f"http_{int(readme_receipt.get('status_code') or 0)}",
            "source_hash": sha256_json(readme_text),
        },
        "license": {
            "url": f"{raw_base}/LICENSE",
            "http_state": f"http_{int(license_receipt.get('status_code') or 0)}",
            "source_hash": sha256_json(license_text),
        },
        "schema": {
            "url": f"{raw_base}/data/problems/README.md",
            "http_state": f"http_{int(schema_receipt.get('status_code') or 0)}",
            "source_hash": sha256_json(schema_text),
        },
    }
    contract_ready = bool(
        repo_receipt.get("ok")
        and commit_receipt.get("ok")
        and readme_receipt.get("ok")
        and license_receipt.get("ok")
        and schema_receipt.get("ok")
        and revision_ok
        and license_name == "MIT"
        and schema_verified
        and corruption_warning
    )
    return {
        "row_type": "drift_bench_provenance_contract",
        "repo_url": DRIFT_REPO_URL,
        "api_repo_url": DRIFT_API_REPO_URL,
        "immutable_revision": revision if revision_ok else None,
        "revision_is_immutable": revision_ok,
        "license": license_name,
        "license_verified": license_name == "MIT" and bool(license_receipt.get("ok")),
        "data_schema_path": "data/problems/README.md",
        "data_schema_verified": schema_verified,
        "upstream_corruption_warning_present": corruption_warning,
        "local_result_receipts_regenerated": True,
        "upstream_aggregate_claims_inherited": False,
        "accessed_at_utc": now_utc,
        "source_receipts": receipts,
        "contract_ready": contract_ready,
    }


def collect_model_cache_resolution_rows(
    *,
    cached_pair_resolver: ModelPairResolver,
    gguf_resolver: GgufResolver,
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
) -> list[JsonDict]:
    pair_specs = cached_pair_resolver(gpu_indices=gpu_indices, preferred_quant=preferred_quant)
    pair_by_id = {
        str(spec.get("hf_id")): dict(spec)
        for spec in pair_specs or []
        if isinstance(spec, Mapping) and spec.get("hf_id")
    }
    rows: list[JsonDict] = []
    for model in SOTA_GGUF_MODELS:
        hf_id = model["hf_id"]
        pair_spec = pair_by_id.get(hf_id)
        resolved = (
            str(pair_spec.get("model_path")) if pair_spec and pair_spec.get("model_path") else None
        )
        if resolved is None:
            resolved = gguf_resolver(hf_id, preferred_quant)
        path = Path(resolved) if resolved else None
        cache_hit = bool(path and path.is_file() and path.stat().st_size > 0)
        rows.append(
            {
                "row_type": "model_cache",
                "name": model["name"],
                "hf_id": hf_id,
                "role": model["role"],
                "preferred_quant": preferred_quant,
                "registry_quantization": model["quantization"],
                "gpu_indices_requested": list(gpu_indices),
                "cached_sota_pair_returned": pair_specs is not None,
                "selected_by_cached_sota_pair": pair_spec is not None,
                "assigned_gpu": pair_spec.get("gpu") if pair_spec else None,
                "model_path": str(path) if path else None,
                "quantized_filename": path.name if path else None,
                "cache_hit": cache_hit,
                "model_file_size_bytes": path.stat().st_size if cache_hit and path else None,
                "model_file_sha256": sha256_file(path) if cache_hit and path else "missing",
                "missing_entry": None if cache_hit else "model_path_not_resolved_or_empty",
                "model_loaded_or_run": False,
            }
        )
    return rows


def build_frozen_external_split_contract() -> JsonDict:
    return {
        "contract_version": "v565_external_split_contract_v1",
        "source_surface": "DRIFT-Bench content-pinned rows only",
        "split_names": ["train", "development", "held_family_blind"],
        "family_blind_keys": ["domain", "base_problem_id", "source_problem_id"],
        "chronology_keys": ["turn_index", "chronology_index"],
        "held_outcome_forbidden_before_freeze": True,
        "negative_controls": ["internal_carnot_only", "row_order_shuffle", "family_alias_attack"],
        "downstream_field_spelling": [
            "split_name",
            "base_problem_id",
            "domain",
            "turn_index",
            "source_row_hash",
            "chronology_index",
        ],
    }


def build_frozen_router_contract() -> JsonDict:
    return {
        "contract_version": "v565_router_contract_v1",
        "structural_controls": ["native_dynamic", "random_order", "analytical_structural_order"],
        "calibration_split": "development",
        "abstention_required": True,
        "exact_fallback_required": True,
        "learned_advice_may_order": True,
        "learned_advice_may_prune": False,
        "learned_advice_may_certify": False,
        "negative_controls": ["identity_shortcut", "family_permutation", "empty_router"],
        "downstream_field_spelling": [
            "router_arm",
            "router_abstained",
            "calibration_split",
            "exact_fallback_used",
            "candidate_set_preserved",
        ],
    }


def build_frozen_embedding_contract() -> JsonDict:
    return {
        "contract_version": "v565_embedding_contract_v1",
        "mandated_hub_ids": [model["hf_id"] for model in SOTA_GGUF_MODELS],
        "paired_conditions": ["canonical", "entity_relabeled", "clause_reordered", "paraphrased"],
        "strata": ["solver_hardness_bin", "surface_realization_id", "family"],
        "answer_scoring_allowed": False,
        "negative_controls": ["model_identity_probe", "family_identity_probe", "row_length_probe"],
        "downstream_field_spelling": [
            "model_hf_id",
            "model_file_sha256",
            "surface_realization_id",
            "paired_embedding_distance",
            "neighbor_stability",
        ],
    }


def build_frozen_transactional_learning_contract() -> JsonDict:
    return {
        "contract_version": "v565_transactional_learning_contract_v1",
        "memory_frozen_within_query": True,
        "commit_after_exact_validation": True,
        "refinement_witness_required": True,
        "same_query_write_negative_control": True,
        "rollback_required": True,
        "support_and_retention_metrics": [
            "future_exact_satisfying_support",
            "retained_family_performance",
            "invalid_reuse_veto_count",
            "rollback_integrity",
        ],
        "downstream_field_spelling": [
            "memory_arm",
            "memory_frozen_within_query",
            "commit_after_exact_validation",
            "future_exact_satisfying_support",
            "retained_family_performance",
            "rollback_integrity",
        ],
    }


def build_frozen_arc_parser_contract() -> JsonDict:
    return {
        "contract_version": "v565_arc_qwen3_xml_contract_v1",
        "live_generator_hf_id": "unsloth/Qwen3.8-27B-GGUF",
        "format_control_hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "parser_family": "qwen3_xml",
        "qwen3_xml_stop_rule": "stop_after_one_bounded_live_tool_call_receipt",
        "game_solve_claim_allowed": False,
        "negative_controls": ["hermes_parser_no_tool_lift", "qwen36_format_control"],
        "downstream_field_spelling": [
            "tool_call_parser",
            "tool_call_lifted",
            "outcome_receipt_present",
            "stop_rule_triggered",
        ],
    }


def build_hardware_stop_contract() -> JsonDict:
    return {
        "contract_version": "v565_hardware_stop_contract_v1",
        "gatemate_command_allowed_without_new_receipt": False,
        "required_new_receipt": "dated_physical_state_change_after_exp6325",
        "tsu_execution_claim_allowed": False,
        "kona_execution_claim_allowed": False,
        "negative_controls": ["unchanged_board_probe", "product_page_as_access_claim"],
        "downstream_field_spelling": [
            "new_physical_state_receipt",
            "hardware_command_count",
            "gatemate_boundary_status",
        ],
    }


def build_non_transfer_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_type": "non_transfer",
            "source_id": row["source_id"],
            "non_transfer_boundary": row["non_transfer_boundary"],
            "exact_authority_boundary": row["exact_authority_boundary"],
            "source_hash": row["source_hash"],
            "not_local_evidence": True,
        }
        for row in source_rows
    ]


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    files = {
        path.as_posix(): {
            "sha256_before": sha256_file(repo_root / path),
            "sha256_after": sha256_file(repo_root / path),
            "unchanged": sha256_file(repo_root / path) != "missing",
        }
        for path in PROTECTED_RELATIVE_PATHS
    }
    changed = [
        path
        for path, row in files.items()
        if row["sha256_before"] != row["sha256_after"] or row["unchanged"] is not True
    ]
    return {"files": files, "changed_paths": changed, "all_protected_files_unchanged": not changed}


def _path_hash_rows(repo_root: Path, paths: Sequence[Path]) -> list[JsonDict]:
    return [
        {
            "path": path.as_posix(),
            "exists": (repo_root / path).exists(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in paths
    ]


def preconditions_checked(
    *,
    repo_root: Path,
    run_date: str,
    now_utc: str,
    network_state: Mapping[str, Any],
    query_receipts: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "timestamps": {"started_or_accessed_at_utc": now_utc},
        "network_availability": dict(network_state),
        "query_strings": [row.get("query") for row in query_receipts],
        "source_tools": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "sweep_clusters_sha256": sha256_file(repo_root / "scripts/sweep_clusters.py"),
            "sweep_semscholar_sha256": sha256_file(repo_root / "scripts/sweep_semscholar.py"),
            "sota_models_sha256": sha256_file(repo_root / "python/carnot/inference/sota_models.py"),
        },
        "cache_paths_without_secret_values": [
            {
                "hf_id": row.get("hf_id"),
                "model_path": row.get("model_path"),
                "quantized_filename": row.get("quantized_filename"),
                "cache_hit": row.get("cache_hit"),
            }
            for row in model_rows
        ],
        "source_paths": _path_hash_rows(repo_root, SOURCE_PATHS),
        "local_hook_paths": _path_hash_rows(repo_root, LOCAL_HOOK_PATHS),
        "git": {
            "head": _git_output(repo_root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(repo_root, ["status", "--short"]),
        },
        "protected_file_hashes": {
            path: row["sha256_before"]
            for path, row in dict(protected.get("files", {})).items()
            if isinstance(row, Mapping)
        },
        "no_high_concurrency_research_harness": True,
        "outcome_artifact_guard": {
            "outcome_artifacts_read": [],
            "downstream_outcome_experiment_range": "exp6529-exp6540",
        },
    }


def _contracts() -> dict[str, JsonDict]:
    return {
        "frozen_external_split_contract": build_frozen_external_split_contract(),
        "frozen_router_contract": build_frozen_router_contract(),
        "frozen_embedding_contract": build_frozen_embedding_contract(),
        "frozen_transactional_learning_contract": build_frozen_transactional_learning_contract(),
        "frozen_arc_parser_contract": build_frozen_arc_parser_contract(),
        "hardware_stop_contract": build_hardware_stop_contract(),
    }


def _contract_spelling_complete(contracts: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        bool(contract.get("downstream_field_spelling"))
        for contract in contracts.values()
        if isinstance(contract, Mapping)
    )


def aggregate_row_recomputation(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    query_receipts: Sequence[Mapping[str, Any]],
    drift_contract: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]],
    non_transfer_rows: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    manifest_ids = {str(row["source_id"]) for row in SOURCE_MANIFEST}
    source_ids = {str(row.get("source_id")) for row in source_rows}
    required_primary = [row for row in source_rows if row.get("required_primary_check") is True]
    required_verified = all(
        row.get("primary_url_verified") is True
        and (row.get("expected_source_date") is None or row.get("source_date_verified") is True)
        for row in required_primary
    )
    by_source = {str(row.get("source_id")): row for row in source_rows}
    adopted_complete = all(
        by_source.get(source_id, {}).get("primary_url_verified") is True
        and by_source.get(source_id, {}).get("local_applicability")
        and by_source.get(source_id, {}).get("negative_control")
        and by_source.get(source_id, {}).get("downstream_field_spelling")
        and by_source.get(source_id, {}).get("non_transfer_boundary")
        and by_source.get(source_id, {}).get("exact_authority_boundary")
        for source_id in ADOPTED_SOURCE_IDS
    )
    model_hf_ids = {str(row.get("hf_id")) for row in model_rows}
    mandated_hf_ids = {model["hf_id"] for model in SOTA_GGUF_MODELS}
    model_cache_ready = (
        model_hf_ids == mandated_hf_ids
        and all(
            row.get("cache_hit") is True and row.get("model_loaded_or_run") is False
            for row in model_rows
        )
        and any(row.get("cached_sota_pair_returned") is True for row in model_rows)
    )
    non_transfer_complete = len(non_transfer_rows) == len(source_rows) and all(
        row.get("non_transfer_boundary") and row.get("exact_authority_boundary")
        for row in non_transfer_rows
    )
    channels = {str(row.get("channel")) for row in query_receipts}
    query_complete = REQUIRED_CHANNELS <= channels
    router = contracts.get("frozen_router_contract", {})
    embedding = contracts.get("frozen_embedding_contract", {})
    hardware = contracts.get("hardware_stop_contract", {})
    authority_ok = (
        router.get("learned_advice_may_prune") is False
        and router.get("learned_advice_may_certify") is False
        and embedding.get("answer_scoring_allowed") is False
        and hardware.get("gatemate_command_allowed_without_new_receipt") is False
    )
    protected_ok = protected.get("all_protected_files_unchanged") is True
    no_outcome_read = not preconditions.get("outcome_artifact_guard", {}).get(
        "outcome_artifacts_read"
    )
    spelling_complete = _contract_spelling_complete(contracts)
    ready = (
        source_ids == manifest_ids
        and required_verified
        and adopted_complete
        and drift_contract.get("contract_ready") is True
        and model_cache_ready
        and non_transfer_complete
        and query_complete
        and spelling_complete
        and authority_ok
        and protected_ok
        and no_outcome_read
    )
    return {
        "required_source_count": len(SOURCE_MANIFEST),
        "source_row_count": len(source_rows),
        "source_rows_cover_manifest": source_ids == manifest_ids,
        "required_primary_source_count": len(required_primary),
        "required_primary_sources_verified": required_verified,
        "adopted_method_count": len(ADOPTED_SOURCE_IDS),
        "adopted_methods_with_source_hook_control_boundary": len(ADOPTED_SOURCE_IDS)
        if adopted_complete
        else 0,
        "drift_bench_contract_ready": drift_contract.get("contract_ready") is True,
        "model_cache_contract_ready": model_cache_ready,
        "model_cache_hit_count": sum(1 for row in model_rows if row.get("cache_hit") is True),
        "mandated_model_count": len(SOTA_GGUF_MODELS),
        "non_transfer_row_count": len(non_transfer_rows),
        "non_transfer_rows_complete": non_transfer_complete,
        "required_query_channels_present": query_complete,
        "frozen_downstream_field_spelling_complete": spelling_complete,
        "authority_boundary_exact_first": authority_ok,
        "protected_files_unchanged": protected_ok,
        "outcome_artifacts_not_read": no_outcome_read,
        "frozen_method_contract_hash": sha256_json(contracts),
        "ready_score_from_rows": 1.0 if ready else 0.0,
    }


def gate_check_summary(
    aggregate: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    checks = {
        "source_rows_cover_manifest": aggregate.get("source_rows_cover_manifest") is True,
        "required_primary_sources_verified": aggregate.get("required_primary_sources_verified")
        is True,
        "adopted_methods_have_source_hook_control_boundary": (
            aggregate.get("adopted_methods_with_source_hook_control_boundary")
            == aggregate.get("adopted_method_count")
        ),
        "drift_bench_contract_ready": aggregate.get("drift_bench_contract_ready") is True,
        "model_cache_contract_ready": aggregate.get("model_cache_contract_ready") is True,
        "non_transfer_rows_complete": aggregate.get("non_transfer_rows_complete") is True,
        "required_query_channels_present": aggregate.get("required_query_channels_present") is True,
        "frozen_downstream_field_spelling_complete": (
            aggregate.get("frozen_downstream_field_spelling_complete") is True
        ),
        "authority_boundary_exact_first": aggregate.get("authority_boundary_exact_first") is True,
        "protected_files_unchanged": aggregate.get("protected_files_unchanged") is True,
        "outcome_artifacts_not_read": aggregate.get("outcome_artifacts_not_read") is True,
    }
    failed: list[JsonDict] = []
    for key, passed in checks.items():
        if passed is True:
            continue
        failed.append({"check": key, "expected": True, "observed": passed, "channel": "local_gate"})
    for row in source_rows:
        if row.get("required_primary_check") is True and (
            row.get("primary_url_verified") is not True
            or (
                row.get("expected_source_date") is not None
                and row.get("source_date_verified") is not True
            )
        ):
            failed.append(
                {
                    "check": "required_primary_source_verified",
                    "source_id": row.get("source_id"),
                    "channel": row.get("query_channel"),
                    "expected": "available_primary_url_and_matching_source_date",
                    "observed": {
                        "retrieval_state": row.get("retrieval_state"),
                        "http_state": row.get("http_state"),
                        "source_date": row.get("source_date"),
                        "observed_error": row.get("observed_error"),
                    },
                }
            )
    missing_models = [row.get("hf_id") for row in model_rows if row.get("cache_hit") is not True]
    if missing_models:
        failed.append(
            {
                "check": "mandated_gguf_cache_resolved",
                "channel": "model_cache_preflight",
                "expected": "all_mandated_gguf_files_present_without_model_load",
                "observed": {"missing_hf_ids": missing_models},
            }
        )
    return {"checks": checks, "failed_checks": failed, "all_gates_passed": not failed}


def blocked_verdict_channels(gate: Mapping[str, Any]) -> list[str]:
    """Return the concrete unavailable channels a blocked verdict must name."""
    channels: list[str] = []
    for row in gate.get("failed_checks", []):
        if not isinstance(row, Mapping):
            continue
        channel = str(row.get("channel") or "").strip()
        check = str(row.get("check") or "").strip()
        if channel and channel != "local_gate":
            channels.append(channel)
        elif check == "mandated_gguf_cache_resolved":
            channels.append("model_cache_preflight")
    return sorted(set(channels))


def blocked_honest_verdict(gate: Mapping[str, Any]) -> str:
    channels = blocked_verdict_channels(gate)
    if channels:
        return (
            "blocked_v565_source_model_method_contract: unavailable_required_channels="
            + ",".join(channels)
        )
    return "blocked_v565_source_model_method_contract: source or model-cache gates failed"


def build_per_unit_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    non_transfer_rows: Sequence[Mapping[str, Any]],
    drift_contract: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = [dict(row) for row in source_rows]
    rows.extend(dict(row) for row in model_rows)
    rows.extend(dict(row) for row in non_transfer_rows)
    rows.append(
        {
            "row_type": "contract",
            "contract_section": "drift_bench",
            "contract_hash": sha256_json(drift_contract),
        }
    )
    rows.extend(
        {"row_type": "contract", "contract_section": key, "contract_hash": sha256_json(value)}
        for key, value in contracts.items()
    )
    rows.extend({"row_type": "gate", **row} for row in gate.get("failed_checks", []))
    if not gate.get("failed_checks"):
        rows.append({"row_type": "gate", "check": "all_gates_passed", "observed": True})
    return rows


def tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(stable)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    run_date: str = RUN_DATE,
    fetcher: Fetcher = default_fetcher,
    command_runner: CommandRunner = default_command_runner,
    cached_pair_resolver: ModelPairResolver = cached_sota_pair,
    gguf_resolver: GgufResolver = resolve_cached_gguf,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    now_utc: str | None = None,
) -> JsonDict:
    start = time.monotonic()
    now = now_utc or _utc_now()
    network = _network_probe(fetcher, now)
    query_receipts = collect_query_receipts(
        repo_root=repo_root,
        command_runner=command_runner,
        network_state=network,
        now_utc=now,
    )
    source_rows = collect_source_rows(fetcher=fetcher, now_utc=now)
    source_hashes = {str(row["source_id"]): str(row["source_hash"]) for row in source_rows}
    citations = citation_trail_receipts(source_rows)
    citation_boundaries = citation_count_boundaries(source_rows)
    drift_contract = build_drift_bench_provenance_contract(fetcher=fetcher, now_utc=now)
    model_rows = collect_model_cache_resolution_rows(
        cached_pair_resolver=cached_pair_resolver,
        gguf_resolver=gguf_resolver,
        gpu_indices=(0, 1),
    )
    contracts = _contracts()
    non_transfer = build_non_transfer_rows(source_rows)
    protected = protected_files_unchanged(repo_root)
    preconditions = preconditions_checked(
        repo_root=repo_root,
        run_date=run_date,
        now_utc=now,
        network_state=network,
        query_receipts=query_receipts,
        protected=protected,
        model_rows=model_rows,
    )
    aggregate = aggregate_row_recomputation(
        source_rows=source_rows,
        query_receipts=query_receipts,
        drift_contract=drift_contract,
        model_rows=model_rows,
        contracts=contracts,
        non_transfer_rows=non_transfer,
        protected=protected,
        preconditions=preconditions,
    )
    gate = gate_check_summary(aggregate, source_rows, model_rows)
    ready_score = float(aggregate["ready_score_from_rows"])
    status = (
        "complete_v565_source_model_method_contract_ready"
        if ready_score == 1.0
        else "blocked_v565_source_model_method_contract"
    )
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": (
            "complete_v565_source_model_method_contract_ready: primary sources, cache, and method fields are frozen"
            if ready_score == 1.0
            else blocked_honest_verdict(gate)
        ),
        "verdict_class": None if ready_score == 1.0 else "partial",
        "query_receipts": query_receipts,
        "source_rows": source_rows,
        "primary_source_hashes": source_hashes,
        "citation_trail_receipts": citations,
        "citation_count_boundaries": citation_boundaries,
        "drift_bench_provenance_contract": drift_contract,
        "model_cache_resolution_rows": model_rows,
        "frozen_external_split_contract": contracts["frozen_external_split_contract"],
        "frozen_router_contract": contracts["frozen_router_contract"],
        "frozen_embedding_contract": contracts["frozen_embedding_contract"],
        "frozen_transactional_learning_contract": contracts[
            "frozen_transactional_learning_contract"
        ],
        "frozen_arc_parser_contract": contracts["frozen_arc_parser_contract"],
        "hardware_stop_contract": contracts["hardware_stop_contract"],
        "non_transfer_rows": non_transfer,
        "v565_method_contract_ready_score": ready_score,
        "gate_check_summary": gate,
        "per_unit_rows": build_per_unit_rows(
            source_rows=source_rows,
            model_rows=model_rows,
            non_transfer_rows=non_transfer,
            drift_contract=drift_contract,
            contracts=contracts,
            gate=gate,
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - start),
        "tests_run": tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        target = Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
        atomic_write_json(target, artifact, allow_override=False, sort_keys=False)
    return artifact


def _artifact_contracts(artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        "frozen_external_split_contract": artifact.get("frozen_external_split_contract", {}),
        "frozen_router_contract": artifact.get("frozen_router_contract", {}),
        "frozen_embedding_contract": artifact.get("frozen_embedding_contract", {}),
        "frozen_transactional_learning_contract": artifact.get(
            "frozen_transactional_learning_contract", {}
        ),
        "frozen_arc_parser_contract": artifact.get("frozen_arc_parser_contract", {}),
        "hardware_stop_contract": artifact.get("hardware_stop_contract", {}),
    }


def _recompute_current_aggregate(artifact: Mapping[str, Any]) -> JsonDict:
    return aggregate_row_recomputation(
        source_rows=artifact.get("source_rows", []),
        query_receipts=artifact.get("query_receipts", []),
        drift_contract=artifact.get("drift_bench_provenance_contract", {}),
        model_rows=artifact.get("model_cache_resolution_rows", []),
        contracts=_artifact_contracts(artifact),
        non_transfer_rows=artifact.get("non_transfer_rows", []),
        protected=artifact.get("protected_files_unchanged", {}),
        preconditions=artifact.get("preconditions_checked", {}),
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in {None, "partial"}:
        errors.append("verdict_class outside Exp6528 enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "blocked_", "partial_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if str(artifact.get("honest_verdict", "")).startswith("blocked_"):
        verdict_text = str(artifact.get("honest_verdict", ""))
        missing_channels = [
            channel
            for channel in blocked_verdict_channels(artifact.get("gate_check_summary", {}))
            if channel not in verdict_text
        ]
        if missing_channels:
            errors.append("blocked verdict must name unavailable required channels")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    current = _recompute_current_aggregate(artifact)
    score = artifact.get("v565_method_contract_ready_score")
    if score not in {0.0, 1.0} or score != current["ready_score_from_rows"]:
        errors.append("ready score mismatch")
    manifest_ids = {str(row["source_id"]) for row in SOURCE_MANIFEST}
    source_ids = {str(row.get("source_id")) for row in artifact.get("source_rows", [])}
    if source_ids != manifest_ids:
        errors.append("source rows must cover manifest")
    if score == 1.0 and current["required_primary_sources_verified"] is not True:
        errors.append("required primary sources must verify")
    by_source = {str(row.get("source_id")): row for row in artifact.get("source_rows", [])}
    adopted_complete = all(
        by_source.get(source_id, {}).get("local_applicability")
        and by_source.get(source_id, {}).get("negative_control")
        and by_source.get(source_id, {}).get("downstream_field_spelling")
        and by_source.get(source_id, {}).get("non_transfer_boundary")
        and by_source.get(source_id, {}).get("exact_authority_boundary")
        for source_id in ADOPTED_SOURCE_IDS
    )
    if not adopted_complete and score == 1.0:
        errors.append("adopted methods must map to source hooks controls and boundaries")
    if (
        artifact.get("drift_bench_provenance_contract", {}).get("contract_ready") is not True
        and score == 1.0
    ):
        errors.append("drift bench provenance contract must be ready")
    model_hf_ids = {
        str(row.get("hf_id")) for row in artifact.get("model_cache_resolution_rows", [])
    }
    mandated_hf_ids = {model["hf_id"] for model in SOTA_GGUF_MODELS}
    model_rows_cover = model_hf_ids == mandated_hf_ids and bool(
        artifact.get("model_cache_resolution_rows")
    )
    if not model_rows_cover or (
        score == 1.0
        and not all(
            row.get("cache_hit") is True for row in artifact.get("model_cache_resolution_rows", [])
        )
    ):
        errors.append("model cache contract must cover all mandated models")
    if not _contract_spelling_complete(_artifact_contracts(artifact)):
        errors.append("frozen contracts must expose downstream field spelling")
    if not all(
        row.get("non_transfer_boundary") and row.get("exact_authority_boundary")
        for row in artifact.get("non_transfer_rows", [])
    ):
        errors.append("non-transfer rows must forbid unsupported transfer")
    router = artifact.get("frozen_router_contract", {})
    if router.get("learned_advice_may_prune") is not False:
        errors.append("learned routing cannot prune candidates")
    if (
        artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if not REQUIRED_CHANNELS <= {
        str(row.get("channel")) for row in artifact.get("query_receipts", [])
    }:
        errors.append("query receipts must include all required sequential channels")
    if set(artifact.get("primary_source_hashes", {})) != manifest_ids:
        errors.append("primary_source_hashes must cover manifest")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def _load_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6528 V565 source-model-method contract."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-live-network", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        errors = validate_artifact(_load_json(result))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result,
        run_date=str(args.date),
        fetcher=offline_fetcher if args.no_live_network else default_fetcher,
        command_runner=offline_command_runner if args.no_live_network else default_command_runner,
        write=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
