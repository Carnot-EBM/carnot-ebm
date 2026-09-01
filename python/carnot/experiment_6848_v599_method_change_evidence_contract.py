"""Build the immutable V599 method-change evidence contract.

Spec refs: REQ-REPORT-6848 and SCENARIO-REPORT-6848-*.

This reducer reads terminal V598 evidence. It does not run the mechanisms that
created that evidence. It also does not import either disputed reducer. The
separation matters because Exp6836 accepted duplicated candidate identities,
while Exp6847 rejected them. A new evidence root must show that disagreement
instead of choosing the more convenient result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.599"
EXPERIMENT_ID = "exp6848-v599-method-change-evidence-contract"
INFERENCE_SUBSTRATE = "deterministic CPU evidence replay"
RESULT_PATH = Path("results/experiment_6848_v599_method_change_evidence_contract.json")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6848_v599_method_change_evidence_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6848_v599_method_change_evidence_contract.py")
TEST_PATH = Path("tests/python/test_experiment_6848_v599_method_change_evidence_contract.py")
NOTE_PATH = Path("docs/research-notes/v599-method-change-contract.md")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
REFERENCE_PATH = Path("research-references.md")

CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

SOURCE_SPECS: dict[str, JsonDict] = {
    source_id: {"source_id": source_id, "path": path}
    for source_id, path in (
        ("exp6836", "results/experiment_6836_typed_obligation_program_fixture.json"),
        ("exp6837", "results/experiment_6837_three_family_output_free_compatibility.json"),
        ("exp6842", "results/experiment_6842_sealed_memory_pathway_portability_audit.json"),
        ("exp6843", "results/experiment_6843_live_arc_evidence_stratum_freeze.json"),
        ("exp6844", "results/experiment_6844_supervisor_action_outcome_credit_audit.json"),
        ("exp6845", "results/experiment_6845_tool_gap_causal_support_audit.json"),
        ("exp6847", "results/experiment_6847_v598_independent_capstone.json"),
    )
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "reproducibility_checksum",
    "rows",
    "producer_auditor_disagreements",
    "terminal_branch_manifest",
    "conductor_skip_manifest",
    "retired_mechanism_manifest",
    "changed_mechanism_manifest",
    "source_access_boundaries",
    "dynamic_evidence_schema",
    "reference_verification_rows",
    "v599_evidence_contract_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

SPEC_REFS = [
    "REQ-REPORT-6848",
    "SCENARIO-REPORT-6848-SOURCE-DRIFT",
    "SCENARIO-REPORT-6848-PRODUCER-AUDITOR-DISAGREEMENT",
    "SCENARIO-REPORT-6848-MISSING-ARTIFACT",
    "SCENARIO-REPORT-6848-STALE-HARDCODED-PATH",
    "SCENARIO-REPORT-6848-TERMINAL-NULL-PRESERVATION",
]

FIELD_PRINCIPLES = {
    "schema": "The schema fixes the V599 evidence-root shape for downstream consumers.",
    "experiment_id": "A unique identity prevents this contract from being confused with V598 evidence.",
    "milestone": "The milestone bounds every decision to V599.",
    "run_date": "The supplied execution date makes the freeze time explicit.",
    "status": "Status reports contract completion without claiming scientific benefit.",
    "result_path": "The roadmap-defined path gives downstream tasks one stable evidence root.",
    "spec_refs": "Requirement anchors connect the artifact to its tests.",
    "random_seed": "A fixed seed records that no sampled decision entered this deterministic replay.",
    "field_principles": "Every top-level field explains why it exists.",
    "preconditions_checked": "Missing, unreadable, nonterminal, and drifted evidence fails closed.",
    "inference_substrate": "The task replays CPU evidence and does not invoke an LLM.",
    "duration_s": "Wall-clock time distinguishes an executed replay from static boilerplate.",
    "source_artifact_hashes": "Content hashes pin immutable V598 inputs and implementation provenance.",
    "reproducibility_checksum": "A stable digest binds all evidence decisions except measured duration.",
    "rows": "Each source, disputed field, method decision, and skip has a checkable row.",
    "producer_auditor_disagreements": "Conflicting readiness values remain visible with controlling authority.",
    "exp6836_independent_recomputation": "Fresh identity counts expose duplicate candidates before V599 scoring.",
    "frozen_v598_results": "Terminal resource, harm, headroom, and obligation results cannot become benefit.",
    "terminal_branch_manifest": "Source verdicts stay distinct from V599 method decisions.",
    "conductor_skip_manifest": "A missing gated artifact is acceptable only with a hash-bound skip record.",
    "retired_mechanism_manifest": "Harmful or authority-invalid mechanisms cannot silently recur.",
    "changed_mechanism_manifest": "Each open V599 branch names the mechanism change and exact consumer.",
    "source_access_boundaries": "Local evidence, network context, and dynamic discovery have separate authority.",
    "dynamic_evidence_schema": "One provenance schema governs hashes, identities, ownership, outcomes, and skips.",
    "reference_verification_rows": "Primary-page receipts verify identifiers while preserving title or date drift.",
    "v599_evidence_contract_ready_score": "This is a completeness gate for Exp6849, Exp6850, Exp6853, and Exp6857, not a benefit score.",
    "gate_check_summary": "Every blocked contract records the failed check and observed value.",
    "verifier_is_oracle": "The reducer audits evidence but cannot validate its own scientific conclusions.",
    "verdict_class": "The closed verdict enum prevents readiness from being relabeled as positivity.",
    "honest_verdict": "A complete_ prefix gives the conductor an unambiguous terminal result.",
}


def _reference(
    identifier: str,
    title: str,
    date: str,
    planner_title: str,
    planner_date: str,
    method_delta: str,
    carnot_hook: str,
) -> JsonDict:
    """Return one primary-page receipt with a bounded Carnot interpretation."""

    arxiv_id = identifier.removeprefix("arXiv:")
    return {
        "title": title,
        "identifier": identifier,
        "date": date,
        "primary_url": f"https://arxiv.org/abs/{arxiv_id}",
        "primary_page": "arXiv abstract record",
        "verified_on": "2026-09-01",
        "identifier_verified": True,
        "planner_title": planner_title,
        "planner_title_matches_primary": planner_title == title,
        "planner_date": planner_date,
        "planner_date_matches_primary": planner_date == date,
        "method_delta": method_delta,
        "carnot_hook": carnot_hook,
        "access_boundary": "context_only_no_dependency",
        "dependency_added": False,
    }


REFERENCE_VERIFICATION_ROWS = [
    _reference(
        "arXiv:2604.27283",
        "Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents",
        "2026-04-30",
        "Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents",
        "2026-04-30",
        "Treat memory use, no-memory, and abstention as risk-sensitive actions.",
        "Replace the harmful residual rule with a bounded selector labeled by exact later outcomes.",
    ),
    _reference(
        "arXiv:2604.15149",
        "LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking",
        "2026-04-16",
        "LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking",
        "2026-04-16",
        "Isomorphic Perturbation Testing detects extensional verifier shortcuts.",
        "Require identity, label, permutation, and duplicate-removal invariance before scoring.",
    ),
    _reference(
        "arXiv:2607.16999",
        "Counterfactual Shapley Credit Assignment",
        "2026-07-18",
        "Counterfactual Shapley Credit Assignment",
        "2026-07-18",
        "Counterfactual coalition values separate policy effects from environmental luck.",
        "Credit only valid action coalitions with exact outcomes and nonzero headroom.",
    ),
    _reference(
        "arXiv:2608.11994",
        "Claim-Level Reliability Assessment for Efficient Test-Time Reasoning",
        "2026-08-12",
        "Claim-Level Reliability Assessment for Efficient Test-Time Reasoning",
        "2026-08-12",
        "Targeted falsification allocates verification to decision-critical claims.",
        "Keep obligation atoms and action receipts as the verification unit.",
    ),
    _reference(
        "arXiv:2606.19808",
        "Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning",
        "2026-06-18",
        "Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning",
        "2026-06-18",
        "Selective verification compares preserve and intervention actions under a budget.",
        "Measure intervention, preserve, and abstain on matched opportunities.",
    ),
    _reference(
        "arXiv:2608.31046",
        "Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement",
        "2026-08-31",
        "Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement",
        "2026-08-31",
        "Teacher-attributed gains can arise from teacher-free tail-token suppression.",
        "Keep model-derived updates as negative controls and freeze GGUF weights.",
    ),
    _reference(
        "arXiv:2608.30461",
        "From Final Artifacts to Trajectories: Retrospective Process Supervision for Evidence-Grounded Long-Form Generation",
        "2026-08-31",
        "RetroGen: Scaling Retrospective Process Supervision for Reliable Agentic Reasoning",
        "2026-08-31",
        "RetroGen reconstructs candidate process traces from final artifacts and evidence.",
        "Use reconstruction only as diagnosis; never replace authentic live receipts.",
    ),
    _reference(
        "arXiv:2608.29596",
        "Towards a Systems Foundation for Agentic Skills: Architecture, Lifecycle, and Security",
        "2026-08-30",
        "Agentic Skills in the Wild: A Comprehensive Study of Reusable Skill Artifacts",
        "2026-08-28",
        "The paper defines a lifecycle for reusable procedural skill artifacts.",
        "Carry source, reachability, intervention, outcome, and retirement provenance together.",
    ),
]


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for content hashes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 representation."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str | None:
    """Hash a readable file, or return None when the path is not a file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind evidence content while excluding checksum recursion and wall-clock noise."""

    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(unsigned))


def spec_anchors(text: str) -> list[str]:
    """Return requirement and scenario identifiers present in text."""

    return re.findall(r"\b(?:REQ|SCENARIO)-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", text)


def _verdict_value(payload: Mapping[str, Any]) -> str:
    """Read plain or principle-annotated honest verdicts without changing them."""

    verdict = payload.get("honest_verdict")
    if isinstance(verdict, Mapping):
        verdict = verdict.get("value")
    return str(verdict or "")


def _is_terminal(payload: Mapping[str, Any]) -> bool:
    """Require the closed class and an explicit terminal verdict prefix."""

    verdict = _verdict_value(payload)
    return payload.get("verdict_class") in CLOSED_VERDICT_CLASSES and verdict.startswith(
        ("complete_", "complete:")
    )


def missing_source_record(spec: Mapping[str, Any]) -> JsonDict:
    """Build the explicit missing record used by precondition failures."""

    return {
        "source_id": str(spec["source_id"]),
        "path": str(spec["path"]),
        "state": "missing",
        "file_sha256": None,
        "payload": None,
        "read_error": "FileNotFoundError",
        "terminal": False,
    }


def load_source_record(repo_root: Path, spec: Mapping[str, Any]) -> JsonDict:
    """Read one V598 artifact without executing any producer code."""

    path = repo_root / str(spec["path"])
    if not path.is_file():
        return missing_source_record(spec)
    file_hash = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "source_id": str(spec["source_id"]),
            "path": str(spec["path"]),
            "state": "invalid",
            "file_sha256": file_hash,
            "payload": None,
            "read_error": type(exc).__name__,
            "terminal": False,
        }
    terminal = isinstance(payload, Mapping) and _is_terminal(payload)
    return {
        "source_id": str(spec["source_id"]),
        "path": str(spec["path"]),
        "state": "present" if terminal else "nonterminal",
        "file_sha256": file_hash,
        "payload": payload,
        "read_error": None,
        "terminal": terminal,
    }


def load_source_records(repo_root: Path) -> dict[str, JsonDict]:
    """Read every required terminal source through the same evidence-only path."""

    return {
        source_id: load_source_record(repo_root, spec) for source_id, spec in SOURCE_SPECS.items()
    }


def _payload(records: Mapping[str, Mapping[str, Any]], source_id: str) -> Mapping[str, Any]:
    """Return a source mapping, or an empty mapping for a blocked build."""

    value = records.get(source_id, {}).get("payload")
    return value if isinstance(value, Mapping) else {}


def _capstone_observed(capstone: Mapping[str, Any], criterion_id: str) -> Any:
    """Read an independent Exp6847 row without importing its reducer."""

    for row in capstone.get("rows", []):
        if isinstance(row, Mapping) and row.get("criterion_id") == criterion_id:
            return row.get("observed_value")
    return None


def recompute_exp6836(
    producer: Mapping[str, Any], capstone: Mapping[str, Any]
) -> tuple[JsonDict, list[JsonDict]]:
    """Recompute candidate identity and compile readiness from stored rows.

    The producer repeats each semantic candidate in its label-swap row. That can
    be useful for a paired fixture, but it is not a unique candidate identity
    manifest. V599 therefore keeps the semantic checks and rejects the readiness
    conclusion. Exp6847 controls the three disputed fields.
    """

    rows = [row for row in producer.get("rows", []) if isinstance(row, Mapping)]
    candidate_ids = [
        str(candidate.get("candidate_id"))
        for row in rows
        for candidate in row.get("candidates", [])
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") is not None
    ]
    counts = Counter(candidate_ids)
    duplicate_rows = [
        {"candidate_id": candidate_id, "occurrences": count}
        for candidate_id, count in sorted(counts.items())
        if count > 1
    ]
    compile_results = producer.get("compile_parity_results")
    compile_results = compile_results if isinstance(compile_results, Mapping) else {}
    compile_rows = [
        row for row in compile_results.get("per_candidate", []) if isinstance(row, Mapping)
    ]
    compile_ids = [str(row.get("candidate_id")) for row in compile_rows]
    semantic_checks = bool(compile_rows) and all(
        row.get("view_atom_identities_equal") is True
        and row.get("diagnostic_atom_ids_match") is True
        and (row.get("energy") == 0) is (row.get("satisfaction_predicate") is True)
        and row.get("memory_admission_guard") is row.get("satisfaction_predicate")
        and row.get("arc_shadow_action_guard") is row.get("satisfaction_predicate")
        for row in compile_rows
    )
    identities_unique = bool(candidate_ids) and len(candidate_ids) == len(set(candidate_ids))
    declared_candidate_count = compile_results.get("candidate_count")
    recomputed_compile = bool(
        semantic_checks
        and identities_unique
        and len(compile_rows) == len(set(compile_ids))
        and declared_candidate_count == len(set(compile_ids))
    )
    capstone_compile = _capstone_observed(capstone, "exp6836.compile_parity")
    capstone_pair = _capstone_observed(capstone, "exp6836.obligation_pair_fixture_ready_score")
    capstone_program = _capstone_observed(capstone, "exp6836.typed_obligation_program_ready_score")
    controlling_compile = capstone_compile if capstone_compile is not None else recomputed_compile
    controlling_pair = capstone_pair if capstone_pair is not None else int(identities_unique)
    controlling_program = (
        capstone_program if capstone_program is not None else int(recomputed_compile)
    )
    recomputation = {
        "candidate_occurrence_count": len(candidate_ids),
        "unique_candidate_identity_count": len(counts),
        "duplicate_candidate_identity_count": len(duplicate_rows),
        "duplicate_candidate_identities": duplicate_rows,
        "compile_receipt_count": len(compile_rows),
        "unique_compile_candidate_identity_count": len(set(compile_ids)),
        "declared_candidate_count": declared_candidate_count,
        "semantic_compile_checks_passed": semantic_checks,
        "candidate_identities_unique": identities_unique,
        "fresh_compile_parity_before_authority_control": recomputed_compile,
        "recomputed_compile_parity": bool(controlling_compile),
        "recomputed_obligation_pair_fixture_ready_score": int(controlling_pair),
        "recomputed_typed_obligation_program_ready_score": int(controlling_program),
        "controlling_authority": "exp6847",
        "producer_reducer_imported": False,
        "auditor_reducer_imported": False,
    }
    producer_compile = all(
        compile_results.get(key) is True
        for key in (
            "all_candidates_exactly_checked",
            "all_views_share_atom_identities",
            "energy_zero_matches_satisfaction",
            "satisfaction_matches_guards",
        )
    )
    disagreements = [
        {
            "field": "compile_parity",
            "producer_value": producer_compile,
            "auditor_value": bool(controlling_compile),
            "controlling_authority": "exp6847",
            "reason": "candidate identities repeat across label-swap rows",
        },
        {
            "field": "obligation_pair_fixture_ready_score",
            "producer_value": producer.get("obligation_pair_fixture_ready_score"),
            "auditor_value": int(controlling_pair),
            "controlling_authority": "exp6847",
            "reason": "unique candidate identity authority did not pass",
        },
        {
            "field": "typed_obligation_program_ready_score",
            "producer_value": producer.get("typed_obligation_program_ready_score"),
            "auditor_value": int(controlling_program),
            "controlling_authority": "exp6847",
            "reason": "compile parity and fixture readiness did not survive independent reduction",
        },
    ]
    return recomputation, disagreements


def _named_check(payload: Mapping[str, Any], check_name: str) -> Any:
    """Find a precondition check by name across the V598 artifact shapes."""

    containers: list[Any] = [payload.get("preconditions_checked")]
    preconditions = payload.get("preconditions_checked")
    if isinstance(preconditions, Mapping):
        containers.append(preconditions.get("checks"))
    gate_summary = payload.get("gate_check_summary")
    if isinstance(gate_summary, Mapping):
        containers.append(gate_summary.get("checks"))
    for container in containers:
        if not isinstance(container, list):
            continue
        for row in container:
            if isinstance(row, Mapping) and row.get("check") == check_name:
                return row.get("observed")
    return None


def freeze_v598_results(records: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Extract only the terminal values that constrain V599 mechanisms."""

    exp6837 = _payload(records, "exp6837")
    exp6842 = _payload(records, "exp6842")
    exp6844 = _payload(records, "exp6844")
    exp6845 = _payload(records, "exp6845")
    residual = exp6842.get("negative_transfer_results", {})
    residual = residual.get("verified_residual_memory", {}) if isinstance(residual, Mapping) else {}
    action_rows = [row for row in exp6844.get("per_game_results", []) if isinstance(row, Mapping)]
    zero_headroom = sum(
        1
        for row in action_rows
        if isinstance(row.get("headroom"), Mapping)
        and row["headroom"].get("nonzero_headroom") is False
    )
    strata = exp6845.get("configuration_strata", {})
    strata_rows = strata.get("strata", []) if isinstance(strata, Mapping) else []
    strata_rows = [row for row in strata_rows if isinstance(row, Mapping)]
    obligation_ledger = exp6845.get("obligation_ledger", {})
    obligation_count = (
        int(obligation_ledger.get("row_count", 0)) if isinstance(obligation_ledger, Mapping) else 0
    )
    return {
        "exp6837_resource_block": {
            "exclusive_gpu_leases": _named_check(exp6837, "exclusive_gpu_leases"),
            "live_canary_per_model": _named_check(exp6837, "live_canary_per_model"),
            "scientific_row_count": len(exp6837.get("rows", [])),
            "compatibility_ready_score": exp6837.get("obligation_compatibility_stream_ready_score"),
        },
        "exp6842_harmful_learning": {
            "held_future_rows": residual.get("held_future_rows"),
            "wins": residual.get("wins"),
            "losses": residual.get("losses"),
            "mean_effect_vs_no_memory": residual.get("mean_effect_vs_no_memory"),
            "continuous_self_learning_ready_score": exp6842.get(
                "continuous_self_learning_ready_score"
            ),
        },
        "exp6844_zero_headroom": {
            "action_row_count": len(action_rows),
            "zero_headroom_action_row_count": zero_headroom,
            "nonzero_headroom_action_row_count": len(action_rows) - zero_headroom,
            "supervisor_effect_eligible_score": exp6844.get("supervisor_effect_eligible_score"),
        },
        "exp6845_zero_obligations": {
            "obligation_row_count": obligation_count,
            "stratum_count": len(strata_rows),
            "zero_obligation_stratum_count": sum(
                1 for row in strata_rows if row.get("obligation_count") == 0
            ),
            "tool_gap_effect_eligible_score": exp6845.get("tool_gap_effect_eligible_score"),
        },
    }


def scan_conductor_skips(repo_root: Path) -> list[JsonDict]:
    """Return the explicit V598 Exp6838 gate skip, bound to its exact log line."""

    path = repo_root / CONDUCTOR_LOG_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if "Independent obligation compatibility shortcut" not in line or "GATE_BLOCK" not in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4:
            continue
        upstream = re.search(r"exp6837", parts[3])
        return [
            {
                "task_id": "exp6838",
                "timestamp": parts[0],
                "outcome": parts[2],
                "reason": parts[3],
                "upstream_task_id": upstream.group(0) if upstream else None,
                "log_path": CONDUCTOR_LOG_PATH.as_posix(),
                "log_line_sha256": sha256_bytes(line.encode()),
            }
        ]
    return []


def validate_pinned_hashes(repo_root: Path, manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Return exact expected and observed values for immutable source drift."""

    failures: list[JsonDict] = []
    for source_id, receipt in manifest.items():
        if not isinstance(receipt, Mapping) or receipt.get("immutable") is not True:
            continue
        expected = receipt.get("file_sha256")
        observed = sha256_file(repo_root / str(receipt.get("path", "")))
        if expected != observed:
            failures.append(
                {
                    "check": f"source_hash.{source_id}",
                    "expected": expected,
                    "observed": observed,
                    "passed": False,
                }
            )
    return failures


def _source_hash_manifest(
    repo_root: Path, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Record frozen evidence and mutable implementation provenance separately."""

    manifest = {
        source_id: {
            "path": str(record.get("path")),
            "file_sha256": record.get("file_sha256"),
            "immutable": True,
        }
        for source_id, record in records.items()
    }
    for source_id, path, immutable in (
        ("conductor_log", CONDUCTOR_LOG_PATH, False),
        ("v599_references", REFERENCE_PATH, True),
        ("reporting_spec", REPORT_SPEC_PATH, False),
        ("implementation_module", MODULE_PATH, False),
        ("experiment_wrapper", WRAPPER_PATH, False),
        ("focused_tests", TEST_PATH, False),
        ("research_note", NOTE_PATH, False),
    ):
        manifest[source_id] = {
            "path": path.as_posix(),
            "file_sha256": sha256_file(repo_root / path),
            "immutable": immutable,
        }
    return manifest


def qualify_dynamic_artifact(repo_root: Path, entry: Mapping[str, Any]) -> JsonDict:
    """Qualify an execution-time artifact and ignore any stored absolute path.

    A stored path describes history. It cannot select future evidence because a
    new clone or a newer terminal receipt can make it stale. Only a repo-relative
    path from the current discovery manifest can become authoritative.
    """

    stored_ignored = bool(entry.get("stored_absolute_path"))
    discovered = entry.get("discovered_path")
    if not discovered:
        return {
            "eligible": False,
            "reason": "execution_time_discovered_path_missing",
            "stored_absolute_path_ignored": stored_ignored,
        }
    relative = Path(str(discovered))
    if relative.is_absolute():
        return {
            "eligible": False,
            "reason": "discovered_path_must_be_repo_relative",
            "stored_absolute_path_ignored": stored_ignored,
        }
    root = repo_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return {
            "eligible": False,
            "reason": "discovered_path_outside_repo",
            "stored_absolute_path_ignored": stored_ignored,
        }
    if not candidate.is_file():
        return {
            "eligible": False,
            "reason": "discovered_artifact_missing",
            "stored_absolute_path_ignored": stored_ignored,
        }
    required = (
        "artifact_family",
        "generator_identity",
        "model_identity",
        "process_ownership",
        "exact_outcome_authority",
    )
    missing = [field for field in required if not entry.get(field)]
    if missing:
        return {
            "eligible": False,
            "reason": "required_provenance_missing",
            "missing_fields": missing,
            "stored_absolute_path_ignored": stored_ignored,
        }
    return {
        "eligible": True,
        "resolved_path": candidate.as_posix(),
        "file_sha256": sha256_file(candidate),
        "stored_absolute_path_ignored": stored_ignored,
    }


def dynamic_evidence_schema() -> JsonDict:
    """Define the single provenance schema used by changed V599 mechanisms."""

    return {
        "schema_id": "carnot.v599.dynamic_evidence_provenance.v1",
        "required": [
            "immutable_hashes",
            "dynamic_artifact_manifest",
            "generator_identity",
            "model_identity",
            "process_ownership",
            "exact_outcome_authority",
            "conductor_skip",
        ],
        "properties": {
            "immutable_hashes": {
                "required_fields": ["source_id", "path", "file_sha256", "immutable"]
            },
            "dynamic_artifact_manifest": {
                "required_fields": [
                    "discovered_path",
                    "artifact_family",
                    "file_sha256",
                    "terminal_class",
                    "discovered_at",
                ]
            },
            "generator_identity": {
                "required_fields": ["generator_id", "implementation_sha256", "configuration_sha256"]
            },
            "model_identity": {
                "required_fields": ["model_id", "model_artifact_sha256", "tokenizer_sha256"]
            },
            "process_ownership": {
                "required_fields": [
                    "pid",
                    "ppid",
                    "process_start",
                    "command_sha256",
                    "task_owned",
                    "lease_id",
                    "teardown_receipt",
                ]
            },
            "exact_outcome_authority": {
                "required_fields": [
                    "authority_kind",
                    "receipt_id",
                    "observed_at",
                    "state_before_sha256",
                    "state_after_sha256",
                    "outcome",
                    "source_sha256",
                ]
            },
            "conductor_skip": {
                "required_fields": [
                    "task_id",
                    "timestamp",
                    "outcome",
                    "reason",
                    "upstream_task_id",
                    "log_line_sha256",
                ]
            },
        },
        "frozen_source_rule": {
            "selection": "contract_path_and_sha256",
            "hash_drift_action": "block",
            "mutable_source_may_replace_frozen_source": False,
        },
        "execution_time_discovery_rule": {
            "selection": "repo_relative_provenance_qualified_manifest_at_execution_time",
            "stored_absolute_paths_authoritative": False,
            "hard_coded_exp6681_allowed": False,
            "missing_provenance_action": "ineligible",
            "cross_configuration_pooling_allowed": False,
        },
    }


def _terminal_branch_manifest(records: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Keep each source verdict separate from the V599 scientific disposition."""

    decisions = {
        "exp6836": ("typed_authority_disqualified", False),
        "exp6837": ("resource_block_preserved_no_model_result", False),
        "exp6842": ("harmful_learning_rule_retired", False),
        "exp6843": ("inventory_null_preserved_dynamic_discovery_required", False),
        "exp6844": ("zero_headroom_effect_block_preserved", False),
        "exp6845": ("zero_obligation_effect_block_preserved", False),
        "exp6847": ("independent_capstone_controls_disputed_fields", False),
    }
    rows: list[JsonDict] = []
    for source_id, record in records.items():
        payload = record.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        disposition, benefit_allowed = decisions[source_id]
        source_class = payload.get("verdict_class")
        rows.append(
            {
                "source_id": source_id,
                "artifact_state": record.get("state"),
                "source_verdict_class": (
                    source_class if source_class in CLOSED_VERDICT_CLASSES else "blocked"
                ),
                "source_honest_verdict": _verdict_value(payload),
                "scientific_disposition": disposition,
                "benefit_claim_allowed": benefit_allowed,
                "verdict_class": (
                    source_class if source_class in CLOSED_VERDICT_CLASSES else "blocked"
                ),
            }
        )
    return rows


def _retired_mechanisms(frozen: Mapping[str, Any]) -> list[JsonDict]:
    """Name the exact V598 mechanisms that cannot be repeated unchanged."""

    return [
        {
            "mechanism_id": "exp6836_producer_self_authorized_candidate_identity",
            "retirement": "retired_as_v599_authority",
            "evidence": "Exp6847 compile parity and both readiness scores control at zero.",
        },
        {
            "mechanism_id": "combined_resource_admission_and_three_family_scoring",
            "retirement": "retired_same_shape_retry",
            "evidence": frozen["exp6837_resource_block"],
        },
        {
            "mechanism_id": "nonselective_verified_residual_memory",
            "retirement": "retired_harmful_learning_rule",
            "evidence": frozen["exp6842_harmful_learning"],
        },
        {
            "mechanism_id": "supervisor_credit_without_nonzero_headroom",
            "retirement": "retired_zero_effect_opportunity",
            "evidence": frozen["exp6844_zero_headroom"],
        },
        {
            "mechanism_id": "tool_gap_credit_without_first_party_obligation",
            "retirement": "retired_zero_obligation_input",
            "evidence": frozen["exp6845_zero_obligations"],
        },
        {
            "mechanism_id": "hard_coded_arc_source_path",
            "retirement": "retired_stale_discovery_rule",
            "evidence": "Execution-time manifests replace pinned Exp6681 selection.",
        },
    ]


def _changed_mechanisms() -> list[JsonDict]:
    """Map the new evidence rule to the four direct readiness consumers."""

    return [
        {
            "mechanism_id": "independent_isomorphic_typed_authority",
            "change": "fresh identities plus permutation, label, and duplicate-removal invariance",
            "consumers": ["exp6849"],
            "contract_gate": "v599_evidence_contract_ready_score",
        },
        {
            "mechanism_id": "separate_task_owned_scoring_admission",
            "change": "lease and one canary per model before scientific scoring",
            "consumers": ["exp6850"],
            "contract_gate": "v599_evidence_contract_ready_score",
        },
        {
            "mechanism_id": "risk_sensitive_memory_opportunity_selection",
            "change": "verified-memory, no-memory, and abstain under exact later outcomes",
            "consumers": ["exp6853", "exp6854", "exp6855", "exp6856"],
            "contract_gate": "v599_evidence_contract_ready_score",
        },
        {
            "mechanism_id": "dynamic_live_arc_receipt_routing",
            "change": "execution-time provenance manifest with headroom and first-party obligations",
            "consumers": ["exp6857", "exp6858", "exp6859"],
            "contract_gate": "v599_evidence_contract_ready_score",
        },
    ]


def _source_boundaries() -> list[JsonDict]:
    """Separate read-only evidence authority from context and discovery."""

    return [
        {
            "source_class": "terminal_v598_artifacts",
            "access": "local_read_only_hash_pinned",
            "authority": "exact stored outcomes and terminal branch state",
        },
        {
            "source_class": "v598_conductor_skip",
            "access": "local_append_only_log_line_hash_pinned",
            "authority": "task skip state only; no scientific result",
        },
        {
            "source_class": "promoted_primary_pages",
            "access": "network_verified_then_frozen_context_only",
            "authority": "title, identifier, date, and method context only",
        },
        {
            "source_class": "future_arc_artifacts",
            "access": "execution_time_repo_relative_discovery_read_only",
            "authority": "only after full provenance-schema qualification",
        },
    ]


def _evidence_rows(
    records: Mapping[str, Mapping[str, Any]],
    disagreements: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
    skips: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build one compact row per source, disputed field, decision, or skip."""

    rows: list[JsonDict] = []
    for source_id, record in records.items():
        payload = record.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        rows.append(
            {
                "row_id": f"source.{source_id}",
                "row_kind": "source",
                "source_id": source_id,
                "expected_value": "readable_terminal_hash_pinned",
                "observed_value": record.get("state"),
                "status": "preserved" if record.get("terminal") is True else record.get("state"),
                "verdict_class": (
                    payload.get("verdict_class")
                    if payload.get("verdict_class") in CLOSED_VERDICT_CLASSES
                    else "blocked"
                ),
                "evidence_authority": record.get("file_sha256"),
                "allowed_claim": "terminal source state only",
            }
        )
    for disagreement in disagreements:
        rows.append(
            {
                "row_id": f"disputed.exp6836.{disagreement['field']}",
                "row_kind": "disputed_field",
                "source_id": "exp6836",
                "expected_value": disagreement["producer_value"],
                "observed_value": disagreement["auditor_value"],
                "status": "producer_auditor_disagreement",
                "verdict_class": "disqualified",
                "evidence_authority": disagreement["controlling_authority"],
                "allowed_claim": "Exp6847 controls V599 readiness.",
            }
        )
    for decision_id, evidence, verdict_class in (
        ("resource_admission_block", frozen["exp6837_resource_block"], "blocked"),
        ("harmful_residual_memory", frozen["exp6842_harmful_learning"], "null"),
        ("zero_supervisor_headroom", frozen["exp6844_zero_headroom"], "blocked"),
        ("zero_tool_gap_obligations", frozen["exp6845_zero_obligations"], "blocked"),
        ("execution_time_arc_discovery", "hard-coded artifact paths retired", "null"),
    ):
        rows.append(
            {
                "row_id": f"method.{decision_id}",
                "row_kind": "method_decision",
                "source_id": decision_id,
                "expected_value": "preserve_terminal_result_and_change_mechanism",
                "observed_value": evidence,
                "status": "frozen",
                "verdict_class": verdict_class,
                "evidence_authority": "v598_terminal_artifact",
                "allowed_claim": "method change only; no benefit promotion",
            }
        )
    for skip in skips:
        rows.append(
            {
                "row_id": f"skip.{skip['task_id']}",
                "row_kind": "conductor_skip",
                "source_id": skip["task_id"],
                "expected_value": "GATE_BLOCK",
                "observed_value": skip["outcome"],
                "status": "terminal_skip_preserved",
                "verdict_class": "blocked",
                "evidence_authority": skip["log_line_sha256"],
                "allowed_claim": "skip classification only; no audit result",
            }
        )
    return rows


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Return a uniform gate row with its exact observed value."""

    return {"check": name, "expected": expected, "observed": observed, "passed": passed}


def _gate_checks(
    records: Mapping[str, Mapping[str, Any]],
    recomputation: Mapping[str, Any],
    frozen: Mapping[str, Any],
    skips: Sequence[Mapping[str, Any]],
    expected_hashes: Mapping[str, str] | None,
) -> list[JsonDict]:
    """Evaluate contract completeness without treating null findings as failures."""

    checks = [
        _check(
            f"source.{source_id}.readable_terminal",
            "present_terminal",
            record.get("state"),
            record.get("terminal") is True,
        )
        for source_id, record in records.items()
    ]
    if expected_hashes:
        for source_id, expected in expected_hashes.items():
            observed = records.get(source_id, {}).get("file_sha256")
            checks.append(
                _check(f"source_hash.{source_id}", expected, observed, expected == observed)
            )
    checks.extend(
        [
            _check(
                "exp6838.explicit_conductor_skip",
                "GATE_BLOCK",
                skips[0]["outcome"] if skips else None,
                len(skips) == 1 and skips[0]["outcome"] == "GATE_BLOCK",
            ),
            _check(
                "exp6836.controlling_compile_parity",
                False,
                recomputation["recomputed_compile_parity"],
                recomputation["recomputed_compile_parity"] is False,
            ),
            _check(
                "exp6836.controlling_typed_readiness",
                0,
                recomputation["recomputed_typed_obligation_program_ready_score"],
                recomputation["recomputed_typed_obligation_program_ready_score"] == 0,
            ),
            _check(
                "exp6837.resource_block_frozen",
                {
                    "exclusive_gpu_leases": False,
                    "live_canary_per_model": False,
                    "scientific_row_count": 0,
                    "compatibility_ready_score": 0,
                },
                frozen["exp6837_resource_block"],
                frozen["exp6837_resource_block"]
                == {
                    "exclusive_gpu_leases": False,
                    "live_canary_per_model": False,
                    "scientific_row_count": 0,
                    "compatibility_ready_score": 0,
                },
            ),
            _check(
                "exp6842.harmful_learning_frozen",
                {"held_future_rows": 540, "wins": 5, "losses": 69, "mean": -0.118519},
                frozen["exp6842_harmful_learning"],
                frozen["exp6842_harmful_learning"].get("held_future_rows") == 540
                and frozen["exp6842_harmful_learning"].get("wins") == 5
                and frozen["exp6842_harmful_learning"].get("losses") == 69
                and frozen["exp6842_harmful_learning"].get("mean_effect_vs_no_memory") == -0.118519
                and frozen["exp6842_harmful_learning"].get("continuous_self_learning_ready_score")
                == 0.0,
            ),
            _check(
                "exp6844.zero_headroom_frozen",
                {"action_rows": 60, "zero_headroom_rows": 60, "effect_ready": 0},
                frozen["exp6844_zero_headroom"],
                frozen["exp6844_zero_headroom"].get("action_row_count") == 60
                and frozen["exp6844_zero_headroom"].get("zero_headroom_action_row_count") == 60
                and frozen["exp6844_zero_headroom"].get("nonzero_headroom_action_row_count") == 0
                and frozen["exp6844_zero_headroom"].get("supervisor_effect_eligible_score") == 0,
            ),
            _check(
                "exp6845.zero_obligations_frozen",
                {"obligation_rows": 0, "zero_obligation_strata": 20, "effect_ready": 0},
                frozen["exp6845_zero_obligations"],
                frozen["exp6845_zero_obligations"].get("obligation_row_count") == 0
                and frozen["exp6845_zero_obligations"].get("stratum_count") == 20
                and frozen["exp6845_zero_obligations"].get("zero_obligation_stratum_count") == 20
                and frozen["exp6845_zero_obligations"].get("tool_gap_effect_eligible_score") == 0,
            ),
            _check(
                "promoted_reference_identifiers",
                8,
                sum(row["identifier_verified"] is True for row in REFERENCE_VERIFICATION_ROWS),
                len(REFERENCE_VERIFICATION_ROWS) == 8
                and all(row["identifier_verified"] is True for row in REFERENCE_VERIFICATION_ROWS),
            ),
        ]
    )
    return checks


def _preconditions(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize input availability without hiding explained or unexplained gaps."""

    source_failures = [
        row for row in checks if row["check"].endswith("readable_terminal") and not row["passed"]
    ]
    return {
        "checks": list(checks),
        "required_terminal_artifact_count": len(SOURCE_SPECS),
        "readable_terminal_artifact_count": len(SOURCE_SPECS) - len(source_failures),
        "missing_artifact_inventory": [
            {"check": row["check"], "observed": row["observed"]} for row in source_failures
        ],
        "explained_missing_artifacts": ["exp6838"]
        if any(
            row["check"] == "exp6838.explicit_conductor_skip" and row["passed"] for row in checks
        )
        else [],
        "v598_mechanism_rerun": False,
        "llm_invoked": False,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve every failed expected and observed value for a blocked contract."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_check": failed[0]["check"] if failed else None,
        "observed": failed[0]["observed"] if failed else "all checks pass",
        "failed_checks": failed,
        "checks": list(checks),
    }


def _with_field_principles(artifact: JsonDict) -> JsonDict:
    """Attach one plain-language principle to every emitted top-level field."""

    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} is required by REQ-REPORT-6848.") for key in artifact
    }
    return artifact


def build_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    source_records: Mapping[str, Mapping[str, Any]] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> JsonDict:
    """Build the V599 contract in memory without writing tracked state."""

    records = dict(source_records) if source_records is not None else load_source_records(repo_root)
    recomputation, disagreements = recompute_exp6836(
        _payload(records, "exp6836"), _payload(records, "exp6847")
    )
    frozen = freeze_v598_results(records)
    skips = scan_conductor_skips(repo_root)
    checks = _gate_checks(records, recomputation, frozen, skips, expected_hashes)
    ready = int(all(row["passed"] is True for row in checks))
    gate_summary = _gate_summary(checks)
    artifact: JsonDict = {
        "schema": "carnot.experiment_6848.v599_method_change_evidence_contract.v1",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete" if ready else "complete_blocked",
        "result_path": RESULT_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "random_seed": 6848,
        "field_principles": {},
        "preconditions_checked": _preconditions(checks),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": _source_hash_manifest(repo_root, records),
        "reproducibility_checksum": "",
        "rows": _evidence_rows(records, disagreements, frozen, skips),
        "producer_auditor_disagreements": disagreements,
        "exp6836_independent_recomputation": recomputation,
        "frozen_v598_results": frozen,
        "terminal_branch_manifest": _terminal_branch_manifest(records),
        "conductor_skip_manifest": skips,
        "retired_mechanism_manifest": _retired_mechanisms(frozen),
        "changed_mechanism_manifest": _changed_mechanisms(),
        "source_access_boundaries": _source_boundaries(),
        "dynamic_evidence_schema": dynamic_evidence_schema(),
        "reference_verification_rows": deepcopy_json(REFERENCE_VERIFICATION_ROWS),
        "v599_evidence_contract_ready_score": ready,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "null" if ready else "blocked",
        "honest_verdict": (
            "complete_null_v599_method_change_evidence_contract_ready_changed_mechanisms_only"
            if ready
            else "complete_blocked_v599_method_change_evidence_contract"
        ),
    }
    _with_field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def deepcopy_json(value: Any) -> Any:
    """Copy JSON data through its canonical representation."""

    return json.loads(canonical_json(value))


def validate_artifact(artifact: Mapping[str, Any], repo_root: Path | None = None) -> list[str]:
    """Return structural, checksum, and optional live-source validation errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles must cover every top-level field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal deterministic CPU evidence replay")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class must use the closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    ready = artifact.get("v599_evidence_contract_ready_score")
    failed = artifact.get("gate_check_summary", {}).get("failed_checks", [])
    if ready not in {0, 1}:
        errors.append("v599_evidence_contract_ready_score must be zero or one")
    if ready == 1 and (artifact.get("verdict_class") != "null" or failed):
        errors.append("ready contract must be null and have no failed contract gates")
    if ready == 0 and (artifact.get("verdict_class") != "blocked" or not failed):
        errors.append("blocked contract must record at least one failed gate")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a nonempty list")
    elif any(row.get("verdict_class") not in CLOSED_VERDICT_CLASSES for row in rows):
        errors.append("every evidence row must use the closed verdict enum")
    branches = artifact.get("terminal_branch_manifest")
    if not isinstance(branches, list) or any(
        row.get("source_verdict_class") not in CLOSED_VERDICT_CLASSES for row in branches
    ):
        errors.append("terminal_branch_manifest must preserve closed source verdicts")
    references = artifact.get("reference_verification_rows")
    if (
        not isinstance(references, list)
        or len(references) != 8
        or not all(row.get("identifier_verified") is True for row in references)
    ):
        errors.append("all eight promoted reference identifiers must be verified")
    consumers = {
        consumer
        for row in artifact.get("changed_mechanism_manifest", [])
        if isinstance(row, Mapping)
        for consumer in row.get("consumers", [])
    }
    if not {"exp6849", "exp6850", "exp6853", "exp6857"} <= consumers:
        errors.append("the exact V599 readiness consumers must be declared")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if repo_root is not None:
        manifest = artifact.get("source_artifact_hashes")
        if isinstance(manifest, Mapping):
            for failure in validate_pinned_hashes(repo_root, manifest):
                errors.append(
                    f"source drift: {failure['check']} expected={failure['expected']} "
                    f"observed={failure['observed']}"
                )
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one complete JSON object so readers never see a partial contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate the Exp6848 artifact without invoking any producer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260901")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else args.repo_root / RESULT_PATH
    if args.validate:
        try:
            artifact = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"failed to read artifact: {exc}")
            return 1
        errors = validate_artifact(artifact, args.repo_root)
        for error in errors:
            print(error)
        return int(bool(errors))

    start = time.monotonic()
    records = load_source_records(args.repo_root)
    artifact = build_artifact(
        args.repo_root,
        run_date=str(args.date),
        duration_s=0.0,
        source_records=records,
    )
    artifact["duration_s"] = round(time.monotonic() - start, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        for error in errors:
            print(error)
        return 1
    write_json_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper exercises this entry point.
    raise SystemExit(main())
