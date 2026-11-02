"""Merge source and semantic relation qualifications into an event bank.

Spec refs: REQ-CONSTRAINT-6915 and SCENARIO-CONSTRAINT-6915-*.

This reducer joins saved rows only. It does not run a model or turn control
success into model evidence. Exact execution is the oracle for admission.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]
VerifyFn = Callable[[str], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/constraint-verification/spec.md"
RESULT_PATH = Path("results/experiment_6915_qualified_relation_event_bank.json")
SOURCE_PATHS = {
    "exp6913": Path("results/experiment_6913_relation_source_tuple_qualification.json"),
    "exp6914": Path("results/experiment_6914_relation_asp_isomorphic_qualification.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6913": "sha256:d46c81632ba9c3a06770f2933cca0803b17035ca13351dd9f9b27ebf61730990",
    "exp6914": "sha256:f93e00c38c0c44a031c87a9256d3b7fa71ee1001e9501e942d143e7e06239b08",
}
REQUIRED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_MODEL_FAMILIES = ("qwen_moe", "gemma_dense", "gemma_moe")
REQUIRED_SEEDS = (6899, 6900, 6901, 6902)
REQUIRED_FAMILIES = (
    "graph_coloring",
    "scheduling",
    "non_monotonic_defaults",
    "contradictions",
    "cardinality_constraints",
)
PERTURBATIONS = (
    "base",
    "entity_renaming",
    "relation_paraphrase",
    "relation_reversal",
    "contradiction_injection",
    "relation_omission",
    "solution_space_restructuring",
)
ENOKI_ARM = "enoki:pinned_openie_encoder"
RULE_ARM = "rule:anchored_lexical_v1"
EXPECTED_CELL_COUNT = 1_400
MINIMUM_EVENTS = 90
MINIMUM_PER_MODEL_FAMILY = 10
RANDOM_SEED = 2609036915
SCHEMA = "carnot.exp6915.qualified_relation_event_bank.v1"
INFERENCE_SUBSTRATE = "deterministic_cpu_qualification_merge_no_llm"
BLOCKED_VERDICT = "complete_blocked_qualified_relation_event_bank"
READY_VERDICT = "complete_circular_positive_qualified_relation_event_bank"
DISQUALIFIED_VERDICT = "complete_disqualified_qualified_relation_event_bank_thresholds_not_met"

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "join_rows",
    "missing_join_rows",
    "duplicate_join_rows",
    "eligibility_rows",
    "admitted_event_rows",
    "rejected_event_rows",
    "rejection_reason_rows",
    "model_summary_rows",
    "family_summary_rows",
    "seed_summary_rows",
    "enoki_control_rows",
    "rule_control_rows",
    "control_substitution_count",
    "source_group_headroom_rows",
    "admitted_event_bank_manifest",
    "fresh_adversarial_rows",
    "reported_vs_recomputed_metrics",
    "random_seed",
    "reproducibility_checksum",
    "qualified_model_relation_event_count",
    "qualified_relation_event_bank_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why its evidence is present.",
    "preconditions_checked": "Unsafe shard inputs stop the merge before eligibility decisions.",
    "inference_substrate": "The exact value records deterministic CPU merging with no model call.",
    "duration_s": "Measured wall time proves that a fresh merge process ran.",
    "source_artifact_hashes": "Exact hashes bind the bank to both immutable qualification shards.",
    "rows": "One row per cell keeps every admitted, rejected, and control decision visible.",
    "join_rows": "One compact row per cell and perturbation makes join evidence replayable.",
    "missing_join_rows": "Missing source or semantic components cannot disappear from the merge.",
    "duplicate_join_rows": "Duplicate identities cannot receive extra credit or replace a row.",
    "eligibility_rows": "Component conjunctions determine cell admission without shard headlines.",
    "admitted_event_rows": "Only qualified model-produced cells enter the event bank.",
    "rejected_event_rows": "Every rejected model cell stays available for audit and diagnosis.",
    "rejection_reason_rows": "Exact reason counts expose the dominant qualification failures.",
    "model_summary_rows": "Each model family keeps its own count and cannot borrow pooled credit.",
    "family_summary_rows": "Each constraint family stays visible before the five-family gate.",
    "seed_summary_rows": "Seed summaries expose repeated or seed-specific failures.",
    "enoki_control_rows": "Enoki evidence remains a control and cannot count as model output.",
    "rule_control_rows": "Lexical rule evidence remains a control and cannot count as model output.",
    "control_substitution_count": "Zero proves that no control entered the model event count.",
    "source_group_headroom_rows": "Each group needs admitted and rejected model cells for headroom.",
    "admitted_event_bank_manifest": "Stable event identities and tuple payloads support later replay.",
    "fresh_adversarial_rows": "Current verifier results prevent stale clean stamps from passing.",
    "reported_vs_recomputed_metrics": "Fresh row replay detects omissions and aggregate drift.",
    "random_seed": "A fixed non-identity seed pins the deterministic merge contract.",
    "reproducibility_checksum": "A stable digest detects decision drift across fresh processes.",
    "qualified_model_relation_event_count": "The count equals admitted model rows directly.",
    "qualified_relation_event_bank_ready_score": "One requires exact joins and every event floor.",
    "gate_check_summary": "Each failed check records exact expected and observed values.",
    "verifier_is_oracle": "True discloses that exact source and solver checks define admission.",
    "verdict_class": "The closed class prevents oracle-backed readiness from claiming positive.",
    "honest_verdict": "A complete prefix marks the result as terminal.",
    "gate_check_summary.checks": "The complete check list prevents hidden gate failures.",
    "gate_check_summary.passed": "One Boolean records whether all listed checks passed.",
    "gate_check_summary.failed_check": "The first failed check gives an actionable name.",
    "gate_check_summary.failed_checks": "All failed check names remain visible.",
    "gate_check_summary.expected": "The first failed check records its required value.",
    "gate_check_summary.observed": "The first failed check records the value that failed.",
    "gate_check.check": "Each check has a stable machine-readable name.",
    "gate_check.expected": "Each check records its exact required value.",
    "gate_check.observed": "Each check records its exact observed value.",
    "gate_check.passed": "Each check records its own equality result.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON in one stable form."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return a repository-style SHA-256 value."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value after stable JSON serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    """Hash exact file bytes."""

    return sha256_bytes(Path(path).read_bytes())


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Build one exact expected-versus-observed gate row."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all failures and expose the first failure in a fixed shape."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = [row for row in copied if row.get("passed") is not True]
    return {
        "checks": copied,
        "passed": not failed,
        "failed_check": failed[0].get("check") if failed else None,
        "failed_checks": [row.get("check") for row in failed],
        "expected": failed[0].get("expected") if failed else "all checks pass",
        "observed": failed[0].get("observed") if failed else "all checks pass",
    }


def expected_cell_identities() -> set[str]:
    """Build the fixed 1,400-cell matrix without reading an observed manifest."""

    fixtures = [f"{family}_{ordinal:02d}" for ordinal in range(20) for family in REQUIRED_FAMILIES]
    identities = {
        f"{model}::{seed}::{fixture}"
        for model in REQUIRED_MODELS
        for seed in REQUIRED_SEEDS
        for fixture in fixtures
    }
    identities.update(
        f"{arm}::deterministic::{fixture}" for arm in (ENOKI_ARM, RULE_ARM) for fixture in fixtures
    )
    return identities


def _manifest(values: Sequence[str]) -> JsonDict:
    ordered = sorted(values)
    return {"count": len(ordered), "sha256": sha256_json(ordered)}


def _seed_label(value: Any) -> str:
    return "deterministic" if value is None else str(value)


def _producer_kind(arm: str) -> str:
    if arm == ENOKI_ARM:
        return "enoki_control"
    if arm == RULE_ARM:
        return "rule_control"
    return "model" if arm.startswith("gguf:") else "unsupported_control"


def _expected_identity(row: Mapping[str, Any]) -> str:
    model_id = str(row.get("model_id", ""))
    return f"{model_id}::{_seed_label(row.get('seed'))}::{row.get('fixture_id')}"


def _critical_flags(report: Mapping[str, Any]) -> list[JsonDict]:
    return [
        deepcopy(dict(row))
        for row in report.get("flags", [])
        if isinstance(row, Mapping) and str(row.get("severity", "")).lower() == "critical"
    ]


def fresh_adversarial_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Keep each current verifier result and all of its flags."""

    rows: list[JsonDict] = []
    for name in sorted(reports):
        report = reports[name]
        flags = [deepcopy(dict(row)) for row in report.get("flags", []) if isinstance(row, Mapping)]
        rows.append(
            {
                "row_type": "fresh_adversarial_summary",
                "source": name,
                "loaded": report.get("loaded") is True,
                "gate_version": report.get("gate_version"),
                "flag_count": len(flags),
                "critical_count": len(_critical_flags(report)),
            }
        )
        rows.extend(
            {"row_type": "fresh_adversarial_finding", "source": name, **row} for row in flags
        )
    return rows


def validate_preconditions(
    *,
    source_shard: Mapping[str, Any],
    semantic_shard: Mapping[str, Any],
    observed_hashes: Mapping[str, str],
    expected_hashes: Mapping[str, str],
    expected_cell_ids: set[str],
    perturbations: Sequence[str],
    fresh_reports: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Check readiness, hashes, identity manifests, and current verifier flags."""

    source_rows = [row for row in source_shard.get("rows", []) if isinstance(row, Mapping)]
    semantic_rows = [row for row in semantic_shard.get("rows", []) if isinstance(row, Mapping)]
    source_ids = [str(row.get("cell_identity")) for row in source_rows]
    semantic_ids = [
        f"{row.get('cell_identity')}::{row.get('perturbation')}" for row in semantic_rows
    ]
    wanted_joins = [
        f"{cell_identity}::{perturbation}"
        for cell_identity in sorted(expected_cell_ids)
        for perturbation in perturbations
    ]
    source_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_id[str(row.get("cell_identity"))].append(row)
    metadata_ok = all(
        len(source_by_id[str(row.get("cell_identity"))]) == 1
        and str(row.get("cell_identity")) in expected_cell_ids
        and _expected_identity(row) == row.get("cell_identity")
        for row in source_rows
    )
    semantic_metadata_ok = all(
        str(row.get("cell_identity")) in source_by_id
        and len(source_by_id[str(row.get("cell_identity"))]) == 1
        and str(row.get("perturbation")) in perturbations
        and row.get("row_id") == f"{row.get('cell_identity')}::{row.get('perturbation')}"
        for row in semantic_rows
    )
    checks = [
        gate_check(
            "source_tuple_shard_ready_score",
            1,
            source_shard.get("source_tuple_shard_ready_score"),
        ),
        gate_check(
            "asp_isomorphic_shard_ready_score",
            1,
            semantic_shard.get("asp_isomorphic_shard_ready_score"),
        ),
        *[
            gate_check(f"source_hash:{name}", expected, observed_hashes.get(name))
            for name, expected in expected_hashes.items()
        ],
        gate_check(
            "source_cell_identity_manifest",
            _manifest(list(expected_cell_ids)),
            _manifest(source_ids),
        ),
        gate_check(
            "semantic_join_identity_manifest",
            _manifest(wanted_joins),
            _manifest(semantic_ids),
        ),
        gate_check("source_identity_metadata", True, metadata_ok),
        gate_check("semantic_identity_metadata", True, semantic_metadata_ok),
        gate_check(
            "source_shard_not_flagged",
            False,
            source_shard.get("flagged_adversarial") is True,
        ),
        gate_check(
            "semantic_shard_not_flagged",
            False,
            semantic_shard.get("flagged_adversarial") is True,
        ),
    ]
    for name in ("exp6913", "exp6914"):
        report = fresh_reports.get(name, {})
        checks.extend(
            [
                gate_check(
                    f"fresh_adversarial_verifier_loaded:{name}",
                    True,
                    report.get("loaded") is True,
                ),
                gate_check(
                    f"fresh_adversarial_critical_count:{name}",
                    0,
                    len(_critical_flags(report)),
                ),
            ]
        )
    return gate_summary(checks)


def _source_reasons(row: Mapping[str, Any]) -> list[str]:
    checks = (
        ("raw_output_check", "source_raw_output_failed"),
        ("parser_check", "source_parser_failed"),
        ("source_offset_check", "source_offsets_failed"),
        ("source_byte_identity_check", "source_byte_identity_failed"),
        ("tuple_type_check", "tuple_type_invalid"),
        ("entity_anchor_check", "entity_anchor_invalid"),
        ("relation_direction_check", "relation_direction_invalid"),
        ("omission_check", "proposal_omission"),
        ("duplicate_check", "duplicate_proposal"),
        ("abstention_check", "invalid_abstention"),
    )
    reasons = [reason for field, reason in checks if row.get(field, {}).get("passed") is not True]
    if row.get("flagged_adversarial") is True:
        reasons.append("source_component_flagged")
    if row.get("terminal") is not True:
        reasons.append("source_row_nonterminal")
    return reasons


def _effect_projection(value: Any) -> JsonDict | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "model_count": value.get("model_count"),
        "models": value.get("models"),
        "models_sha256": value.get("models_sha256"),
        "satisfiable": value.get("satisfiable"),
    }


def _effects_equal(left: Any, right: Any) -> bool:
    return _effect_projection(left) is not None and _effect_projection(left) == _effect_projection(
        right
    )


def _expected_effect_matches(row: Mapping[str, Any], perturbation: str) -> bool:
    expected = row.get("expected_effect")
    observed = row.get("observed_effect")
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        return False
    kind = expected.get("kind")
    if perturbation in {"base", "entity_renaming", "relation_paraphrase"}:
        return kind in {
            "sealed_exact_models",
            "renamed_sealed_models",
            "paraphrased_sealed_models",
        } and all(
            observed.get(key) == expected.get(key)
            for key in ("model_count", "models", "satisfiable")
        )
    if perturbation == "contradiction_injection":
        return kind == "contradiction" and observed.get("satisfiable") is False
    if perturbation == "relation_omission":
        return kind == "all_proposal_relation_facts_removed" and row.get("tuples") == []
    if perturbation == "solution_space_restructuring":
        return (
            kind == "same_projected_models_and_satisfiability_different_model_count"
            and isinstance(row.get("auxiliary_atom"), str)
            and row.get("shortcut_detected") is False
        )
    return False


def _semantic_reasons(row: Mapping[str, Any], perturbation: str) -> list[str]:
    reasons: list[str] = []
    if row.get("flagged_adversarial") is True:
        reasons.append(f"semantic_component_flagged:{perturbation}")
    if row.get("terminal") is not True:
        reasons.append(f"semantic_row_nonterminal:{perturbation}")
    if row.get("transform_valid") is not True:
        reasons.append(f"transform_invalid:{perturbation}")
    if perturbation == "relation_reversal":
        directional_rejection = (
            row.get("compilation_status") == "unsupported_atom"
            and bool(row.get("unsupported_atoms"))
            and row.get("program") is None
            and row.get("expected_effect", {}).get("kind") == "directional_vocabulary_rejection"
            and row.get("primary_solver_receipt", {}).get("status") == "not_run"
            and row.get("independent_solver_receipt", {}).get("status") == "not_run"
            and row.get("pair_requirement_met") is True
            and row.get("expected_effect_met") is True
        )
        if not directional_rejection:
            reasons.append(f"perturbation_behavior_failed:{perturbation}")
        return reasons
    unsupported = row.get("compilation_status") != "compiled" or bool(row.get("unsupported_atoms"))
    if unsupported:
        reasons.append(f"unsupported_component:{perturbation}")
    atoms = row.get("atoms")
    program_atoms = row.get("program_atoms")
    atom_membership = (
        isinstance(atoms, list)
        and isinstance(program_atoms, list)
        and set(map(str, atoms)) <= set(map(str, program_atoms))
    )
    if row.get("exact_atom_valid") is not True or not atom_membership:
        reasons.append(f"exact_atom_invalid:{perturbation}")
    program_hash = row.get("program_sha256")
    primary_receipt = row.get("primary_solver_receipt", {})
    independent_receipt = row.get("independent_solver_receipt", {})
    parity = (
        not unsupported
        and row.get("solver_parity") is True
        and isinstance(primary_receipt, Mapping)
        and isinstance(independent_receipt, Mapping)
        and primary_receipt.get("status") == "complete"
        and independent_receipt.get("status") == "complete"
        and primary_receipt.get("program_sha256") == program_hash
        and independent_receipt.get("program_sha256") == program_hash
        and _effects_equal(row.get("primary_effect"), row.get("independent_effect"))
    )
    if not parity:
        reasons.append(f"solver_parity_failed:{perturbation}")
    behavior = (
        row.get("expected_effect_met") is True
        and row.get("pair_requirement_met") is True
        and _expected_effect_matches(row, perturbation)
    )
    if perturbation in {"entity_renaming", "relation_paraphrase"}:
        behavior = behavior and row.get("isomorphic_invariant") is True
    if not behavior:
        reasons.append(f"perturbation_behavior_failed:{perturbation}")
    return reasons


def _identity_reasons(
    source: Mapping[str, Any], semantic: Mapping[str, Any], perturbation: str
) -> list[str]:
    fields = ("cell_identity", "arm", "model_id", "fixture_id", "family", "split")
    reasons = [
        f"identity_mismatch:{field}" for field in fields if source.get(field) != semantic.get(field)
    ]
    if _seed_label(source.get("seed")) != _seed_label(semantic.get("seed")):
        reasons.append("identity_mismatch:seed")
    if semantic.get("row_id") != f"{source.get('cell_identity')}::{perturbation}":
        reasons.append("perturbation_identity_mismatch")
    return reasons


def build_join_evidence(
    source_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    expected_cell_ids: set[str],
    perturbations: Sequence[str],
) -> JsonDict:
    """Build one deterministic join receipt for each expected cell and perturbation."""

    source_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    semantic_index: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_index[str(row.get("cell_identity"))].append(row)
    for row in semantic_rows:
        semantic_index[(str(row.get("cell_identity")), str(row.get("perturbation")))].append(row)
    missing_rows: list[JsonDict] = []
    duplicate_rows: list[JsonDict] = []
    for cell_identity in sorted(expected_cell_ids):
        source_count = len(source_index[cell_identity])
        if source_count == 0:
            missing_rows.append(
                {
                    "cell_identity": cell_identity,
                    "perturbation_id": None,
                    "source_occurrence_count": 0,
                    "semantic_occurrence_count": sum(
                        len(semantic_index[(cell_identity, perturbation)])
                        for perturbation in perturbations
                    ),
                    "reason": "missing_source_join_row",
                }
            )
        elif source_count > 1:
            duplicate_rows.append(
                {
                    "cell_identity": cell_identity,
                    "perturbation_id": None,
                    "occurrence_count": source_count,
                    "reason": "duplicate_source_cell_id",
                }
            )
        for perturbation in perturbations:
            semantic_count = len(semantic_index[(cell_identity, perturbation)])
            if semantic_count == 0:
                missing_rows.append(
                    {
                        "cell_identity": cell_identity,
                        "perturbation_id": perturbation,
                        "source_occurrence_count": source_count,
                        "semantic_occurrence_count": 0,
                        "reason": "missing_semantic_join_row",
                    }
                )
            elif semantic_count > 1:
                duplicate_rows.append(
                    {
                        "cell_identity": cell_identity,
                        "perturbation_id": perturbation,
                        "occurrence_count": semantic_count,
                        "reason": "duplicate_semantic_join_id",
                    }
                )
    join_rows = []
    for cell_identity in sorted(expected_cell_ids):
        for perturbation in perturbations:
            sources = source_index[cell_identity]
            semantics = semantic_index[(cell_identity, perturbation)]
            reasons: list[str] = []
            if len(sources) == 0:
                reasons.append("missing_source_join_row")
            elif len(sources) > 1:
                reasons.append("duplicate_source_cell_id")
            if len(semantics) == 0:
                reasons.append("missing_semantic_join_row")
            elif len(semantics) > 1:
                reasons.append("duplicate_semantic_join_id")
            source = sources[0] if sources else {}
            semantic = semantics[0] if semantics else {}
            if len(sources) == 1 and len(semantics) == 1:
                reasons.extend(_identity_reasons(source, semantic, perturbation))
                reasons.extend(_source_reasons(source))
                reasons.extend(_semantic_reasons(semantic, perturbation))
            arm = str(source.get("arm", semantic.get("arm", "")))
            join_rows.append(
                {
                    "row_id": f"{cell_identity}::{perturbation}",
                    "cell_identity": cell_identity,
                    "perturbation_id": perturbation,
                    "source_occurrence_count": len(sources),
                    "semantic_occurrence_count": len(semantics),
                    "arm": arm,
                    "model_id": source.get("model_id", semantic.get("model_id")),
                    "model_family": source.get("model_family"),
                    "seed": source.get("seed", semantic.get("seed")),
                    "fixture_id": source.get("fixture_id", semantic.get("fixture_id")),
                    "group_id": source.get("group_id"),
                    "family": source.get("family", semantic.get("family")),
                    "split": source.get("split", semantic.get("split")),
                    "producer_kind": _producer_kind(arm),
                    "source_text_hash": source.get("source_text_hash"),
                    "source_bytes_b64": source.get("source_bytes_b64"),
                    "source_tuple_fields": deepcopy(source.get("parsed_tuple_fields", [])),
                    "program_sha256": semantic.get("program_sha256"),
                    "primary_effect": deepcopy(semantic.get("primary_effect")),
                    "expected_effect": deepcopy(semantic.get("expected_effect")),
                    "reasons": list(dict.fromkeys(reasons)),
                    "decision": "qualified" if not reasons else "rejected",
                    "terminal": True,
                }
            )
    return {
        "join_rows": join_rows,
        "missing_join_rows": missing_rows,
        "duplicate_join_rows": duplicate_rows,
    }


def derive_eligibility_rows(
    join_rows: Sequence[Mapping[str, Any]], perturbations: Sequence[str]
) -> list[JsonDict]:
    """Reduce joined perturbations to one model or control decision per cell."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in join_rows:
        grouped[str(row.get("cell_identity"))].append(row)
    result = []
    wanted = set(perturbations)
    for cell_identity in sorted(grouped):
        members = grouped[cell_identity]
        first = members[0]
        observed = {str(row.get("perturbation_id")) for row in members}
        reasons = [str(reason) for row in members for reason in row.get("reasons", [])]
        if observed != wanted or len(members) != len(perturbations):
            reasons.append("incomplete_perturbation_join")
        reasons = list(dict.fromkeys(reasons))
        producer_kind = str(first.get("producer_kind"))
        model_produced = producer_kind == "model"
        eligible = model_produced and not reasons
        if not model_produced:
            reasons.append("control_not_model_produced")
        base = next(
            (row for row in members if row.get("perturbation_id") == "base"),
            {},
        )
        result.append(
            {
                "row_type": "joined_cell_eligibility",
                "cell_identity": cell_identity,
                "arm": first.get("arm"),
                "model_id": first.get("model_id"),
                "model_family": first.get("model_family"),
                "seed": first.get("seed"),
                "fixture_id": first.get("fixture_id"),
                "group_id": first.get("group_id"),
                "family": first.get("family"),
                "split": first.get("split"),
                "producer_kind": producer_kind,
                "model_produced": model_produced,
                "perturbation_ids": [
                    perturbation for perturbation in perturbations if perturbation in observed
                ],
                "source_text_hash": first.get("source_text_hash"),
                "source_bytes_b64": first.get("source_bytes_b64"),
                "source_tuple_fields": deepcopy(first.get("source_tuple_fields", [])),
                "base_program_sha256": base.get("program_sha256"),
                "base_primary_effect": deepcopy(base.get("primary_effect")),
                "eligible": eligible,
                "decision": "admitted"
                if eligible
                else "rejected"
                if model_produced
                else "control_only",
                "reasons": reasons,
                "terminal": True,
            }
        )
    return result


def _model_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("model_produced") is True]


def _summary_rows(rows: Sequence[Mapping[str, Any]], field: str) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _model_rows(rows):
        grouped[_seed_label(row.get(field))].append(row)
    result = []
    for value in sorted(grouped):
        members = grouped[value]
        admitted = sum(row.get("eligible") is True for row in members)
        item = {
            field: value,
            "candidate_event_count": len(members),
            "admitted_event_count": admitted,
            "rejected_event_count": len(members) - admitted,
        }
        if field == "model_id":
            item["model_family"] = members[0].get("model_family")
        result.append(item)
    return result


def summarize_eligibility(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build all counts directly from terminal cell eligibility rows."""

    model_rows = _model_rows(rows)
    admitted = [deepcopy(dict(row)) for row in model_rows if row.get("eligible") is True]
    rejected = [deepcopy(dict(row)) for row in model_rows if row.get("eligible") is not True]
    controls = [row for row in rows if row.get("model_produced") is not True]
    reason_counts = Counter(str(reason) for row in rejected for reason in row.get("reasons", []))
    group_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in model_rows:
        group_rows[str(row.get("group_id"))].append(row)
    headroom = []
    for group_id in sorted(group_rows):
        members = group_rows[group_id]
        admitted_count = sum(row.get("eligible") is True for row in members)
        rejected_count = len(members) - admitted_count
        headroom.append(
            {
                "source_group": group_id,
                "candidate_event_count": len(members),
                "admitted_event_count": admitted_count,
                "rejected_event_count": rejected_count,
                "positive_headroom": admitted_count > 0 and rejected_count > 0,
            }
        )
    control_substitution_count = sum(
        row.get("eligible") is True or row.get("decision") == "admitted" for row in controls
    )
    manifest = [
        {
            "event_id": sha256_json(
                {
                    "cell_identity": row["cell_identity"],
                    "source_tuple_fields": row.get("source_tuple_fields"),
                    "base_program_sha256": row.get("base_program_sha256"),
                }
            ),
            "cell_identity": row["cell_identity"],
            "model_id": row.get("model_id"),
            "model_family": row.get("model_family"),
            "seed": row.get("seed"),
            "family": row.get("family"),
            "source_group": row.get("group_id"),
            "split": row.get("split"),
            "source_text_hash": row.get("source_text_hash"),
            "source_bytes_b64": row.get("source_bytes_b64"),
            "source_tuple_fields": deepcopy(row.get("source_tuple_fields", [])),
            "base_program_sha256": row.get("base_program_sha256"),
            "required_perturbations": deepcopy(row.get("perturbation_ids", [])),
        }
        for row in admitted
    ]
    return {
        "admitted_event_rows": admitted,
        "rejected_event_rows": rejected,
        "rejection_reason_rows": [
            {"reason": reason, "rejected_event_count": reason_counts[reason]}
            for reason in sorted(reason_counts)
        ],
        "model_summary_rows": _summary_rows(rows, "model_id"),
        "family_summary_rows": _summary_rows(rows, "family"),
        "seed_summary_rows": _summary_rows(rows, "seed"),
        "enoki_control_rows": [
            deepcopy(dict(row)) for row in controls if row.get("producer_kind") == "enoki_control"
        ],
        "rule_control_rows": [
            deepcopy(dict(row)) for row in controls if row.get("producer_kind") == "rule_control"
        ],
        "control_substitution_count": control_substitution_count,
        "source_group_headroom_rows": headroom,
        "admitted_event_bank_manifest": manifest,
        "qualified_model_relation_event_count": len(admitted),
    }


def _threshold_checks(
    *,
    metrics: Mapping[str, Any],
    join_evidence: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    expected_cell_count: int,
    perturbation_count: int,
    required_model_families: Sequence[str],
    required_families: Sequence[str],
    minimum_events: int,
    minimum_per_model_family: int,
) -> list[JsonDict]:
    model_family_counts: Counter[str] = Counter()
    for row in metrics["admitted_event_rows"]:
        model_family_counts[str(row.get("model_family"))] += 1
    family_counts = {
        str(row.get("family")): int(row.get("admitted_event_count", 0))
        for row in metrics["family_summary_rows"]
    }
    checks = [
        gate_check("preconditions_passed", True, preconditions_checked.get("passed") is True),
        gate_check("missing_join_row_count", 0, len(join_evidence["missing_join_rows"])),
        gate_check("duplicate_join_row_count", 0, len(join_evidence["duplicate_join_rows"])),
        gate_check(
            "complete_join_row_count",
            expected_cell_count * perturbation_count,
            len(join_evidence["join_rows"]),
        ),
        gate_check(
            "qualified_model_relation_event_count",
            f">={minimum_events}",
            metrics["qualified_model_relation_event_count"],
            passed=metrics["qualified_model_relation_event_count"] >= minimum_events,
        ),
    ]
    checks.extend(
        gate_check(
            f"minimum_events:model_family:{family}",
            f">={minimum_per_model_family}",
            model_family_counts[family],
            passed=model_family_counts[family] >= minimum_per_model_family,
        )
        for family in required_model_families
    )
    checks.extend(
        gate_check(
            f"constraint_family_present:{family}",
            ">=1",
            family_counts.get(family, 0),
            passed=family_counts.get(family, 0) >= 1,
        )
        for family in required_families
    )
    checks.extend(
        [
            gate_check(
                "source_group_positive_headroom",
                True,
                bool(metrics["source_group_headroom_rows"])
                and all(row["positive_headroom"] for row in metrics["source_group_headroom_rows"]),
            ),
            gate_check("control_substitution_count", 0, metrics["control_substitution_count"]),
        ]
    )
    return checks


def replay_reported_metrics(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute every output summary from eligibility rows."""

    eligibility = artifact.get("eligibility_rows", [])
    recomputed = summarize_eligibility(eligibility)
    names = tuple(recomputed)
    reported = {name: deepcopy(artifact.get(name)) for name in names}
    agreement = reported == recomputed and artifact.get("rows") == eligibility
    return {
        "reported_metrics_sha256": sha256_json(reported),
        "recomputed_metrics_sha256": sha256_json(recomputed),
        "agreement": agreement,
        "comparisons": [
            {
                "metric": name,
                "passed": reported[name] == recomputed[name],
            }
            for name in names
        ],
    }


def _attach_principles(artifact: JsonDict) -> None:
    principles = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves one required merge receipt.")
        for key in artifact
    }
    for key in REQUIRED_ARTIFACT_FIELDS:
        principles[key] = FIELD_PRINCIPLES[key]
    for key, principle in FIELD_PRINCIPLES.items():
        if key.startswith("gate_check"):
            principles[key] = principle
    for row in artifact.get("gate_check_summary", {}).get("checks", []):
        principles[f"gate:{row.get('check')}"] = (
            "This exact expected-versus-observed check prevents unsupported bank readiness."
        )
    artifact["field_principles"] = principles


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding wall time, principles, and this hash."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
        }
    )


def blocked_artifact(
    *,
    date: str,
    duration_s: float,
    source_artifact_hashes: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    fresh_adversarial_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the full schema when a shard cannot enter the merge."""

    summary = deepcopy(dict(preconditions_checked))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6915,
        "run_date": date,
        "status": "blocked",
        "field_principles": {},
        "preconditions_checked": summary,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "qualification_thresholds": {
            "minimum_events": MINIMUM_EVENTS,
            "minimum_per_model_family": MINIMUM_PER_MODEL_FAMILY,
            "required_model_families": list(REQUIRED_MODEL_FAMILIES),
            "required_families": list(REQUIRED_FAMILIES),
            "perturbations": list(PERTURBATIONS),
            "expected_cell_count": EXPECTED_CELL_COUNT,
        },
        "rows": [],
        "join_rows": [],
        "missing_join_rows": [],
        "duplicate_join_rows": [],
        "eligibility_rows": [],
        "admitted_event_rows": [],
        "rejected_event_rows": [],
        "rejection_reason_rows": [],
        "model_summary_rows": [],
        "family_summary_rows": [],
        "seed_summary_rows": [],
        "enoki_control_rows": [],
        "rule_control_rows": [],
        "control_substitution_count": 0,
        "source_group_headroom_rows": [],
        "admitted_event_bank_manifest": [],
        "fresh_adversarial_rows": [deepcopy(dict(row)) for row in fresh_adversarial_rows],
        "reported_vs_recomputed_metrics": {
            "reported_metrics_sha256": sha256_json({}),
            "recomputed_metrics_sha256": sha256_json({}),
            "agreement": True,
            "comparisons": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "qualified_model_relation_event_count": 0,
        "qualified_relation_event_bank_ready_score": 0,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    duration_s: float,
    source_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    expected_cell_ids: set[str],
    perturbations: Sequence[str],
    required_model_families: Sequence[str],
    required_families: Sequence[str],
    minimum_events: int,
    minimum_per_model_family: int,
    fresh_adversarial_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Join both shards, derive eligibility, and evaluate the bank thresholds."""

    if preconditions_checked.get("passed") is not True:
        return blocked_artifact(
            date=date,
            duration_s=duration_s,
            source_artifact_hashes=source_artifact_hashes,
            preconditions_checked=preconditions_checked,
            fresh_adversarial_rows=fresh_adversarial_rows,
        )
    join_evidence = build_join_evidence(
        source_rows, semantic_rows, expected_cell_ids, perturbations
    )
    eligibility = derive_eligibility_rows(join_evidence["join_rows"], perturbations)
    metrics = summarize_eligibility(eligibility)
    checks = _threshold_checks(
        metrics=metrics,
        join_evidence=join_evidence,
        preconditions_checked=preconditions_checked,
        expected_cell_count=len(expected_cell_ids),
        perturbation_count=len(perturbations),
        required_model_families=required_model_families,
        required_families=required_families,
        minimum_events=minimum_events,
        minimum_per_model_family=minimum_per_model_family,
    )
    thresholds = {
        "minimum_events": int(minimum_events),
        "minimum_per_model_family": int(minimum_per_model_family),
        "required_model_families": list(required_model_families),
        "required_families": list(required_families),
        "perturbations": list(perturbations),
        "expected_cell_count": len(expected_cell_ids),
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6915,
        "run_date": date,
        "status": "complete",
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions_checked)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "qualification_thresholds": thresholds,
        "rows": deepcopy(eligibility),
        "join_rows": join_evidence["join_rows"],
        "missing_join_rows": join_evidence["missing_join_rows"],
        "duplicate_join_rows": join_evidence["duplicate_join_rows"],
        "eligibility_rows": eligibility,
        **metrics,
        "fresh_adversarial_rows": [deepcopy(dict(row)) for row in fresh_adversarial_rows],
        "reported_vs_recomputed_metrics": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "qualified_relation_event_bank_ready_score": 0,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "disqualified",
        "honest_verdict": DISQUALIFIED_VERDICT,
    }
    replay = replay_reported_metrics(artifact)
    checks.append(gate_check("reported_aggregates_match_rows", True, replay["agreement"]))
    summary = gate_summary(checks)
    ready = int(summary["passed"])
    artifact.update(
        {
            "reported_vs_recomputed_metrics": replay,
            "qualified_relation_event_bank_ready_score": ready,
            "gate_check_summary": summary,
            "verdict_class": "circular_positive" if ready else "disqualified",
            "honest_verdict": READY_VERDICT if ready else DISQUALIFIED_VERDICT,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, row coverage, aggregate replay, gates, and checksum."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    principles = artifact.get("field_principles", {})
    gate_principles = {
        f"gate:{row.get('check')}"
        for row in artifact.get("gate_check_summary", {}).get("checks", [])
        if isinstance(row, Mapping)
    }
    if (
        not isinstance(principles, Mapping)
        or not set(artifact) <= set(principles)
        or not gate_principles <= set(principles)
        or not {key for key in FIELD_PRINCIPLES if key.startswith("gate_check")} <= set(principles)
    ):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in {
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("verdict_class") == "positive":
        errors.append("positive_verdict_forbidden")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("status") != "blocked":
        thresholds = artifact.get("qualification_thresholds", {})
        perturbations = thresholds.get("perturbations", [])
        eligibility = artifact.get("eligibility_rows", [])
        cell_ids = {str(row.get("cell_identity")) for row in eligibility}
        join_ids = {str(row.get("cell_identity")) for row in artifact.get("join_rows", [])}
        if (
            artifact.get("rows") != eligibility
            or len(eligibility) != thresholds.get("expected_cell_count")
            or cell_ids != join_ids
            or len(artifact.get("join_rows", [])) != len(eligibility) * len(perturbations)
        ):
            errors.append("row_coverage")
        replay = replay_reported_metrics(artifact)
        if not replay["agreement"]:
            errors.append("aggregate_disagreement")
        metrics = summarize_eligibility(eligibility)
        checks = _threshold_checks(
            metrics=metrics,
            join_evidence={
                "join_rows": artifact.get("join_rows", []),
                "missing_join_rows": artifact.get("missing_join_rows", []),
                "duplicate_join_rows": artifact.get("duplicate_join_rows", []),
            },
            preconditions_checked=artifact.get("preconditions_checked", {}),
            expected_cell_count=int(thresholds.get("expected_cell_count", 0)),
            perturbation_count=len(perturbations),
            required_model_families=thresholds.get("required_model_families", []),
            required_families=thresholds.get("required_families", []),
            minimum_events=int(thresholds.get("minimum_events", 0)),
            minimum_per_model_family=int(thresholds.get("minimum_per_model_family", 0)),
        )
        checks.append(gate_check("reported_aggregates_match_rows", True, replay["agreement"]))
        expected_summary = gate_summary(checks)
        if artifact.get("gate_check_summary") != expected_summary:
            errors.append("gate_summary_disagreement")
        ready = int(expected_summary["passed"])
        if artifact.get("qualified_relation_event_bank_ready_score") != ready:
            errors.append("readiness_disagreement")
        if artifact.get("qualified_model_relation_event_count") != len(
            artifact.get("admitted_event_rows", [])
        ):
            errors.append("event_count_disagreement")
        recomputed_rows = derive_eligibility_rows(artifact.get("join_rows", []), perturbations)
        if recomputed_rows != eligibility:
            errors.append("eligibility_inversion")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(set(errors))


def _load_current_verifier(path: str) -> Mapping[str, Any]:
    """Load the current verifier without importing it during module import."""

    verifier_path = REPO_ROOT / "scripts/adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_exp6915_adversarial", verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("adversarial_verifier_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def _read_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source_artifact_must_be_object")
    return value


def _safe_hash(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError:
        return "missing"


def _safe_verify(path: Path, verify_fn: VerifyFn) -> Mapping[str, Any]:
    try:
        return verify_fn(str(path))
    except Exception as exc:  # noqa: BLE001 - a failed safety check must block the merge.
        return {"loaded": False, "flags": [], "error": f"{type(exc).__name__}: {exc}"}


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path = RESULT_PATH,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
    expected_cell_ids: set[str] | None = None,
    perturbations: Sequence[str] = PERTURBATIONS,
    verify_fn: VerifyFn = _load_current_verifier,
    clock: Callable[[], float] = time.monotonic,
) -> JsonDict:
    """Read both shards and always write one terminal merge artifact."""

    started = clock()
    paths = {name: root / path for name, path in source_paths.items()}
    observed_hashes = {name: _safe_hash(path) for name, path in paths.items()}
    shards: dict[str, JsonDict] = {}
    read_checks = []
    for name in ("exp6913", "exp6914"):
        try:
            shards[name] = _read_object(paths[name])
            read_checks.append(gate_check(f"source_readable:{name}", True, True))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            shards[name] = {}
            read_checks.append(
                gate_check(f"source_readable:{name}", True, f"{type(exc).__name__}: {exc}")
            )
    reports = {
        name: _safe_verify(paths[name], verify_fn)
        if paths[name].is_file()
        else {"loaded": False, "flags": []}
        for name in ("exp6913", "exp6914")
    }
    wanted_ids = expected_cell_ids if expected_cell_ids is not None else expected_cell_identities()
    preconditions = validate_preconditions(
        source_shard=shards["exp6913"],
        semantic_shard=shards["exp6914"],
        observed_hashes=observed_hashes,
        expected_hashes=expected_hashes,
        expected_cell_ids=wanted_ids,
        perturbations=perturbations,
        fresh_reports=reports,
    )
    preconditions = gate_summary([*preconditions["checks"], *read_checks])
    hash_rows = {
        name: {
            "path": source_paths[name].as_posix(),
            "expected_sha256": expected_hashes.get(name),
            "observed_sha256": observed_hashes.get(name),
        }
        for name in source_paths
    }
    adversarial_rows = fresh_adversarial_rows(reports)
    artifact = build_artifact(
        date=date,
        duration_s=clock() - started,
        source_rows=[row for row in shards["exp6913"].get("rows", []) if isinstance(row, Mapping)],
        semantic_rows=[
            row for row in shards["exp6914"].get("rows", []) if isinstance(row, Mapping)
        ],
        source_artifact_hashes=hash_rows,
        preconditions_checked=preconditions,
        expected_cell_ids=wanted_ids,
        perturbations=perturbations,
        required_model_families=REQUIRED_MODEL_FAMILIES,
        required_families=REQUIRED_FAMILIES,
        minimum_events=MINIMUM_EVENTS,
        minimum_per_model_family=MINIMUM_PER_MODEL_FAMILY,
        fresh_adversarial_rows=adversarial_rows,
    )
    artifact["duration_s"] = round(clock() - started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _attach_principles(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("artifact_validation:" + ",".join(errors))
    output = output_path if output_path.is_absolute() else root / output_path
    _write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the exact dated command required by the experiment contract."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260903")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    run(date=args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper is the required command surface.
    raise SystemExit(main())
