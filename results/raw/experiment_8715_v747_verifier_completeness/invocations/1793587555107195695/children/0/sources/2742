"""Build the V634 typed-grounding executor and sealed fixture contract.

This task does not call a model. It diagnoses the frozen V633 evidence, builds
a fresh source-backed panel, and checks exact typed semantics before a later
generation task can use the public sidecar.

Spec refs: REQ-VERIFY-7195 and SCENARIO-VERIFY-7195-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import yaml

from carnot import experiment_7180_v633_symbolic_edit_fixture as fixture_builder
from carnot import experiment_7181_v633_qwen38_symbolic_traces as trace_capture
from carnot import experiment_7182_v633_grounding_energy_audit as old_audit
from carnot.experiment_artifacts import atomic_write_bytes, atomic_write_json
from carnot.paths import repo_root as find_repo_root
from carnot.verify.experiment_7195_source_relation_executor import (
    EntityBinding,
    ExecutionResult,
    TypedRelation,
    execute_relation,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260910"
RANDOM_SEED = 7_195_001
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

RESULT_PATH = Path("results/experiment_7195_v634_typed_grounding.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7195_v634_typed_grounding.json")
PUBLIC_VIEW_PATH = Path("results/experiment_7195_v634_typed_grounding_public.jsonl")
AUTHORITY_SIDECAR_PATH = Path("results/experiment_7195_v634_typed_grounding_authority.jsonl")

ERROR_CATEGORIES = (
    "parse_failure",
    "source_omission",
    "entity_binding",
    "relation_orientation",
    "negation",
    "unresolved_evidence",
)
VARIANTS = fixture_builder.VARIANTS

SOURCE_PATHS = {
    "exp7158_artifact": Path("results/experiment_7158_v630_entity_evidence_fixture.json"),
    "exp7180_artifact": Path("results/experiment_7180_v633_symbolic_edit_fixture.json"),
    "exp7180_public": Path(
        "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"
    ),
    "exp7180_authority": Path("results/experiment_7180_v633_symbolic_edit_fixture_authority.jsonl"),
    "exp7181_artifact": Path("results/experiment_7181_v633_qwen38_symbolic_traces.json"),
    "exp7182_artifact": Path("results/experiment_7182_v633_grounding_energy_audit.json"),
    "exp7180_module": Path("python/carnot/experiment_7180_v633_symbolic_edit_fixture.py"),
    "exp7181_module": Path("python/carnot/experiment_7181_v633_qwen38_symbolic_traces.py"),
    "exp7182_module": Path("python/carnot/experiment_7182_v633_grounding_energy_audit.py"),
    "executor_module": Path("python/carnot/verify/experiment_7195_source_relation_executor.py"),
    "experiment_module": Path("python/carnot/experiment_7195_v634_typed_grounding.py"),
    "entrypoint": Path("scripts/experiments/experiment_7195_v634_typed_grounding.py"),
    "focused_tests": Path("tests/python/test_experiment_7195_v634_typed_grounding.py"),
    "constraint_spec": Path("openspec/capabilities/constraint-verification/spec.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
}

UPSTREAM_FIELDS: dict[str, JsonDict] = {
    "exp7158_artifact": deepcopy(fixture_builder.EXP7158_EXPECTED_FIELDS),
    "exp7180_artifact": {
        "status": "complete",
        "run_date": "20260910",
        "fixture_ready_score": 1,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_symbolic_edit_fixture_ready_no_live_verifier_result",
    },
    "exp7181_artifact": {
        "status": "complete",
        "run_date": "20260910",
        "trace_capture_complete_score": 1,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_transport_capture_no_correctness_claim",
    },
    "exp7182_artifact": {
        "status": "complete",
        "run_date": "20260910",
        "grounding_measurement_complete_score": 1,
        "grounding_value_score": 0,
        "verdict_class": "null",
        "honest_verdict": "complete_null_grounding_energy_pilot_value_gate_not_met",
    },
}

FIELD_PRINCIPLES = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260910, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": "Every blocked verdict names the failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when verification uses the same correctness authority; separate implementations alone do not remove circularity.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings, including nulls; blocked_* for external blocks. Never promote infrastructure readiness as scientific benefit.",
    "typed_executor_ready_score": "Readiness requires deterministic semantics, separation and frozen prompts.",
    "error_decomposition_rows": "Observed failures justify a changed mechanism.",
    "fixture_manifest": "Record all 192 rows and base-group split hashes.",
    "public_view_path": "The generator sees raw sources and claims only.",
    "authority_sidecar_path": "Truth stays evaluator-only until scoring.",
    "semantic_mutation_rows": "Wrong semantics must fail even when syntax is valid.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

FROZEN_GROUP_PLAN = {
    "calibration": {
        "families": list(fixture_builder.CALIBRATION_FAMILIES),
        "bases_per_family": 8,
    },
    "held_out": {
        "families": list(fixture_builder.EVALUATION_FAMILIES),
        "bases_per_family": 8,
    },
    "variants": list(VARIANTS),
    "frozen_before_generation": True,
}


def canonical_json(value: Any) -> str:
    """Use one compact JSON spelling for every content hash."""

    return fixture_builder.canonical_json(value)


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize public and private rows with stable bytes."""

    return fixture_builder.jsonl_bytes(rows)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes and keep the algorithm name in the receipt."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a source without normalizing its bytes."""

    return sha256_bytes(path.read_bytes())


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding duration and the checksum itself."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _progress(phase: int, event: str, detail: str) -> None:  # pragma: no cover
    """Flush each real phase boundary for the outer task monitor."""

    print(f"exp7195 phase={phase} event={event} detail={detail}", flush=True)


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read an exact JSONL sidecar and require object rows."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL row is not an object")
    return rows


def _fresh_token(prefix: str, *parts: str) -> str:
    """Create one seed-bound identity that cannot reuse V633 names."""

    digest = hashlib.sha256("|".join((str(RANDOM_SEED), *parts)).encode()).hexdigest()[:20]
    return f"{prefix}-7195-{digest}"


def _fresh_upstream(upstream: Mapping[str, Any]) -> JsonDict:
    """Replace source identities before the shipped builder selects rows."""

    copied = deepcopy(dict(upstream))
    for row_index, row in enumerate(copied["rows"]):
        original_base = str(row["base_id"])
        row["base_id"] = _fresh_token("base", original_base, str(row_index))
        row["source_fixture_id"] = _fresh_token(
            "source", str(row["source_fixture_id"]), original_base
        )
        for entity_index, entity in enumerate(row["claim_entities"]):
            fresh = _fresh_token("entity", original_base, str(entity_index))
            entity["canonical_name"] = fresh
            entity["entity_id"] = fresh
            entity["aliases"] = [fresh]
    return copied


def build_fresh_panel(upstream: Mapping[str, Any]) -> JsonDict:
    """Reuse the Exp7180 builder after seed-bound identity replacement."""

    return fixture_builder.materialize_fixture(_fresh_upstream(upstream))


def build_public_rows(internal_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose only raw source and claim text to later generation work."""

    return [
        {
            "unit_id": row["unit_id"],
            "source_text": row["source_text"],
            "claim_text": row["claim_text"],
        }
        for row in internal_rows
    ]


def _entity_id(base_id: str, surface: str) -> str:
    """Give a surface one explicit panel-local identifier."""

    return _fresh_token("typed", base_id, surface)


def _typed_relation(
    value: Mapping[str, Any], entity_ids: Mapping[str, str], text: str
) -> TypedRelation:
    """Convert one exact fixture tuple into the executor input type."""

    return TypedRelation(
        entity_ids[str(value["subject"])],
        str(value["relation"]),
        entity_ids[str(value["object"])],
        str(value["polarity"]),
        0,
        len(text.encode("utf-8")),
    )


def build_authority_rows(panel: Mapping[str, Any]) -> list[JsonDict]:
    """Keep typed inputs, truth, split, and edit metadata evaluator-only."""

    exact_by_id = {str(row["unit_id"]): row for row in panel["authority_rows"]}
    authority_rows: list[JsonDict] = []
    for row in panel["internal_rows"]:
        base_id = str(row["base_id"])
        surfaces = {
            str(row["claim_tuple"]["subject"]),
            str(row["claim_tuple"]["object"]),
        }
        entity_ids = {surface: _entity_id(base_id, surface) for surface in surfaces}
        source_bytes = str(row["source_text"]).encode("utf-8")
        bindings: list[EntityBinding] = []
        evidence = row["evidence_tuple"]
        source_relations: list[TypedRelation] = []
        if isinstance(evidence, Mapping):
            for surface in sorted(surfaces):
                needle = surface.encode("utf-8")
                start = source_bytes.index(needle)
                bindings.append(
                    EntityBinding(entity_ids[surface], surface, start, start + len(needle))
                )
            source_relations.append(_typed_relation(evidence, entity_ids, str(row["source_text"])))
        claim = _typed_relation(row["claim_tuple"], entity_ids, str(row["claim_text"]))
        expected = (
            "unknown"
            if evidence is None
            else (
                "supported"
                if exact_by_id[str(row["unit_id"])]["expected_response"]["direct_decision"]
                == "supported"
                else "contradicted"
            )
        )
        authority_rows.append(
            {
                "unit_id": row["unit_id"],
                "base_id": base_id,
                "split": row["split"],
                "relation_family": row["relation_family"],
                "variant": row["variant"],
                "edit_detail": row["edit_detail"],
                "support_label": exact_by_id[str(row["unit_id"])]["expected_response"][
                    "direct_decision"
                ],
                "expected_executor_decision": expected,
                "entity_bindings": [asdict(binding) for binding in bindings],
                "source_relations": [asdict(relation) for relation in source_relations],
                "claim_relation": asdict(claim),
                "source_artifact_row_sha256": row["source_artifact_row_sha256"],
            }
        )
    return authority_rows


def public_view_errors(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Reject any producer shape, roster, identity, or private-field leak."""

    errors: list[str] = []
    expected_keys = {"unit_id", "source_text", "claim_text"}
    public_ids = [str(row.get("unit_id")) for row in public_rows]
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if len(public_rows) != 192 or any(set(row) != expected_keys for row in public_rows):
        errors.append("public_view_shape")
    if len(public_ids) != len(set(public_ids)) or public_ids != authority_ids:
        errors.append("public_authority_roster")
    forbidden = {
        "split",
        "relation_family",
        "variant",
        "edit_detail",
        "support_label",
        "expected_executor_decision",
        "claim_relation",
        "source_relations",
        "canonical_answer",
        "corpus_label",
    }
    if any(key in row for row in public_rows for key in forbidden):
        errors.append("public_authority_leak")
    if any(re.fullmatch(r"u-[0-9a-f]{24}", unit_id) is None for unit_id in public_ids):
        errors.append("public_unit_id_not_opaque")
    return errors


def freeze_generation_contract(
    public_rows: Sequence[Mapping[str, Any]], split_manifest: Mapping[str, Any]
) -> JsonDict:
    """Freeze separated prompts, syntax control, budgets, and scoring policy."""

    relation_schema = {
        "entity_bindings": [
            {
                "entity_id": "string",
                "surface": "string",
                "source_start": "integer",
                "source_end": "integer",
            }
        ],
        "relations": [
            {
                "subject_id": "string",
                "operator": "string",
                "object_id": "string",
                "polarity": "positive | negative",
                "source_start": "integer",
                "source_end": "integer",
            }
        ],
        "missing_fields": ["string"],
    }
    direct_schema = {"decision": "supported | unsupported | abstain"}
    atomic_prompts = {
        "source": {
            "visible_fields": ["source_text"],
            "template": "Extract typed entities and relations from SOURCE only. SOURCE:\n{source_text}",
            "schema": relation_schema,
            "max_tokens": 128,
        },
        "claim": {
            "visible_fields": ["claim_text"],
            "template": "Extract one typed relation from CLAIM only. CLAIM:\n{claim_text}",
            "schema": relation_schema,
            "max_tokens": 64,
        },
        "direct": {
            "visible_fields": ["source_text", "claim_text"],
            "template": "Judge support from SOURCE and CLAIM only. SOURCE:\n{source_text}\nCLAIM:\n{claim_text}",
            "schema": direct_schema,
            "max_tokens": 16,
        },
    }
    contract: JsonDict = {
        "frozen_before_held_out_truth": True,
        "atomic_prompts": atomic_prompts,
        "grammar_only_control": {
            "schemas": {"typed_relation": relation_schema, "direct": direct_schema},
            "semantic_constraints": [],
        },
        "scoring_policy": {
            "outcomes": ["supported", "contradicted", "unknown"],
            "unknown_is_false": False,
            "unknown_is_abstention": True,
            "full_denominator": 128,
            "tuning_split": "calibration",
            "held_out_label_access_during_freeze": False,
        },
        "sample_budget": {
            "base_cases": 48,
            "calibration_base_cases": 16,
            "held_out_base_cases": 32,
            "variants_per_base": 4,
            "rows": 192,
            "source_calls_max_tokens": 128,
            "claim_calls_max_tokens": 64,
            "direct_calls_max_tokens": 16,
        },
        "group_plan": deepcopy(FROZEN_GROUP_PLAN),
        "public_roster_sha256": sha256_bytes(jsonl_bytes(public_rows)),
        "split_manifest_sha256": sha256_bytes(canonical_json(split_manifest).encode("utf-8")),
    }
    contract["contract_sha256"] = sha256_bytes(canonical_json(contract).encode("utf-8"))
    return contract


def _tuple_mismatch(
    expected: Mapping[str, Any] | None,
    observed: Mapping[str, Any] | None,
    mode: str,
) -> bool:
    """Compare one tuple axis while keeping diagnosis categories separate."""

    if expected is None or observed is None:
        return False
    if mode == "entities":
        return {expected["subject"], expected["object"]} != {
            observed["subject"],
            observed["object"],
        }
    if mode == "orientation":
        return (
            expected["relation"] != observed["relation"]
            or expected["subject"] != observed["subject"]
            or expected["object"] != observed["object"]
        )
    return expected["polarity"] != observed["polarity"]


def decompose_old_errors(
    trace: Mapping[str, Any],
    audit: Mapping[str, Any],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Reconstruct V633 failure modes without changing its frozen rule."""

    features = old_audit.build_candidate_features(trace["rows"], public_rows)
    authority = {str(row["unit_id"]): row for row in authority_rows}
    raw_row_hashes = {
        str(row["unit_id"]): sha256_bytes(canonical_json(row).encode("utf-8"))
        for row in trace["rows"]
    }
    hashes: dict[str, list[str]] = {name: [] for name in ERROR_CATEGORIES}
    unit_ids: dict[str, list[str]] = {name: [] for name in ERROR_CATEGORIES}

    def add(category: str, feature: Mapping[str, Any]) -> None:
        hashes[category].append(raw_row_hashes[str(feature["unit_id"])])
        unit_ids[category].append(str(feature["unit_id"]))

    for feature in features:
        expected_row = authority[str(feature["unit_id"])]
        if expected_row["split"] != "evaluation":
            continue
        expected = expected_row["expected_response"]
        observed = feature["parsed_response"]
        if feature["parse_status"] != "valid":
            add("parse_failure", feature)
        if expected["evidence_tuple"] is None:
            add("source_omission", feature)
        if isinstance(observed, Mapping):
            expected_tuples = (expected["claim_tuple"], expected["evidence_tuple"])
            observed_tuples = (observed["claim_tuple"], observed["evidence_tuple"])
            if any(
                _tuple_mismatch(left, right, "entities")
                for left, right in zip(expected_tuples, observed_tuples, strict=True)
            ):
                add("entity_binding", feature)
            if any(
                _tuple_mismatch(left, right, "orientation")
                for left, right in zip(expected_tuples, observed_tuples, strict=True)
            ):
                add("relation_orientation", feature)
            if any(
                _tuple_mismatch(left, right, "polarity")
                for left, right in zip(expected_tuples, observed_tuples, strict=True)
            ):
                add("negation", feature)
            if expected["evidence_tuple"] is not None and observed["evidence_tuple"] is None:
                add("unresolved_evidence", feature)
        if feature["parse_error"] == "source_span_invalid":
            add("unresolved_evidence", feature)

    definitions = {
        "parse_failure": "The exact Exp7181 parser rejected the raw completion.",
        "source_omission": "The sealed source was empty for an evidence-deletion variant.",
        "entity_binding": "Parsed entity sets differed from evaluator-only typed entities.",
        "relation_orientation": "Parsed operator or ordered arguments differed from the typed tuple.",
        "negation": "Parsed polarity differed from the evaluator-only typed tuple.",
        "unresolved_evidence": "Evidence was absent after parsing or its source span was invalid.",
    }
    return [
        {
            "category": category,
            "definition": definitions[category],
            "observed_count": len(unit_ids[category]),
            "unit_ids": unit_ids[category],
            "raw_row_hashes": hashes[category],
            "old_parse_failure_denominator": 128,
            "old_grounding_value_score": audit["grounding_value_score"],
            "old_energy_accuracy": audit["arm_metrics"]["energy_from_extracted_tuples"]["accuracy"],
            "old_direct_accuracy": audit["arm_metrics"]["baseline_direct"]["accuracy"],
            "old_harmful_flip_count": audit["arm_metrics"]["energy_from_extracted_tuples"][
                "harmful_flip_count"
            ],
            "old_rule_changed": False,
        }
        for category in ERROR_CATEGORIES
    ]


def _load_typed(
    row: Mapping[str, Any],
) -> tuple[tuple[EntityBinding, ...], tuple[TypedRelation, ...], TypedRelation]:
    """Rebuild immutable executor inputs from the private JSON row."""

    bindings = tuple(EntityBinding(**value) for value in row["entity_bindings"])
    relations = tuple(TypedRelation(**value) for value in row["source_relations"])
    claim = TypedRelation(**row["claim_relation"])
    return bindings, relations, claim


def execute_panel(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Run public bytes through typed execution and independent label scoring."""

    authority = {str(row["unit_id"]): row for row in authority_rows}
    rows: list[JsonDict] = []
    for public in public_rows:
        private = authority[str(public["unit_id"])]
        bindings, relations, claim = _load_typed(private)
        result = execute_relation(
            str(public["source_text"]).encode("utf-8"), bindings, relations, claim
        )
        expected = str(private["expected_executor_decision"])
        passed = result.decision == expected and result.abstention == (expected == "unknown")
        rows.append(
            {
                "unit_id": public["unit_id"],
                "base_id": private["base_id"],
                "arm": "typed_executor_exact_input",
                "seed": RANDOM_SEED,
                "metric": int(passed),
                "error": None if passed else "semantic_disagreement",
                "abstention": result.abstention,
                "prediction": result.decision,
                "expected_prediction": expected,
                "uncertainty_reasons": list(result.uncertainty_reasons),
                "matched_relation_indexes": list(result.matched_relation_indexes),
                "source_text_sha256": sha256_bytes(str(public["source_text"]).encode("utf-8")),
                "claim_text_sha256": sha256_bytes(str(public["claim_text"]).encode("utf-8")),
            }
        )
    return rows


def _result_row(name: str, expected: str, result: ExecutionResult) -> JsonDict:
    """Convert one adverse semantic outcome into an auditable test row."""

    passed = result.decision == expected and result.abstention == (expected == "unknown")
    return {
        "mutation": name,
        "syntax_valid": True,
        "expected_decision": expected,
        "observed_decision": result.decision,
        "abstention": result.abstention,
        "uncertainty_reasons": list(result.uncertainty_reasons),
        "matched_relation_indexes": list(result.matched_relation_indexes),
        "passed": passed,
    }


def semantic_mutation_checks() -> list[JsonDict]:
    """Exercise exact, adverse, incomplete, and unsupported typed inputs."""

    source = b"entity-a precedes entity-b."
    a = EntityBinding("a", "entity-a", 0, 8)
    b = EntityBinding("b", "entity-b", 18, 26)
    bindings = (a, b)
    evidence = TypedRelation("a", "precedes", "b", "positive", 0, len(source))

    def run(relations: tuple[TypedRelation, ...], claim: TypedRelation) -> ExecutionResult:
        return execute_relation(source, bindings, relations, claim)

    rows = [
        _result_row("exact", "supported", run((evidence,), evidence)),
        _result_row(
            "inverse_equivalent",
            "supported",
            run((evidence,), TypedRelation("b", "follows", "a", "positive", 0, len(source))),
        ),
        _result_row(
            "reversed_arguments",
            "contradicted",
            run((evidence,), TypedRelation("b", "precedes", "a", "positive", 0, len(source))),
        ),
        _result_row(
            "negation",
            "contradicted",
            run((evidence,), TypedRelation("a", "precedes", "b", "negative", 0, len(source))),
        ),
        _result_row("removed_evidence", "unknown", run((), evidence)),
    ]

    duplicate_source = b"Alex precedes Alex."
    duplicate_bindings = (
        EntityBinding("left", "Alex", 0, 4),
        EntityBinding("right", "Alex", 14, 18),
    )
    duplicate_relation = TypedRelation(
        "left", "precedes", "right", "positive", 0, len(duplicate_source)
    )
    rows.append(
        _result_row(
            "duplicate_names",
            "unknown",
            execute_relation(
                duplicate_source,
                duplicate_bindings,
                (duplicate_relation,),
                duplicate_relation,
            ),
        )
    )
    rows.append(
        _result_row(
            "invalid_offsets",
            "unknown",
            run(
                (TypedRelation("a", "precedes", "b", "positive", 0, len(source) + 1),),
                evidence,
            ),
        )
    )
    contradictory_source = b"entity-a precedes entity-b and does not precede entity-b."
    contradictory_bindings = (
        EntityBinding("a", "entity-a", 0, 8),
        EntityBinding("b", "entity-b", 18, 26),
    )
    positive = TypedRelation("a", "precedes", "b", "positive", 0, len(contradictory_source))
    negative = TypedRelation("a", "precedes", "b", "negative", 0, len(contradictory_source))
    rows.append(
        _result_row(
            "contradictory_evidence",
            "unknown",
            execute_relation(
                contradictory_source,
                contradictory_bindings,
                (positive, negative),
                positive,
            ),
        )
    )
    unsupported_source = b"entity-a touches entity-b."
    unsupported_bindings = (
        EntityBinding("a", "entity-a", 0, 8),
        EntityBinding("b", "entity-b", 17, 25),
    )
    unsupported = TypedRelation("a", "touches", "b", "positive", 0, len(unsupported_source))
    rows.append(
        _result_row(
            "unsupported_operator",
            "unknown",
            execute_relation(
                unsupported_source,
                unsupported_bindings,
                (unsupported,),
                unsupported,
            ),
        )
    )
    return rows


def build_fixture_manifest(
    panel: Mapping[str, Any],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
    sidecar_hashes: Mapping[str, Any],
) -> JsonDict:
    """Bind every row, base group, split, and frozen generation contract."""

    public = {str(row["unit_id"]): row for row in public_rows}
    authority = {str(row["unit_id"]): row for row in authority_rows}
    row_manifest = [
        {
            "unit_id": unit_id,
            "public_row_sha256": sha256_bytes(canonical_json(public[unit_id]).encode("utf-8")),
            "authority_row_sha256": sha256_bytes(
                canonical_json(authority[unit_id]).encode("utf-8")
            ),
        }
        for unit_id in public
    ]
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in authority_rows:
        grouped[str(row["base_id"])].append(dict(row))
    base_groups = []
    for base_id in sorted(grouped):
        group = grouped[base_id]
        base_groups.append(
            {
                "base_id": base_id,
                "split": group[0]["split"],
                "relation_family": group[0]["relation_family"],
                "unit_ids": [row["unit_id"] for row in group],
                "group_sha256": sha256_bytes(canonical_json(group).encode("utf-8")),
            }
        )
    split_hashes = {
        split: sha256_bytes(
            canonical_json([row for row in base_groups if row["split"] == split]).encode("utf-8")
        )
        for split in ("calibration", "evaluation")
    }
    return {
        "seed": RANDOM_SEED,
        "row_count": len(public_rows),
        "base_count": len(base_groups),
        "calibration_base_count": panel["split_manifest"]["calibration_base_count"],
        "held_out_base_count": panel["split_manifest"]["evaluation_base_count"],
        "variants_per_base": len(VARIANTS),
        "group_plan": deepcopy(FROZEN_GROUP_PLAN),
        "group_plan_sha256": sha256_bytes(canonical_json(FROZEN_GROUP_PLAN).encode("utf-8")),
        "split_manifest": deepcopy(panel["split_manifest"]),
        "base_group_split_hashes": split_hashes,
        "base_groups": base_groups,
        "row_manifest": row_manifest,
        "sidecar_hashes": dict(sidecar_hashes),
        "frozen_generation_contract": freeze_generation_contract(
            public_rows, panel["split_manifest"]
        ),
    }


def _gate(
    check: str, upstream: str, field: str, expected: Any, observed: Any, passed: bool
) -> JsonDict:
    """Record both sides of one prerequisite without inferred repair."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project the first failure into the stable terminal gate shape."""

    if failure is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_preconditions_and_semantic_checks_pass",
            "observed_value": "all_preconditions_and_semantic_checks_pass",
            "passed": True,
        }
    return {
        "failed_check": failure["check"],
        "upstream": failure["upstream"],
        "field": failure["field"],
        "expected_value": failure["expected_value"],
        "observed_value": failure["observed_value"],
        "passed": False,
    }


def _unwrap(value: Any) -> Any:
    """Read a principle-wrapped field without treating its object as true."""

    if isinstance(value, Mapping) and "principle" in value and "value" in value:
        return value["value"]
    return value


def _is_quarantined(artifact: Mapping[str, Any]) -> bool:
    """Reject the fabrication gate's structured quarantine determination."""

    return bool(_unwrap(artifact.get("flagged_adversarial")))


def _manifest_hits(value: Any, wanted: frozenset[str]) -> set[str]:
    """Find excluded upstream IDs anywhere in the manifest's nested records."""

    hits: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"experiment_id", "experiment_ids"}:
                candidates = child if isinstance(child, list) else [child]
                hits.update(str(candidate) for candidate in candidates if str(candidate) in wanted)
            hits.update(_manifest_hits(child, wanted))
    elif isinstance(value, list):
        for child in value:
            hits.update(_manifest_hits(child, wanted))
    return hits


def _resolved_paths(root: Path, overrides: Mapping[str, Path] | None) -> dict[str, Path]:
    """Resolve fixed sources while allowing tests to replace one input."""

    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    if overrides:
        paths.update({name: Path(path) for name, path in overrides.items()})
    return paths


def _load_artifact(path: Path) -> JsonDict:
    """Read one upstream JSON object after its bytes are hashed."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact root is not an object")
    return value


def _preconditions(
    root: Path,
    run_date: str,
    paths: Mapping[str, Path],
    output_paths: Sequence[Path],
) -> tuple[list[JsonDict], JsonDict | None, dict[str, str], dict[str, JsonDict]]:
    """Check source bytes, quarantine, exact gates, tools, and destinations."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    loaded: dict[str, JsonDict] = {}

    def record(row: JsonDict) -> bool:
        checks.append(row)
        return bool(row["passed"])

    date_gate = _gate(
        "run_date", "experiment_7195", "run_date", RUN_DATE, run_date, run_date == RUN_DATE
    )
    if not record(date_gate):
        return checks, date_gate, hashes, loaded
    for name in SOURCE_PATHS:
        path = paths[name]
        passed = path.is_file() and os.access(path, os.R_OK)
        row = _gate(
            f"{name}_path",
            str(SOURCE_PATHS[name]),
            "path",
            "readable_file",
            str(path.resolve()) if passed else "missing_or_unreadable",
            passed,
        )
        if not record(row):
            return checks, row, hashes, loaded
        hashes[name] = sha256_file(path)
    spec_text = paths["constraint_spec"].read_text(encoding="utf-8")
    requirement = "REQ-VERIFY-7195" if "REQ-VERIFY-7195" in spec_text else "missing"
    row = _gate(
        "constraint_spec_requirement",
        str(SOURCE_PATHS["constraint_spec"]),
        "requirement",
        "REQ-VERIFY-7195",
        requirement,
        requirement == "REQ-VERIFY-7195",
    )
    if not record(row):
        return checks, row, hashes, loaded

    for name in UPSTREAM_FIELDS:
        try:
            loaded[name] = _load_artifact(paths[name])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            row = _gate(
                f"{name}_json",
                str(SOURCE_PATHS[name]),
                "json_object",
                True,
                type(exc).__name__,
                False,
            )
            record(row)
            return checks, row, hashes, loaded
        quarantined = _is_quarantined(loaded[name])
        row = _gate(
            f"{name}_quarantine",
            str(SOURCE_PATHS[name]),
            "flagged_adversarial",
            False,
            quarantined,
            not quarantined,
        )
        if not record(row):
            return checks, row, hashes, loaded

    manifest = yaml.safe_load(paths["exclusion_manifest"].read_text(encoding="utf-8"))
    manifest_hits = sorted(_manifest_hits(manifest, frozenset({"7158", "7180", "7181", "7182"})))
    row = _gate(
        "upstream_manifest_quarantine",
        str(SOURCE_PATHS["exclusion_manifest"]),
        "excluded_upstream_ids",
        [],
        manifest_hits,
        not manifest_hits,
    )
    if not record(row):
        return checks, row, hashes, loaded

    checksums = {
        "exp7158_artifact": fixture_builder.EXP7158_EXPECTED_FIELDS["reproducibility_checksum"],
        "exp7180_artifact": fixture_builder.artifact_checksum(loaded["exp7180_artifact"]),
        "exp7181_artifact": trace_capture.artifact_checksum(loaded["exp7181_artifact"]),
        "exp7182_artifact": old_audit.artifact_checksum(loaded["exp7182_artifact"]),
    }
    for name, expected_fields in UPSTREAM_FIELDS.items():
        observed_fields = {field: loaded[name].get(field) for field in expected_fields}
        row = _gate(
            f"{name}_terminal_fields",
            str(SOURCE_PATHS[name]),
            "terminal_gate_fields",
            expected_fields,
            observed_fields,
            observed_fields == expected_fields,
        )
        if not record(row):
            return checks, row, hashes, loaded
        observed_checksum = loaded[name].get("reproducibility_checksum")
        expected_checksum = checksums[name]
        row = _gate(
            f"{name}_checksum",
            str(SOURCE_PATHS[name]),
            "reproducibility_checksum",
            expected_checksum,
            observed_checksum,
            observed_checksum == expected_checksum,
        )
        if not record(row):
            return checks, row, hashes, loaded

    sidecar_gate = _gate(
        "exp7180_sidecar_rows",
        "experiment_7180",
        "public_and_authority_row_counts",
        [192, 192],
        [len(read_jsonl(paths["exp7180_public"])), len(read_jsonl(paths["exp7180_authority"]))],
        len(read_jsonl(paths["exp7180_public"]))
        == len(read_jsonl(paths["exp7180_authority"]))
        == 192,
    )
    if not record(sidecar_gate):
        return checks, sidecar_gate, hashes, loaded
    python_gate = _gate(
        "python_executable",
        ".venv/bin/python",
        "executable",
        "executable_file",
        sys.executable,
        Path(sys.executable).is_file() and os.access(sys.executable, os.X_OK),
    )
    if not record(python_gate):
        return checks, python_gate, hashes, loaded
    for output in output_paths:
        passed = output.parent.is_dir() and os.access(output.parent, os.W_OK)
        row = _gate(
            "output_directory",
            str(output),
            "parent",
            "existing_writable_directory",
            str(output.parent.resolve()) if passed else "missing_or_not_writable",
            passed,
        )
        if not record(row):
            return checks, row, hashes, loaded
    return checks, None, hashes, loaded


def _display_path(root: Path, path: Path) -> str:
    """Use repository-relative paths in production and exact test paths."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _base_artifact(root: Path, run_date: str, public_path: Path, authority_path: Path) -> JsonDict:
    """Create the schema-complete running checkpoint before fallible work."""

    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_rows": 192,
            "completed_rows": 0,
            "planned_base_cases": 48,
            "completed_base_cases": 0,
            "independent_held_out_base_cases": 32,
            "variants_per_base": 4,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(
            _gate("typed_executor_complete", "experiment_7195", "status", True, False, False)
        ),
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_typed_grounding_prototype",
        "typed_executor_ready_score": 0,
        "error_decomposition_rows": [],
        "fixture_manifest": {},
        "public_view_path": _display_path(root, public_path),
        "authority_sidecar_path": _display_path(root, authority_path),
        "semantic_mutation_rows": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _blocked_artifact(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any],
    hashes: Mapping[str, str],
    duration_s: float,
) -> JsonDict:
    """Finish an external block without claiming executor computation."""

    artifact.update(
        {
            "status": "blocked",
            "preconditions_checked": list(checks),
            "duration_s": duration_s,
            "source_artifact_hashes": dict(hashes),
            "gate_check_summary": _gate_summary(failure),
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failure['check']}_no_typed_executor_run",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    public_view_path: Path | None = None,
    authority_sidecar_path: Path | None = None,
    source_paths: Mapping[str, Path] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Gate sources, build the panel, execute semantics, and write atomically."""

    started = time.monotonic()
    root = Path(root).resolve()
    result_path = root / (output_path or RESULT_PATH)
    checkpoint = root / (checkpoint_path or CHECKPOINT_PATH)
    public_path = root / (public_view_path or PUBLIC_VIEW_PATH)
    authority_path = root / (authority_sidecar_path or AUTHORITY_SIDECAR_PATH)
    paths = _resolved_paths(root, source_paths)

    _progress(0, "start", "write schema-complete checkpoint before checks")
    artifact = _base_artifact(root, run_date, public_path, authority_path)
    atomic_write_json(checkpoint, artifact, allow_override=False, sort_keys=True)
    _progress(0, "end", "running checkpoint is durable")

    _progress(1, "start", "check bytes gates quarantine tools and output directories")
    checks, failure, hashes, loaded = _preconditions(
        root, run_date, paths, (result_path, checkpoint, public_path, authority_path)
    )
    _progress(1, "end", f"preconditions={len(checks)} failed={failure is not None}")
    measured = duration_s if duration_s is not None else time.monotonic() - started
    if failure is not None:
        blocked = _blocked_artifact(artifact, checks, failure, hashes, measured)
        _progress(7, "write_start", "atomically write terminal blocked artifact")
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        _progress(7, "write_end", "terminal blocked artifact is durable")
        return blocked

    _progress(2, "start", "reconstruct frozen V633 errors without rescoring")
    old_public = read_jsonl(paths["exp7180_public"])
    old_authority = read_jsonl(paths["exp7180_authority"])
    error_rows = decompose_old_errors(
        loaded["exp7181_artifact"], loaded["exp7182_artifact"], old_public, old_authority
    )
    _progress(2, "end", f"diagnostic_categories={len(error_rows)}")

    _progress(3, "start", "build fresh sealed panel with shipped exact fixture builder")
    panel = build_fresh_panel(loaded["exp7158_artifact"])
    public_rows = build_public_rows(panel["internal_rows"])
    authority_rows = build_authority_rows(panel)
    view_errors = public_view_errors(public_rows, authority_rows)
    _progress(3, "end", f"rows={len(public_rows)} view_errors={len(view_errors)}")

    _progress(4, "benchmark_start", "execute all exact typed panel rows")
    rows = execute_panel(public_rows, authority_rows)
    _progress(
        4,
        "benchmark_end",
        f"completed={len(rows)} disagreements={sum(not row['metric'] for row in rows)}",
    )

    _progress(5, "benchmark_start", "run adverse semantic mutation suite")
    mutation_rows = semantic_mutation_checks()
    _progress(
        5,
        "benchmark_end",
        f"mutations={len(mutation_rows)} failures={sum(not row['passed'] for row in mutation_rows)}",
    )

    public_bytes = jsonl_bytes(public_rows)
    authority_bytes = jsonl_bytes(authority_rows)
    sidecar_hashes = {
        "public_view_sha256": sha256_bytes(public_bytes),
        "authority_sidecar_sha256": sha256_bytes(authority_bytes),
        "public_view_row_count": len(public_rows),
        "authority_sidecar_row_count": len(authority_rows),
        "authority_mutation_preserves_public_bytes": True,
    }
    fixture_manifest = build_fixture_manifest(panel, public_rows, authority_rows, sidecar_hashes)
    ready = (
        not view_errors
        and len(rows) == 192
        and all(row["metric"] == 1 for row in rows)
        and all(row["passed"] for row in mutation_rows)
        and all(row["observed_count"] > 0 for row in error_rows)
    )
    measured = duration_s if duration_s is not None else time.monotonic() - started
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": checks,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "duration_s": measured,
            "source_artifact_hashes": hashes,
            "rows": rows,
            "sample_size_budget": {
                "planned_rows": 192,
                "completed_rows": len(rows),
                "planned_base_cases": 48,
                "completed_base_cases": len({row["base_id"] for row in rows}),
                "independent_held_out_base_cases": 32,
                "variants_per_base": 4,
                "exclusions": [],
            },
            "gate_check_summary": _gate_summary(None),
            "verdict_class": "circular_positive" if ready else "disqualified",
            "honest_verdict": (
                "complete_circular_positive_typed_executor_ready_no_independent_value_claim"
                if ready
                else "complete_disqualified_typed_executor_readiness_checks_failed"
            ),
            "typed_executor_ready_score": int(ready),
            "error_decomposition_rows": error_rows,
            "fixture_manifest": fixture_manifest,
            "semantic_mutation_rows": mutation_rows,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)

    _progress(6, "validation_start", "cold-check in-memory terminal contract")
    in_memory_errors = _validate_core(artifact, public_rows, authority_rows, panel)
    _progress(6, "validation_end", f"errors={len(in_memory_errors)}")
    if in_memory_errors:  # pragma: no cover - tests exercise each core check directly.
        raise ValueError(f"invalid Exp7195 artifact before write: {in_memory_errors}")

    _progress(7, "write_start", "atomically write sidecars and terminal artifact")
    atomic_write_bytes(public_path, public_bytes, allow_override=False)
    atomic_write_bytes(authority_path, authority_bytes, allow_override=False)
    atomic_write_json(checkpoint, artifact, allow_override=False, sort_keys=True)
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(7, "write_end", "complete terminal artifact is durable")
    return artifact


def _validate_core(
    artifact: Mapping[str, Any],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
) -> list[str]:
    """Recompute all complete-state evidence without trusting summary fields."""

    errors: list[str] = []
    if public_view_errors(public_rows, authority_rows):
        errors.append("public_view_invalid")
    expected_rows = execute_panel(public_rows, authority_rows)
    if artifact.get("rows") != expected_rows:
        errors.append("rows_mismatch")
    mutations = semantic_mutation_checks()
    if artifact.get("semantic_mutation_rows") != mutations:
        errors.append("semantic_mutation_rows_mismatch")
    sidecars = artifact.get("fixture_manifest", {}).get("sidecar_hashes", {})
    expected_manifest = build_fixture_manifest(panel, public_rows, authority_rows, sidecars)
    if artifact.get("fixture_manifest") != expected_manifest:
        errors.append("fixture_manifest_mismatch")
    ready = (
        not errors
        and all(row["metric"] == 1 for row in expected_rows)
        and all(row["passed"] for row in mutations)
    )
    if artifact.get("typed_executor_ready_score") != int(ready):
        errors.append("typed_executor_ready_score_mismatch")
    return errors


def validate_artifact(
    value: Mapping[str, Any] | str | Path,
    *,
    root: Path | None = None,
    public_view_path: Path | None = None,
    authority_sidecar_path: Path | None = None,
) -> list[str]:
    """Cold-check terminal state, sources, sidecars, semantics, and checksum."""

    if isinstance(value, (str, Path)):
        try:
            artifact = _load_artifact(Path(value))
        except (OSError, json.JSONDecodeError, ValueError):
            return ["artifact_unreadable"]
    elif isinstance(value, Mapping):
        artifact = dict(value)
    else:
        return ["artifact_not_object"]
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        return ["artifact_fields_mismatch"]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_invocation_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    failed = next(
        (row for row in artifact.get("preconditions_checked", []) if row.get("passed") is False),
        None,
    )
    if failed is not None:
        if artifact.get("status") != "blocked":
            errors.append("blocked_status_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("gate_check_summary") != _gate_summary(failed):
            errors.append("blocked_gate_summary_mismatch")
        if artifact.get("typed_executor_ready_score") != 0:
            errors.append("blocked_readiness_mismatch")
        return errors

    if artifact.get("status") != "complete":
        errors.append("complete_status_mismatch")
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("complete_substrate_class_mismatch")
    if artifact.get("gate_check_summary") != _gate_summary(None):
        errors.append("complete_gate_summary_mismatch")
    if artifact.get("verdict_class") != "circular_positive":
        errors.append("complete_verdict_class_mismatch")
    if (
        artifact.get("honest_verdict")
        != "complete_circular_positive_typed_executor_ready_no_independent_value_claim"
    ):
        errors.append("complete_honest_verdict_mismatch")

    repo = find_repo_root() if root is None else Path(root).resolve()
    paths = _resolved_paths(repo, None)
    if not all(path.is_file() for path in paths.values()):
        errors.append("source_artifact_missing")
        return errors
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if artifact.get("source_artifact_hashes") != observed_hashes:
        errors.append("source_artifact_hashes_mismatch")
    public_path = (
        Path(public_view_path)
        if public_view_path is not None
        else repo / str(artifact["public_view_path"])
    )
    authority_path = (
        Path(authority_sidecar_path)
        if authority_sidecar_path is not None
        else repo / str(artifact["authority_sidecar_path"])
    )
    try:
        public_rows = read_jsonl(public_path)
        authority_rows = read_jsonl(authority_path)
        upstream = _load_artifact(paths["exp7158_artifact"])
    except (OSError, json.JSONDecodeError, ValueError):
        errors.append("sidecar_or_upstream_unreadable")
        return errors
    observed_sidecar_hashes = {
        "public_view_sha256": sha256_file(public_path),
        "authority_sidecar_sha256": sha256_file(authority_path),
        "public_view_row_count": len(public_rows),
        "authority_sidecar_row_count": len(authority_rows),
        "authority_mutation_preserves_public_bytes": True,
    }
    if artifact.get("fixture_manifest", {}).get("sidecar_hashes") != observed_sidecar_hashes:
        errors.append("sidecar_hashes_mismatch")
    panel = build_fresh_panel(upstream)
    expected_public = build_public_rows(panel["internal_rows"])
    expected_authority = build_authority_rows(panel)
    if public_rows != expected_public:
        errors.append("public_view_rows_mismatch")
    if authority_rows != expected_authority:
        errors.append("authority_sidecar_rows_mismatch")
    errors.extend(_validate_core(artifact, public_rows, authority_rows, panel))
    old_errors = decompose_old_errors(
        _load_artifact(paths["exp7181_artifact"]),
        _load_artifact(paths["exp7182_artifact"]),
        read_jsonl(paths["exp7180_public"]),
        read_jsonl(paths["exp7180_authority"]),
    )
    if artifact.get("error_decomposition_rows") != old_errors:
        errors.append("error_decomposition_rows_mismatch")
    budget = artifact.get("sample_size_budget", {})
    if budget.get("completed_rows") != 192 or budget.get("completed_base_cases") != 48:
        errors.append("sample_size_budget_mismatch")
    return list(dict.fromkeys(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Build the fixed-date result or cold-validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--public-view", type=Path, default=PUBLIC_VIEW_PATH)
    parser.add_argument("--authority-sidecar", type=Path, default=AUTHORITY_SIDECAR_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(6, "subprocess_start", "cold artifact validation")
        errors = validate_artifact(
            args.validate,
            public_view_path=args.public_view,
            authority_sidecar_path=args.authority_sidecar,
        )
        _progress(6, "subprocess_end", f"cold artifact validation errors={len(errors)}")
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date != RUN_DATE:
        print(f"run date must be {RUN_DATE}", file=sys.stderr, flush=True)
        return 2
    artifact = build_artifact(
        find_repo_root(),
        args.date,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        public_view_path=args.public_view,
        authority_sidecar_path=args.authority_sidecar,
    )
    errors = validate_artifact(
        artifact,
        public_view_path=args.public_view,
        authority_sidecar_path=args.authority_sidecar,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "typed_executor_ready_score": artifact["typed_executor_ready_score"],
                "verdict_class": artifact["verdict_class"],
                "valid": not errors,
                "errors": errors,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
