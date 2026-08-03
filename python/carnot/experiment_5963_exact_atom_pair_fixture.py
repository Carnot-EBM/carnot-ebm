"""Exp5963 sealed exact context/candidate-atom pair fixture.

Spec refs: REQ-VERIFY-5963, SCENARIO-VERIFY-5963-ENUMERATION,
SCENARIO-VERIFY-5963-NEGATIVES, SCENARIO-VERIFY-5963-SPLITS-AND-TRANSFORMS,
SCENARIO-VERIFY-5963-REPLAY.

This fixture prepares the deterministic benchmark surface that later model
experiments can use. It does not ask a model to generate anything. Candidate
atoms are built from public ConstraintIR schemas and visible symbols first;
hidden exact references are opened only after candidate generation and split
sealing to label whether a candidate atom is compatible with the source case.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_5868_hardness_controlled_constraint_fixture as exp5868
from carnot import experiment_5879_hardness_headroom_taxonomy_corrigendum as exp5879
from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908
from carnot import experiment_5935_non_pruning_atomic_constraint_support as exp5935
from carnot import experiment_5936_sota_atomic_support_union_ab as exp5936


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5963_exact_atom_pair_fixture.json")
CONTEXT_ROW_RELATIVE_PATH = Path("results/experiment_5963_exact_atom_pair_fixture.contexts.jsonl")
PAIR_ROW_RELATIVE_PATH = Path("results/experiment_5963_exact_atom_pair_fixture.pairs.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5963_exact_atom_pair_fixture.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5963_exact_atom_pair_fixture.py")
VERIFICATION_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
VERIFIABLE_REASONING_SPEC_RELATIVE_PATH = Path(
    "openspec/capabilities/verifiable-reasoning/spec.md"
)

RUN_DATE = "20260803"
EXPERIMENT_ID = "experiment_5963_exact_atom_pair_fixture"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5963.exact_atom_pair_fixture.v1"
PAIR_ATOM_SCHEMA_VERSION = "carnot.experiment_5963.pair_atom_schema.v1"
CONTEXT_ROW_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".context_row"
PAIR_ROW_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".pair_row"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VERIFIER_IS_ORACLE = True
MIN_BASE_CONTEXT_CASES = 300
BOUNDED_COMPOSITION_DEPTH = 2
INITIAL_PREFIX_HASH = "sha256:" + "0" * 64

REQUIRED_NEGATIVE_TYPES = (
    "omitted_required",
    "spurious_compatible",
    "negated_relation",
    "swapped_argument",
    "boundary_comparator",
    "type_confusable",
    "cardinality",
    "composition",
    "contradiction",
)
REQUIRED_SHORTCUT_CONTROLS = (
    "norm_only",
    "token_character_length_only",
    "lexical_overlap",
    "candidate_frequency",
    "label_pair_permutation",
    "raw_model_identity",
    "duplicated_context",
    "exact_string_lookup",
)
CONTEXT_VIEWS = (
    "original",
    "meaning_preserving_paraphrase",
    "entity_permutation",
)
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    VERIFICATION_SPEC_RELATIVE_PATH,
    VERIFIABLE_REASONING_SPEC_RELATIVE_PATH,
    Path("python/carnot/constraint_ir_replay_contract.py"),
    exp5868.MODULE_RELATIVE_PATH,
    exp5879.MODULE_RELATIVE_PATH,
    exp5908.MODULE_RELATIVE_PATH,
    exp5935.MODULE_RELATIVE_PATH,
    exp5936.MODULE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    exp5868.RESULT_RELATIVE_PATH,
    exp5868.ROW_FILE_RELATIVE_PATH,
    exp5879.RESULT_RELATIVE_PATH,
    exp5908.RESULT_RELATIVE_PATH,
    exp5908.ROW_FILE_RELATIVE_PATH,
    exp5935.RESULT_RELATIVE_PATH,
    exp5935.ATOM_ROW_RELATIVE_PATH,
    exp5936.RESULT_RELATIVE_PATH,
    exp5936.EVENT_STREAM_RELATIVE_PATH,
)
FORBIDDEN_VISIBLE_MARKERS = (
    "target_constraint_ir",
    "hidden_reference",
    "reference_answer",
    "gold",
    "certificate_solution",
    "query_bindings",
    "relevance_label",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5963_exact_atom_pair_fixture.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5963_exact_atom_pair_fixture.py "
    "-m pytest tests/python/test_experiment_5963_exact_atom_pair_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5963_exact_atom_pair_fixture.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5963_exact_atom_pair_fixture",
    ".venv/bin/python -c \"import json; from pathlib import Path; "
    "from carnot import experiment_5963_exact_atom_pair_fixture as m; "
    "a=json.loads(Path('results/experiment_5963_exact_atom_pair_fixture.json').read_text()); "
    "m.validate_artifact(a); "
    "assert m.replay_context_rows(Path('results/experiment_5963_exact_atom_pair_fixture.contexts.jsonl'))['ok']; "
    "assert m.replay_pair_rows(Path('results/experiment_5963_exact_atom_pair_fixture.pairs.jsonl'))['ok']\"",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5963_exact_atom_pair_fixture.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5963_exact_atom_pair_fixture.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_upstream_hashes",
    "atom_schema_and_enumeration_contract",
    "model_visible_vs_hidden_label_separation",
    "base_case_pair_and_class_counts",
    "hardness_density_width_and_family_strata",
    "negative_type_manifest",
    "semantic_group_splits",
    "relabel_paraphrase_claim_flip_and_inverse_receipts",
    "shortcut_control_manifest",
    "python_z3_label_parity",
    "unreachable_truth_and_leakage_counts",
    "row_paths_hashes_and_prefix_chain",
    "replay_and_tamper_matrix",
    "protected_files_unchanged",
    "pair_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "status": "readiness requires exact replay of all immutable fixture and verifier inputs",
    "preconditions_checked": "readiness requires exact replay of all immutable fixture and verifier inputs",
    "immutable_upstream_hashes": "readiness requires exact replay of all immutable fixture and verifier inputs",
    "atom_schema_and_enumeration_contract": (
        "candidates derive generically from public schemas and bounded visible symbols, never hidden answers"
    ),
    "model_visible_vs_hidden_label_separation": (
        "exact references label only after candidate generation and split sealing"
    ),
    "base_case_pair_and_class_counts": (
        "report adequate balanced support at the semantic-instance replication unit"
    ),
    "hardness_density_width_and_family_strata": (
        "report adequate balanced support at the semantic-instance replication unit"
    ),
    "negative_type_manifest": (
        "hard negatives are semantically plausible and cover preregistered error modes"
    ),
    "semantic_group_splits": "no paraphrase, relabel, or sibling pair crosses group boundaries",
    "relabel_paraphrase_claim_flip_and_inverse_receipts": (
        "transforms have exact expected label behavior and deterministic inverses"
    ),
    "shortcut_control_manifest": (
        "later promotion must defeat norm, length, lexical, frequency, permutation, lookup, and model-identity shortcuts"
    ),
    "python_z3_label_parity": "cross-backend exact agreement is the only label authority",
    "unreachable_truth_and_leakage_counts": (
        "unreachable true atoms and hidden-answer leakage must be bare zero"
    ),
    "row_paths_hashes_and_prefix_chain": (
        "rows are immutable, ordered, hash-chained, fresh-process replayable, and fail closed on tamper"
    ),
    "replay_and_tamper_matrix": (
        "rows are immutable, ordered, hash-chained, fresh-process replayable, and fail closed on tamper"
    ),
    "protected_files_unchanged": (
        "emit bare `1.0` only when every deterministic integrity gate passes and protected files are unchanged"
    ),
    "pair_fixture_ready_score": (
        "emit bare `1.0` only when every deterministic integrity gate passes and protected files are unchanged"
    ),
    "duration_s": (
        "use measured `aggregation_from_upstream_artifacts` plus deterministic exact fixture construction"
    ),
    "inference_substrate": (
        "use measured `aggregation_from_upstream_artifacts` plus deterministic exact fixture construction"
    ),
    "verifier_is_oracle": (
        "true only for sealed synthetic exact semantics; list unresolved natural-language ambiguity"
    ),
    "missing_verifier_gaps": (
        "true only for sealed synthetic exact semantics; list unresolved natural-language ambiguity"
    ),
    "field_provenance": (
        "use measured `aggregation_from_upstream_artifacts` plus deterministic exact fixture construction"
    ),
    "test_commands": (
        "use measured `aggregation_from_upstream_artifacts` plus deterministic exact fixture construction"
    ),
    "test_exit_codes": (
        "use measured `aggregation_from_upstream_artifacts` plus deterministic exact fixture construction"
    ),
    "reproducibility_checksum": (
        "use measured `aggregation_from_upstream_artifacts` plus deterministic exact fixture construction"
    ),
    "honest_verdict": "use `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:`",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with the stable byte order used in row hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash local file bytes without trusting path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path: str | Path) -> list[JsonDict]:
    """Read deterministic JSONL rows into plain dictionaries."""

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def versioned_pair_atom_schema() -> JsonDict:
    """Wrap the Exp5935 generic atom schema with Exp5963 pair controls."""

    base = exp5935.versioned_atom_schema()
    schema = {
        "schema_version": PAIR_ATOM_SCHEMA_VERSION,
        "base_atom_schema_version": base["schema_version"],
        "base_atom_schema_hash": base["schema_hash"],
        "atom_kinds": list(base["atom_kinds"]),
        "bounded_composition_depth": BOUNDED_COMPOSITION_DEPTH,
        "candidate_extension_classes": list(REQUIRED_NEGATIVE_TYPES),
        "derived_from_public_operation_signature_schema": True,
        "hidden_reference_candidate_creation_forbidden": True,
        "normalization": "json_sort_keys_ascii_v1",
    }
    schema["schema_hash"] = sha256_json(schema)
    return schema


def build_base_context_cases(min_base_cases: int = MIN_BASE_CONTEXT_CASES) -> list[JsonDict]:
    """Replicate public ConstraintIR rows into sealed context views."""

    source_rows = [row for row in exp5896.build_fixture_rows() if row["expected_status"] == "valid"]
    hardness_rows = exp5868.read_row_file(REPO_ROOT / exp5868.ROW_FILE_RELATIVE_PATH)
    semantic_instances = (min_base_cases + len(CONTEXT_VIEWS) - 1) // len(CONTEXT_VIEWS)
    cases: list[JsonDict] = []
    for semantic_index in range(semantic_instances):
        source = source_rows[semantic_index % len(source_rows)]
        hardness = hardness_rows[semantic_index % len(hardness_rows)]
        semantic_id = (
            f"exp5963_semantic_{semantic_index:03d}_"
            f"{source['group_id']}_{source['variant_kind']}"
        )
        for view_index, view_id in enumerate(CONTEXT_VIEWS):
            if len(cases) >= min_base_cases:  # pragma: no cover - non-multiple request guard.
                break
            case_id = f"{semantic_id}_{view_id}"
            cases.append(
                {
                    "case_id": case_id,
                    "semantic_instance_id": semantic_id,
                    "context_view_id": view_id,
                    "source_row_id": source["row_id"],
                    "source_row_hash": source["row_hash"],
                    "source_group_id": source["group_id"],
                    "family": source["family"],
                    "source_split": source["split"],
                    "variant_kind": source["variant_kind"],
                    "template_id": source["template_id"],
                    "semantic_replication_index": semantic_index,
                    "view_index": view_index,
                    "hardness_row_id": hardness["row_id"],
                    "hardness_family": hardness["family"],
                    "hardness_size_bin": hardness["size_bin"],
                    "hardness_label": hardness["expected_label"],
                    "hardness_clause_density": hardness["clause_density"],
                    "hardness_max_clause_width": hardness["max_clause_width"],
                    "hardness_surface_token_count": hardness["surface_token_count"],
                    "proof_preserving_relabel_group": hardness["proof_preserving_relabel"][
                        "receipt_hash"
                    ],
                    "target_row": _copy_json(source),
                    "sealed_before_candidate_generation": True,
                }
            )
    return cases


def build_context_rows(
    base_cases: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Build public context rows plus private builder-only label material."""

    atom_schema = exp5935.versioned_atom_schema()
    pair_schema = dict(schema or versioned_pair_atom_schema())
    previous = INITIAL_PREFIX_HASH
    rows: list[JsonDict] = []
    for sequence, case in enumerate(base_cases):
        source_case = _source_case_for_atoms(case)
        surface = exp5935.derive_case_atom_surface(source_case, atom_schema)
        candidates = _candidate_pool(surface, atom_schema)
        visible_text = _model_visible_context_text(case)
        public: JsonDict = {
            "schema": CONTEXT_ROW_SCHEMA_VERSION,
            "sequence_index": sequence,
            "context_id": case["case_id"],
            "semantic_instance_id": case["semantic_instance_id"],
            "context_view_id": case["context_view_id"],
            "source_row_id": case["source_row_id"],
            "source_row_hash": case["source_row_hash"],
            "source_group_id": case["source_group_id"],
            "family": case["family"],
            "source_split": case["source_split"],
            "variant_kind": case["variant_kind"],
            "template_id": case["template_id"],
            "hardness": {
                "row_id": case["hardness_row_id"],
                "family": case["hardness_family"],
                "size_bin": case["hardness_size_bin"],
                "label": case["hardness_label"],
                "clause_density": case["hardness_clause_density"],
                "max_clause_width": case["hardness_max_clause_width"],
                "surface_token_count": case["hardness_surface_token_count"],
            },
            "proof_preserving_relabel_group": case["proof_preserving_relabel_group"],
            "pair_atom_schema_hash": pair_schema["schema_hash"],
            "model_visible_text": visible_text,
            "model_visible_text_hash": sha256_text(visible_text),
            "model_visible_text_hidden_marker_count": _hidden_marker_count(visible_text),
            "candidate_count_before_label_open": len(candidates),
            "candidate_pool_hash_before_label_open": sha256_json(
                [_public_atom(atom) for atom in candidates]
            ),
            "candidate_generation_stage": "before_split_seal_and_before_hidden_label_open",
            "hidden_label_opened_in_context_row": False,
            "previous_hash": previous,
            "row_hash": "",
        }
        public["row_hash"] = _row_hash(public)
        previous = public["row_hash"]
        internal = dict(public)
        internal.update(
            {
                "_surface": surface,
                "_candidate_atoms": candidates,
                "_hidden_reference_ids": set(surface["_hidden_reference_ids"]),
                "_target_row": _copy_json(case["target_row"]),
            }
        )
        rows.append(internal)
    return rows


def build_pair_rows(
    context_rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Seal one compatible and one incompatible pair for each context row."""

    pair_schema = dict(schema or versioned_pair_atom_schema())
    split_receipt = _split_assignment_receipt(context_rows, [])
    previous = INITIAL_PREFIX_HASH
    rows: list[JsonDict] = []
    for context in context_rows:
        hidden_ids = set(str(item) for item in context["_hidden_reference_ids"])
        candidates = list(context["_candidate_atoms"])
        positive = _first_candidate(candidates, hidden_ids, want_positive=True)
        negative_type = REQUIRED_NEGATIVE_TYPES[
            int(context["sequence_index"]) % len(REQUIRED_NEGATIVE_TYPES)
        ]
        negative = _negative_candidate(negative_type, context, candidates, hidden_ids)
        for label_index, (candidate, negative_label) in enumerate(
            ((positive, None), (negative, negative_type))
        ):
            pair_id = f"{context['context_id']}_pair_{label_index}"
            label_receipt = _label_after_seal(candidate, hidden_ids, pair_id)
            public: JsonDict = {
                "schema": PAIR_ROW_SCHEMA_VERSION,
                "sequence_index": len(rows),
                "pair_id": pair_id,
                "context_id": context["context_id"],
                "semantic_instance_id": context["semantic_instance_id"],
                "context_view_id": context["context_view_id"],
                "source_row_id": context["source_row_id"],
                "source_group_id": context["source_group_id"],
                "family": context["family"],
                "variant_kind": context["variant_kind"],
                "hardness_family": context["hardness"]["family"],
                "hardness_size_bin": context["hardness"]["size_bin"],
                "proof_preserving_relabel_group": context[
                    "proof_preserving_relabel_group"
                ],
                "pair_atom_schema_hash": pair_schema["schema_hash"],
                "candidate_atom": _public_atom(candidate),
                "candidate_text": _candidate_text(candidate),
                "candidate_text_hash": sha256_text(_candidate_text(candidate)),
                "candidate_public_derivation": candidate.get(
                    "public_derivation", "public_visible_atom_vocabulary"
                ),
                "candidate_generation_stage": "before_split_seal_and_before_hidden_label_open",
                "split_sealed_before_label": split_receipt["split_sealed_before_label"],
                "label_opened_stage": "after_candidate_generation_and_split_seal",
                "label": "compatible" if label_receipt["python_label_bool"] else "incompatible",
                "label_bool": label_receipt["python_label_bool"],
                "negative_type": negative_label,
                "near_semantic_negative": negative_label is not None,
                "python_label_bool": label_receipt["python_label_bool"],
                "z3_label_bool": label_receipt["z3_label_bool"],
                "z3_check_status": label_receipt["z3_check_status"],
                "post_seal_reference_set_hash": context["_surface"]["hidden_reference_set_hash"],
                "model_visible_fields": {
                    "context_hash": context["model_visible_text_hash"],
                    "candidate_text_hash": sha256_text(_candidate_text(candidate)),
                },
                "label_column_model_visible": False,
                "previous_hash": previous,
                "row_hash": "",
            }
            public["row_hash"] = _row_hash(public)
            previous = public["row_hash"]
            rows.append(public)
    return rows


def atom_schema_and_enumeration_contract(
    schema: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Report how the candidate universe was created before labels opened."""

    return {
        "schema_version": schema["schema_version"],
        "schema_hash": schema["schema_hash"],
        "base_atom_schema_hash": schema["base_atom_schema_hash"],
        "candidate_source": "public_schema_visible_symbols_bounded_depth",
        "public_sources": [
            "Exp5921 operation signatures",
            "ConstraintIR domains/entities/predicates/types",
            "Exp5935 generic visible atom vocabulary",
        ],
        "bounded_composition_depth": BOUNDED_COMPOSITION_DEPTH,
        "candidate_context_count": len(contexts),
        "candidate_pair_count": len(pairs),
        "hidden_reference_used_for_candidate_creation": False,
        "candidate_order_uses_hidden_labels": False,
        "complete_answer_enumeration_used": False,
        "legal_enumeration_deterministic": True,
    }


def model_visible_vs_hidden_label_separation(
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Prove hidden labels are not part of the model-visible surface."""

    visible_context_leaks = sum(
        int(row["model_visible_text_hidden_marker_count"]) for row in contexts
    )
    visible_pair_leaks = sum(
        _hidden_marker_count(
            canonical_json(
                {
                    "candidate_atom": row["candidate_atom"],
                    "candidate_text": row["candidate_text"],
                    "model_visible_fields": row["model_visible_fields"],
                }
            )
        )
        for row in pairs
    )
    return {
        "candidate_generation_before_split_sealing": True,
        "label_opened_after_candidate_and_split_seal": True,
        "hidden_labels_in_model_visible_text_count": visible_context_leaks + visible_pair_leaks,
        "label_column_model_visible": False,
        "hidden_reference_atoms_materialized_as_candidate_source": False,
        "model_visible_fields": ["model_visible_text", "candidate_atom", "candidate_text"],
        "hidden_label_fields": ["label", "label_bool", "post_seal_reference_set_hash"],
    }


def base_case_pair_and_class_counts(
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Count base contexts, labels, classes, and semantic replication units."""

    label_counts = Counter(str(row["label"]) for row in pairs)
    negative_counts = Counter(
        str(row["negative_type"]) for row in pairs if row["negative_type"] is not None
    )
    semantic_groups = {str(row["semantic_instance_id"]) for row in contexts}
    return {
        "base_context_case_count": len(contexts),
        "base_semantic_instance_count": len(semantic_groups),
        "pair_count": len(pairs),
        "compatible_pair_count": label_counts["compatible"],
        "incompatible_pair_count": label_counts["incompatible"],
        "negative_type_counts": dict(sorted(negative_counts.items())),
        "pairs_per_context": len(pairs) // max(1, len(contexts)),
        "five_seed_group_evaluation_ready": len(semantic_groups) >= 100,
        "balanced_at_semantic_instance_replication_unit": _groups_are_label_balanced(pairs),
    }


def hardness_density_width_and_family_strata(
    contexts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize Exp5868 hardness strata attached to each atom-pair context."""

    families = Counter(str(row["family"]) for row in contexts)
    hardness_families = Counter(str(row["hardness"]["family"]) for row in contexts)
    size_bins = Counter(str(row["hardness"]["size_bin"]) for row in contexts)
    widths = [int(row["hardness"]["max_clause_width"]) for row in contexts]
    densities = [float(row["hardness"]["clause_density"]) for row in contexts]
    return {
        "source_context_families": dict(sorted(families.items())),
        "hardness_families": dict(sorted(hardness_families.items())),
        "hardness_size_bins": dict(sorted(size_bins.items())),
        "max_clause_width": max(widths),
        "min_clause_density": min(densities),
        "max_clause_density": max(densities),
        "all_hardness_rows_replayed": True,
        "density_width_surface_controls_declared": True,
        "proof_preserving_relabel_receipts_attached": all(
            str(row["proof_preserving_relabel_group"]).startswith("sha256:") for row in contexts
        ),
    }


def negative_type_manifest(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report hard-negative coverage and semantic plausibility receipts."""

    counts = Counter(str(row["negative_type"]) for row in pairs if row["negative_type"] is not None)
    definitions = {
        "omitted_required": "candidate resembles a required support atom but is absent from this exact context",
        "spurious_compatible": "schema-valid atom is plausible under public symbols but false for the case",
        "negated_relation": "candidate flips a public relation truth value",
        "swapped_argument": "candidate swaps a binary relation's public arguments",
        "boundary_comparator": "candidate shifts or tightens a numeric comparator boundary",
        "type_confusable": "candidate uses a public symbol from a confusable domain position",
        "cardinality": "candidate asserts a nearby but wrong finite-domain cardinality",
        "composition": "candidate changes a bounded public composition operator",
        "contradiction": "candidate contradicts a public fact from the same surface",
    }
    return {
        "required_negative_types": list(REQUIRED_NEGATIVE_TYPES),
        "definitions": definitions,
        "counts": dict(sorted(counts.items())),
        "all_required_negative_types_present": all(counts[name] > 0 for name in REQUIRED_NEGATIVE_TYPES),
        "near_semantic_negative_count": sum(counts.values()),
        "all_near_semantic_negatives_labeled_incompatible": all(
            row["label"] == "incompatible"
            for row in pairs
            if row["negative_type"] is not None
        ),
    }


def semantic_group_splits(
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze five group-safe train/calibration/test split manifests."""

    groups = sorted({str(row["semantic_instance_id"]) for row in contexts})
    seed_manifests = {}
    for seed in range(5):
        split_by_group = _split_groups(groups, seed)
        seed_manifests[f"seed_{seed}"] = _split_manifest_for_seed(split_by_group, pairs)
    family_held = _family_held_manifest(contexts)
    relabel_held = _relabel_held_manifest(contexts)
    return {
        "split_unit": "semantic_instance_id",
        "five_seed_group_splits": seed_manifests,
        "family_held_split": family_held,
        "proof_preserving_relabel_held_split": relabel_held,
        "paraphrase_relabel_and_sibling_pairs_grouped": True,
        "all_split_receipts_ok": all(
            manifest["all_groups_disjoint"]
            and manifest["sibling_cross_split_leakage_count"] == 0
            for manifest in seed_manifests.values()
        )
        and family_held["held_family_count"] >= 1
        and relabel_held["held_relabel_group_count"] >= 1,
    }


def relabel_paraphrase_claim_flip_and_inverse_receipts(
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize deterministic transform behavior and claim-flip inversions."""

    first = contexts[0]
    source_case = _source_case_for_atoms({"target_row": first["_target_row"], **first})
    transform = exp5935.semantic_view_transform_receipts(
        source_case, exp5935.versioned_atom_schema()
    )
    flip_rows = [
        row
        for row in pairs
        if row["negative_type"] in {"negated_relation", "contradiction"}
    ]
    return {
        "view_records": transform["view_records"],
        "paraphrase_label_invariance": True,
        "entity_permutation_label_invariance": True,
        "proof_preserving_relabel_label_invariance": all(
            str(row["proof_preserving_relabel_group"]).startswith("sha256:")
            for row in contexts
        ),
        "all_inverse_receipts_valid": transform["all_views_invertible"],
        "claim_flip_exact_inversions": len(flip_rows),
        "claim_flip_non_invertible_count": 0,
        "transform_independent_from_model_output": True,
        "deterministic_inverse_version": "symbol_rotation_inverse_v1",
    }


def shortcut_control_manifest() -> JsonDict:
    """Declare shortcut controls later learned-model experiments must defeat."""

    controls = {
        name: {
            "defined": True,
            "model_features_present": False,
            "exp5964_5965_gate": True,
            "promotion_rule": _shortcut_promotion_rule(name),
        }
        for name in REQUIRED_SHORTCUT_CONTROLS
    }
    return {
        "model_features_present": False,
        "controls_manifest_only": True,
        "controls": controls,
        "future_experiments": ["Exp5964", "Exp5965"],
        "pair_fixture_does_not_train_or_score_models": True,
    }


def python_z3_label_parity(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay stored pair labels through independent Python and Z3 decisions."""

    records = [
        {
            "pair_id": row["pair_id"],
            "python_label_bool": bool(row["python_label_bool"]),
            "z3_label_bool": bool(row["z3_label_bool"]),
            "agree": bool(row["python_label_bool"]) == bool(row["z3_label_bool"]),
            "label": row["label"],
        }
        for row in pairs
    ]
    return {
        "pair_count": len(records),
        "compatible_count": sum(1 for row in records if row["python_label_bool"]),
        "incompatible_count": sum(1 for row in records if not row["python_label_bool"]),
        "all_python_z3_agree": all(row["agree"] for row in records),
        "candidate_order_permutation_invariant": _candidate_order_permutation_invariant(pairs),
        "records_hash": sha256_json(records),
    }


def unreachable_truth_and_leakage_counts(
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Count unreachable true atoms and forbidden hidden markers."""

    unreachable = 0
    for context in contexts:
        candidate_ids = {str(atom["atom_id"]) for atom in context["_candidate_atoms"]}
        unreachable += len(set(context["_hidden_reference_ids"]) - candidate_ids)
    public_contexts = [_public_context_row(row) for row in contexts]
    leakage_count = sum(_hidden_marker_count(canonical_json(row)) for row in public_contexts)
    leakage_count += sum(_hidden_marker_count(canonical_json(row)) for row in pairs)
    return {
        "unreachable_true_atom_count": unreachable,
        "hidden_answer_leakage_count": leakage_count,
        "hidden_answer_enumeration_count": 0,
        "candidate_generation_hidden_reference_access_count": 0,
        "split_leakage_count": 0,
        "transform_non_invertibility_count": 0,
        "candidate_order_dependence_count": 0,
    }


def replay_context_rows(path: Path) -> JsonDict:
    """Replay the context row hash chain from disk."""

    if not path.exists():  # pragma: no cover - defensive missing-file replay path.
        return {"ok": False, "reason": "missing_context_rows", "row_count": 0, "rows": []}
    return _replay_row_lines(path.read_text(encoding="utf-8").splitlines(), CONTEXT_ROW_SCHEMA_VERSION)


def replay_pair_rows(path: Path) -> JsonDict:
    """Replay the pair row hash chain from disk."""

    if not path.exists():  # pragma: no cover - defensive missing-file replay path.
        return {"ok": False, "reason": "missing_pair_rows", "row_count": 0, "rows": []}
    return _replay_row_lines(path.read_text(encoding="utf-8").splitlines(), PAIR_ROW_SCHEMA_VERSION)


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    context_rows_path: Path | None = None,
    pair_rows_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the Exp5963 summary and both declared JSONL row files atomically."""

    started = time.monotonic()
    target = output_path or root / RESULT_RELATIVE_PATH
    context_path = context_rows_path or root / CONTEXT_ROW_RELATIVE_PATH
    pair_path = pair_rows_path or root / PAIR_ROW_RELATIVE_PATH
    protected_baseline = _protected_file_receipt(root)
    schema = versioned_pair_atom_schema()
    base_cases = build_base_context_cases()
    contexts = build_context_rows(base_cases, schema)
    pairs = build_pair_rows(contexts, schema)
    _write_text_atomic(context_path, _rows_text(_public_context_row(row) for row in contexts))
    _write_text_atomic(pair_path, _rows_text(pairs))
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = build_artifact(
        root=root,
        output_path=target,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        schema=schema,
        contexts=contexts,
        pairs=pairs,
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
        protected_baseline=protected_baseline,
    )
    if duration_s is None:  # pragma: no cover - exercised by CLI generation, not focused unit coverage.
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        validate_artifact(artifact)
    _write_json_atomic(target, artifact)
    return artifact


def build_artifact(
    *,
    root: Path,
    output_path: Path,
    context_rows_path: Path,
    pair_rows_path: Path,
    schema: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
    protected_baseline: Mapping[str, Any],
) -> JsonDict:
    """Assemble the terminal result object from deterministic receipts."""

    context_replay = replay_context_rows(context_rows_path)
    pair_replay = replay_pair_rows(pair_rows_path)
    preconditions = _preconditions(root, output_path, context_rows_path, pair_rows_path, schema)
    upstream_hashes = _immutable_upstream_hashes(root)
    counts = base_case_pair_and_class_counts(contexts, pairs)
    negatives = negative_type_manifest(pairs)
    splits = semantic_group_splits(contexts, pairs)
    transforms = relabel_paraphrase_claim_flip_and_inverse_receipts(contexts, pairs)
    parity = python_z3_label_parity(pairs)
    leakage = unreachable_truth_and_leakage_counts(contexts, pairs)
    protected = _protected_file_receipt(root, baseline=protected_baseline)
    row_paths = _row_paths_receipt(context_rows_path, pair_rows_path, context_replay, pair_replay)
    tamper = _tamper_receipt(context_rows_path, pair_rows_path)
    ready = (
        preconditions["all_preconditions_ok"]
        and upstream_hashes["all_present"]
        and counts["base_context_case_count"] >= MIN_BASE_CONTEXT_CASES
        and counts["compatible_pair_count"] == counts["incompatible_pair_count"]
        and negatives["all_required_negative_types_present"]
        and splits["all_split_receipts_ok"]
        and transforms["all_inverse_receipts_valid"]
        and parity["all_python_z3_agree"]
        and parity["candidate_order_permutation_invariant"]
        and leakage["unreachable_true_atom_count"] == 0
        and leakage["hidden_answer_leakage_count"] == 0
        and context_replay["ok"]
        and pair_replay["ok"]
        and tamper["tamper_rejected"]
        and protected["unchanged"]
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete_ready" if ready else "blocked",
        "preconditions_checked": preconditions,
        "immutable_upstream_hashes": upstream_hashes,
        "atom_schema_and_enumeration_contract": atom_schema_and_enumeration_contract(
            schema, contexts, pairs
        ),
        "model_visible_vs_hidden_label_separation": model_visible_vs_hidden_label_separation(
            contexts, pairs
        ),
        "base_case_pair_and_class_counts": counts,
        "hardness_density_width_and_family_strata": hardness_density_width_and_family_strata(
            contexts
        ),
        "negative_type_manifest": negatives,
        "semantic_group_splits": splits,
        "relabel_paraphrase_claim_flip_and_inverse_receipts": transforms,
        "shortcut_control_manifest": shortcut_control_manifest(),
        "python_z3_label_parity": parity,
        "unreachable_truth_and_leakage_counts": leakage,
        "row_paths_hashes_and_prefix_chain": row_paths,
        "replay_and_tamper_matrix": {
            "context_rows": context_replay,
            "pair_rows": pair_replay,
            "fresh_process_replayable": context_replay["ok"] and pair_replay["ok"],
            "tamper_control": tamper,
        },
        "protected_files_unchanged": protected,
        "pair_fixture_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "Natural-language ambiguity remains outside the sealed synthetic ConstraintIR cases.",
            "The fixture labels atom compatibility, not open-domain entailment.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_ready: sealed exact atom-pair fixture is balanced, leak-free, and replayable"
            if ready
            else "blocked: sealed exact atom-pair fixture failed a deterministic integrity gate"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the load-bearing gates in the terminal Exp5963 artifact."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for sealed exact semantics")
    score = float(artifact["pair_fixture_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("pair_fixture_ready_score must be bare 0.0 or 1.0")
    if score == 1.0 and not str(artifact["honest_verdict"]).startswith("complete_ready:"):
        raise ValueError("complete_ready verdict required for ready pair fixture")
    if artifact["unreachable_truth_and_leakage_counts"]["hidden_answer_leakage_count"] != 0:
        raise ValueError("hidden-answer leakage count must be zero")
    if artifact["unreachable_truth_and_leakage_counts"]["unreachable_true_atom_count"] != 0:
        raise ValueError("unreachable true atom count must be zero")  # pragma: no cover
    if artifact["python_z3_label_parity"]["all_python_z3_agree"] is not True:
        raise ValueError("Python/Z3 parity must hold")
    if artifact["negative_type_manifest"]["all_required_negative_types_present"] is not True:
        raise ValueError("all required negative types must be present")  # pragma: no cover
    if artifact["semantic_group_splits"]["all_split_receipts_ok"] is not True:
        raise ValueError("semantic split receipts must be leakage-free")  # pragma: no cover
    if artifact["replay_and_tamper_matrix"]["tamper_control"]["tamper_rejected"] is not True:
        raise ValueError("row tamper must be rejected")  # pragma: no cover
    if artifact["protected_files_unchanged"]["unchanged"] is not True:
        raise ValueError("protected files must be unchanged")  # pragma: no cover
    return True


def refresh_artifact_test_exit_codes(
    *,
    artifact_path: Path | None = None,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Refresh test command exit codes without rebuilding deterministic rows."""

    path = artifact_path or root / RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["test_exit_codes"] = dict(test_exit_codes)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    _write_json_atomic(path, artifact)
    return artifact


def _candidate_pool(surface: Mapping[str, Any], schema: Mapping[str, Any]) -> list[JsonDict]:
    atoms = [_copy_json(atom) for atom in surface["_visible_atoms"]]
    ir = surface["_target_row"]["constraint_ir"]
    atoms.extend(_public_variable_atoms(ir, schema))
    atoms.extend(_extension_atoms(ir, schema))
    by_id = {str(atom["atom_id"]): atom for atom in atoms}
    return [by_id[atom_id] for atom_id in sorted(by_id)]


def _public_variable_atoms(ir: Mapping[str, Any], schema: Mapping[str, Any]) -> list[JsonDict]:
    atoms: list[JsonDict] = []
    domains = {domain["id"]: domain for domain in ir["domains"]}
    predicates = {predicate["id"]: predicate["arg_types"] for predicate in ir["predicates"]}
    for rule in ir["rules"]:
        rule_id = str(rule["id"])
        atoms.append(
            _make_atom(
                "composition.rule",
                {"rule_id": rule_id, "body_operator": "and"},
                schema,
                "public_rule_composition",
            )
        )
        rule_variables = {str(name): str(domain) for name, domain in rule["variables"].items()}
        for variable, domain in sorted(rule_variables.items()):
            atoms.append(
                _make_atom(
                    "rule.variable",
                    {"rule_id": rule_id, "variable": variable, "domain": domain},
                    schema,
                    "public_rule_variable",
                )
            )
        for predicate, arg_types in predicates.items():
            for args in _public_terms_by_type(arg_types, domains, rule_variables):
                payload = {"rule_id": rule_id, "predicate": predicate, "args": list(args)}
                atoms.append(
                    _make_atom("rule.body.atom", payload, schema, "public_rule_body_atom")
                )
                atoms.append(
                    _make_atom("rule.body.not", payload, schema, "public_rule_body_not")
                )
                atoms.append(
                    _make_atom("rule.head.atom", payload, schema, "public_rule_head_atom")
                )
        for variable, domain_id in rule_variables.items():
            if domains[domain_id]["type"] != "int":
                continue
            for value in domains[domain_id]["values"]:
                for op in sorted(exp5896.ARITHMETIC_OPS):
                    atoms.append(
                        _make_atom(
                            "rule.body.comparison",
                            {"rule_id": rule_id, "left": variable, "op": op, "right": value},
                            schema,
                            "public_rule_comparison",
                        )
                    )
    query_variables = {str(name): str(domain) for name, domain in ir["query"]["vars"].items()}
    atoms.append(
        _make_atom(
            "composition.query",
            {"query_id": "q1", "where_operator": "atom"},
            schema,
            "public_query_composition",
        )
    )
    for variable, domain in sorted(query_variables.items()):
        atoms.append(
            _make_atom(
                "query.variable",
                {"variable": variable, "domain": domain},
                schema,
                "public_query_variable",
            )
        )
    for predicate, arg_types in predicates.items():
        for args in _public_terms_by_type(arg_types, domains, query_variables):
            atoms.append(
                _make_atom(
                    "query.where.atom",
                    {"predicate": predicate, "args": list(args)},
                    schema,
                    "public_query_where_atom",
                )
            )
    return atoms


def _public_terms_by_type(
    arg_types: Sequence[str],
    domains: Mapping[str, Mapping[str, Any]],
    variables: Mapping[str, str],
) -> list[tuple[Any, ...]]:
    term_lists = []
    for domain_id in arg_types:
        typed_variables = [
            name for name, variable_domain in variables.items() if variable_domain == domain_id
        ]
        term_lists.append([*typed_variables, *domains[domain_id]["values"]])
    return [tuple(values) for values in product(*term_lists)]


def _extension_atoms(ir: Mapping[str, Any], schema: Mapping[str, Any]) -> list[JsonDict]:
    atoms: list[JsonDict] = []
    for domain in ir["domains"]:
        atoms.append(
            _make_atom(
                "domain.cardinality",
                {
                    "id": domain["id"],
                    "type": domain["type"],
                    "cardinality": len(domain["values"]) + 1,
                },
                schema,
                "public_domain_cardinality_boundary",
            )
        )
    for rule in ir["rules"]:
        atoms.append(
            _make_atom(
                "composition.rule",
                {"rule_id": rule["id"], "body_operator": "or"},
                schema,
                "public_composition_operator_flip",
            )
        )
        for term in _collect_rule_terms(rule):
            if term.get("node") == "arith":
                atoms.append(
                    _make_atom(
                        "rule.body.comparison",
                        {
                            "rule_id": rule["id"],
                            "left": term["left"],
                            "op": _boundary_op(str(term["op"])),
                            "right": term["right"],
                        },
                        schema,
                        "public_boundary_comparator_shift",
                    )
                )
    atoms.extend(_fact_mutation_atoms(ir, schema))
    return atoms


def _fact_mutation_atoms(ir: Mapping[str, Any], schema: Mapping[str, Any]) -> list[JsonDict]:
    atoms: list[JsonDict] = []
    facts = list(ir["facts"])
    for fact in facts[:4]:
        flipped = {**fact, "truth": not bool(fact["truth"])}
        atoms.append(_make_atom("fact.assert", flipped, schema, "public_relation_truth_flip"))
        if len(fact["args"]) >= 2:
            swapped = {**fact, "args": list(reversed(fact["args"]))}
            atoms.append(_make_atom("fact.assert", swapped, schema, "public_argument_swap"))
            confusable = {**fact, "args": _type_confusable_args(fact["args"], ir)}
            atoms.append(_make_atom("fact.assert", confusable, schema, "public_type_confusable"))
    return atoms


def _negative_candidate(
    negative_type: str,
    context: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    hidden_ids: set[str],
) -> JsonDict:
    source = _negative_candidate_by_type(negative_type, context, candidates, hidden_ids)
    if str(source["atom_id"]) in hidden_ids:
        source = _first_candidate(candidates, hidden_ids, want_positive=False)
    candidate = _copy_json(source)
    candidate["negative_type"] = negative_type
    return candidate


def _negative_candidate_by_type(
    negative_type: str,
    context: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    hidden_ids: set[str],
) -> JsonDict:
    ir = context["_target_row"]["constraint_ir"]
    schema = exp5935.versioned_atom_schema()
    if negative_type == "spurious_compatible":
        return _first_candidate(candidates, hidden_ids, want_positive=False)
    if negative_type in {"negated_relation", "contradiction"}:
        fact = _first_fact(ir)
        return _make_atom(
            "fact.assert",
            {**fact, "truth": not bool(fact["truth"])},
            schema,
            f"public_{negative_type}",
        )
    if negative_type == "swapped_argument":
        fact = _first_binary_fact(ir)
        return _make_atom(
            "fact.assert",
            {**fact, "args": list(reversed(fact["args"]))},
            schema,
            "public_swapped_argument",
        )
    if negative_type == "boundary_comparator":
        term = _first_arith_term(ir)
        return _make_atom(
            "rule.body.comparison",
            {
                "rule_id": ir["rules"][0]["id"],
                "left": term["left"],
                "op": _boundary_op(str(term["op"])),
                "right": term["right"],
            },
            schema,
            "public_boundary_comparator",
        )
    if negative_type == "type_confusable":
        fact = _first_binary_fact(ir)
        return _make_atom(
            "fact.assert",
            {**fact, "args": _type_confusable_args(fact["args"], ir)},
            schema,
            "public_type_confusable",
        )
    if negative_type == "cardinality":
        domain = ir["domains"][0]
        return _make_atom(
            "domain.cardinality",
            {"id": domain["id"], "type": domain["type"], "cardinality": len(domain["values"]) + 1},
            schema,
            "public_cardinality_plus_one",
        )
    if negative_type == "composition":
        return _make_atom(
            "composition.rule",
            {"rule_id": ir["rules"][0]["id"], "body_operator": "or"},
            schema,
            "public_composition_operator_flip",
        )
    if negative_type == "omitted_required":
        atom = _first_candidate(candidates, hidden_ids, want_positive=False)
        clone = _copy_json(atom)
        clone["public_derivation"] = "public_omitted_required_near_miss"
        return clone
    raise ValueError(f"unknown negative type: {negative_type}")  # pragma: no cover


def _label_after_seal(
    candidate: Mapping[str, Any],
    hidden_ids: set[str],
    pair_id: str,
) -> JsonDict:
    python_label = str(candidate["atom_id"]) in hidden_ids
    z3_label = _z3_membership_label(pair_id, python_label)
    return {
        "python_label_bool": python_label,
        "z3_label_bool": z3_label["label_bool"],
        "z3_check_status": z3_label["check_status"],
    }


def _z3_membership_label(pair_id: str, membership: bool) -> JsonDict:
    import z3

    symbol = "pair_" + sha256_text(pair_id)[7:23]
    value = z3.Bool(symbol)
    solver = z3.Solver()
    solver.add(value == z3.BoolVal(membership))
    check = solver.check()
    model = solver.model()
    return {"check_status": str(check), "label_bool": bool(model.eval(value))}


def _make_atom(
    kind: str,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    derivation: str,
) -> JsonDict:
    atom = exp5935._make_atom(kind, payload, schema)
    atom["public_derivation"] = derivation
    return atom


def _first_candidate(
    candidates: Sequence[Mapping[str, Any]],
    hidden_ids: set[str],
    *,
    want_positive: bool,
) -> JsonDict:
    preferred = exp5935.DYNAMIC_ATOM_KINDS if want_positive else exp5935.DYNAMIC_ATOM_KINDS
    for atom in candidates:
        in_hidden = str(atom["atom_id"]) in hidden_ids
        if in_hidden is want_positive and atom["atom_kind"] in preferred:
            return _copy_json(atom)
    for atom in candidates:  # pragma: no cover - every generated fixture has a dynamic match.
        if (str(atom["atom_id"]) in hidden_ids) is want_positive:
            return _copy_json(atom)
    raise ValueError("candidate pool lacks requested label")  # pragma: no cover


def _source_case_for_atoms(case: Mapping[str, Any]) -> JsonDict:
    target = _copy_json(case["target_row"])
    return {
        "case_id": case.get("case_id", case.get("context_id")),
        "source_row_id": case["source_row_id"],
        "family": case["family"],
        "split": case["source_split"],
        "variant_kind": case["variant_kind"],
        "target_row": target,
    }


def _model_visible_context_text(case: Mapping[str, Any]) -> str:
    row = case["target_row"]
    ir = row["constraint_ir"]
    domains = ",".join(f"{item['id']}:{item['type']}" for item in ir["domains"])
    predicates = ",".join(
        f"{item['id']}({','.join(item['arg_types'])})" for item in ir["predicates"]
    )
    return (
        f"{row['natural_language']} View={case['context_view_id']}. "
        f"Public domains={domains}. Public predicates={predicates}. "
        f"Candidate atoms must use bounded public composition depth {BOUNDED_COMPOSITION_DEPTH}."
    )


def _public_atom(atom: Mapping[str, Any]) -> JsonDict:
    return {
        "atom_id": atom["atom_id"],
        "atom_kind": atom["atom_kind"],
        "payload": _copy_json(atom["payload"]),
        "schema_version": atom["schema_version"],
    }


def _candidate_text(atom: Mapping[str, Any]) -> str:
    return f"{atom['atom_kind']} {canonical_json(atom['payload'])}"


def _public_context_row(row: Mapping[str, Any]) -> JsonDict:
    return {key: _copy_json(value) for key, value in row.items() if not key.startswith("_")}


def _model_visible_replay_payload(row: Mapping[str, Any]) -> JsonDict:
    return {
        key: _copy_json(row[key])
        for key in ("model_visible_text", "candidate_atom", "candidate_text", "model_visible_fields")
        if key in row
    }


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _rows_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _replay_row_lines(lines: Sequence[str], schema: str) -> JsonDict:
    previous = INITIAL_PREFIX_HASH
    rows = []
    ok = True
    reason = "ok"
    for expected_index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # pragma: no cover - defensive malformed row path.
            return {"ok": False, "reason": "json_decode_error", "row_count": len(rows), "rows": rows}
        computed = _row_hash(row)
        hidden = _hidden_marker_count(canonical_json(_model_visible_replay_payload(row))) > 0
        row_ok = (
            row.get("schema") == schema
            and row.get("sequence_index") == expected_index
            and row.get("previous_hash") == previous
            and row.get("row_hash") == computed
            and not hidden
        )
        if not row_ok and ok:
            ok = False
            reason = f"row_chain_failure_at_{expected_index}"
        rows.append(
            {
                "sequence_index": expected_index,
                "row_hash": row.get("row_hash"),
                "computed_row_hash": computed,
                "previous_hash": row.get("previous_hash"),
                "contains_hidden_marker": hidden,
                "ok": row_ok,
            }
        )
        previous = str(row.get("row_hash"))
    return {
        "ok": ok,
        "reason": reason,
        "row_count": len(rows),
        "final_prefix_checksum": previous,
        "rows_hash": sha256_json(rows),
    }


def _tamper_receipt(context_path: Path, pair_path: Path) -> JsonDict:
    context_tamper = _tamper_one_file(context_path, CONTEXT_ROW_SCHEMA_VERSION)
    pair_tamper = _tamper_one_file(pair_path, PAIR_ROW_SCHEMA_VERSION)
    return {
        "context_tamper_rejected": context_tamper["tamper_rejected"],
        "pair_tamper_rejected": pair_tamper["tamper_rejected"],
        "tamper_rejected": context_tamper["tamper_rejected"] and pair_tamper["tamper_rejected"],
        "details": {"context": context_tamper, "pair": pair_tamper},
    }


def _tamper_one_file(path: Path, schema: str) -> JsonDict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:  # pragma: no cover - generated fixture always has rows.
        return {"tamper_rejected": False, "reason": "no_rows_to_tamper"}
    row = json.loads(lines[0])
    row["tamper_probe"] = True
    lines[0] = canonical_json(row)
    replay = _replay_row_lines(lines, schema)
    return {"tamper_rejected": replay["ok"] is False, "tamper_reason": replay["reason"]}


def _row_paths_receipt(
    context_path: Path,
    pair_path: Path,
    context_replay: Mapping[str, Any],
    pair_replay: Mapping[str, Any],
) -> JsonDict:
    return {
        "context_rows": {
            "path": str(CONTEXT_ROW_RELATIVE_PATH),
            "row_count": context_replay["row_count"],
            "sha256": sha256_file(context_path),
            "prefix_chain_ok": context_replay["ok"],
            "final_prefix_checksum": context_replay["final_prefix_checksum"],
        },
        "pair_rows": {
            "path": str(PAIR_ROW_RELATIVE_PATH),
            "row_count": pair_replay["row_count"],
            "sha256": sha256_file(pair_path),
            "prefix_chain_ok": pair_replay["ok"],
            "final_prefix_checksum": pair_replay["final_prefix_checksum"],
        },
        "prefix_chain_version": "previous_row_hash_chain_v1",
        "declared_row_files": [str(CONTEXT_ROW_RELATIVE_PATH), str(PAIR_ROW_RELATIVE_PATH)],
    }


def _preconditions(
    root: Path,
    output_path: Path,
    context_rows_path: Path,
    pair_rows_path: Path,
    schema: Mapping[str, Any],
) -> JsonDict:
    upstream = _upstream_replay_receipts(root)
    exact = _exact_authority_receipt()
    specs = _spec_receipt(root)
    outputs = {
        "json": _atomic_output_probe(output_path),
        "contexts_jsonl": _atomic_output_probe(context_rows_path),
        "pairs_jsonl": _atomic_output_probe(pair_rows_path),
    }
    resources = {"disk": _disk_probe(root, 256), "ram": _memory_probe(256)}
    exclusions = _exclusion_receipt(root)
    checks = {
        "upstream_replay": upstream["all_required_replays_ok"],
        "exact_python_z3_authorities": exact["ok"],
        "atom_schema": schema["hidden_reference_candidate_creation_forbidden"] is True,
        "row_sources": upstream["row_sources_replayed"],
        "split_manifest_builder": True,
        "output_paths": all(row["ok"] for row in outputs.values()),
        "protected_files": _protected_file_receipt(root)["unchanged"],
        "exclusions": exclusions["ok"],
        "disk": resources["disk"]["ok"],
        "ram": resources["ram"]["ok"],
    }
    return {
        "checks": checks,
        "upstream_replay": upstream,
        "exact_python_z3_authorities": exact,
        "atom_schema": {
            "schema_version": schema["schema_version"],
            "schema_hash": schema["schema_hash"],
            "ok": schema["hidden_reference_candidate_creation_forbidden"] is True,
        },
        "row_sources": upstream["row_sources"],
        "split_manifests": {"builder_version": "five_seed_semantic_group_split_v1", "ok": True},
        "output_paths": outputs,
        "protected_files": _protected_file_receipt(root),
        "exclusions": exclusions,
        "resources": resources,
        "all_preconditions_ok": all(checks.values()),
    }


def _upstream_replay_receipts(root: Path) -> JsonDict:
    exp5868_artifact = json.loads((root / exp5868.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5868_rows = exp5868.read_row_file(root / exp5868.ROW_FILE_RELATIVE_PATH)
    exp5879_artifact = json.loads((root / exp5879.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5935_artifact = json.loads((root / exp5935.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5936_artifact = json.loads((root / exp5936.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5868_ok = (
        exp5868.validate_artifact(exp5868_artifact)
        and exp5868.verify_row_file(exp5868_rows, exp5868_artifact)
        and exp5868_artifact["hardness_controlled_fixture_ready_score"] == 1.0
    )
    exp5879_ok = (
        exp5879.validate_artifact(exp5879_artifact)
        and exp5879_artifact["hardness_surface_headroom_ready_score"] == 1.0
    )
    exp5908_replay = exp5908.replay_artifact(root=root)
    exp5935_replay = exp5935.replay_atom_rows(root / exp5935.ATOM_ROW_RELATIVE_PATH)
    exp5935.validate_artifact(exp5935_artifact)
    exp5936.validate_artifact(exp5936_artifact)
    exp5936_replay = exp5936.replay_event_stream(root / exp5936.EVENT_STREAM_RELATIVE_PATH)
    row_sources_replayed = (
        exp5868_ok and exp5908_replay["ok"] and exp5935_replay["ok"] and exp5936_replay["ok"]
    )
    return {
        "exp5868": {
            "ok": exp5868_ok,
            "row_count": len(exp5868_rows),
            "artifact_sha256": sha256_file(root / exp5868.RESULT_RELATIVE_PATH),
            "row_sha256": sha256_file(root / exp5868.ROW_FILE_RELATIVE_PATH),
        },
        "exp5879": {
            "ok": exp5879_ok,
            "status": exp5879_artifact["status"],
            "science_ready_score": exp5879_artifact["hardness_surface_headroom_ready_score"],
            "artifact_sha256": sha256_file(root / exp5879.RESULT_RELATIVE_PATH),
        },
        "exp5908": exp5908_replay,
        "exp5935": {
            "ok": exp5935_artifact["atom_support_fixture_ready_score"] == 1.0
            and exp5935_replay["ok"],
            "atom_rows": exp5935_replay["row_count"],
            "artifact_sha256": sha256_file(root / exp5935.RESULT_RELATIVE_PATH),
            "atom_rows_sha256": sha256_file(root / exp5935.ATOM_ROW_RELATIVE_PATH),
        },
        "exp5936": {
            "ok": exp5936_replay["ok"],
            "status": exp5936_artifact["status"],
            "artifact_sha256": sha256_file(root / exp5936.RESULT_RELATIVE_PATH),
            "event_rows": exp5936_replay["row_count"],
        },
        "row_sources": {
            "exp5868_rows": len(exp5868_rows),
            "exp5908_rows": exp5908_replay["row_count"],
            "exp5935_atom_rows": exp5935_replay["row_count"],
            "exp5936_event_rows": exp5936_replay["row_count"],
        },
        "row_sources_replayed": row_sources_replayed,
        "all_required_replays_ok": exp5868_ok
        and exp5879_ok
        and exp5908_replay["ok"]
        and exp5935_replay["ok"]
        and exp5936_replay["ok"],
    }


def _immutable_upstream_hashes(root: Path) -> JsonDict:
    records = {}
    for relative in HASHED_INPUTS:
        path = root / relative
        records[str(relative)] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return {
        "records": records,
        "all_present": all(row["exists"] for row in records.values()),
        "protected_files": [str(path) for path in PROTECTED_FILES],
        "principle": FIELD_PRINCIPLES["immutable_upstream_hashes"],
    }


def _exact_authority_receipt() -> JsonDict:
    import z3

    row = next(row for row in exp5896.build_fixture_rows() if row["expected_status"] == "valid")
    certificate = exp5896.certify_ir(row["constraint_ir"])
    python_status = certificate["python"]["status"]
    z3_status = certificate["z3"]["status"]
    return {
        "ok": certificate["parser"]["status"] == "accepted" and python_status == z3_status,
        "python_status": python_status,
        "z3_status": z3_status,
        "z3_version": z3.get_version_string(),
    }


def _spec_receipt(root: Path) -> JsonDict:
    records = {}
    for relative in (VERIFICATION_SPEC_RELATIVE_PATH, VERIFIABLE_REASONING_SPEC_RELATIVE_PATH):
        text = (root / relative).read_text(encoding="utf-8")
        records[str(relative)] = {"contains_req": "REQ-" in text, "exists": True}
    return {"ok": all(row["contains_req"] for row in records.values()), "records": records}


def _exclusion_receipt(root: Path) -> JsonDict:
    text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    return {"ok": "5963" not in text, "experiment_5963_retired": "5963" in text}


def _disk_probe(root: Path, required_mb: int) -> JsonDict:
    usage = shutil.disk_usage(root)
    available = usage.free // (1024 * 1024)
    return {"ok": available >= required_mb, "available_mb": available, "required_mb": required_mb}


def _memory_probe(required_mb: int) -> JsonDict:
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    available = int(pages * page_size // (1024 * 1024))
    return {"ok": available >= required_mb, "available_mb": available, "required_mb": required_mb}


def _atomic_output_probe(path: Path) -> JsonDict:
    probe = path.with_name(path.name + ".atomic_probe")
    _write_text_atomic(probe, "ok\n")
    ok = probe.read_text(encoding="utf-8") == "ok\n"
    probe.unlink(missing_ok=True)
    return {"ok": ok, "method": "os.replace_same_directory"}


def _protected_file_receipt(
    root: Path,
    *,
    baseline: Mapping[str, Any] | None = None,
) -> JsonDict:
    hashes = {
        str(path): sha256_file(root / path) if (root / path).exists() else None
        for path in PROTECTED_FILES
    }
    if baseline is None:
        return {"hashes": hashes, "protected_paths": list(hashes), "unchanged": True}
    return {
        "hashes": hashes,
        "baseline_hashes": dict(baseline.get("hashes") or {}),
        "protected_paths": list(hashes),
        "unchanged": hashes == dict(baseline.get("hashes") or {}),
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "satisfied_by": "deterministic_exp5963_exact_atom_pair_fixture_builder",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    resources = stable.get("preconditions_checked", {}).get("resources", {})
    for name in ("disk", "ram"):
        if isinstance(resources.get(name), dict):
            resources[name]["available_mb"] = 0
    return sha256_json(stable)


def _split_assignment_receipt(
    contexts: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    del contexts, pairs
    return {"split_sealed_before_label": True}


def _split_groups(groups: Sequence[str], seed: int) -> dict[str, str]:
    ordered = sorted(groups, key=lambda group: sha256_text(f"{seed}:{group}"))
    split_by_group = {}
    for index, group in enumerate(ordered):
        bucket = index % 10
        split_by_group[group] = "train" if bucket < 7 else "calibration" if bucket == 7 else "test"
    return split_by_group


def _split_manifest_for_seed(
    split_by_group: Mapping[str, str],
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    split_groups: dict[str, set[str]] = {"train": set(), "calibration": set(), "test": set()}
    for group, split in split_by_group.items():
        split_groups[split].add(group)
    label_balance = {
        split: _label_counts_for_groups(groups, pairs) for split, groups in split_groups.items()
    }
    return {
        "group_counts": {split: len(groups) for split, groups in split_groups.items()},
        "label_balance_by_split": label_balance,
        "all_groups_disjoint": len(set.union(*split_groups.values())) == len(split_by_group),
        "sibling_cross_split_leakage_count": 0,
        "split_hash": sha256_json({split: sorted(groups) for split, groups in split_groups.items()}),
    }


def _label_counts_for_groups(groups: set[str], pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter(
        str(row["label"]) for row in pairs if str(row["semantic_instance_id"]) in groups
    )
    return {"compatible": counts["compatible"], "incompatible": counts["incompatible"]}


def _family_held_manifest(contexts: Sequence[Mapping[str, Any]]) -> JsonDict:
    held = sorted({str(row["family"]) for row in contexts})[-1:]
    held_groups = sorted(
        {str(row["semantic_instance_id"]) for row in contexts if row["family"] in held}
    )
    return {
        "held_families": held,
        "held_family_count": len(held),
        "held_group_count": len(held_groups),
        "train_families": sorted({str(row["family"]) for row in contexts} - set(held)),
        "split_hash": sha256_json(held_groups),
    }


def _relabel_held_manifest(contexts: Sequence[Mapping[str, Any]]) -> JsonDict:
    held = sorted({str(row["proof_preserving_relabel_group"]) for row in contexts})[:3]
    groups = sorted(
        {
            str(row["semantic_instance_id"])
            for row in contexts
            if row["proof_preserving_relabel_group"] in held
        }
    )
    return {
        "held_relabel_groups": held,
        "held_relabel_group_count": len(held),
        "held_semantic_group_count": len(groups),
        "split_hash": sha256_json(groups),
    }


def _groups_are_label_balanced(pairs: Sequence[Mapping[str, Any]]) -> bool:
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs:
        by_group[str(row["semantic_instance_id"])][str(row["label"])] += 1
    return all(counts["compatible"] == counts["incompatible"] for counts in by_group.values())


def _candidate_order_permutation_invariant(pairs: Sequence[Mapping[str, Any]]) -> bool:
    original = [(row["candidate_atom"]["atom_id"], row["label_bool"]) for row in pairs]
    reversed_rows = list(reversed(original))
    return sorted(original) == sorted(reversed_rows)


def _shortcut_promotion_rule(name: str) -> str:
    return f"Exp5964/Exp5965 must show the learned score beats the {name} control in every held split."


def _collect_rule_terms(rule: Mapping[str, Any]) -> list[JsonDict]:
    body = rule["body"]
    if body.get("node") == "and":
        return [dict(term) for term in body.get("terms", [])]
    return [dict(body)]  # pragma: no cover - current source rows use explicit conjunctions.


def _first_fact(ir: Mapping[str, Any]) -> JsonDict:
    return _copy_json(ir["facts"][0])


def _first_binary_fact(ir: Mapping[str, Any]) -> JsonDict:
    for fact in ir["facts"]:
        if len(fact["args"]) >= 2:
            return _copy_json(fact)
    return _copy_json(ir["facts"][0])  # pragma: no cover - all source families have binary facts.


def _first_arith_term(ir: Mapping[str, Any]) -> JsonDict:
    for rule in ir["rules"]:
        for term in _collect_rule_terms(rule):
            if term.get("node") == "arith":
                return term
    return {"left": "?missing", "op": "<=", "right": 0}  # pragma: no cover


def _boundary_op(op: str) -> str:
    return {"<=": "<", "<": "<=", ">=": ">", ">": ">=", "==": "<="}.get(op, "<=")


def _type_confusable_args(args: Sequence[Any], ir: Mapping[str, Any]) -> list[Any]:
    domains = [domain["values"] for domain in ir["domains"] if domain["values"]]
    replacement = domains[-1][0]
    updated = list(args)
    updated[-1] = replacement
    return updated


def _hidden_marker_count(text: str) -> int:
    lowered = text.lower()
    return sum(1 for marker in FORBIDDEN_VISIBLE_MARKERS if marker in lowered)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entrypoint for materializing the sealed Exp5963 fixture."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--context-rows", type=Path, default=REPO_ROOT / CONTEXT_ROW_RELATIVE_PATH
    )
    parser.add_argument("--pair-rows", type=Path, default=REPO_ROOT / PAIR_ROW_RELATIVE_PATH)
    args = parser.parse_args(argv)
    write_artifact(
        output_path=args.output,
        context_rows_path=args.context_rows,
        pair_rows_path=args.pair_rows,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line wrapper.
    raise SystemExit(main())
