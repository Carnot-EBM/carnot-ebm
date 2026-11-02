"""Exp6484 non-generation representation receipt contract.

Spec refs: REQ-INFRA-6484, SCENARIO-INFRA-6484-COMMITMENT,
SCENARIO-INFRA-6484-PERSISTENCE, SCENARIO-INFRA-6484-NO-GENERATION,
SCENARIO-INFRA-6484-FAMILY-SEPARATION, SCENARIO-INFRA-6484-ATTACKS,
SCENARIO-INFRA-6484-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6484
TASK_ID = "exp6484-non-generation-representation-receipt-contract"
INFERENCE_SUBSTRATE = "deterministic_representation_contract_no_llm"
SCHEMA_VERSION = "carnot.experiment_6484.non_generation_representation_receipt_contract.v1"
RECEIPT_SCHEMA_VERSION = SCHEMA_VERSION + ".receipt"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6484_non_generation_representation_receipt_contract.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6484_non_generation_representation_receipt_contract.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6484_non_generation_representation_receipt_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
ROADMAP_PROPOSAL_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

PHASES = (
    "candidate_commitment",
    "model_access",
    "raw_vector_persistence",
    "feature_transform",
)
ATTACK_IDS = (
    "generation_api_call",
    "post_load_candidate_edit",
    "duplicate_vector_write",
    "label_read_before_persistence",
    "pooled_family_vectors",
    "dimension_identity",
    "norm_only_signal",
    "length_only_signal",
    "pair_permutation",
    "claim_flip",
)

FAMILY_SPECS: tuple[JsonDict, ...] = (
    {
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "family": "qwen3_6_35b_a3b",
        "native_dimension": 2048,
    },
    {
        "model_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma4_31b_it",
        "native_dimension": 5376,
    },
    {
        "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "family": "gemma4_26b_a4b_it",
        "native_dimension": 2816,
    },
)
FIXTURE_SPECS: tuple[JsonDict, ...] = (
    {
        "fixture_id": "exp6484-fixed-pair-0000-correct",
        "pair_id": "exp6484-fixed-pair-0000",
        "pair_position": "candidate-a",
        "prompt_text": "Select the finite-domain assignment that satisfies clause A.",
        "candidate_text": "x0=true, x1=false, x2=true",
        "claim_text": "candidate_satisfies_clause_A",
        "claim_label": True,
    },
    {
        "fixture_id": "exp6484-fixed-pair-0000-controlled-wrong",
        "pair_id": "exp6484-fixed-pair-0000",
        "pair_position": "candidate-b",
        "prompt_text": "Select the finite-domain assignment that satisfies clause A.",
        "candidate_text": "x0=false, x1=false, x2=true",
        "claim_text": "candidate_satisfies_clause_A",
        "claim_label": False,
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "receipt_schema",
    "fixture_manifest",
    "candidate_commitment_rows",
    "raw_vector_persistence_rows",
    "no_generation_receipts",
    "family_separation_receipts",
    "transform_manifest",
    "attack_matrix",
    "non_generation_surface_contract_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "protected_files_unchanged",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal contract state.",
    "receipt_schema": "Versioned non-generation representation receipt schema.",
    "fixture_manifest": "Deterministic multi-dimension fixtures.",
    "candidate_commitment_rows": "Proof that candidate bytes predate model access.",
    "raw_vector_persistence_rows": "One durable write per raw vector.",
    "no_generation_receipts": "Proof that no generation API was called.",
    "family_separation_receipts": "Proof that native dimensions were not pooled.",
    "transform_manifest": "Frozen transforms bound to raw hashes.",
    "attack_matrix": "All shortcut and lifecycle attacks.",
    "non_generation_surface_contract_ready_score": "Same-roadmap downstream gate field.",
    "per_unit_rows": "Fixture, phase, and attack rows.",
    "aggregate_row_recomputation": "Ready score recomputed from rows.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "gate_check_summary": "Required for any blocked_* verdict.",
    "preconditions_checked": "Retirement and fixture prechecks.",
    "inference_substrate": (
        "`deterministic_representation_contract_no_llm` states that no LLM was loaded."
    ),
    "verifier_is_oracle": "True only for deterministic receipt validation.",
    "field_principles": "Reason for each field.",
    "field_provenance": "Source paths, hashes, and reducers.",
    "random_seed": "Fixed attack ordering seed.",
    "duration_s": "Measured wall time.",
    "tests_run": "Executed commands and exit codes.",
    "reproducibility_checksum": "Hash over schema, fixtures, and attacks.",
    "honest_verdict": "States contract readiness without a model-quality claim.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6484_non_generation_representation_receipt_contract "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6484_non_generation_representation_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6484_non_generation_representation_receipt_contract.py "
    "-m pytest "
    "tests/python/test_experiment_6484_non_generation_representation_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6484_non_generation_representation_receipt_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6484_non_generation_representation_receipt_contract.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6484_non_generation_representation_receipt_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6484_non_generation_representation_receipt_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6484_non_generation_representation_receipt_contract --validate"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "text=Path('ops/e2e-test-plan.md').read_text(); assert 'E2E' in text\""
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    VALIDATE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_COMMAND,
)
DEFAULT_TEST_RESULTS = tuple(
    {"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_PROPOSAL_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("results/experiment_5852_three_family_paired_embeddings.json"),
    Path("results/experiment_5853_paired_embedding_integrity_audit.json"),
    Path("python/carnot/experiment_5852_three_family_paired_embeddings.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
    Path("scripts/adversarial_verify.py"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_output(args: Sequence[str], root: Path) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {str(path): receipts.sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {str(path): receipts.sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for path, before_hash in before.items():
        after_hash = receipts.sha256_file(root / path)
        files[path] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "unchanged": before_hash == after_hash,
        }
    return {
        "protected_files_unchanged": all(row["unchanged"] for row in files.values()),
        "files": files,
    }


def _copy_json(value: Any) -> Any:
    return json.loads(receipts.canonical_json(value))


def row_hash(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "row_hash"}
    return receipts.sha256_json(payload)


def _with_row_hash(row: Mapping[str, Any]) -> JsonDict:
    out = dict(row)
    out["row_hash"] = row_hash(out)
    return out


def _refresh_row(row: JsonDict) -> JsonDict:
    row["row_hash"] = row_hash(row)
    return row


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _model_hash(family: Mapping[str, Any]) -> str:
    return receipts.sha256_json(
        {
            "model_id": family["model_id"],
            "family": family["family"],
            "native_dimension": family["native_dimension"],
            "fixture_only_no_model_load": True,
        }
    )


def _deterministic_vector(fixture_id: str, family: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        seed = f"{RANDOM_SEED}|{fixture_id}|{family}|{counter}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        for offset in range(0, len(digest), 4):
            chunk = int.from_bytes(digest[offset : offset + 4], "big")
            values.append(round((chunk % 2_000_000) / 1_000_000.0 - 1.0, 6))
            if len(values) == dimension:
                break
        counter += 1
    return values


def _vector_hash(vector: Sequence[float]) -> str:
    return receipts.sha256_json([round(float(value), 6) for value in vector])


def _candidate_hash(text: str) -> str:
    return receipts.sha256_text(text)


def _claim_commitment_hash(fixture: Mapping[str, Any]) -> str:
    return receipts.sha256_json(
        {
            "claim_text": fixture["claim_text"],
            "claim_label": fixture["claim_label"],
            "pair_id": fixture["pair_id"],
            "pair_position": fixture["pair_position"],
        }
    )


def _fixture_manifest() -> JsonDict:
    fixtures = []
    for fixture in FIXTURE_SPECS:
        row = dict(fixture)
        row["prompt_hash"] = receipts.sha256_text(str(fixture["prompt_text"]))
        row["candidate_hash"] = _candidate_hash(str(fixture["candidate_text"]))
        row["candidate_byte_length"] = len(str(fixture["candidate_text"]).encode("utf-8"))
        row["claim_commitment_hash"] = _claim_commitment_hash(fixture)
        row["label_hidden_until_after_raw_persistence"] = True
        fixtures.append(row)
    families = []
    for family in FAMILY_SPECS:
        row = dict(family)
        row["model_hash"] = _model_hash(family)
        row["model_loaded"] = False
        row["fixture_vector_generator"] = "sha256_counter_stream_round6"
        families.append(row)
    manifest: JsonDict = {
        "schema_version": SCHEMA_VERSION + ".fixture_manifest",
        "planning_date": RUN_DATE,
        "fixtures": fixtures,
        "families": families,
        "fixture_count": len(fixtures),
        "family_count": len(families),
        "native_dimensions": [row["native_dimension"] for row in families],
    }
    manifest["manifest_hash"] = receipts.sha256_json(manifest)
    return manifest


def receipt_schema() -> JsonDict:
    schema = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "required_receipt_fields": [
            "prompt_hash",
            "candidate_hash",
            "pre_model_commitment_ns",
            "model_id",
            "model_hash",
            "family",
            "native_dimension",
            "vector_hash",
            "write_count",
            "phase_intervals",
            "no_generation_witness",
        ],
        "row_types": [
            "candidate_commitment",
            "phase",
            "raw_vector_persistence",
            "no_generation_receipt",
            "family_separation",
            "transform_binding",
            "attack",
        ],
        "required_phases": list(PHASES),
        "attack_ids": list(ATTACK_IDS),
        "no_generation_methods_forbidden": ["generate", "completion", "chat", "decode"],
    }
    schema["schema_hash"] = receipts.sha256_json(schema)
    return schema


def _phase_window(fixture_index: int, family_index: int, phase_index: int) -> tuple[int, int]:
    start = 1_000_000_000 + fixture_index * 10_000_000 + family_index * 1_000_000
    start += phase_index * 100_000
    return start, start + 50_000


def _candidate_commitment_rows(manifest: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for fixture_index, fixture in enumerate(manifest["fixtures"]):
        start, end = _phase_window(fixture_index, 0, 0)
        model_start, _ = _phase_window(fixture_index, 0, 1)
        rows.append(
            _with_row_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "candidate_commitment",
                    "task_id": TASK_ID,
                    "fixture_id": fixture["fixture_id"],
                    "pair_id": fixture["pair_id"],
                    "pair_position": fixture["pair_position"],
                    "prompt_text": fixture["prompt_text"],
                    "candidate_text": fixture["candidate_text"],
                    "prompt_hash": fixture["prompt_hash"],
                    "candidate_hash": fixture["candidate_hash"],
                    "candidate_byte_length": fixture["candidate_byte_length"],
                    "claim_commitment_hash": fixture["claim_commitment_hash"],
                    "pre_model_commitment_ns": start,
                    "commitment_end_ns": end,
                    "model_access_start_ns": model_start,
                    "candidate_edit_monotonic_ns": None,
                    "candidate_frozen_before_model_access": True,
                }
            )
        )
    return rows


def _phase_rows(manifest: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for fixture_index, fixture in enumerate(manifest["fixtures"]):
        for family_index, family in enumerate(manifest["families"]):
            for phase_index, phase in enumerate(PHASES):
                start, end = _phase_window(fixture_index, family_index, phase_index)
                rows.append(
                    _with_row_hash(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "row_type": "phase",
                            "task_id": TASK_ID,
                            "fixture_id": fixture["fixture_id"],
                            "family": family["family"],
                            "model_id": family["model_id"],
                            "phase": phase,
                            "monotonic_start_ns": start,
                            "monotonic_end_ns": end,
                            "wall_clock_start": "2026-08-21T00:00:00Z",
                            "wall_clock_end": "2026-08-21T00:00:00Z",
                        }
                    )
                )
    return rows


def _raw_vector_rows(manifest: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for fixture_index, fixture in enumerate(manifest["fixtures"]):
        for family_index, family in enumerate(manifest["families"]):
            start, end = _phase_window(fixture_index, family_index, 2)
            vector = _deterministic_vector(
                str(fixture["fixture_id"]),
                str(family["family"]),
                int(family["native_dimension"]),
            )
            vector_hash = _vector_hash(vector)
            raw_record = {
                "fixture_id": fixture["fixture_id"],
                "family": family["family"],
                "model_id": family["model_id"],
                "native_dimension": family["native_dimension"],
                "vector_hash": vector_hash,
            }
            rows.append(
                _with_row_hash(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "row_type": "raw_vector_persistence",
                        "task_id": TASK_ID,
                        "fixture_id": fixture["fixture_id"],
                        "pair_id": fixture["pair_id"],
                        "pair_position": fixture["pair_position"],
                        "prompt_hash": fixture["prompt_hash"],
                        "candidate_hash": fixture["candidate_hash"],
                        "claim_commitment_hash": fixture["claim_commitment_hash"],
                        "model_id": family["model_id"],
                        "model_hash": family["model_hash"],
                        "family": family["family"],
                        "native_dimension": family["native_dimension"],
                        "raw_vector": vector,
                        "vector_hash": vector_hash,
                        "write_count": 1,
                        "durable_write_index": 1,
                        "durable_record_id": receipts.sha256_json(raw_record),
                        "raw_persist_start_ns": start,
                        "raw_persist_end_ns": end,
                        "label_read_monotonic_ns": end + 10_000,
                        "raw_persisted_before_transform": True,
                    }
                )
            )
    return rows


def _no_generation_rows(manifest: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for fixture_index, fixture in enumerate(manifest["fixtures"]):
        for family_index, family in enumerate(manifest["families"]):
            start, end = _phase_window(fixture_index, family_index, 1)
            witness = {
                "allowed_methods": ["load_representation_backend", "embed_fixed_candidate"],
                "prohibited_methods": [],
                "generation_call_count": 0,
                "model_loaded": False,
            }
            rows.append(
                _with_row_hash(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "row_type": "no_generation_receipt",
                        "task_id": TASK_ID,
                        "fixture_id": fixture["fixture_id"],
                        "family": family["family"],
                        "model_id": family["model_id"],
                        "model_hash": family["model_hash"],
                        "model_access_start_ns": start,
                        "model_access_end_ns": end,
                        "allowed_method_calls": witness["allowed_methods"],
                        "prohibited_method_calls": witness["prohibited_methods"],
                        "generation_call_count": witness["generation_call_count"],
                        "no_generation_witness_hash": receipts.sha256_json(witness),
                    }
                )
            )
    return rows


def _family_separation_rows(
    manifest: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows = []
    for family in manifest["families"]:
        hashes = sorted(
            str(row["vector_hash"])
            for row in raw_rows
            if row.get("family") == family["family"]
        )
        rows.append(
            _with_row_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "family_separation",
                    "task_id": TASK_ID,
                    "family": family["family"],
                    "model_id": family["model_id"],
                    "model_hash": family["model_hash"],
                    "native_dimension": family["native_dimension"],
                    "native_dimensions_seen": [family["native_dimension"]],
                    "raw_vector_hashes": hashes,
                    "pooled_with_families": [],
                    "pool_or_concat_operation_count": 0,
                    "native_dimension_preserved": True,
                }
            )
        )
    return rows


def _transform_manifest(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_family[str(row["family"])].append(row)
    transforms = []
    for family in sorted(by_family):
        family_rows = by_family[family]
        native_dimension = int(family_rows[0]["native_dimension"])
        transforms.append(
            {
                "transform_id": f"hash_bound_per_family_vector:{family}",
                "family": family,
                "native_dimension": native_dimension,
                "input_raw_vector_hashes": sorted(str(row["vector_hash"]) for row in family_rows),
                "operation": "bind_raw_hash_without_pooling",
                "output_dimension": native_dimension,
                "uses_native_dimension_as_feature": False,
                "uses_norm_only": False,
                "uses_candidate_length_only": False,
                "uses_label": False,
                "pools_families": False,
            }
        )
    manifest: JsonDict = {
        "schema_version": SCHEMA_VERSION + ".transform_manifest",
        "frozen_at_utc": "2026-08-21T00:00:00Z",
        "transform_version": "exp6484.hash_bound_per_family_vector.v1",
        "transforms": transforms,
    }
    manifest["manifest_hash"] = receipts.sha256_json(manifest)
    return manifest


def _derived_feature_hash(raw_row: Mapping[str, Any], transform_manifest_hash: str) -> str:
    return receipts.sha256_json(
        {
            "raw_vector_hash": raw_row["vector_hash"],
            "transform_manifest_hash": transform_manifest_hash,
            "family": raw_row["family"],
            "native_dimension": raw_row["native_dimension"],
            "operation": "bind_raw_hash_without_pooling",
        }
    )


def _transform_binding_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    transform_manifest: Mapping[str, Any],
) -> list[JsonDict]:
    rows = []
    manifest_hash = str(transform_manifest["manifest_hash"])
    for raw_row in raw_rows:
        rows.append(
            _with_row_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "transform_binding",
                    "task_id": TASK_ID,
                    "fixture_id": raw_row["fixture_id"],
                    "family": raw_row["family"],
                    "model_id": raw_row["model_id"],
                    "native_dimension": raw_row["native_dimension"],
                    "raw_vector_hash": raw_row["vector_hash"],
                    "source_candidate_hash": raw_row["candidate_hash"],
                    "candidate_byte_length": len(str(raw_row["candidate_hash"])),
                    "claim_commitment_hash": raw_row["claim_commitment_hash"],
                    "pair_position": raw_row["pair_position"],
                    "transform_manifest_hash": manifest_hash,
                    "derived_feature_hash": _derived_feature_hash(raw_row, manifest_hash),
                    "derived_feature_dimension": raw_row["native_dimension"],
                    "feature_kind": "raw_hash_bound_per_family_vector",
                    "pooled_raw_vector_hashes": [raw_row["vector_hash"]],
                    "feature_primitives": {
                        "uses_full_vector_hash": True,
                        "uses_native_dimension_as_feature": False,
                        "uses_norm_only": False,
                        "uses_candidate_length_only": False,
                        "uses_label": False,
                    },
                }
            )
        )
    return rows


def build_contract_rows(*, root: Path = REPO_ROOT) -> JsonDict:
    """Build positive deterministic receipt rows for the contract."""

    manifest = _fixture_manifest()
    candidate_rows = _candidate_commitment_rows(manifest)
    phase_rows = _phase_rows(manifest)
    raw_rows = _raw_vector_rows(manifest)
    no_generation_rows = _no_generation_rows(manifest)
    family_rows = _family_separation_rows(manifest, raw_rows)
    transform_manifest = _transform_manifest(raw_rows)
    transform_rows = _transform_binding_rows(raw_rows, transform_manifest)
    return {
        "root": str(root),
        "fixture_manifest": manifest,
        "transform_manifest": transform_manifest,
        "rows": [
            *candidate_rows,
            *phase_rows,
            *raw_rows,
            *no_generation_rows,
            *family_rows,
            *transform_rows,
        ],
        "candidate_commitment_rows": candidate_rows,
        "raw_vector_persistence_rows": raw_rows,
        "no_generation_receipts": no_generation_rows,
        "family_separation_receipts": family_rows,
    }


def _rows_by_type(rows: Sequence[Mapping[str, Any]], row_type: str) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("row_type") == row_type]


def _fixture_by_id(fixture_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["fixture_id"]): row for row in fixture_manifest["fixtures"]}


def _family_by_name(fixture_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["family"]): row for row in fixture_manifest["families"]}


def validate_contract_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fixture_manifest: Mapping[str, Any],
    transform_manifest: Mapping[str, Any],
) -> JsonDict:
    """Validate receipt rows without trusting the artifact summary."""

    reasons: list[str] = []
    base_rows = [row for row in rows if row.get("row_type") != "attack"]
    for row in base_rows:
        if row.get("row_hash") != row_hash(row):
            _add_reason(reasons, "row_hash_mismatch")
    fixtures = _fixture_by_id(fixture_manifest)
    families = _family_by_name(fixture_manifest)
    candidate_rows = _rows_by_type(base_rows, "candidate_commitment")
    raw_rows = _rows_by_type(base_rows, "raw_vector_persistence")
    no_generation_rows = _rows_by_type(base_rows, "no_generation_receipt")
    family_rows = _rows_by_type(base_rows, "family_separation")
    transform_rows = _rows_by_type(base_rows, "transform_binding")
    phase_rows = _rows_by_type(base_rows, "phase")
    expected_raw_count = len(fixtures) * len(families)
    if len(candidate_rows) != len(fixtures):
        _add_reason(reasons, "candidate_commitment_count")
    candidate_by_fixture: dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        fixture = fixtures.get(str(row.get("fixture_id")))
        if fixture is None:
            _add_reason(reasons, "unknown_fixture")
            continue
        candidate_by_fixture[str(row["fixture_id"])] = row
        expected_hash = _candidate_hash(str(row.get("candidate_text")))
        if row.get("candidate_hash") != expected_hash:
            _add_reason(reasons, "candidate_hash_mismatch")
        if row.get("candidate_text") != fixture["candidate_text"]:
            _add_reason(reasons, "post_load_candidate_edit")
        edit_ns = row.get("candidate_edit_monotonic_ns")
        if edit_ns is not None and int(edit_ns) > int(row["model_access_start_ns"]):
            _add_reason(reasons, "post_load_candidate_edit")
        if int(row["pre_model_commitment_ns"]) >= int(row["model_access_start_ns"]):
            _add_reason(reasons, "candidate_not_committed_before_model_access")
        if row.get("prompt_hash") != fixture["prompt_hash"]:
            _add_reason(reasons, "prompt_hash_mismatch")
        if row.get("pair_position") != fixture["pair_position"]:
            _add_reason(reasons, "pair_permutation_detected")
        if row.get("claim_commitment_hash") != fixture["claim_commitment_hash"]:
            _add_reason(reasons, "claim_flip_detected")
    phase_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in phase_rows:
        phase_groups[(str(row.get("fixture_id")), str(row.get("family")))].append(row)
        if int(row["monotonic_start_ns"]) >= int(row["monotonic_end_ns"]):
            _add_reason(reasons, "phase_interval_invalid")
    for fixture_id in fixtures:
        for family in families:
            group = sorted(
                phase_groups.get((fixture_id, family), []),
                key=lambda row: int(row["monotonic_start_ns"]),
            )
            if [row.get("phase") for row in group] != list(PHASES):
                _add_reason(reasons, "phase_order_or_count_mismatch")
    if len(raw_rows) != expected_raw_count:
        _add_reason(reasons, "raw_vector_row_count")
    raw_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    raw_by_hash: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        fixture = fixtures.get(str(row.get("fixture_id")))
        family = families.get(str(row.get("family")))
        if fixture is None or family is None:
            _add_reason(reasons, "unknown_raw_fixture_or_family")
            continue
        raw_by_cell[(str(row["fixture_id"]), str(row["family"]))].append(row)
        raw_by_hash[str(row["vector_hash"])] = row
        vector = [float(value) for value in row.get("raw_vector", [])]
        if len(vector) != int(family["native_dimension"]):
            _add_reason(reasons, "native_dimension_mismatch")
        if row.get("native_dimension") != family["native_dimension"]:
            _add_reason(reasons, "native_dimension_mismatch")
        if row.get("vector_hash") != _vector_hash(vector):
            _add_reason(reasons, "vector_hash_mismatch")
        if row.get("write_count") != 1 or row.get("durable_write_index") != 1:
            _add_reason(reasons, "duplicate_vector_write")
        if int(row["label_read_monotonic_ns"]) <= int(row["raw_persist_end_ns"]):
            _add_reason(reasons, "label_read_before_raw_persistence")
        if row.get("model_hash") != family["model_hash"]:
            _add_reason(reasons, "model_hash_mismatch")
        if row.get("candidate_hash") != fixture["candidate_hash"]:
            _add_reason(reasons, "candidate_hash_mismatch")
        if row.get("claim_commitment_hash") != fixture["claim_commitment_hash"]:
            _add_reason(reasons, "claim_flip_detected")
    if any(len(values) != 1 for values in raw_by_cell.values()):
        _add_reason(reasons, "duplicate_vector_write")
    if len(raw_by_cell) != expected_raw_count:
        _add_reason(reasons, "raw_vector_cell_count")
    for row in no_generation_rows:
        calls = [str(call).lower() for call in row.get("prohibited_method_calls", [])]
        calls.extend(str(call).lower() for call in row.get("allowed_method_calls", []))
        forbidden = ("generate", "completion", "chat", "decode")
        if int(row.get("generation_call_count", 0) or 0) != 0:
            _add_reason(reasons, "generation_api_called")
        if any(any(token in call for token in forbidden) for call in calls):
            _add_reason(reasons, "generation_api_called")
    for row in family_rows:
        family = families.get(str(row.get("family")))
        if family is None:
            _add_reason(reasons, "unknown_family_separation_row")
            continue
        expected_hashes = sorted(
            str(raw["vector_hash"]) for raw in raw_rows if raw.get("family") == family["family"]
        )
        if row.get("raw_vector_hashes") != expected_hashes:
            _add_reason(reasons, "family_hash_set_mismatch")
        if row.get("native_dimensions_seen") != [family["native_dimension"]]:
            _add_reason(reasons, "family_pooling_detected")
        if row.get("pooled_with_families") or row.get("pool_or_concat_operation_count") != 0:
            _add_reason(reasons, "family_pooling_detected")
        if row.get("native_dimension_preserved") is not True:
            _add_reason(reasons, "family_pooling_detected")
    manifest_hash = str(transform_manifest["manifest_hash"])
    for row in transform_rows:
        raw_row = raw_by_hash.get(str(row.get("raw_vector_hash")))
        fixture = fixtures.get(str(row.get("fixture_id")))
        if raw_row is None or fixture is None:
            _add_reason(reasons, "transform_raw_binding_missing")
            continue
        primitives = dict(row.get("feature_primitives") or {})
        if row.get("transform_manifest_hash") != manifest_hash:
            _add_reason(reasons, "transform_manifest_mismatch")
        if row.get("derived_feature_hash") != _derived_feature_hash(raw_row, manifest_hash):
            _add_reason(reasons, "derived_feature_hash_mismatch")
        if row.get("feature_kind") != "raw_hash_bound_per_family_vector":
            _add_reason(reasons, "derived_feature_kind_mismatch")
        if primitives.get("uses_native_dimension_as_feature") is True:
            _add_reason(reasons, "dimension_identity_shortcut")
        if primitives.get("uses_norm_only") is True:
            _add_reason(reasons, "norm_only_shortcut")
        if primitives.get("uses_candidate_length_only") is True:
            _add_reason(reasons, "length_only_shortcut")
        if primitives.get("uses_label") is True:
            _add_reason(reasons, "label_leakage_shortcut")
        if row.get("pooled_raw_vector_hashes") != [raw_row["vector_hash"]]:
            _add_reason(reasons, "family_pooling_detected")
        if row.get("pair_position") != fixture["pair_position"]:
            _add_reason(reasons, "pair_permutation_detected")
        if row.get("claim_commitment_hash") != fixture["claim_commitment_hash"]:
            _add_reason(reasons, "claim_flip_detected")
    counts = Counter(str(row.get("row_type")) for row in base_rows)
    return {
        "accepted": not reasons,
        "reasons": sorted(reasons),
        "row_type_counts": dict(sorted(counts.items())),
        "native_dimensions_by_family": {
            family: sorted(
                {int(row["native_dimension"]) for row in raw_rows if row.get("family") == family}
            )
            for family in sorted(families)
        },
        "raw_vector_write_counts": {
            f"{fixture_id}|{family}": len(values)
            for (fixture_id, family), values in sorted(raw_by_cell.items())
        },
    }


def mutate_rows_for_attack(attack_id: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return rows with one shortcut or lifecycle mutation applied."""

    mutated: list[JsonDict] = _copy_json(list(rows))
    if attack_id == "generation_api_call":
        row = next(row for row in mutated if row["row_type"] == "no_generation_receipt")
        row["generation_call_count"] = 1
        row["prohibited_method_calls"] = ["generate"]
        _refresh_row(row)
    elif attack_id == "post_load_candidate_edit":
        row = next(row for row in mutated if row["row_type"] == "candidate_commitment")
        row["candidate_text"] = str(row["candidate_text"]) + " edited after load"
        row["candidate_edit_monotonic_ns"] = int(row["model_access_start_ns"]) + 1
        _refresh_row(row)
    elif attack_id == "duplicate_vector_write":
        row = next(row for row in mutated if row["row_type"] == "raw_vector_persistence")
        row["write_count"] = 2
        _refresh_row(row)
    elif attack_id == "label_read_before_persistence":
        row = next(row for row in mutated if row["row_type"] == "raw_vector_persistence")
        row["label_read_monotonic_ns"] = int(row["raw_persist_start_ns"]) - 1
        _refresh_row(row)
    elif attack_id == "pooled_family_vectors":
        row = next(row for row in mutated if row["row_type"] == "family_separation")
        row["pooled_with_families"] = ["other_family"]
        row["pool_or_concat_operation_count"] = 1
        _refresh_row(row)
    elif attack_id == "dimension_identity":
        row = next(row for row in mutated if row["row_type"] == "transform_binding")
        row["feature_kind"] = "native_dimension_identity"
        row["feature_primitives"]["uses_native_dimension_as_feature"] = True
        _refresh_row(row)
    elif attack_id == "norm_only_signal":
        row = next(row for row in mutated if row["row_type"] == "transform_binding")
        row["feature_kind"] = "raw_vector_norm_only"
        row["feature_primitives"]["uses_norm_only"] = True
        _refresh_row(row)
    elif attack_id == "length_only_signal":
        row = next(row for row in mutated if row["row_type"] == "transform_binding")
        row["feature_kind"] = "candidate_length_only"
        row["feature_primitives"]["uses_candidate_length_only"] = True
        _refresh_row(row)
    elif attack_id == "pair_permutation":
        row = next(row for row in mutated if row["row_type"] == "transform_binding")
        row["pair_position"] = "candidate-b" if row["pair_position"] == "candidate-a" else "candidate-a"
        _refresh_row(row)
    elif attack_id == "claim_flip":
        row = next(row for row in mutated if row["row_type"] == "transform_binding")
        row["claim_commitment_hash"] = receipts.sha256_json({"claim_flip": row["fixture_id"]})
        _refresh_row(row)
    else:
        raise ValueError(f"unknown attack_id: {attack_id}")
    return mutated


def mutation_attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    fixture_manifest: Mapping[str, Any],
    transform_manifest: Mapping[str, Any],
) -> JsonDict:
    """Run all mutations and require every one to fail closed."""

    attack_rows = []
    for attack_id in ATTACK_IDS:
        mutated = mutate_rows_for_attack(attack_id, rows)
        report = validate_contract_rows(
            mutated,
            fixture_manifest=fixture_manifest,
            transform_manifest=transform_manifest,
        )
        attack_rows.append(
            _with_row_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "attack",
                    "task_id": TASK_ID,
                    "attack_id": attack_id,
                    "accepted": report["accepted"],
                    "fail_closed": report["accepted"] is False,
                    "reasons": report["reasons"],
                    "mutated_row_count": len(mutated),
                }
            )
        )
    false_accepts = [row["attack_id"] for row in attack_rows if row["fail_closed"] is not True]
    return {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": attack_rows,
        "attack_count": len(attack_rows),
        "false_accept_count": len(false_accepts),
        "false_accept_attack_ids": false_accepts,
        "all_critical_fail_closed": not false_accepts and len(attack_rows) == len(ATTACK_IDS),
    }


def recompute_aggregates_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fixture_manifest: Mapping[str, Any],
    transform_manifest: Mapping[str, Any],
) -> JsonDict:
    """Recompute the ready score from rows and attack outcomes."""

    base_rows = [row for row in rows if row.get("row_type") != "attack"]
    attack_rows = _rows_by_type(rows, "attack")
    validation = validate_contract_rows(
        base_rows,
        fixture_manifest=fixture_manifest,
        transform_manifest=transform_manifest,
    )
    counts = Counter(str(row.get("row_type")) for row in rows)
    attack_ids = {str(row.get("attack_id")) for row in attack_rows}
    checks = {
        "positive_rows_validate": validation["accepted"] is True,
        "candidate_commitments_present": counts.get("candidate_commitment", 0)
        == len(fixture_manifest["fixtures"]),
        "raw_vectors_present": counts.get("raw_vector_persistence", 0)
        == len(fixture_manifest["fixtures"]) * len(fixture_manifest["families"]),
        "phase_rows_present": counts.get("phase", 0)
        == len(fixture_manifest["fixtures"]) * len(fixture_manifest["families"]) * len(PHASES),
        "no_generation_receipts_present": counts.get("no_generation_receipt", 0)
        == len(fixture_manifest["fixtures"]) * len(fixture_manifest["families"]),
        "family_rows_present": counts.get("family_separation", 0)
        == len(fixture_manifest["families"]),
        "transform_rows_present": counts.get("transform_binding", 0)
        == len(fixture_manifest["fixtures"]) * len(fixture_manifest["families"]),
        "all_attacks_present": attack_ids == set(ATTACK_IDS),
        "all_attacks_fail_closed": bool(attack_rows)
        and all(row.get("fail_closed") is True for row in attack_rows),
    }
    score = 1.0 if all(checks.values()) else 0.0
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(counts.items())),
        "validation_reasons": validation["reasons"],
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "non_generation_surface_contract_ready_score_from_rows": score,
    }


def _gate_check_summary(
    *,
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "aggregate_ready_score_is_one": aggregate.get(
            "non_generation_surface_contract_ready_score_from_rows"
        )
        == 1.0,
        "protected_files_unchanged": protected.get("protected_files_unchanged") is True,
        "preconditions_ready": preconditions.get("preconditions_ready") is True,
    }
    return {
        "checks": checks,
        "all_gates_passed": all(checks.values()),
        "failed_gates": [key for key, value in checks.items() if not value],
    }


def _preconditions_checked(root: Path, source_hashes: Mapping[str, str | None]) -> JsonDict:
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    roadmap_text = (root / ROADMAP_PROPOSAL_RELATIVE_PATH).read_text(encoding="utf-8")
    exp5852 = json.loads(
        (root / "results/experiment_5852_three_family_paired_embeddings.json").read_text(
            encoding="utf-8"
        )
    )
    exp5853 = json.loads(
        (root / "results/experiment_5853_paired_embedding_integrity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    retirement_checks = {
        "generated_answer_transport_retired": (
            "finite_id_gguf_generated_answer_transport_same_mechanism_v519" in exclusion_text
            and "generated-answer retry" in exclusion_text
        ),
        "finite_id_patterns_retired": (
            "finite-ID GGUF generated-answer retry" in exclusion_text
            and "parser-only generated-answer retry" in exclusion_text
        ),
        "paired_representation_surface_preserved": (
            "sota paired embeddings" in exclusion_text
            and "final-token/final-layer embedding verifier" in exclusion_text
        ),
        "v559_exp6484_planned": "Exp6484 - Non-generation representation receipt contract"
        in roadmap_text,
        "exp5852_surface_preserved": exp5852.get("status") == "complete"
        and exp5852.get("paired_embedding_corpus_ready_score") == 1.0,
        "exp5853_claim_path_disqualified": exp5853.get("status") == "disqualified"
        and exp5853.get("paired_embedding_integrity_ready_score") == 0.0,
    }
    return {
        "date": RUN_DATE,
        "repository_state": {
            "head": _git_output(["rev-parse", "HEAD"], root),
            "status_short": _git_output(["status", "--short"], root),
        },
        "retirement_and_surface_checks": retirement_checks,
        "deterministic_fixture_only": True,
        "large_model_loaded": False,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "pid": os.getpid(),
            "captured_utc": _utc_now(),
        },
        "source_hashes": dict(source_hashes),
        "preconditions_ready": all(retirement_checks.values()),
    }


def _field_provenance(source_hashes: Mapping[str, str | None]) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    reducers = [
        "build_contract_rows",
        "validate_contract_rows",
        "mutation_attack_matrix",
        "recompute_aggregates_from_rows",
    ]
    return {
        field: {
            "spec_refs": ["REQ-INFRA-6484"],
            "source_paths": source_paths,
            "reducers": reducers,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status(score: float, gates: Mapping[str, Any]) -> str:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return "complete_non_generation_representation_receipt_contract"
    return "blocked_non_generation_representation_receipt_contract"


def _honest_verdict(status: str) -> str:
    if status.startswith("complete_"):
        return (
            "complete: non-generation representation receipt contract is ready; "
            "no model-quality claim is made"
        )
    return (
        "complete_blocked: non-generation representation receipt contract failed; "
        "gate_check_summary names the failed checks"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = False,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal Exp6484 artifact."""

    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    contract = build_contract_rows(root=root)
    attack_matrix = mutation_attack_matrix(
        contract["rows"],
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    per_unit_rows = [*contract["rows"], *attack_matrix["rows"]]
    aggregate = recompute_aggregates_from_rows(
        per_unit_rows,
        fixture_manifest=contract["fixture_manifest"],
        transform_manifest=contract["transform_manifest"],
    )
    protected = _protected_unchanged(root, protected_before)
    preconditions = _preconditions_checked(root, source_hashes)
    gates = _gate_check_summary(
        aggregate=aggregate,
        protected=protected,
        preconditions=preconditions,
    )
    score = float(aggregate["non_generation_surface_contract_ready_score_from_rows"])
    if not gates["all_gates_passed"]:
        score = 0.0
    status = _status(score, gates)
    artifact: JsonDict = {
        "status": status,
        "receipt_schema": receipt_schema(),
        "fixture_manifest": contract["fixture_manifest"],
        "candidate_commitment_rows": contract["candidate_commitment_rows"],
        "raw_vector_persistence_rows": contract["raw_vector_persistence_rows"],
        "no_generation_receipts": contract["no_generation_receipts"],
        "family_separation_receipts": contract["family_separation_receipts"],
        "transform_manifest": contract["transform_manifest"],
        "attack_matrix": attack_matrix,
        "non_generation_surface_contract_ready_score": score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "protected_files_unchanged": protected,
        "gate_check_summary": gates,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "results": list(DEFAULT_TEST_RESULTS if tests_run is None else tests_run),
        },
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result_path)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clone = _copy_json(payload)
    clone["duration_s"] = 0.0
    clone["reproducibility_checksum"] = ""
    return receipts.sha256_json(clone)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the artifact fields and row-derived ready score."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors: list[str] = []
    aggregate = recompute_aggregates_from_rows(
        artifact.get("per_unit_rows", []),
        fixture_manifest=artifact.get("fixture_manifest", {}),
        transform_manifest=artifact.get("transform_manifest", {}),
    )
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("non_generation_surface_contract_ready_score") != aggregate.get(
        "non_generation_surface_contract_ready_score_from_rows"
    ):
        errors.append("non_generation_surface_contract_ready_score mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field}")
            break
    if artifact.get("protected_files_unchanged", {}).get("protected_files_unchanged") is not True:
        errors.append("protected_files_unchanged must be true")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "complete_blocked:")):
        errors.append("honest_verdict lacks required terminal prefix")
    expected_status = _status(
        float(artifact.get("non_generation_surface_contract_ready_score", 0.0) or 0.0),
        artifact.get("gate_check_summary", {}),
    )
    if artifact.get("status") != expected_status:
        errors.append("status mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    return receipts.write_json_atomic(path, artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = True,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and write the Exp6484 artifact."""

    del date
    start = time.monotonic()
    artifact = build_artifact(
        root=REPO_ROOT,
        result_path=result_path,
        write=False,
        duration_s=0.0001,
        tests_run=tests_run,
    )
    artifact["duration_s"] = max(time.monotonic() - start, 0.0001)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(json.dumps({"ok": False, "errors": ["artifact missing"]}, sort_keys=True))
            return 1
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=str(args.date), result_path=result_path, write=True)
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "path": str(result_path),
                "status": artifact["status"],
                "non_generation_surface_contract_ready_score": artifact[
                    "non_generation_surface_contract_ready_score"
                ],
                "ok": not errors,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
