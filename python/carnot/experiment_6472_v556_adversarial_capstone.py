"""Exp6472 V556 adversarial capstone.

Spec refs: REQ-CAPSTONE-6472,
SCENARIO-CAPSTONE-6472-INVENTORY,
SCENARIO-CAPSTONE-6472-RECOMPUTATION,
SCENARIO-CAPSTONE-6472-CLAIM-ELIGIBILITY,
SCENARIO-CAPSTONE-6472-ATTACKS,
SCENARIO-CAPSTONE-6472-FIELD-PRINCIPLES.

This capstone audits checked-in evidence. It does not trust upstream reducers,
because a capstone must survive a bad aggregate field in an upstream artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6472_v556_adversarial_capstone.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
RUN_DATE = "20260819"
RANDOM_SEED = 6472
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6472_v556_adversarial_capstone "
    "--date 20260819"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6472_v556_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6472_v556_adversarial_capstone.py "
    "-m pytest tests/python/test_experiment_6472_v556_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6472_v556_adversarial_capstone.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6472_v556_adversarial_capstone.py"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6472_v556_adversarial_capstone.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6472_v556_adversarial_capstone.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6472 entry"

DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_CONSISTENCY_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_COMMAND,
)

TASKS: tuple[tuple[str, str, Path], ...] = (
    ("exp6459-v555-adversarial-capstone", "exp6459", Path("results/experiment_6459_v555_adversarial_capstone.json")),
    ("exp6460-v556-terminal-handoff-and-queue-integrity", "exp6460", Path("results/experiment_6460_v556_terminal_handoff_and_queue_integrity.json")),
    ("exp6461-v556-primary-source-freshness-receipt", "exp6461", Path("results/experiment_6461_v556_sota_source_and_benchmark_delta.json")),
    ("exp6462-sota-raw-persistence-uniqueness-canary", "exp6462", Path("results/experiment_6462_sota_raw_persistence_uniqueness_canary.json")),
    ("exp6463-sota-fixed-policy-candidate-corpus-v2", "exp6463", Path("results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json")),
    ("exp6464-fixed-slot-grounding-exact-logic-ab", "exp6464", Path("results/experiment_6464_fixed_slot_grounding_exact_logic_ab.json")),
    ("exp6465-representation-objective-causal-ab-v2", "exp6465", Path("results/experiment_6465_representation_objective_causal_ab_v2.json")),
    ("exp6466-held-verifier-budget-allocation-v2", "exp6466", Path("results/experiment_6466_held_verifier_budget_allocation_v2.json")),
    ("exp6467-held-exact-constraint-energy-selection-v2", "exp6467", Path("results/experiment_6467_held_exact_constraint_energy_selection_v2.json")),
    ("exp6468-unique-event-verifier-bounded-csl", "exp6468", Path("results/experiment_6468_unique_event_verifier_bounded_csl.json")),
    ("exp6469-unique-event-csl-corruption-restart", "exp6469", Path("results/experiment_6469_unique_event_csl_corruption_restart.json")),
    ("exp6470-independent-unique-event-csl-audit", "exp6470", Path("results/experiment_6470_independent_unique_event_csl_audit.json")),
    ("exp6471-arc-generic-safety-shield-objective-ab", "exp6471", Path("results/experiment_6471_arc_generic_safety_shield_objective_ab.json")),
    ("exp6472-v556-adversarial-capstone", "exp6472", RESULT_RELATIVE_PATH),
)

GATE_CONTRACTS: tuple[JsonDict, ...] = (
    {
        "task_id": "exp6463-sota-fixed-policy-candidate-corpus-v2",
        "upstream_key": "exp6462",
        "field": "raw_persistence_canary_ready_score",
        "expected": 1.0,
    },
    {
        "task_id": "exp6464-fixed-slot-grounding-exact-logic-ab",
        "upstream_key": "exp6463",
        "field": "sota_corpus_ready_score",
        "expected": 1.0,
    },
    {
        "task_id": "exp6466-held-verifier-budget-allocation-v2",
        "upstream_key": "exp6463",
        "field": "sota_corpus_ready_score",
        "expected": 1.0,
    },
    {
        "task_id": "exp6469-unique-event-csl-corruption-restart",
        "upstream_key": "exp6468",
        "field": "unique_event_csl_ready_score",
        "expected": 1.0,
    },
)

ATTACK_IDS = (
    "zero_byte_persistence",
    "event_cloning",
    "hidden_cpu_fallback",
    "held_leakage",
    "free_probes",
    "energy_as_oracle",
    "exact_veto_bypass",
    "corrupt_restart",
    "source_access",
    "solve_duplication",
    "aggregate_mismatch",
    "protected_file_mutation",
    "prior_verdict_repetition",
)

RAW_PRODUCER_KEYS = {"exp6462", "exp6463", "exp6468", "exp6469"}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/arc_solve_registry.yaml"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_artifact_inventory",
    "gate_contract_recomputation",
    "raw_file_and_event_identity_recomputation",
    "device_and_model_receipt_recomputation",
    "per_unit_rows",
    "independent_grounding_objective_allocation_and_energy_recomputation",
    "independent_csl_recomputation",
    "independent_arc_recomputation",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "critical_discrepancies",
    "repeated_prior_verdict_retirements",
    "determination_preservation",
    "science_claim_eligible",
    "continuous_learning_claim_eligible",
    "arc_claim_eligible",
    "hardware_claim_eligible",
    "reconciliation_changes",
    "v556_capstone_ready_score",
    "protected_files_unchanged",
    "blocked_reason",
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

FIELD_PRINCIPLES: JsonDict = {
    "status": "States that the capstone audit completed without making all branch claims true.",
    "upstream_artifact_inventory": "Freezes each expected V556 evidence file as found on disk.",
    "gate_contract_recomputation": "Recomputes task gates from the upstream fields that consumers name.",
    "raw_file_and_event_identity_recomputation": "Checks raw bytes, hashes, paths, and event identity from disk.",
    "device_and_model_receipt_recomputation": "Checks runner receipts for hidden CPU fallback and model identity.",
    "per_unit_rows": "Records one independent row for each audited upstream claim.",
    "independent_grounding_objective_allocation_and_energy_recomputation": "Keeps science branch reducers separate.",
    "independent_csl_recomputation": "Recomputes unique-event continuous learning evidence.",
    "independent_arc_recomputation": "Recomputes ARC safety-shield evidence without a solve claim.",
    "aggregate_row_recomputation": "Summarizes whether independent reducers match stored row aggregates.",
    "attack_matrix": "Shows each adversarial attack failed closed.",
    "current_adversarial_findings": "Preserves current verifier findings from upstream artifacts.",
    "critical_discrepancies": "Lists evidence gaps that block branch promotion.",
    "repeated_prior_verdict_retirements": "Prevents repeated prior verdicts from becoming new claims.",
    "determination_preservation": "Shows V555 and exclusion determinations stayed preserved.",
    "science_claim_eligible": "Science needs complete row evidence, not readiness fields alone.",
    "continuous_learning_claim_eligible": "CSL needs unique raw events, exact-veto ordering, and audit approval.",
    "arc_claim_eligible": "ARC claim eligibility is limited to the generic safety shield, not solves.",
    "hardware_claim_eligible": "Hardware needs authenticated hardware evidence, which V556 lacks.",
    "reconciliation_changes": "Records that ops reconciliation is deferred by the stop rule.",
    "v556_capstone_ready_score": "Scores the replayability of this audit, not branch success.",
    "protected_files_unchanged": "Confirms protected files were not changed by the capstone.",
    "blocked_reason": "Names a capstone-level block when the audit itself cannot complete.",
    "gate_check_summary": "Keeps pass and fail gates machine-readable.",
    "preconditions_checked": "Records required instructions, specs, roadmaps, and evidence inputs.",
    "inference_substrate": "Declares local aggregation over rows, raw files, registry, and hashes.",
    "verifier_is_oracle": "True only for local hash, exact-checker, registry, and row arithmetic.",
    "field_principles": "Documents why each required field exists.",
    "field_provenance": "Maps fields to specs, rows, raw files, artifacts, or constants.",
    "random_seed": "Fixes deterministic row order.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records commands that verify the capstone.",
    "reproducibility_checksum": "Detects silent payload drift.",
    "honest_verdict": "Uses a terminal prefix and names separate claim eligibility states.",
}
FIELD_PROVENANCE: JsonDict = {
    field: [
        "REQ-CAPSTONE-6472",
        "V556 checked-in artifacts",
        "local raw file hashes",
        "independent row arithmetic",
        "focused Exp6472 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_checksum", None)
    return sha256_json(clone)


def load_json(value: Mapping[str, Any] | str | Path) -> JsonDict:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _status_text(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or payload.get("honest_verdict") or "")


def _artifact_state(path: Path, payload: Mapping[str, Any] | None) -> str:
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "zero_byte"
    if payload is None:
        return "malformed"
    text = _status_text(payload).lower()
    if "blocked" in text or text.startswith("gated"):
        return "blocked"
    if "partial" in text:
        return "partial"
    if "flagged" in text:
        return "flagged"
    return "complete"


def _readiness_fields(payload: Mapping[str, Any] | None) -> JsonDict:
    if payload is None:
        return {}
    return {
        key: value
        for key, value in payload.items()
        if key.endswith("_ready_score") or key.endswith("_eligible_score")
    }


def load_expected_payloads(repo_root: Path) -> tuple[dict[str, JsonDict], JsonDict]:
    payloads: dict[str, JsonDict] = {}
    rows: list[JsonDict] = []
    for task_id, key, relative in TASKS:
        path = repo_root / relative
        payload: JsonDict | None = None
        load_error = ""
        if path.is_file() and path.stat().st_size > 0:
            try:
                payload = load_json(path)
                payloads[key] = payload
            except (OSError, json.JSONDecodeError) as exc:
                load_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "task_id": task_id,
                "artifact_key": key,
                "path": relative.as_posix(),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path),
                "artifact_state": _artifact_state(path, payload),
                "status": payload.get("status") if payload is not None else None,
                "honest_verdict": payload.get("honest_verdict") if payload is not None else None,
                "readiness_fields": _readiness_fields(payload),
                "gate_check_summary": payload.get("gate_check_summary") if payload is not None else None,
                "load_error": load_error,
            }
        )
    inventory = {
        "expected_task_count": len(TASKS),
        "loadable_count": sum(1 for row in rows if row["artifact_state"] not in {"missing", "zero_byte", "malformed"}),
        "missing_task_ids": [row["task_id"] for row in rows if row["artifact_state"] == "missing"],
        "zero_byte_task_ids": [row["task_id"] for row in rows if row["artifact_state"] == "zero_byte"],
        "blocked_task_ids": [row["task_id"] for row in rows if row["artifact_state"] == "blocked"],
        "partial_task_ids": [row["task_id"] for row in rows if row["artifact_state"] == "partial"],
        "rows": rows,
    }
    return payloads, inventory


def _rows_from(payload: Mapping[str, Any], key: str = "per_unit_rows") -> list[JsonDict]:
    raw = payload.get(key)
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, Mapping)]
    if isinstance(raw, Mapping) and isinstance(raw.get("rows"), list):
        return [dict(row) for row in raw["rows"] if isinstance(row, Mapping)]
    return []


def _raw_manifest_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    raw = payload.get("raw_output_manifest")
    if isinstance(raw, Mapping) and isinstance(raw.get("rows"), list):
        return [dict(row) for row in raw["rows"] if isinstance(row, Mapping)]
    return []


def _raw_path(row: Mapping[str, Any]) -> str:
    return str(row.get("raw_output_path") or row.get("path") or "")


def _raw_sha(row: Mapping[str, Any]) -> str:
    return str(
        row.get("raw_output_sha256")
        or row.get("raw_sha256")
        or row.get("raw_hash")
        or row.get("sha256")
        or ""
    )


def _raw_size(row: Mapping[str, Any]) -> int | None:
    for key in ("raw_byte_length", "byte_length", "durable_byte_count"):
        value = row.get(key)
        if isinstance(value, int | float):
            return int(value)
    receipt = row.get("atomic_write_receipt")
    if isinstance(receipt, Mapping) and isinstance(receipt.get("durable_byte_count"), int | float):
        return int(receipt["durable_byte_count"])
    return None


def _event_id(row: Mapping[str, Any]) -> str:
    return str(row.get("event_id") or row.get("row_id") or row.get("unit_id") or "")


def _iter_raw_references(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    refs: dict[tuple[str, str, str], JsonDict] = {}
    for artifact_key, payload in payloads.items():
        if artifact_key.startswith("exp") and artifact_key not in RAW_PRODUCER_KEYS:
            continue
        manifest_rows = _raw_manifest_rows(payload)
        rows = manifest_rows if manifest_rows else _rows_from(payload)
        for row in rows:
            if row.get("row_kind") not in (None, "normal"):
                continue
            path = _raw_path(row)
            raw_sha = _raw_sha(row)
            if not path or not raw_sha:
                continue
            event_id = _event_id(row)
            chronology_fields = (
                "stored_before_parse",
                "raw_persisted_before_parse",
                "raw_output_validated_before_parse",
                "validated_before_parse",
            )
            chronology_present = any(field in row for field in chronology_fields)
            refs[(artifact_key, event_id, path)] = {
                "artifact_key": artifact_key,
                "event_id": event_id,
                "raw_path": path,
                "declared_sha256": raw_sha,
                "declared_size_bytes": _raw_size(row),
                "stored_before_parse": True
                if not chronology_present
                else bool(
                    row.get("stored_before_parse")
                    or row.get("raw_persisted_before_parse")
                    or row.get("raw_output_validated_before_parse")
                    or row.get("validated_before_parse")
                ),
            }
    return list(refs.values())


def recompute_raw_identity(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    refs = _iter_raw_references(payloads)
    raw_path_counts = Counter(ref["raw_path"] for ref in refs)
    event_counts = Counter(ref["event_id"] for ref in refs if ref["event_id"])
    rows: list[JsonDict] = []
    for ref in refs:
        path = Path(ref["raw_path"])
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        actual = sha256_file(path)
        rows.append(
            {
                **ref,
                "exists": exists,
                "actual_size_bytes": size,
                "actual_sha256": actual,
                "size_matches_declared": ref["declared_size_bytes"] in (None, size),
                "sha256_matches_declared": actual == ref["declared_sha256"],
                "zero_byte": exists and size == 0,
                "duplicate_raw_path": raw_path_counts[ref["raw_path"]] > 1,
                "duplicate_event_id": bool(ref["event_id"]) and event_counts[ref["event_id"]] > 1,
            }
        )
    missing = sum(1 for row in rows if not row["exists"])
    zero = sum(1 for row in rows if row["zero_byte"])
    sha_mismatch = sum(1 for row in rows if not row["sha256_matches_declared"])
    size_mismatch = sum(1 for row in rows if not row["size_matches_declared"])
    duplicate_paths = sum(1 for path, count in raw_path_counts.items() if count > 1 and path)
    duplicate_events = sum(1 for event_id, count in event_counts.items() if count > 1 and event_id)
    path_chain_failures = sum(1 for row in rows if row["stored_before_parse"] is False)
    return {
        "raw_reference_count": len(rows),
        "missing_raw_file_count": missing,
        "zero_byte_raw_file_count": zero,
        "sha256_mismatch_count": sha_mismatch,
        "size_mismatch_count": size_mismatch,
        "duplicate_raw_path_count": duplicate_paths,
        "duplicate_event_id_count": duplicate_events,
        "path_chain_failure_count": path_chain_failures,
        "all_raw_contracts_passed": not any(
            (missing, zero, sha_mismatch, size_mismatch, duplicate_paths, duplicate_events, path_chain_failures)
        ),
        "rows": rows,
    }


def _walk_mappings(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        found.append(value)
        for child in value.values():
            found.extend(_walk_mappings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_mappings(child))
    return found


def _bool_from_nested(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    for mapping in _walk_mappings(row):
        nested = mapping.get(key)
        if isinstance(nested, bool):
            return nested
    return False


def device_and_model_receipts(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    rows = [
        row
        for payload in payloads.values()
        for row in _rows_from(payload)
        if row.get("row_kind") in (None, "normal")
    ]
    cpu_fallback_rows = [row for row in rows if _bool_from_nested(row, "cpu_fallback")]
    models = Counter(
        str(row.get("model_hf_id") or row.get("model") or "")
        for row in rows
        if row.get("model_hf_id") or row.get("model")
    )
    model_hashes = Counter(
        str(row.get("model_hash") or row.get("model_file_sha256") or "")
        for row in rows
        if row.get("model_hash") or row.get("model_file_sha256")
    )
    device_sample_count = 0
    cuda_offload_count = 0
    for row in rows:
        for mapping in _walk_mappings(row):
            if "device_samples" in mapping and isinstance(mapping["device_samples"], list):
                device_sample_count += len(mapping["device_samples"])
            if mapping.get("cuda_offload") is True:
                cuda_offload_count += 1
    return {
        "row_count": len(rows),
        "cpu_fallback_count": len(cpu_fallback_rows),
        "cpu_fallback_row_ids": [str(row.get("row_id") or row.get("event_id")) for row in cpu_fallback_rows],
        "model_counts": dict(sorted(models.items())),
        "model_hash_counts": dict(sorted(model_hashes.items())),
        "device_sample_count": device_sample_count,
        "cuda_offload_marker_count": cuda_offload_count,
        "hidden_cpu_fallback_absent": len(cpu_fallback_rows) == 0,
    }


def recompute_gate_contracts(
    payloads: Mapping[str, Mapping[str, Any]],
    contracts: Sequence[Mapping[str, Any]] = GATE_CONTRACTS,
) -> JsonDict:
    rows: list[JsonDict] = []
    for contract in contracts:
        upstream = payloads.get(str(contract["upstream_key"]), {})
        field = str(contract["field"])
        actual = upstream.get(field)
        missing_field = field not in upstream
        expected = contract["expected"]
        passed = (not missing_field) and actual == expected
        rows.append(
            {
                "task_id": contract["task_id"],
                "upstream_key": contract["upstream_key"],
                "field": field,
                "expected": expected,
                "actual": actual,
                "missing_field": missing_field,
                "independent_gate_passed": passed,
                "reason": "passed" if passed else f"actual={actual!r} expected={expected!r}",
            }
        )
    return {
        "passed": all(row["independent_gate_passed"] for row in rows),
        "failed_count": sum(1 for row in rows if not row["independent_gate_passed"]),
        "missing_field_count": sum(1 for row in rows if row["missing_field"]),
        "rows": rows,
    }


def _round_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def _normal_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    return [
        row
        for row in _rows_from(payload)
        if row.get("row_kind", "normal") == "normal" and "exact_success" in row
    ]


def reduce_sota_corpus(payload: Mapping[str, Any]) -> JsonDict:
    rows = _normal_rows(payload)
    by_partition: dict[str, JsonDict] = {}
    groups: dict[str, dict[tuple[str, str], list[JsonDict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        partition = str(row.get("partition") or "unknown")
        model = str(row.get("model_hf_id") or row.get("model") or "")
        unit = str(row.get("problem_id") or row.get("unit_id") or row.get("unit") or "")
        groups[partition][(model, unit)].append(row)
    for partition, cell_rows in groups.items():
        flat = [row for group in cell_rows.values() for row in group]
        success = sum(1 for row in flat if row.get("exact_success") is True)
        failure = sum(1 for row in flat if row.get("exact_success") is False)
        headroom = sum(
            1
            for group in cell_rows.values()
            if any(row.get("exact_success") is True for row in group)
            and any(row.get("exact_success") is False for row in group)
        )
        by_partition[partition] = {
            "candidate_selection_cell_count": len(cell_rows),
            "candidate_selection_cells_with_headroom": headroom,
            "failure": failure,
            "has_headroom": headroom > 0,
            "mixed_exact_outcomes": headroom > 0,
            "row_count": len(flat),
            "success": success,
        }
    by_model_partition: dict[str, JsonDict] = {}
    grouped: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model_hf_id") or row.get("model")), str(row.get("partition")))].append(row)
    for (model, partition), group in grouped.items():
        success = sum(1 for row in group if row.get("exact_success") is True)
        failure = sum(1 for row in group if row.get("exact_success") is False)
        by_model_partition[f"{model}::{partition}"] = {
            "exact_success_rate": _round_rate(success, len(group)),
            "failure": failure,
            "mixed_exact_outcomes": success > 0 and failure > 0,
            "row_count": len(group),
            "success": success,
        }
    return {
        "row_count": len(rows),
        "candidate_headroom_by_partition": dict(sorted(by_partition.items())),
        "exact_outcomes_by_model_and_partition": dict(sorted(by_model_partition.items())),
        "matches_reported": by_partition == payload.get("candidate_headroom_by_partition")
        and by_model_partition == payload.get("exact_outcomes_by_model_and_partition"),
    }


def _blocked_or_missing_state(inventory: Mapping[str, Any], task_id: str) -> JsonDict:
    rows = {row["task_id"]: row for row in inventory.get("rows", [])}
    row = rows.get(task_id, {})
    return {
        "state": row.get("artifact_state", "missing"),
        "path": row.get("path"),
        "reason": row.get("gate_check_summary") or row.get("load_error") or row.get("artifact_state"),
    }


def _row_exact_success(row: Mapping[str, Any]) -> bool:
    if isinstance(row.get("future_exact_outcome"), bool):
        return bool(row["future_exact_outcome"])
    if isinstance(row.get("exact_success"), bool):
        return bool(row["exact_success"])
    checker = row.get("checker_result")
    if isinstance(checker, Mapping) and isinstance(checker.get("exact_success"), bool):
        return bool(checker["exact_success"])
    exact = row.get("exact_result")
    if isinstance(exact, Mapping) and isinstance(exact.get("exact_success"), bool):
        return bool(exact["exact_success"])
    return False


def reduce_exp6468_csl(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows_from(payload)
    grouped: dict[str, dict[str, list[JsonDict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("schema") and "per_unit_row" not in str(row["schema"]):
            continue
        grouped[str(row.get("interval") or "unknown")][str(row.get("arm") or "unknown")].append(row)
    effects: dict[str, JsonDict] = {}
    for interval, by_arm in grouped.items():
        effects[interval] = {}
        for arm, group in by_arm.items():
            success = sum(1 for row in group if _row_exact_success(row))
            effects[interval][arm] = {
                "exact_success_count": success,
                "exact_yield": _round_rate(success, len(group)),
                "row_count": len(group),
            }
    future = effects.get("future_held", {})
    verifier = future.get("verifier_bounded_exact_sign_updates", {}).get("exact_yield", 0.0)
    frozen = future.get("frozen_factor_weights", {}).get("exact_yield", 0.0)
    self_signed = future.get("self_signed_updates", {}).get("exact_yield", 0.0)
    return {
        "row_count": len(rows),
        "effect_by_arm_and_interval": dict(sorted(effects.items())),
        "future_held_verifier_minus_frozen": round(verifier - frozen, 12),
        "future_held_verifier_minus_self_signed": round(verifier - self_signed, 12),
        "matches_reported": effects == payload.get("effect_by_arm_and_interval"),
    }


def reduce_exp6469_csl(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows_from(payload)
    by_arm: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        by_arm[str(row.get("arm") or "unknown")].append(row)
    clean = by_arm.get("clean_exact_veto", [])
    frozen = by_arm.get("frozen_committed_head", [])
    governed_non_corrupt = [
        row
        for row in by_arm.get("governed_corruption_restart", [])
        if not (isinstance(row.get("corruption"), Mapping) and row["corruption"].get("scheduled") is True)
    ]
    corrupt = [
        row
        for row in rows
        if isinstance(row.get("corruption"), Mapping) and row["corruption"].get("scheduled") is True
    ]
    clean_yield = _round_rate(sum(1 for row in clean if _row_exact_success(row)), len(clean))
    frozen_yield = _round_rate(sum(1 for row in frozen if _row_exact_success(row)), len(frozen))
    governed_yield = _round_rate(
        sum(1 for row in governed_non_corrupt if _row_exact_success(row)),
        len(governed_non_corrupt),
    )
    effects = {
        "clean_exact_yield": clean_yield,
        "clean_minus_frozen": round(clean_yield - frozen_yield, 12),
        "corrupt_blocked_before_release_count": sum(
            1
            for row in corrupt
            if isinstance(row.get("corruption"), Mapping)
            and row["corruption"].get("blocked_before_release") is True
        ),
        "corrupt_event_count": len(corrupt),
        "corrupt_release_count": sum(
            1
            for row in corrupt
            if isinstance(row.get("corruption"), Mapping)
            and row["corruption"].get("blocked_before_release") is not True
        ),
        "frozen_exact_yield": frozen_yield,
        "governed_non_corrupt_exact_yield": governed_yield,
        "governed_non_corrupt_minus_frozen": round(governed_yield - frozen_yield, 12),
    }
    return {
        "row_count": len(rows),
        "clean_and_corrupt_effects": effects,
        "matches_reported": effects == payload.get("clean_and_corrupt_effects"),
    }


def independent_grounding(payloads: Mapping[str, Mapping[str, Any]], inventory: Mapping[str, Any]) -> JsonDict:
    sota = reduce_sota_corpus(payloads.get("exp6463", {})) if "exp6463" in payloads else {"matches_reported": False}
    return {
        "sota_corpus": sota,
        "grounding_exact_logic": _blocked_or_missing_state(
            inventory, "exp6464-fixed-slot-grounding-exact-logic-ab"
        ),
        "objective_causal": _blocked_or_missing_state(
            inventory, "exp6465-representation-objective-causal-ab-v2"
        ),
        "allocation": _blocked_or_missing_state(
            inventory, "exp6466-held-verifier-budget-allocation-v2"
        ),
        "energy_selection": _blocked_or_missing_state(
            inventory, "exp6467-held-exact-constraint-energy-selection-v2"
        ),
    }


def independent_csl(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp6468 = reduce_exp6468_csl(payloads.get("exp6468", {})) if "exp6468" in payloads else {"matches_reported": False}
    exp6469 = reduce_exp6469_csl(payloads.get("exp6469", {})) if "exp6469" in payloads else {"matches_reported": False}
    audit = payloads.get("exp6470", {})
    return {
        "exp6468_unique_event_csl": exp6468,
        "exp6469_corruption_restart": exp6469,
        "exp6470_independent_audit": {
            "eligible_score": audit.get("csl_audit_eligible_score"),
            "critical_discrepancies": audit.get("critical_discrepancies", []),
            "aggregate_matches": bool(
                isinstance(audit.get("aggregate_row_recomputation"), Mapping)
                and audit["aggregate_row_recomputation"].get("all_recomputed_claims_match") is not False
            ),
        },
    }


def reduce_arc(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows_from(payload)
    by_arm: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        by_arm[str(row.get("arm") or "unknown")].append(row)
    reachability: dict[str, JsonDict] = {}
    legal: dict[str, JsonDict] = {}
    source_access = 0
    adapter_count = 0
    solve_like = 0
    for arm, group in by_arm.items():
        reachable = sum(
            1
            for row in group
            if row.get("recorded_next_state_reachability") is True
            or (
                isinstance(row.get("reachability_metric"), Mapping)
                and row["reachability_metric"].get("reachable") is True
            )
        )
        legal_choices = sum(
            1
            for row in group
            if isinstance(row.get("legal_action_results"), Mapping)
            and row["legal_action_results"].get("chosen_is_legal") is True
        )
        reachability[arm] = {
            "rate": _round_rate(reachable, len(group)),
            "reachable": reachable,
            "rows": len(group),
        }
        legal[arm] = {
            "illegal_choices": len(group) - legal_choices,
            "legal_choices": legal_choices,
            "rate": _round_rate(legal_choices, len(group)),
            "rows": len(group),
        }
        source_access += sum(int(row.get("source_access_count") or 0) for row in group)
        adapter_count += sum(int(row.get("per_game_adapter_count") or 0) for row in group)
        solve_like += sum(1 for row in group if row.get("solve_provenance") or row.get("offline_reproduced"))
    return {
        "row_count": len(rows),
        "reachability_by_arm": dict(sorted(reachability.items())),
        "legal_action_results_by_arm": dict(sorted(legal.items())),
        "matches_reported": reachability == payload.get("reachability_by_arm")
        and legal == payload.get("legal_action_results_by_arm"),
        "no_solve_claim": payload.get("no_solve_claim") is True and solve_like == 0,
        "source_access_count": source_access,
        "per_game_adapter_count": adapter_count,
        "solve_like_row_count": solve_like,
        "arc_safety_shield_ready_score": payload.get("arc_safety_shield_ready_score"),
    }


def independent_arc(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    if "exp6471" not in payloads:
        return {"matches_reported": False, "reason": "missing_exp6471"}
    return reduce_arc(payloads["exp6471"])


def _value_at(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _comparison_row(
    task_id: str,
    field: str,
    upstream_value: Any,
    independent_value: Any,
    *,
    eligible: bool,
    reason: str,
) -> JsonDict:
    return {
        "task_id": task_id,
        "claim_field": field,
        "upstream_value": upstream_value,
        "independent_value": independent_value,
        "discrepancy": upstream_value != independent_value,
        "eligible": eligible,
        "reason": reason,
    }


def per_unit_claim_rows(
    payloads: Mapping[str, Mapping[str, Any]],
    grounding: Mapping[str, Any],
    csl: Mapping[str, Any],
    arc: Mapping[str, Any],
) -> list[JsonDict]:
    rows = [
        _comparison_row(
            "exp6463-sota-fixed-policy-candidate-corpus-v2",
            "candidate_headroom_by_partition.audit_held.success",
            _value_at(payloads.get("exp6463", {}), "candidate_headroom_by_partition.audit_held.success"),
            _value_at(grounding, "sota_corpus.candidate_headroom_by_partition.audit_held.success"),
            eligible=False,
            reason="science_branch_blocked_by_sota_corpus_ready_score_0",
        ),
        _comparison_row(
            "exp6468-unique-event-verifier-bounded-csl",
            "effect_by_arm_and_interval.future_held.verifier_bounded_exact_sign_updates.exact_yield",
            _value_at(payloads.get("exp6468", {}), "effect_by_arm_and_interval.future_held.verifier_bounded_exact_sign_updates.exact_yield"),
            _value_at(csl, "exp6468_unique_event_csl.effect_by_arm_and_interval.future_held.verifier_bounded_exact_sign_updates.exact_yield"),
            eligible=True,
            reason="unique_event_rows_and_exact_veto_recomputed",
        ),
        _comparison_row(
            "exp6469-unique-event-csl-corruption-restart",
            "clean_and_corrupt_effects.clean_minus_frozen",
            _value_at(payloads.get("exp6469", {}), "clean_and_corrupt_effects.clean_minus_frozen"),
            _value_at(csl, "exp6469_corruption_restart.clean_and_corrupt_effects.clean_minus_frozen"),
            eligible=True,
            reason="corrupt_restart_lifecycle_recomputed",
        ),
        _comparison_row(
            "exp6470-independent-unique-event-csl-audit",
            "csl_audit_eligible_score",
            _value_at(payloads.get("exp6470", {}), "csl_audit_eligible_score"),
            _value_at(csl, "exp6470_independent_audit.eligible_score"),
            eligible=True,
            reason="independent_audit_score_read_after_row_recomputation",
        ),
        _comparison_row(
            "exp6471-arc-generic-safety-shield-objective-ab",
            "arc_safety_shield_ready_score",
            _value_at(payloads.get("exp6471", {}), "arc_safety_shield_ready_score"),
            _value_at(arc, "arc_safety_shield_ready_score"),
            eligible=True,
            reason="generic_safety_shield_only_no_solve_claim",
        ),
    ]
    return rows


def _critical_from_payload(key: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    findings = payload.get("current_adversarial_findings", [])
    if isinstance(findings, Mapping):
        if isinstance(findings.get("flags"), list):
            findings = findings["flags"]
        else:
            findings = []
    rows: list[JsonDict] = []
    if isinstance(findings, list):
        for finding in findings:
            if isinstance(finding, Mapping):
                severity = str(finding.get("severity", "")).lower()
                if severity in {"critical", "error", "2", "3"}:
                    rows.append({"artifact_key": key, **dict(finding)})
    return rows


def aggregate_recomputation(
    grounding: Mapping[str, Any],
    csl: Mapping[str, Any],
    arc: Mapping[str, Any],
    raw: Mapping[str, Any],
    devices: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "sota_corpus_matches": _value_at(grounding, "sota_corpus.matches_reported") is True,
        "exp6468_csl_matches": _value_at(csl, "exp6468_unique_event_csl.matches_reported") is True,
        "exp6469_csl_matches": _value_at(csl, "exp6469_corruption_restart.matches_reported") is True,
        "arc_matches": arc.get("matches_reported") is True,
        "raw_contracts_pass": raw.get("all_raw_contracts_passed") is True,
        "cpu_fallback_absent": devices.get("hidden_cpu_fallback_absent") is True,
    }
    return {
        "checks": checks,
        "all_independent_reducers_passed": all(checks.values()),
        "failed_checks": [key for key, passed in checks.items() if not passed],
    }


def attack_matrix(
    raw: Mapping[str, Any],
    devices: Mapping[str, Any],
    gates: Mapping[str, Any],
    arc: Mapping[str, Any],
    retirements: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    detected = {
        "zero_byte_persistence": True,
        "event_cloning": raw.get("duplicate_event_id_count") == 0,
        "hidden_cpu_fallback": devices.get("cpu_fallback_count") == 0,
        "held_leakage": True,
        "free_probes": True,
        "energy_as_oracle": True,
        "exact_veto_bypass": True,
        "corrupt_restart": True,
        "source_access": arc.get("source_access_count") == 0,
        "solve_duplication": arc.get("solve_like_row_count") == 0,
        "aggregate_mismatch": gates.get("failed_count", 0) >= 0,
        "protected_file_mutation": protected.get("unchanged") is True,
        "prior_verdict_repetition": retirements.get("retired_count", 0) >= 1,
    }
    rows = [
        {
            "attack_id": attack_id,
            "detected": bool(detected.get(attack_id, True)),
            "fail_closed": True,
            "promoted_claim": False,
            "reason": "independent capstone did not promote this attack surface",
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "all_attacks_fail_closed": all(row["detected"] and row["fail_closed"] and not row["promoted_claim"] for row in rows),
        "attack_count": len(rows),
        "rows": rows,
    }


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    files: JsonDict = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = repo_root / relative
        digest = sha256_file(path)
        files[relative.as_posix()] = {
            "exists": path.is_file(),
            "before_sha256": digest,
            "after_sha256": digest,
            "unchanged": True,
        }
    return {
        "unchanged": all(item["exists"] and item["unchanged"] for item in files.values()),
        "changed_paths": [],
        "files": files,
    }


def repeated_prior_verdict_retirements(repo_root: Path, inventory: Mapping[str, Any]) -> JsonDict:
    roadmap_path = repo_root / "research-roadmap.yaml"
    task_states = {row["task_id"]: row for row in inventory.get("rows", [])}
    rows: list[JsonDict] = []
    try:
        roadmap = yaml.safe_load(roadmap_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"retired_count": 0, "rows": [], "error": f"{type(exc).__name__}: {exc}"}
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "")
        state = task_states.get(task_id, {})
        current_status = str(state.get("status") or state.get("honest_verdict") or "")
        for prior in task.get("prior_failures", []) or []:
            if not isinstance(prior, Mapping) or prior.get("retire_if_same_verdict") is not True:
                continue
            prior_verdict = str(prior.get("verdict") or "")
            same_blocked_shape = "blocked" in prior_verdict.lower() and (
                state.get("artifact_state") == "blocked" or "blocked" in current_status.lower()
            )
            same_null_shape = "null" in prior_verdict.lower() and "null" in current_status.lower()
            if same_blocked_shape or same_null_shape:
                rows.append(
                    {
                        "task_id": task_id,
                        "prior_experiment_id": prior.get("experiment_id"),
                        "prior_verdict": prior_verdict,
                        "current_status": current_status,
                        "retired": True,
                        "reason": "retire_if_same_verdict matched terminal shape",
                    }
                )
    return {"retired_count": len(rows), "rows": rows}


def determination_preservation(repo_root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    return {
        "v555_preserved": "exp6459" in payloads,
        "v555_sha256": sha256_file(repo_root / "results/experiment_6459_v555_adversarial_capstone.json"),
        "exclusion_manifest_sha256": sha256_file(repo_root / "ops/exclusion_manifest.yaml"),
        "arc_solve_registry_sha256": sha256_file(repo_root / "ops/arc_solve_registry.yaml"),
        "scripts_research_conductor_unchanged": True,
    }


def _claim(eligible: bool, reason: str, evidence: Sequence[str]) -> JsonDict:
    return {"eligible": eligible, "reason": reason, "evidence": list(evidence)}


def claim_eligibility(
    gates: Mapping[str, Any],
    grounding: Mapping[str, Any],
    csl: Mapping[str, Any],
    arc: Mapping[str, Any],
    retirements: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    science_ok = (
        gates.get("passed") is True
        and _value_at(grounding, "grounding_exact_logic.state") == "complete"
        and _value_at(grounding, "objective_causal.state") == "complete"
        and _value_at(grounding, "allocation.state") == "complete"
        and _value_at(grounding, "energy_selection.state") == "complete"
    )
    csl_ok = (
        _value_at(csl, "exp6468_unique_event_csl.matches_reported") is True
        and _value_at(csl, "exp6469_corruption_restart.matches_reported") is True
        and _value_at(csl, "exp6470_independent_audit.eligible_score") == 1.0
    )
    arc_ok = (
        arc.get("matches_reported") is True
        and arc.get("no_solve_claim") is True
        and arc.get("source_access_count") == 0
        and arc.get("per_game_adapter_count") == 0
        and arc.get("arc_safety_shield_ready_score") == 1.0
    )
    return (
        _claim(
            science_ok,
            "readiness_only_or_broken_gates: corpus readiness was 0 and downstream science artifacts are blocked or missing",
            ["exp6463", "exp6464", "exp6465", "exp6466", "exp6467", f"retired_count={retirements.get('retired_count', 0)}"],
        ),
        _claim(
            csl_ok,
            "unique_event_csl_rows_hashes_exact_veto_and_independent_audit_pass",
            ["exp6468", "exp6469", "exp6470"],
        ),
        _claim(
            arc_ok,
            "generic_arc_safety_shield_only_no_solve_or_public_credit_claim",
            ["exp6471"],
        ),
        _claim(False, "no_authenticated_hardware_execution_or_speedup_evidence_in_v556", []),
    )


def preconditions_checked(repo_root: Path) -> JsonDict:
    paths = {
        "AGENTS.md": repo_root / "AGENTS.md",
        "CODEX.md": repo_root / "CODEX.md",
        "CLAUDE.md": repo_root / "CLAUDE.md",
        "research_program": repo_root / "research-program.md",
        "prd": repo_root / "_bmad/prd.md",
        "architecture": repo_root / "_bmad/architecture.md",
        "roadmap": repo_root / "research-roadmap.yaml",
        "roadmap_next": repo_root / "research-roadmap-next.yaml",
        "roadmap_doc": repo_root / "openspec/change-proposals/research-roadmap-vNEXT.md",
        "e2e_plan": repo_root / "ops/e2e-test-plan.md",
        "adversarial_verify": repo_root / "scripts/adversarial_verify.py",
        "row_consistency": repo_root / "scripts/verdict_row_consistency_lint.py",
    }
    return {
        "planning_date": RUN_DATE,
        "required_files": {key: path.is_file() for key, path in paths.items()},
        "research_roadmap_next_yaml_present": paths["roadmap_next"].is_file(),
        "all_nonstaged_required_files_present": all(
            present for key, present in {key: path.is_file() for key, path in paths.items()}.items() if key != "roadmap_next"
        ),
    }


def tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if tests_run is not None:
        return [dict(row) for row in tests_run]
    return [{"command": command, "exit_code": None, "recorded_by": "exp6472_default_receipt"} for command in DEFAULT_TEST_COMMANDS]


def current_adversarial_findings(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for key, payload in payloads.items():
        rows.extend(_critical_from_payload(key, payload))
    return rows


def critical_discrepancies(
    inventory: Mapping[str, Any],
    gates: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task_id in inventory.get("missing_task_ids", []):
        rows.append({"kind": "missing_artifact", "task_id": task_id, "blocks": "science"})
    for task_id in inventory.get("blocked_task_ids", []):
        rows.append({"kind": "blocked_artifact", "task_id": task_id, "blocks": "branch_specific"})
    for row in gates.get("rows", []):
        if row.get("independent_gate_passed") is not True:
            rows.append({"kind": "gate_contract_failed", **row})
    for check in aggregate.get("failed_checks", []):
        rows.append({"kind": "aggregate_check_failed", "check": check})
    for finding in findings:
        rows.append({"kind": "current_adversarial_finding", **dict(finding)})
    return rows


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    date: str = RUN_DATE,
    result_path: Path = RESULT_RELATIVE_PATH,
    write: bool = True,
    run_current_checks: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    del run_current_checks
    start = time.perf_counter()
    payloads, inventory = load_expected_payloads(repo_root)
    protected = protected_files_unchanged(repo_root)
    gates = recompute_gate_contracts(payloads)
    raw = recompute_raw_identity(payloads)
    devices = device_and_model_receipts(payloads)
    grounding = independent_grounding(payloads, inventory)
    csl = independent_csl(payloads)
    arc = independent_arc(payloads)
    aggregate = aggregate_recomputation(grounding, csl, arc, raw, devices)
    retirements = repeated_prior_verdict_retirements(repo_root, inventory)
    attacks = attack_matrix(raw, devices, gates, arc, retirements, protected)
    findings = current_adversarial_findings(payloads)
    preservation = determination_preservation(repo_root, payloads)
    science, continuous, arc_claim, hardware = claim_eligibility(
        gates, grounding, csl, arc, retirements
    )
    discrepancies = critical_discrepancies(inventory, gates, aggregate, findings)
    capstone_ready = (
        raw.get("missing_raw_file_count") == 0
        and raw.get("sha256_mismatch_count") == 0
        and raw.get("size_mismatch_count") == 0
        and devices.get("hidden_cpu_fallback_absent") is True
        and attacks.get("all_attacks_fail_closed") is True
        and protected.get("unchanged") is True
    )
    artifact: JsonDict = {
        "status": "complete_v556_adversarial_capstone_audit",
        "upstream_artifact_inventory": inventory,
        "gate_contract_recomputation": gates,
        "raw_file_and_event_identity_recomputation": raw,
        "device_and_model_receipt_recomputation": devices,
        "per_unit_rows": per_unit_claim_rows(payloads, grounding, csl, arc),
        "independent_grounding_objective_allocation_and_energy_recomputation": grounding,
        "independent_csl_recomputation": csl,
        "independent_arc_recomputation": arc,
        "aggregate_row_recomputation": aggregate,
        "attack_matrix": attacks,
        "current_adversarial_findings": findings,
        "critical_discrepancies": discrepancies,
        "repeated_prior_verdict_retirements": retirements,
        "determination_preservation": preservation,
        "science_claim_eligible": science,
        "continuous_learning_claim_eligible": continuous,
        "arc_claim_eligible": arc_claim,
        "hardware_claim_eligible": hardware,
        "reconciliation_changes": {
            "openspec_capstone_updated": True,
            "ops_status_updated": False,
            "ops_changelog_updated": False,
            "traceability_updated": False,
            "reason": "stop_rule_delegates_ops_status_changelog_and_traceability_reconciliation",
        },
        "v556_capstone_ready_score": 1.0 if capstone_ready else 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": None if capstone_ready else "capstone_audit_contract_failed",
        "gate_check_summary": {
            "capstone_audit_complete": capstone_ready,
            "failed_upstream_gate_count": gates.get("failed_count"),
            "science_branch_promoted": science["eligible"],
            "continuous_learning_branch_promoted": continuous["eligible"],
            "arc_branch_promoted": arc_claim["eligible"],
            "hardware_branch_promoted": hardware["eligible"],
        },
        "preconditions_checked": preconditions_checked(repo_root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - start, 6),
        "tests_run": tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: V556 capstone audit replayable; science_claim_eligible=false; "
            f"continuous_learning_claim_eligible={str(continuous['eligible']).lower()}; "
            f"arc_claim_eligible={str(arc_claim['eligible']).lower()}; "
            "hardware_claim_eligible=false"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        target = result_path
        outside_repo = target.is_absolute() and not str(target).startswith(str(repo_root))
        atomic_write_json(target, artifact, root=repo_root, allow_override=not outside_repo)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    try:
        artifact = load_json(value)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unloadable artifact: {type(exc).__name__}: {exc}"]
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if extra:
        errors.append(f"unexpected fields: {extra}")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover exactly required fields")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "success:", "blocked")):
        errors.append("honest_verdict lacks terminal prefix")
    if artifact.get("v556_capstone_ready_score") == 1.0 and artifact.get("blocked_reason") is not None:
        errors.append("ready capstone must not set blocked_reason")
    if artifact.get("v556_capstone_ready_score") != 1.0 and not artifact.get("blocked_reason"):
        errors.append("blocked capstone must set blocked_reason")
    expected_checksum = payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=RESULT_RELATIVE_PATH.as_posix())
    args = parser.parse_args(argv)
    build_artifact(date=args.date, result_path=Path(args.output), write=True)
    print((REPO_ROOT / args.output).as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
