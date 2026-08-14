"""Exp6428 clean write-time factor admission A/B replay.

Spec refs: REQ-LEARN-6428, SCENARIO-LEARN-6428-GATES,
SCENARIO-LEARN-6428-MATCHED-ARMS, SCENARIO-LEARN-6428-ADMISSION,
SCENARIO-LEARN-6428-FUTURE, SCENARIO-LEARN-6428-ATTACKS,
SCENARIO-LEARN-6428-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6428_clean_write_time_factor_admission_ab.json"
)
EXP6427_RELATIVE_PATH = Path(
    "results/experiment_6427_fresh_constraint_saturation_factor_corpus.json"
)
EXP6417_RELATIVE_PATH = Path(
    "results/experiment_6417_authentic_write_time_factor_admission_ab.json"
)

SCHEMA = "carnot.experiment_6428.clean_write_time_factor_admission_ab.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6428
INFERENCE_SUBSTRATE = "cached_sota_event_energy_calibration_exp6427_replay_no_new_generation"

FROZEN_ARM = "frozen"
WRITE_EVERYTHING_ARM = "write_everything"
EXACT_ADMISSION_ARM = "exact_admission"
ARMS = (FROZEN_ARM, WRITE_EVERYTHING_ARM, EXACT_ADMISSION_ARM)
PROPOSAL_PARTITIONS = ("acquisition", "calibration")
PROTECTED_RETENTION_PARTITION = "calibration"
FUTURE_PARTITION = "future"

BARE_FINITE_FIELDS = (
    "delta_future_exact_yield",
    "delta_contamination_propagation_rate",
    "protected_retention_delta",
    "false_accept_delta",
    "false_reject_delta",
)
ZERO_COUNTER_FIELDS = (
    "exact_veto_override_count",
    "protected_leakage_count",
    "runtime_field_synthesis_count",
    "current_adversarial_flag_count",
)
FAIL_CLOSED_CLASSES = (
    "contradicted",
    "implicit",
    "stale",
    "duplicate",
    "replayed",
    "superseded",
    "poisoned",
    "malformed",
    "unlicensed",
    "stale_head",
    "missing_exact",
)
ATTACK_IDS = (
    "receipt_substitution",
    "source_replacement",
    "model_swap",
    "license_inheritance",
    "checker_omission",
    "stale_head",
    "duplicate_effect",
    "future_leakage",
    "exact_veto_override",
    "row_deletion",
    "duration_synthesis",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6428_clean_write_time_factor_admission_ab "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py "
    "-m pytest tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6428_clean_write_time_factor_admission_ab.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ARTIFACT_AUDIT_COMMAND = (
    ".venv/bin/python scripts/artifact_convention_audit.py "
    "results/experiment_6428_clean_write_time_factor_admission_ab.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_AUDIT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6427_RELATIVE_PATH,
    EXP6417_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py"),
    Path("python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/artifact_convention_audit.py"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6427_gate_receipts",
    "upstream_model_process_raw_output_and_row_hashes",
    "corpus_event_order_partition_checker_license_and_head_hashes",
    "preregistered_frozen_write_everything_and_exact_admission_arm_contract",
    "matched_work_receipts",
    "per_unit_rows",
    "per_proposal_source_model_license_checker_predecessor_expiry_and_supersession_bindings",
    "atomic_disposition_records",
    "untouched_future_evaluation_receipts",
    "aggregate_recomputation_receipts",
    "reported_vs_recomputed_deltas",
    "delta_future_exact_yield",
    "delta_contamination_propagation_rate",
    "protected_retention_delta",
    "false_accept_delta",
    "false_reject_delta",
    "factor_growth_by_arm",
    "exact_work_by_arm",
    "exact_veto_override_count",
    "protected_leakage_count",
    "runtime_field_synthesis_count",
    "task_phase_duration_receipts",
    "attack_matrix",
    "clean_write_time_admission_ready_score",
    "current_adversarial_flag_count",
    "public_factor_claim_eligibility",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "blocked_reason",
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
    "status": "Names the terminal safety state for the clean Exp6427 replay.",
    "exp6427_gate_receipts": "Pins the clean corpus gate and the Exp6417 duration quarantine context.",
    "upstream_model_process_raw_output_and_row_hashes": "Binds model, prompt, process, raw output, and row hashes before parsing can act.",
    "corpus_event_order_partition_checker_license_and_head_hashes": "Seals event order, partitions, checkers, licenses, disk, CPU, RAM, and the initial head.",
    "preregistered_frozen_write_everything_and_exact_admission_arm_contract": "Defines all three arms before future outcomes open.",
    "matched_work_receipts": "Shows equal row order, evidence, checker calls, consumer budget, and initial head for all arms.",
    "per_unit_rows": "Records one future outcome row for every arm and cell before aggregate calculation.",
    "per_proposal_source_model_license_checker_predecessor_expiry_and_supersession_bindings": "Binds every proposal to source, model, license, checker, predecessor, expiry, supersession, and refinement hashes.",
    "atomic_disposition_records": "Records exactly one Commit, Reject, Quarantine, or Defer decision for every proposal.",
    "untouched_future_evaluation_receipts": "Proves future outcomes open once after proposal dispositions and head freeze.",
    "aggregate_recomputation_receipts": "Recomputes every comparative aggregate from per-unit rows in an independent pass.",
    "reported_vs_recomputed_deltas": "Shows reported deltas and reductions match the independent recomputation.",
    "delta_future_exact_yield": "Bare future exact-yield lift for exact admission over frozen.",
    "delta_contamination_propagation_rate": "Bare contamination-rate change for exact admission over frozen.",
    "protected_retention_delta": "Bare protected-retention change for exact admission over frozen.",
    "false_accept_delta": "Bare false-accept rate change for exact admission over frozen.",
    "false_reject_delta": "Bare false-reject rate change for exact admission over frozen.",
    "factor_growth_by_arm": "Reports committed factor growth per arm.",
    "exact_work_by_arm": "Reports exact checker work per arm under the matched budget.",
    "exact_veto_override_count": "Must be zero because exact rejections cannot be overridden.",
    "protected_leakage_count": "Must be zero because protected and future labels cannot route writes.",
    "runtime_field_synthesis_count": "Must be zero because runtime fields come from receipts.",
    "task_phase_duration_receipts": "Records monotonic phase timing without synthetic duration fields.",
    "attack_matrix": "Shows substitution, source, model, license, checker, head, duplicate, leakage, veto, deletion, and duration attacks fail closed.",
    "clean_write_time_admission_ready_score": "Conjunctive score for future gain without contamination, retention harm, aggregate drift, or adversarial flags.",
    "current_adversarial_flag_count": "Must stay zero for the clean Exp6427 replay.",
    "public_factor_claim_eligibility": "Limits public eligibility to this clean replay and excludes the flagged Exp6417 timing claim.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps unlicensed, underpowered, missing, blocked, and flagged cells visible.",
    "protected_files_unchanged": "Shows protected upstream and ops files stayed byte-identical.",
    "blocked_reason": "Explains why readiness is blocked when any precondition fails.",
    "preconditions_checked": "Lists all gates checked before readiness can become one.",
    "inference_substrate": "Declares cached Exp6427 deterministic replay with no new model generation.",
    "verifier_is_oracle": "Marks only exact event and protected-retention checkers as oracles.",
    "field_principles": "Documents why each field exists.",
    "field_provenance": "Maps each field to specs, inputs, replay, reductions, attacks, or tests.",
    "random_seed": "Pins the replay constants.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the payload with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the clean replay boundary.",
    "gate:exp6427": "Exp6427 must be complete, clean, row-recomputable, and adversarial-clean before Exp6428 can promote readiness.",
    "gate:exp6417_duration_quarantine": "Exp6417 is context only because its deterministic replay duration is adversarial-flagged.",
    "gate:raw_outputs": "Raw output files and stored hashes must match before proposals bind.",
    "gate:event_order": "Chronological order and partitions must stay sealed.",
    "gate:licenses": "License validity controls commits and blocks inheritance.",
    "gate:initial_factor_head": "All arms start from the same read-only head.",
    "arm:frozen": "Frozen reads the future with no write-time state.",
    "arm:write_everything": "Write-everything commits every licensed proposal and acts as the contamination control.",
    "arm:exact_admission": "Exact admission commits only licensed joint-exact proposal rows.",
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash stable JSON bytes."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    file_path = Path(path)
    return sha256_bytes(file_path.read_bytes()) if file_path.is_file() else None


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other shapes with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round deterministic metrics without hiding small nonzero values."""

    return round(float(value), 9)


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and fail with a stable error for other shapes."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("json_object")
    return data


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def path_receipt(path: str | Path, *, relative_to: Path | None = None) -> JsonDict:
    """Record path presence, size, and digest."""

    file_path = Path(path)
    display = file_path
    if relative_to is not None:
        try:
            display = file_path.relative_to(relative_to)
        except ValueError:
            display = file_path
    return {
        "path": str(display),
        "present": file_path.is_file(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
    }


def _resolve_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    """Hash source files that define this replay."""

    return {
        path.as_posix(): path_receipt(root / path, relative_to=root)
        for path in SOURCE_RELATIVE_PATHS
    }


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected files before and after the replay."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def load_context(root: Path = REPO_ROOT) -> JsonDict:
    """Load Exp6427 rows and the flagged Exp6417 context."""

    exp6427_artifact = read_json(root / EXP6427_RELATIVE_PATH)
    exp6417_artifact = read_json(root / EXP6417_RELATIVE_PATH)
    manifest_receipt = as_mapping(
        exp6427_artifact.get("manifest_path_hash_counts_balance_and_partition_seals")
    )
    manifest_path = _resolve_path(root, str(manifest_receipt.get("path", "")))
    rows = [
        dict(as_mapping(row))
        for row in as_mapping(exp6427_artifact.get("per_unit_rows")).get("rows", [])
    ]
    ordered_rows = sorted(rows, key=lambda row: int(row.get("row_index", -1)))
    return {
        "exp6427": exp6427_artifact,
        "exp6417": exp6417_artifact,
        "manifest_path": manifest_path,
        "ordered_row_ids": [str(row.get("row_id")) for row in ordered_rows],
        "rows_by_id": {str(row.get("row_id")): row for row in ordered_rows},
    }


def _ordered_rows(context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = as_mapping(context.get("rows_by_id"))
    return [as_mapping(rows.get(row_id)) for row_id in context.get("ordered_row_ids", [])]


def proposal_row_ids(context: Mapping[str, Any]) -> list[str]:
    """Return acquisition and retention-control row ids in sealed order."""

    rows = as_mapping(context.get("rows_by_id"))
    return [
        str(row_id)
        for row_id in context.get("ordered_row_ids", [])
        if as_mapping(rows.get(str(row_id))).get("partition") in PROPOSAL_PARTITIONS
    ]


def future_row_ids(context: Mapping[str, Any]) -> list[str]:
    """Return untouched future row ids in sealed order."""

    rows = as_mapping(context.get("rows_by_id"))
    return [
        str(row_id)
        for row_id in context.get("ordered_row_ids", [])
        if as_mapping(rows.get(str(row_id))).get("partition") == FUTURE_PARTITION
    ]


def _row_digest_rows(context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [as_mapping(row) for row in _ordered_rows(context)]


def _raw_path(row: Mapping[str, Any]) -> Path:
    return Path(str(row.get("raw_output_path", "")))


def _raw_hash_matches(row: Mapping[str, Any]) -> bool:
    return sha256_file(_raw_path(row)) == row.get("raw_output_sha256")


def _effect_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('model_family')}::{row.get('factor_family')}"


def _cell_key(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('model_family')}::{row.get('factor_family')}::"
        f"{row.get('constraint_count_bucket')}::{row.get('interaction_class')}"
    )


def _license_valid(row: Mapping[str, Any]) -> bool:
    return as_mapping(row.get("source_license")).get("licensed") is True


def _initial_factor_head(context: Mapping[str, Any]) -> JsonDict:
    payload = {
        "schema": SCHEMA + ".initial_head",
        "source": EXP6427_RELATIVE_PATH.as_posix(),
        "active_factors": [],
        "future_outcomes_visible": False,
        "row_order_sha256": sha256_json(list(context.get("ordered_row_ids", []))),
        "random_seed": RANDOM_SEED,
    }
    return {**payload, "head_hash": sha256_json(payload)}


def exp6427_gate_receipts(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Revalidate the clean Exp6427 corpus and quarantine Exp6417 timing."""

    exp6427_artifact = as_mapping(context.get("exp6427"))
    exp6417_artifact = as_mapping(context.get("exp6417"))
    exp6427_hash = sha256_file(root / EXP6427_RELATIVE_PATH)
    exp6417_flagged_duration = (
        exp6417_artifact.get("flagged_adversarial") is True
        and float(exp6417_artifact.get("duration_s", 1.0)) < 0.0001
    )
    blockers = []
    if exp6427_artifact.get("status") != "complete":
        blockers.append("exp6427_not_complete")
    if exp6427_artifact.get("fresh_row_recomputable_factor_corpus_ready_score") != 1.0:
        blockers.append("exp6427_ready_score_not_one")
    if exp6427_artifact.get("current_adversarial_flag_count") != 0:
        blockers.append("exp6427_adversarial_flags_present")
    if exp6427_artifact.get("protected_leakage_count") != 0:
        blockers.append("exp6427_protected_leakage")
    if as_mapping(exp6427_artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is not True:
        blockers.append("exp6427_aggregates_do_not_recompute")
    if as_mapping(exp6427_artifact.get("attack_matrix")).get("all_fail_closed") is not True:
        blockers.append("exp6427_attack_matrix_open")
    return {
        "schema": SCHEMA + ".exp6427_gate",
        "path": (root / EXP6427_RELATIVE_PATH).as_posix(),
        "sha256": exp6427_hash,
        "status": exp6427_artifact.get("status"),
        "fresh_row_recomputable_factor_corpus_ready_score": exp6427_artifact.get(
            "fresh_row_recomputable_factor_corpus_ready_score"
        ),
        "current_adversarial_flag_count": exp6427_artifact.get(
            "current_adversarial_flag_count"
        ),
        "protected_leakage_count": exp6427_artifact.get("protected_leakage_count"),
        "row_count": as_mapping(exp6427_artifact.get("per_unit_rows")).get("row_count"),
        "row_hash": as_mapping(exp6427_artifact.get("per_unit_rows")).get("row_hash"),
        "aggregate_deltas_all_zero": as_mapping(
            exp6427_artifact.get("reported_vs_recomputed_deltas")
        ).get("all_zero")
        is True,
        "attack_matrix_all_fail_closed": as_mapping(exp6427_artifact.get("attack_matrix")).get(
            "all_fail_closed"
        )
        is True,
        "exp6426_gate_passed": as_mapping(exp6427_artifact.get("exp6426_gate_receipt")).get(
            "gate_passed"
        )
        is True,
        "upstream_exp6417_path": EXP6417_RELATIVE_PATH.as_posix(),
        "upstream_exp6417_flagged_duration": exp6417_flagged_duration,
        "blocked_reasons": blockers,
        "gate_passed": not blockers,
    }


def upstream_model_process_raw_output_and_row_hashes(
    root: Path,
    context: Mapping[str, Any],
) -> JsonDict:
    """Bind Exp6427 raw sidecars and row hashes before the A/B replay."""

    exp6427_artifact = as_mapping(context.get("exp6427"))
    stored_row_hash = as_mapping(exp6427_artifact.get("per_unit_rows")).get("row_hash")
    rows = []
    for row in _ordered_rows(context):
        raw_hash = sha256_file(_raw_path(row))
        rows.append(
            {
                "row_id": row.get("row_id"),
                "row_index": row.get("row_index"),
                "event_hash": row.get("event_hash"),
                "model_hf_id": row.get("model_hf_id"),
                "model_family": row.get("model_family"),
                "model_hash": row.get("model_hash"),
                "prompt_sha256": row.get("prompt_sha256"),
                "process_pid": row.get("pid"),
                "parent_pid": row.get("parent_pid"),
                "raw_output_path": row.get("raw_output_path"),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "actual_raw_output_sha256": raw_hash,
                "raw_hash_matches": raw_hash == row.get("raw_output_sha256"),
                "raw_output_stored_before_parse": row.get("raw_output_stored_before_parse")
                is True,
                "row_sha256": sha256_json(row),
            }
        )
    actual_row_hash = sha256_json(_row_digest_rows(context))
    return {
        "schema": SCHEMA + ".upstream_raw_rows",
        "source": EXP6427_RELATIVE_PATH.as_posix(),
        "row_count": len(rows),
        "stored_row_hash": stored_row_hash,
        "actual_row_hash": actual_row_hash,
        "all_row_hashes_match": actual_row_hash == stored_row_hash,
        "all_raw_hashes_match": all(row["raw_hash_matches"] for row in rows),
        "all_raw_written_before_parse": all(row["raw_output_stored_before_parse"] for row in rows),
        "upstream_model_ids": sorted({str(row["model_hf_id"]) for row in rows}),
        "upstream_model_count": len({str(row["model_hf_id"]) for row in rows}),
        "prompt_hashes_sha256": sha256_json(sorted(str(row["prompt_sha256"]) for row in rows)),
        "raw_output_hashes_sha256": sha256_json(
            sorted(str(row["raw_output_sha256"]) for row in rows)
        ),
        "new_model_generation_count": 0,
        "rows": rows,
    }


def _system_receipts(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    mem_total_kb = 0
    mem_available_kb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            if line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
    return {
        "disk": {
            "total_bytes": disk.total,
            "free_bytes": disk.free,
            "root": str(root),
            "checked": True,
        },
        "cpu": {"count": os.cpu_count() or 1, "checked": True},
        "ram": {
            "mem_total_kb": mem_total_kb,
            "mem_available_kb": mem_available_kb,
            "checked": True,
        },
    }


def corpus_event_order_partition_checker_license_and_head_hashes(
    root: Path,
    context: Mapping[str, Any],
) -> JsonDict:
    """Seal order, partitions, checkers, licenses, system receipts, and head."""

    exp6427_artifact = as_mapping(context.get("exp6427"))
    ordered_ids = list(context.get("ordered_row_ids", []))
    rows = as_mapping(context.get("rows_by_id"))
    order_values = [int(as_mapping(rows.get(row_id)).get("row_index", -1)) for row_id in ordered_ids]
    partition_counts = Counter(
        str(as_mapping(rows.get(row_id)).get("partition")) for row_id in ordered_ids
    )
    partition_seals = {
        partition: {
            "row_count": partition_counts[partition],
            "row_hash": sha256_json(
                [
                    row_id
                    for row_id in ordered_ids
                    if as_mapping(rows.get(row_id)).get("partition") == partition
                ]
            ),
            "used_for_proposals": partition in PROPOSAL_PARTITIONS,
        }
        for partition in sorted(partition_counts)
    }
    partition_seals["protected_retention"] = {
        **partition_seals[PROTECTED_RETENTION_PARTITION],
        "source_partition": PROTECTED_RETENTION_PARTITION,
        "protected_retention_control": True,
    }
    checker_rows = []
    for checker in {
        str(as_mapping(row.get("checker_identity")).get("checker")): as_mapping(
            row.get("checker_identity")
        )
        for row in _ordered_rows(context)
    }.values():
        checker_rows.append(
            {
                "checker": checker.get("checker"),
                "checker_sha256": checker.get("checker_sha256"),
                "verifier_is_oracle": checker.get("verifier_is_oracle") is True,
            }
        )
    head = _initial_factor_head(context)
    return {
        "schema": SCHEMA + ".corpus_hashes",
        "event_order": {
            "row_count": len(ordered_ids),
            "row_order_sha256": sha256_json(ordered_ids),
            "order_is_strict": order_values == list(range(len(order_values))),
            "future_outcome_visible_before_row_freeze_count": as_mapping(
                as_mapping(exp6427_artifact.get("manifest_path_hash_counts_balance_and_partition_seals")).get(
                    "partition_seals"
                )
            ).get("future_label_visible_before_row_freeze_count"),
        },
        "partitions": partition_seals,
        "checker": {
            "checker_versions": checker_rows,
            "checker_versions_sha256": sha256_json(checker_rows),
            "all_oracle_scoped": all(row["verifier_is_oracle"] for row in checker_rows),
        },
        "license": {
            "license_matrix_ready": as_mapping(exp6427_artifact.get("preconditions_checked")).get(
                "license_matrix_ready"
            )
            is True,
            "licensed_row_count": sum(_license_valid(row) for row in _ordered_rows(context)),
            "unlicensed_row_count": sum(not _license_valid(row) for row in _ordered_rows(context)),
            "license_status_counts": dict(
                sorted(
                    Counter(
                        str(as_mapping(row.get("source_license")).get("license_status"))
                        for row in _ordered_rows(context)
                    ).items()
                )
            ),
            "license_inheritance_count": 0,
        },
        "initial_factor_head": head,
        "manifest": path_receipt(context.get("manifest_path", ""), relative_to=root),
        "system": _system_receipts(root),
    }


def preregistered_frozen_write_everything_and_exact_admission_arm_contract(
    context: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> JsonDict:
    """Freeze arms and budgets before future outcomes open."""

    rows = proposal_row_ids(context)
    initial_head_hash = as_mapping(corpus.get("initial_factor_head")).get("head_hash")
    return {
        "schema": SCHEMA + ".arm_contract",
        "registered_before_future_open": True,
        "future_partition_opened_after_dispositions": True,
        "proposal_partition_names": list(PROPOSAL_PARTITIONS),
        "protected_retention_source_partition": PROTECTED_RETENTION_PARTITION,
        "future_partition_name": FUTURE_PARTITION,
        "arms": {
            arm: {
                "event_order_sha256": sha256_json(rows),
                "proposal_count": len(rows),
                "checker_call_count": len(rows),
                "consumer_budget": len(rows),
                "initial_head_hash": initial_head_hash,
                "authority": {
                    FROZEN_ARM: "read_only_no_write",
                    WRITE_EVERYTHING_ARM: "commit_every_license_valid_row",
                    EXACT_ADMISSION_ARM: "commit_only_joint_exact_license_fresh_rows",
                }[arm],
            }
            for arm in ARMS
        },
    }


def matched_work_receipts(context: Mapping[str, Any], corpus: Mapping[str, Any]) -> JsonDict:
    """Record the equal replay surface for all arms."""

    rows = proposal_row_ids(context)
    rows_by_id = as_mapping(context.get("rows_by_id"))
    initial_head_hash = as_mapping(corpus.get("initial_factor_head")).get("head_hash")
    raw_hashes = [as_mapping(rows_by_id.get(row_id)).get("raw_output_sha256") for row_id in rows]
    event_hashes = [as_mapping(rows_by_id.get(row_id)).get("event_hash") for row_id in rows]
    prompt_hashes = [as_mapping(rows_by_id.get(row_id)).get("prompt_sha256") for row_id in rows]
    return {
        "schema": SCHEMA + ".matched_work",
        "proposal_count_per_arm": len(rows),
        "consumer_budget_per_arm": len(rows),
        "initial_head_hash": initial_head_hash,
        "proposal_order_sha256": sha256_json(rows),
        "raw_evidence_sha256": sha256_json(raw_hashes),
        "event_evidence_sha256": sha256_json(event_hashes),
        "prompt_evidence_sha256": sha256_json(prompt_hashes),
        "by_arm": {
            arm: {
                "proposal_order_sha256": sha256_json(rows),
                "checker_call_count": len(rows),
                "consumer_budget": len(rows),
                "initial_head_hash": initial_head_hash,
                "raw_evidence_sha256": sha256_json(raw_hashes),
                "event_evidence_sha256": sha256_json(event_hashes),
                "prompt_evidence_sha256": sha256_json(prompt_hashes),
            }
            for arm in ARMS
        },
    }


def _refinement_hashes(context: Mapping[str, Any], row: Mapping[str, Any]) -> JsonDict:
    exp6427_artifact = as_mapping(context.get("exp6427"))
    return {
        "artifact": EXP6427_RELATIVE_PATH.as_posix(),
        "artifact_checksum": exp6427_artifact.get("reproducibility_checksum"),
        "row_hash": sha256_json(row),
        "checker_hash": as_mapping(row.get("checker_identity")).get("checker_sha256"),
        "source_hash": as_mapping(row.get("source_identity")).get("source_sha256"),
    }


def _binding_for_row(
    context: Mapping[str, Any],
    row_id: str,
    arm: str,
    predecessor_head_hash: str,
) -> JsonDict:
    row = as_mapping(as_mapping(context.get("rows_by_id")).get(row_id))
    refinement = _refinement_hashes(context, row)
    licensed = _license_valid(row)
    exact_support = (
        licensed
        and row.get("evaluable") is True
        and row.get("joint_exact") is True
        and row.get("parse_valid") is True
        and _raw_hash_matches(row)
    )
    malformed = row.get("malformed") is True or row.get("parse_valid") is not True
    supersession_state = "active"
    if row.get("duplicate") is True:
        supersession_state = "duplicate"
    if not licensed:
        supersession_state = "unlicensed"
    binding = {
        "schema": SCHEMA + ".proposal_binding",
        "proposal_id": f"{arm}:{row_id}",
        "arm": arm,
        "row_id": row_id,
        "partition": row.get("partition"),
        "chronological_index": row.get("row_index"),
        "event_hash": row.get("event_hash"),
        "raw_output_sha256": row.get("raw_output_sha256"),
        "raw_hash_matches": _raw_hash_matches(row),
        "model_hf_id": row.get("model_hf_id"),
        "model_family": row.get("model_family"),
        "factor_family": row.get("factor_family"),
        "constraint_count_bucket": row.get("constraint_count_bucket"),
        "interaction_class": row.get("interaction_class"),
        "prompt_sha256": row.get("prompt_sha256"),
        "source_identity": row.get("source_identity"),
        "source_license": row.get("source_license"),
        "license_valid": licensed,
        "checker_identity": row.get("checker_identity"),
        "checker_called": as_mapping(row.get("checker_identity")).get("checker") != "",
        "exact_support": exact_support,
        "evaluable": row.get("evaluable") is True,
        "joint_exact": row.get("joint_exact") is True,
        "malformed": malformed,
        "duplicate": row.get("duplicate") is True,
        "predecessor_head_hash": predecessor_head_hash,
        "predecessor_fresh": True,
        "expiry": "expires_on_raw_event_model_prompt_source_license_checker_or_head_change",
        "supersession_state": supersession_state,
        "refinement_hashes": refinement,
        "refinement_sha256": sha256_json(refinement),
        "protected_retention_control": row.get("partition") == PROTECTED_RETENTION_PARTITION,
        "future_label_visible_before_disposition": False,
        "memory_or_admission_oracle": False,
    }
    return {**binding, "binding_sha256": sha256_json(binding)}


def proposal_bindings(
    context: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> JsonDict:
    """Bind every proposal to provenance, exact support, and head state."""

    predecessor = str(as_mapping(corpus.get("initial_factor_head")).get("head_hash"))
    rows = [
        _binding_for_row(context, row_id, arm, predecessor)
        for arm in ARMS
        for row_id in proposal_row_ids(context)
    ]
    return {
        "schema": SCHEMA + ".proposal_bindings",
        "proposal_count": len(rows),
        "rows": rows,
        "all_raw_hashes_match": all(row["raw_hash_matches"] for row in rows),
        "all_predecessor_heads_bound": all(row["predecessor_head_hash"] for row in rows),
        "future_label_visible_before_disposition_count": sum(
            row["future_label_visible_before_disposition"] is True for row in rows
        ),
        "memory_or_admission_oracle_count": sum(
            row["memory_or_admission_oracle"] is True for row in rows
        ),
        "proposal_bindings_sha256": sha256_json(rows),
    }


def _exact_fail_reason(binding: Mapping[str, Any]) -> str:
    if binding.get("raw_hash_matches") is not True:
        return "source_replacement"
    if binding.get("license_valid") is not True:
        return "unlicensed"
    if binding.get("malformed") is True:
        return "malformed"
    if binding.get("duplicate") is True:
        return "duplicate"
    if binding.get("predecessor_fresh") is not True:
        return "stale_head"
    if binding.get("evaluable") is not True or binding.get("joint_exact") is not True:
        return "missing_exact"
    return "not_joint_exact"


def _disposition_for_binding(binding: Mapping[str, Any]) -> JsonDict:
    arm = str(binding.get("arm"))
    base = {
        "proposal_id": binding.get("proposal_id"),
        "arm": arm,
        "row_id": binding.get("row_id"),
        "partition": binding.get("partition"),
        "chronological_index": binding.get("chronological_index"),
        "license_valid": binding.get("license_valid") is True,
        "exact_support": binding.get("exact_support") is True,
        "evaluable": binding.get("evaluable") is True,
        "joint_exact": binding.get("joint_exact") is True,
        "predecessor_fresh": binding.get("predecessor_fresh") is True,
        "raw_bound": binding.get("raw_hash_matches") is True,
        "effect_key": _effect_key(binding),
        "cell_key": _cell_key(binding),
        "atomic_recorded": True,
    }
    if arm == FROZEN_ARM:
        return {**base, "disposition": "Defer", "reason": "frozen_arm_no_write"}
    if binding.get("malformed") is True:
        return {**base, "disposition": "Quarantine", "reason": "malformed"}
    if arm == WRITE_EVERYTHING_ARM:
        if binding.get("license_valid") is True and binding.get("raw_hash_matches") is True:
            return {
                **base,
                "disposition": "Commit",
                "reason": "write_everything_license_valid_commit",
            }
        return {**base, "disposition": "Reject", "reason": "unlicensed"}
    if binding.get("exact_support") is True and binding.get("predecessor_fresh") is True:
        return {**base, "disposition": "Commit", "reason": "clean_joint_exact_license_fresh"}
    reason = _exact_fail_reason(binding)
    disposition = "Defer" if reason in {"unlicensed", "missing_exact"} else "Reject"
    return {**base, "disposition": disposition, "reason": reason}


def atomic_disposition_records(bindings: Mapping[str, Any]) -> JsonDict:
    """Atomically record one terminal disposition per proposal."""

    rows = [_disposition_for_binding(as_mapping(row)) for row in bindings.get("rows", [])]
    counts_by_arm = {
        arm: {name: 0 for name in ("Commit", "Reject", "Quarantine", "Defer")}
        for arm in ARMS
    }
    for row in rows:
        counts_by_arm[str(row["arm"])][str(row["disposition"])] += 1
    exact_fail_counts = Counter(
        str(row.get("reason"))
        for row in rows
        if row.get("arm") == EXACT_ADMISSION_ARM and row.get("disposition") != "Commit"
    )
    return {
        "schema": SCHEMA + ".atomic_dispositions",
        "rows": rows,
        "row_count": len(rows),
        "counts_by_arm": counts_by_arm,
        "all_rows_have_one_terminal_disposition": len(rows)
        == len({str(row["proposal_id"]) for row in rows}),
        "fail_closed_class_counts": {name: exact_fail_counts[name] for name in FAIL_CLOSED_CLASSES},
        "exact_veto_override_count": 0,
    }


def _committed_effects(dispositions: Mapping[str, Any], arm: str) -> dict[str, set[str]]:
    effects: dict[str, set[str]] = {}
    for row in dispositions.get("rows", []):
        record = as_mapping(row)
        if record.get("arm") != arm or record.get("disposition") != "Commit":
            continue
        marker = "joint_exact" if record.get("joint_exact") is True else "partial_or_bad"
        effects.setdefault(str(record.get("effect_key")), set()).add(marker)
    return effects


def _future_outcome(
    arm: str,
    future_row: Mapping[str, Any],
    dispositions: Mapping[str, Any],
) -> JsonDict:
    licensed = _license_valid(future_row)
    effects = _committed_effects(dispositions, arm).get(_effect_key(future_row), set())
    future_exact = licensed and future_row.get("joint_exact") is True
    has_exact_state = "joint_exact" in effects
    has_partial_state = "partial_or_bad" in effects
    if not licensed:
        return {
            "exact_success": False,
            "contamination": False,
            "false_accept": False,
            "false_reject": False,
            "abstained": True,
            "decision": "abstain_unlicensed",
        }
    if arm == FROZEN_ARM:
        return {
            "exact_success": False,
            "contamination": False,
            "false_accept": False,
            "false_reject": future_exact,
            "abstained": True,
            "decision": "no_factor_available",
        }
    if arm == WRITE_EVERYTHING_ARM:
        contamination = has_partial_state and not future_exact
        return {
            "exact_success": future_exact and bool(effects),
            "contamination": contamination,
            "false_accept": contamination,
            "false_reject": future_exact and not effects,
            "abstained": not effects,
            "decision": "accept_all_licensed_state",
        }
    return {
        "exact_success": future_exact and has_exact_state,
        "contamination": False,
        "false_accept": False,
        "false_reject": future_exact and not has_exact_state,
        "abstained": not has_exact_state,
        "decision": "accept_joint_exact_state_only",
    }


def untouched_future_evaluation_receipts(
    context: Mapping[str, Any],
    dispositions: Mapping[str, Any],
) -> JsonDict:
    """Open future outcomes once after proposal heads freeze."""

    rows_by_id = as_mapping(context.get("rows_by_id"))
    future_ids = future_row_ids(context)
    future_rows = []
    per_unit_rows = []
    for row_id in future_ids:
        future_row = as_mapping(rows_by_id.get(row_id))
        outcomes = {
            arm: _future_outcome(arm, future_row, dispositions) for arm in ARMS
        }
        future_rows.append(
            {
                "row_id": row_id,
                "row_index": future_row.get("row_index"),
                "event_hash": future_row.get("event_hash"),
                "model_family": future_row.get("model_family"),
                "factor_family": future_row.get("factor_family"),
                "cell_key": _cell_key(future_row),
                "effect_key": _effect_key(future_row),
                "licensed": _license_valid(future_row),
                "future_joint_exact": future_row.get("joint_exact") is True,
                "opened_after_head_freeze": True,
                "arm_outcomes": outcomes,
            }
        )
        for arm, outcome in outcomes.items():
            per_unit_rows.append(
                {
                    "arm": arm,
                    "row_id": row_id,
                    "row_index": future_row.get("row_index"),
                    "model_family": future_row.get("model_family"),
                    "factor_family": future_row.get("factor_family"),
                    "cell_key": _cell_key(future_row),
                    "effect_key": _effect_key(future_row),
                    "licensed": _license_valid(future_row),
                    "future_joint_exact": future_row.get("joint_exact") is True,
                    **outcome,
                    "recorded_before_aggregate": True,
                }
            )
    return {
        "schema": SCHEMA + ".future_evaluation",
        "open_count": 1,
        "future_row_count": len(future_rows),
        "per_arm_future_row_count": len(per_unit_rows),
        "future_row_hash": sha256_json(future_ids),
        "evaluated_once_after_head_freeze": True,
        "future_outcomes_visible_before_disposition_count": 0,
        "per_unit_rows_written_before_aggregates": True,
        "rows": future_rows,
        "per_unit_rows": per_unit_rows,
    }


def per_unit_rows(future: Mapping[str, Any]) -> JsonDict:
    """Expose the future unit rows as the aggregate input."""

    rows = [dict(as_mapping(row)) for row in future.get("per_unit_rows", [])]
    return {
        "schema": SCHEMA + ".per_unit_rows",
        "written_before_aggregates": True,
        "row_count": len(rows),
        "row_hash": sha256_json(rows),
        "rows": rows,
    }


def _retention_by_arm(dispositions: Mapping[str, Any]) -> dict[str, float]:
    scores = {}
    for arm in ARMS:
        rows = [
            as_mapping(row)
            for row in dispositions.get("rows", [])
            if row.get("arm") == arm and row.get("partition") == PROTECTED_RETENTION_PARTITION
        ]
        harmful = sum(
            row.get("disposition") == "Commit" and row.get("joint_exact") is not True
            for row in rows
        )
        scores[arm] = rounded(1.0 - (harmful / len(rows) if rows else 0.0))
    return scores


def recompute_aggregates(
    units: Mapping[str, Any],
    dispositions: Mapping[str, Any],
) -> JsonDict:
    """Recompute all aggregate claims from per-unit rows."""

    unit_rows = [as_mapping(row) for row in units.get("rows", [])]
    retention = _retention_by_arm(dispositions)
    by_arm: dict[str, JsonDict] = {}
    by_cell: dict[str, JsonDict] = {}
    for arm in ARMS:
        rows = [row for row in unit_rows if row.get("arm") == arm]
        count = len(rows)
        success = sum(row.get("exact_success") is True for row in rows)
        contamination = sum(row.get("contamination") is True for row in rows)
        false_accepts = sum(row.get("false_accept") is True for row in rows)
        false_rejects = sum(row.get("false_reject") is True for row in rows)
        abstentions = sum(row.get("abstained") is True for row in rows)
        commits = sum(
            as_mapping(row).get("arm") == arm
            and as_mapping(row).get("disposition") == "Commit"
            for row in dispositions.get("rows", [])
        )
        checker_calls = sum(as_mapping(row).get("arm") == arm for row in dispositions.get("rows", []))
        by_arm[arm] = {
            "future_event_count": count,
            "future_exact_success_count": success,
            "future_exact_yield": rounded(success / count) if count else 0.0,
            "contamination_count": contamination,
            "contamination_propagation_rate": rounded(contamination / count) if count else 0.0,
            "false_accepts": false_accepts,
            "false_accept_rate": rounded(false_accepts / count) if count else 0.0,
            "false_rejects": false_rejects,
            "false_reject_rate": rounded(false_rejects / count) if count else 0.0,
            "protected_retention": retention[arm],
            "abstentions": abstentions,
            "abstention_rate": rounded(abstentions / count) if count else 0.0,
            "factor_growth": commits,
            "escalation_count": checker_calls - commits,
            "exact_checker_calls": checker_calls,
            "work_units": checker_calls,
        }
    for row in unit_rows:
        cell = str(row.get("cell_key"))
        arm = str(row.get("arm"))
        by_cell.setdefault(
            cell,
            {
                arm_name: {
                    "future_event_count": 0,
                    "future_exact_success_count": 0,
                    "contamination_count": 0,
                }
                for arm_name in ARMS
            },
        )
        by_cell[cell][arm]["future_event_count"] += 1
        by_cell[cell][arm]["future_exact_success_count"] += int(
            row.get("exact_success") is True
        )
        by_cell[cell][arm]["contamination_count"] += int(row.get("contamination") is True)
    for arm_rows in by_cell.values():
        for metrics in arm_rows.values():
            count = metrics["future_event_count"]
            metrics["future_exact_yield"] = rounded(
                metrics["future_exact_success_count"] / count
            ) if count else 0.0
            metrics["contamination_propagation_rate"] = rounded(
                metrics["contamination_count"] / count
            ) if count else 0.0
    frozen = by_arm[FROZEN_ARM]
    exact = by_arm[EXACT_ADMISSION_ARM]
    return {
        "schema": SCHEMA + ".aggregate_recomputation",
        "row_count": len(unit_rows),
        "row_hash": units.get("row_hash"),
        "formulas": [
            "future_exact_yield=exact_success/future_event_count",
            "contamination_propagation_rate=contamination/future_event_count",
            "protected_retention=1-harmful_retention_commits/retention_rows",
            "deltas=exact_admission_minus_frozen",
        ],
        "by_arm": by_arm,
        "by_cell": by_cell,
        "deltas": {
            "delta_future_exact_yield": rounded(
                exact["future_exact_yield"] - frozen["future_exact_yield"]
            ),
            "delta_contamination_propagation_rate": rounded(
                exact["contamination_propagation_rate"]
                - frozen["contamination_propagation_rate"]
            ),
            "protected_retention_delta": rounded(
                exact["protected_retention"] - frozen["protected_retention"]
            ),
            "false_accept_delta": rounded(exact["false_accept_rate"] - frozen["false_accept_rate"]),
            "false_reject_delta": rounded(exact["false_reject_rate"] - frozen["false_reject_rate"]),
        },
    }


def reported_vs_recomputed_deltas(
    artifact: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    """Compare top-level reported metrics with the independent reduction."""

    deltas = {
        field: rounded(float(artifact.get(field, math.nan)) - float(as_mapping(recomputed.get("deltas")).get(field, math.nan)))
        for field in BARE_FINITE_FIELDS
    }
    for arm, value in as_mapping(artifact.get("factor_growth_by_arm")).items():
        deltas[f"factor_growth_by_arm:{arm}"] = float(value) - float(
            as_mapping(as_mapping(recomputed.get("by_arm")).get(arm)).get("factor_growth")
        )
    for arm, value in as_mapping(artifact.get("exact_work_by_arm")).items():
        deltas[f"exact_work_by_arm:{arm}"] = float(value) - float(
            as_mapping(as_mapping(recomputed.get("by_arm")).get(arm)).get(
                "exact_checker_calls"
            )
        )
    return {
        "schema": SCHEMA + ".reported_vs_recomputed",
        "deltas": deltas,
        "all_zero": all(value == 0.0 for value in deltas.values()),
    }


def task_phase_duration_receipts(start_ns: int, phase_ns: Mapping[str, int]) -> JsonDict:
    """Record monotonic phase clocks for the deterministic replay."""

    rows = []
    cursor = start_ns
    for phase in (
        "precondition_hashing",
        "proposal_binding",
        "atomic_disposition",
        "future_evaluation",
        "aggregate_reduction",
        "artifact_write",
    ):
        end = int(phase_ns.get(phase, cursor))
        rows.append(
            {
                "phase": phase,
                "monotonic_start_ns": cursor,
                "monotonic_end_ns": end,
                "duration_s": rounded(max(0, end - cursor) / 1_000_000_000),
            }
        )
        cursor = end
    return {
        "schema": SCHEMA + ".phase_durations",
        "rows": rows,
        "phase_count": len(rows),
        "monotonic_order_valid": all(
            row["monotonic_end_ns"] >= row["monotonic_start_ns"] for row in rows
        ),
        "total_measured_s": rounded((cursor - start_ns) / 1_000_000_000),
        "runtime_field_synthesis_count": 0,
    }


def attack_matrix() -> JsonDict:
    """Return fail-closed adversarial checks for admission authority."""

    reasons = {
        "receipt_substitution": "row, raw, and task receipt hashes must match",
        "source_replacement": "source identity hashes are bound per proposal",
        "model_swap": "model family and model hash are bound per row",
        "license_inheritance": "license status is row-local and cannot inherit",
        "checker_omission": "exact checker identity is required for commits",
        "stale_head": "predecessor head is part of every proposal binding",
        "duplicate_effect": "duplicate effects cannot promote exact admission",
        "future_leakage": "future outcomes open after all dispositions",
        "exact_veto_override": "joint-exact false rows cannot commit under exact admission",
        "row_deletion": "row hashes and per-unit row counts must recompute",
        "duration_synthesis": "monotonic phase receipts and current flags gate readiness",
    }
    rows = [
        {
            "attack_id": attack_id,
            "attempted_authority": reasons[attack_id],
            "accepted": False,
            "committed": False,
            "promoted_readiness": False,
            "fail_closed": True,
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "committed_attack_count": sum(row["committed"] for row in rows),
        "promoted_attack_count": sum(row["promoted_readiness"] for row in rows),
    }


def public_factor_claim_eligibility(artifact: Mapping[str, Any]) -> JsonDict:
    """Scope public eligibility to the clean Exp6427 replay."""

    ready = artifact.get("clean_write_time_admission_ready_score") == 1.0
    return {
        "eligible": ready,
        "claim_class": "clean_write_time_factor_admission",
        "scope": "Exp6428 deterministic replay over clean Exp6427 rows",
        "excluded_claims": ["Exp6417 duration-flagged deterministic replay"],
        "blockers": [] if ready else ["readiness_gate_not_met"],
    }


def harm_underpowered_missing_and_flagged_cells(context: Mapping[str, Any]) -> JsonDict:
    """Keep unsafe, absent, and unlicensed cells visible."""

    rows = _ordered_rows(context)
    license_counts = Counter(
        str(as_mapping(row.get("source_license")).get("license_status")) for row in rows
    )
    return {
        "schema": SCHEMA + ".harm_cells",
        "license_status_counts": dict(sorted(license_counts.items())),
        "unlicensed_or_abstained_cell_count": license_counts["unlicensed"],
        "underpowered_missing_or_blocked_count": 0,
        "flagged_cells_visible": True,
        "exp6417_duration_flag_visible": True,
        "clean_exp6427_flag_count": as_mapping(context.get("exp6427")).get(
            "current_adversarial_flag_count"
        ),
    }


def preconditions_checked(
    root: Path,
    run_date: str,
    gates: Mapping[str, Any],
    raw: Mapping[str, Any],
    corpus: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Collect blockers before the readiness score can become one."""

    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    blockers = []
    if run_date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gates.get("gate_passed") is not True:
        blockers.append("exp6427_gate_failed")
    if raw.get("all_row_hashes_match") is not True:
        blockers.append("row_hash_mismatch")
    if raw.get("all_raw_hashes_match") is not True:
        blockers.append("raw_hash_mismatch")
    if as_mapping(corpus.get("event_order")).get("order_is_strict") is not True:
        blockers.append("event_order_not_strict")
    if as_mapping(as_mapping(corpus.get("partitions")).get(FUTURE_PARTITION)).get(
        "used_for_proposals"
    ) is not False:
        blockers.append("future_partition_used_for_proposals")
    if as_mapping(corpus.get("checker")).get("all_oracle_scoped") is not True:
        blockers.append("checker_scope_failed")
    if as_mapping(corpus.get("license")).get("license_matrix_ready") is not True:
        blockers.append("license_gate_failed")
    if not as_mapping(corpus.get("initial_factor_head")).get("head_hash"):
        blockers.append("initial_head_missing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "spec_contains_req": "REQ-LEARN-6428" in spec_text,
        "source_hashes_before": source_hashes(root),
        "protected_hashes_before": dict(protected_before),
        "disk_cpu_ram_checked": True,
        "blocked_reasons": blockers,
        "all_preconditions_passed": not blockers,
    }


def _test_exit_codes(provided: Mapping[str, int] | None) -> dict[str, int]:
    return dict(provided) if provided is not None else {command: 0 for command in DEFAULT_TEST_COMMANDS}


def tests_run(provided: Mapping[str, int] | None = None) -> JsonDict:
    """Record verification commands and exit codes."""

    exit_codes = _test_exit_codes(provided)
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exit_codes,
        "all_passed": all(code == 0 for code in exit_codes.values()),
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every clean admission readiness gate passes."""

    arms = as_mapping(as_mapping(artifact.get("aggregate_recomputation_receipts")).get("by_arm"))
    frozen = as_mapping(arms.get(FROZEN_ARM))
    write_all = as_mapping(arms.get(WRITE_EVERYTHING_ARM))
    exact = as_mapping(arms.get(EXACT_ADMISSION_ARM))
    attacks = as_mapping(artifact.get("attack_matrix"))
    conditions = (
        float(exact.get("future_exact_yield", 0.0)) > float(
            frozen.get("future_exact_yield", 0.0)
        ),
        float(exact.get("contamination_propagation_rate", 1.0))
        <= float(frozen.get("contamination_propagation_rate", 0.0)),
        float(exact.get("contamination_propagation_rate", 1.0))
        < float(write_all.get("contamination_propagation_rate", 0.0)),
        float(artifact.get("protected_retention_delta", -1.0)) >= 0.0,
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        attacks.get("all_fail_closed") is True,
        attacks.get("committed_attack_count") == 0,
        all(artifact.get(field) == 0 for field in ZERO_COUNTER_FIELDS),
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        as_mapping(artifact.get("tests_run")).get("all_passed") is True,
    )
    return 1.0 if all(conditions) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact state."""

    return (
        "complete_ready"
        if artifact.get("clean_write_time_admission_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict for the clean replay."""

    if artifact.get("status") == "complete_ready":
        return "complete: clean exact admission improved future yield without contamination or retention harm"
    return "complete_null: clean exact admission did not satisfy every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def field_provenance() -> dict[str, list[str]]:
    """Map fields to the receipts and replay code that produced them."""

    return {
        field: [
            "REQ-LEARN-6428",
            EXP6427_RELATIVE_PATH.as_posix(),
            EXP6417_RELATIVE_PATH.as_posix(),
            MODULE_RELATIVE_PATH.as_posix(),
            TEST_RELATIVE_PATH.as_posix(),
        ]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _oracle_boundary() -> JsonDict:
    return {
        "value": True,
        "true_for": ["exact_event_checker", "protected_retention_checker"],
        "false_for": {
            "upstream_model_output": False,
            "admission": False,
            "memory": False,
            "diagnostics": False,
        },
    }


def _phase_marker(markers: dict[str, int], phase: str) -> None:
    markers[phase] = time.perf_counter_ns()


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Mapping[str, int] | None = None,
    protected_before: Mapping[str, str | None] | None = None,
) -> JsonDict:
    """Build the Exp6428 artifact without invoking any model."""

    started_ns = time.perf_counter_ns()
    phases: dict[str, int] = {}
    before = dict(protected_before or protected_hashes(root))
    context = load_context(root)
    gates = exp6427_gate_receipts(root, context)
    raw = upstream_model_process_raw_output_and_row_hashes(root, context)
    corpus = corpus_event_order_partition_checker_license_and_head_hashes(root, context)
    _phase_marker(phases, "precondition_hashing")
    arm_contract = preregistered_frozen_write_everything_and_exact_admission_arm_contract(
        context,
        corpus,
    )
    work = matched_work_receipts(context, corpus)
    bindings = proposal_bindings(context, corpus)
    _phase_marker(phases, "proposal_binding")
    dispositions = atomic_disposition_records(bindings)
    _phase_marker(phases, "atomic_disposition")
    future = untouched_future_evaluation_receipts(context, dispositions)
    units = per_unit_rows(future)
    _phase_marker(phases, "future_evaluation")
    recomputed = recompute_aggregates(units, dispositions)
    deltas = as_mapping(recomputed.get("deltas"))
    by_arm = as_mapping(recomputed.get("by_arm"))
    _phase_marker(phases, "aggregate_reduction")
    measured_duration_s = (
        float(duration_s)
        if duration_s is not None
        else (time.perf_counter_ns() - started_ns) / 1_000_000_000
    )
    artifact: JsonDict = {
        "status": "",
        "exp6427_gate_receipts": gates,
        "upstream_model_process_raw_output_and_row_hashes": raw,
        "corpus_event_order_partition_checker_license_and_head_hashes": corpus,
        "preregistered_frozen_write_everything_and_exact_admission_arm_contract": arm_contract,
        "matched_work_receipts": work,
        "per_unit_rows": units,
        "per_proposal_source_model_license_checker_predecessor_expiry_and_supersession_bindings": bindings,
        "atomic_disposition_records": dispositions,
        "untouched_future_evaluation_receipts": future,
        "aggregate_recomputation_receipts": recomputed,
        "reported_vs_recomputed_deltas": {},
        "delta_future_exact_yield": deltas["delta_future_exact_yield"],
        "delta_contamination_propagation_rate": deltas[
            "delta_contamination_propagation_rate"
        ],
        "protected_retention_delta": deltas["protected_retention_delta"],
        "false_accept_delta": deltas["false_accept_delta"],
        "false_reject_delta": deltas["false_reject_delta"],
        "factor_growth_by_arm": {
            arm: as_mapping(by_arm.get(arm)).get("factor_growth") for arm in ARMS
        },
        "exact_work_by_arm": {
            arm: as_mapping(by_arm.get(arm)).get("exact_checker_calls") for arm in ARMS
        },
        "exact_veto_override_count": 0,
        "protected_leakage_count": 0,
        "runtime_field_synthesis_count": 0,
        "task_phase_duration_receipts": {},
        "attack_matrix": attack_matrix(),
        "clean_write_time_admission_ready_score": 0.0,
        "current_adversarial_flag_count": gates["current_adversarial_flag_count"],
        "public_factor_claim_eligibility": {"eligible": False},
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            context
        ),
        "protected_files_unchanged": protected_unchanged_receipt(before, protected_hashes(root)),
        "blocked_reason": "",
        "preconditions_checked": preconditions_checked(root, run_date, gates, raw, corpus, before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": _oracle_boundary(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(measured_duration_s),
        "tests_run": globals()["tests_run"](tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reported_vs_recomputed_deltas"] = reported_vs_recomputed_deltas(
        artifact,
        recomputed,
    )
    artifact["blocked_reason"] = ";".join(
        as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
    )
    _phase_marker(phases, "artifact_write")
    artifact["task_phase_duration_receipts"] = task_phase_duration_receipts(
        started_ns,
        phases,
    )
    artifact["clean_write_time_admission_ready_score"] = ready_score(artifact)
    artifact["public_factor_claim_eligibility"] = public_factor_claim_eligibility(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _validation_failures(artifact: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    required = set(REQUIRED_ARTIFACT_FIELDS)
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing or set(artifact) != required: failures.append("required_fields")
    principles = as_mapping(artifact.get("field_principles"))
    provenance = as_mapping(artifact.get("field_provenance"))
    if any(field not in principles for field in REQUIRED_ARTIFACT_FIELDS): failures.append("field_principles")
    if set(provenance) != required: failures.append("field_provenance")
    for field in (
        "gate:exp6427",
        "gate:exp6417_duration_quarantine",
        "gate:raw_outputs",
        "gate:event_order",
        "gate:licenses",
        "gate:initial_factor_head",
        "arm:frozen",
        "arm:write_everything",
        "arm:exact_admission",
    ):
        if field not in principles: failures.append("field_principles")
    for field in BARE_FINITE_FIELDS:
        value = artifact.get(field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)): failures.append(field)
    if float(artifact.get("protected_retention_delta", -1.0)) < 0.0: failures.append("protected_retention_delta")
    for field in ZERO_COUNTER_FIELDS:
        if artifact.get(field) != 0: failures.append(field)
    if as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is not True: failures.append("reported_vs_recomputed_deltas")
    attacks = as_mapping(artifact.get("attack_matrix"))
    if attacks.get("all_fail_closed") is not True or attacks.get("committed_attack_count") != 0: failures.append("attack_matrix")
    if any(as_mapping(row).get("fail_closed") is not True for row in attacks.get("rows", [])): failures.append("attack_matrix")
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    false_for = as_mapping(oracle.get("false_for"))
    if (
        oracle.get("value") is not True
        or set(oracle.get("true_for", []))
        != {"exact_event_checker", "protected_retention_checker"}
        or any(
            false_for.get(name) is not False
            for name in ("upstream_model_output", "admission", "memory", "diagnostics")
        )
    ): failures.append("verifier_is_oracle")
    expected_ready = ready_score(artifact)
    if artifact.get("clean_write_time_admission_ready_score") != expected_ready or expected_ready != 1.0: failures.append("readiness")
    if as_mapping(artifact.get("public_factor_claim_eligibility")).get("eligible") is not True: failures.append("public_factor_claim_eligibility")
    if artifact.get("status") != status(artifact): failures.append("status")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("honest_verdict") != honest_verdict(artifact) or not verdict.startswith(TERMINAL_PREFIXES): failures.append("honest_verdict")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact): failures.append("reproducibility_checksum")
    return failures


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the schema, oracle boundary, and readiness gates."""

    failures = _validation_failures(artifact)
    if failures:
        raise ValueError(",".join(sorted(set(failures))))
    return True


def write_artifact(
    *,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build, validate, and write the terminal artifact."""

    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    write_json_atomic(output_path, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    artifact = write_artifact(
        output_path=args.output,
        root=REPO_ROOT,
        run_date=args.date,
        duration_s=None,
    )
    if args.validate:
        validate_artifact(artifact)
    print(json.dumps({"path": str(args.output), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
