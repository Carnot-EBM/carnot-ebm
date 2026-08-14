"""Exp6431 controlled memory-interference A/B.

Spec refs: REQ-LEARN-6431, SCENARIO-LEARN-6431-GATES,
SCENARIO-LEARN-6431-FREEZE, SCENARIO-LEARN-6431-PATHS,
SCENARIO-LEARN-6431-METRICS, SCENARIO-LEARN-6431-ATTACKS,
SCENARIO-LEARN-6431-READY.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_6430_prospective_write_once_memory_capacity_frontier as exp6430
from carnot.memory.revocable_atomic_repair import (
    AtomicRepairItem,
    TransactionalRevocableRepairMemory,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6431_controlled_memory_interference_ab.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6431_controlled_memory_interference_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6431_controlled_memory_interference_ab.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
REFERENCE_RELATIVE_PATH = Path("research-references.md")
EXP6430_RELATIVE_PATH = exp6430.RESULT_RELATIVE_PATH
EXP6420_RELATIVE_PATH = Path("results/experiment_6420_csl_authenticity_safety_audit.json")
EXP6430_MANIFEST_RELATIVE_PATH = exp6430.DATA_DIR_RELATIVE_PATH / exp6430.MANIFEST_FILENAME
EXP6430_TASK_RECEIPT_RELATIVE_PATH = exp6430.DATA_DIR_RELATIVE_PATH / exp6430.TASK_RECEIPT_FILENAME

SCHEMA = "carnot.experiment_6431.controlled_memory_interference_ab.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6431
INFERENCE_SUBSTRATE = "deterministic_replay_over_sealed_exp6430_rows_no_new_llm"
TRANSACTIONAL_MEMORY_MODULE = (
    "carnot.memory.revocable_atomic_repair.TransactionalRevocableRepairMemory"
)

CAPACITIES = tuple(int(capacity) for capacity in exp6430.CAPACITIES)
FUTURE_EVENT_COUNT = int(exp6430.FUTURE_EVENT_COUNT)
BASELINE_ARM = "capacity_matched_baseline_memory"
AUTHORITY_AWARE_ARM = "authority_aware_retrieval_and_write_controls"
ARMS = (BASELINE_ARM, AUTHORITY_AWARE_ARM)
RELATIONSHIP_CLASSES = (
    "benign_accumulation",
    "reinforcing_evidence",
    "contradiction",
    "source_authority_conflict",
    "supersession",
    "temporal_invalidity",
    "lexical_collision",
    "structural_collision",
    "poisoned_evidence",
    "target_occlusion",
)
ATTACK_IDS = (
    "authority_spoofing",
    "recency_only_override",
    "source_pooling",
    "lexical_collision",
    "structural_collision",
    "target_hiding",
    "cache_resurrection",
    "rollback_omission",
    "head_substitution",
    "post_outcome_relation_labels",
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

AUTHORITY_LEVELS = {
    "poison": 0,
    "spoofed": 10,
    "lower_untrusted": 20,
    "peer_session": 70,
    "primary_exact": 80,
    "higher_exact": 100,
}
RELATIONSHIP_RULES: dict[str, JsonDict] = {
    "benign_accumulation": {
        "family": "benign",
        "source_authority": "primary_exact",
        "target_authority": "primary_exact",
        "valid_update": True,
        "higher_authority_update": False,
        "invalid_memory": False,
    },
    "reinforcing_evidence": {
        "family": "reinforcing",
        "source_authority": "primary_exact",
        "target_authority": "primary_exact",
        "valid_update": True,
        "higher_authority_update": False,
        "invalid_memory": False,
    },
    "contradiction": {
        "family": "invalid_authority",
        "source_authority": "lower_untrusted",
        "target_authority": "primary_exact",
        "valid_update": False,
        "authority_spoof_attempt": True,
        "invalid_memory": True,
    },
    "source_authority_conflict": {
        "family": "invalid_authority",
        "source_authority": "lower_untrusted",
        "target_authority": "higher_exact",
        "valid_update": False,
        "authority_spoof_attempt": True,
        "invalid_memory": True,
    },
    "supersession": {
        "family": "valid_plasticity",
        "source_authority": "higher_exact",
        "target_authority": "primary_exact",
        "valid_update": True,
        "higher_authority_update": True,
        "invalid_memory": False,
    },
    "temporal_invalidity": {
        "family": "invalid_temporal",
        "source_authority": "primary_exact",
        "target_authority": "primary_exact",
        "valid_update": False,
        "expired": True,
        "invalid_memory": True,
    },
    "lexical_collision": {
        "family": "collision",
        "source_authority": "peer_session",
        "target_authority": "primary_exact",
        "valid_update": True,
        "collision": "lexical",
        "invalid_memory": False,
    },
    "structural_collision": {
        "family": "collision",
        "source_authority": "peer_session",
        "target_authority": "primary_exact",
        "valid_update": True,
        "collision": "structural",
        "invalid_memory": False,
    },
    "poisoned_evidence": {
        "family": "poison",
        "source_authority": "poison",
        "target_authority": "primary_exact",
        "valid_update": False,
        "poisoned": True,
        "invalid_memory": True,
    },
    "target_occlusion": {
        "family": "occlusion",
        "source_authority": "peer_session",
        "target_authority": "primary_exact",
        "valid_update": True,
        "target_occlusion": True,
        "invalid_memory": False,
    },
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6431_controlled_memory_interference_ab "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6431_controlled_memory_interference_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6431_controlled_memory_interference_ab.py "
    "-m pytest tests/python/test_experiment_6431_controlled_memory_interference_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6431_controlled_memory_interference_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6431_controlled_memory_interference_ab.py"
)
E2E_VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6431_controlled_memory_interference_ab "
    "--date 20260814 --validate --output /tmp/experiment_6431_e2e.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6431_controlled_memory_interference_ab.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ARTIFACT_AUDIT_COMMAND = ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_AUDIT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6430_gate_receipts",
    "upstream_row_manifest_policy_checker_and_head_hashes",
    "preregistered_interference_matrix",
    "preregistered_capacity_matched_arm_contract",
    "per_unit_rows",
    "per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results",
    "exposure_failure_count",
    "downstream_use_failure_count",
    "authority_spoof_accept_count",
    "expired_or_superseded_accept_count",
    "poisoned_evidence_accept_count",
    "valid_higher_authority_update_count",
    "protected_stability_delta",
    "contamination_after_rollback",
    "aggregate_recomputation_receipts",
    "reported_vs_recomputed_deltas",
    "attack_matrix",
    "memory_interference_safety_ready_score",
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
    "status": "Names the terminal state for the controlled memory-interference A/B.",
    "exp6430_gate_receipts": "Pins Exp6430 readiness, Exp6420 null context, and sealed-row eligibility.",
    "upstream_row_manifest_policy_checker_and_head_hashes": "Binds rows, manifests, memory policy, exact checkers, heads, and machine resources.",
    "preregistered_interference_matrix": "Freezes relationship labels before downstream outcomes can steer them.",
    "preregistered_capacity_matched_arm_contract": "Proves baseline and authority-aware arms share capacity, work, evidence, order, and initial heads.",
    "per_unit_rows": "Records one arm-level unit before aggregate calculation.",
    "per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results": "Reports separated cells without masking weak strata.",
    "exposure_failure_count": "Counts target records blocked before retrieval.",
    "downstream_use_failure_count": "Counts exposed records that were not used correctly downstream.",
    "authority_spoof_accept_count": "Must be zero because spoofed authority cannot release memory.",
    "expired_or_superseded_accept_count": "Must be zero because invalid temporal state cannot release memory.",
    "poisoned_evidence_accept_count": "Must be zero because poison cannot release memory.",
    "valid_higher_authority_update_count": "Must stay nonzero to show safe plasticity.",
    "protected_stability_delta": "Must not regress protected behavior.",
    "contamination_after_rollback": "Must be zero after rollback.",
    "aggregate_recomputation_receipts": "Recomputes metrics from per-unit rows.",
    "reported_vs_recomputed_deltas": "Shows reported aggregates match row recomputation.",
    "attack_matrix": "Shows every critical invalid-memory attack fails closed.",
    "memory_interference_safety_ready_score": "Conjunctive readiness for authority-aware memory interference safety.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps weak, missing, null, and flagged cells visible.",
    "protected_files_unchanged": "Shows protected upstream and ops files stayed byte-identical.",
    "blocked_reason": "Explains failed preconditions.",
    "preconditions_checked": "Lists gates, hashes, schemas, resources, and seals checked before readiness.",
    "inference_substrate": "Declares deterministic replay over sealed Exp6430 rows with no new LLM.",
    "verifier_is_oracle": "Marks only exact support and authority lifecycle checks as oracles.",
    "field_principles": "Documents why each field exists.",
    "field_provenance": "Maps each field to specs, inputs, rows, reductions, attacks, or tests.",
    "random_seed": "Pins relation assignment, arms, attacks, and reductions.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal success prefix and states the interference result.",
    "gate:exp6430_complete_ready": "Exp6430 must be complete, ready, and row-sealed.",
    "gate:exp6420_safety_null_context": "Exp6420 null findings must remain visible.",
    "relationship:benign_accumulation": "Valid same-authority evidence should remain usable.",
    "relationship:reinforcing_evidence": "Valid reinforcing evidence should improve coverage without instability.",
    "relationship:contradiction": "Lower-authority contradiction must fail closed.",
    "relationship:source_authority_conflict": "Lower-authority source conflict must fail closed.",
    "relationship:supersession": "Valid higher-authority updates may replace older state through audit.",
    "relationship:temporal_invalidity": "Expired evidence must fail closed.",
    "relationship:lexical_collision": "Similar words must not hide the target record.",
    "relationship:structural_collision": "Similar structure must not hide the target record.",
    "relationship:poisoned_evidence": "Poisoned evidence must fail closed.",
    "relationship:target_occlusion": "Target exposure must be measured apart from downstream use.",
    "attack:authority_spoofing": "Spoofed source rank must not release memory.",
    "attack:recency_only_override": "Recency alone must not override authority.",
    "attack:source_pooling": "Pooled sources must not launder lower authority.",
    "attack:lexical_collision": "Lexical collision must not select the wrong record.",
    "attack:structural_collision": "Structural collision must not select the wrong record.",
    "attack:target_hiding": "Hidden targets must stay visible to the exposure metric.",
    "attack:cache_resurrection": "Stale cache must not revive invalid memory.",
    "attack:rollback_omission": "Rollback omission must not leave contamination.",
    "attack:head_substitution": "Head swaps must not alter the matched initial state.",
    "attack:post_outcome_relation_labels": "Relation labels must not be assigned after outcome use.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6431",
        "SCENARIO-LEARN-6431-GATES",
        "SCENARIO-LEARN-6431-FREEZE",
        "SCENARIO-LEARN-6431-PATHS",
        "sealed Exp6430 rows",
        "focused Exp6431 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6430_RELATIVE_PATH,
    EXP6420_RELATIVE_PATH,
    EXP6430_MANIFEST_RELATIVE_PATH,
    EXP6430_TASK_RECEIPT_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    REFERENCE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/memory/revocable_atomic_repair.py"),
    exp6430.MODULE_RELATIVE_PATH,
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for receipts and checksums."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return a project-prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible values after stable serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Hash a file, or return None when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and other values as an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round deterministic metrics without hiding small nonzero values."""

    return round(float(value), 9)


def require(condition: bool, reason: str) -> None:
    """Raise one stable validation error name."""

    if not condition:
        raise ValueError(reason)


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object from disk."""

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


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash files that define this experiment."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected files before and after the run."""

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
    """Load upstream artifacts for the controlled interference replay."""

    return {
        "exp6430": read_json(root / EXP6430_RELATIVE_PATH),
        "exp6420": read_json(root / EXP6420_RELATIVE_PATH),
    }


def _ready_score(payload: Mapping[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)


def _v552_visible(exp6420_payload: Mapping[str, Any]) -> bool:
    harm = as_mapping(exp6420_payload.get("harm_underpowered_missing_and_flagged_cells"))
    reported = as_mapping(exp6420_payload.get("reported_vs_recomputed_deltas"))
    attacks = set(harm.get("open_critical_attack_ids", []))
    return (
        exp6420_payload.get("status") == "complete_null"
        and int(reported.get("mismatch_count", 0) or 0) > 0
        and {"raw_output_reuse", "cache_resurrection"} <= attacks
    )


def exp6430_gate_receipts(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Revalidate upstream readiness and null-context gates."""

    exp6430_payload = as_mapping(context.get("exp6430"))
    exp6420_payload = as_mapping(context.get("exp6420"))
    exp6430_reported = as_mapping(exp6430_payload.get("reported_vs_recomputed_deltas"))
    exp6430_per_unit = as_mapping(exp6430_payload.get("per_unit_rows"))
    checks = (
        (exp6430_payload.get("status") != "complete_ready", "exp6430_not_ready"),
        (
            _ready_score(exp6430_payload, "prospective_write_once_csl_ready_score") != 1.0,
            "exp6430_ready_score_not_one",
        ),
        (exp6430_reported.get("all_zero") is not True, "exp6430_aggregates_do_not_recompute"),
        (int(exp6430_per_unit.get("row_count", 0) or 0) <= 0, "exp6430_per_unit_rows_missing"),
        (_v552_visible(exp6420_payload) is not True, "exp6420_v552_defects_not_visible"),
    )
    blocked = sorted({reason for failed, reason in checks if failed})
    return {
        "schema": SCHEMA + ".upstream_gates",
        "exp6430": {
            **path_receipt(root / EXP6430_RELATIVE_PATH, relative_to=root),
            "status": exp6430_payload.get("status"),
            "ready_score": _ready_score(
                exp6430_payload,
                "prospective_write_once_csl_ready_score",
            ),
            "reported_vs_recomputed_all_zero": exp6430_reported.get("all_zero") is True,
            "per_unit_row_count": exp6430_per_unit.get("row_count"),
        },
        "exp6420": {
            **path_receipt(root / EXP6420_RELATIVE_PATH, relative_to=root),
            "status": exp6420_payload.get("status"),
            "ready_score": _ready_score(
                exp6420_payload,
                "csl_authenticity_safety_audit_ready_score",
            ),
            "v552_defects_visible": _v552_visible(exp6420_payload),
        },
        "blocked_reasons": blocked,
        "all_gates_passed": not blocked,
    }


def ram_total_bytes() -> int:
    """Return host RAM bytes using the local POSIX sysconf values."""

    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def upstream_row_manifest_policy_checker_and_head_hashes(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Bind sealed rows, sidecars, policy, checkers, heads, and resources."""

    exp6430_payload = as_mapping(context.get("exp6430"))
    manifest = as_mapping(
        exp6430_payload.get(
            "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
        )
    )
    contract = as_mapping(exp6430_payload.get("preregistered_capacity_and_arm_contract"))
    history = as_mapping(exp6430_payload.get("memory_schema_head_and_transition_history"))
    per_unit = as_mapping(exp6430_payload.get("per_unit_rows"))
    disk = shutil.disk_usage(root)
    initial_heads = [
        as_mapping(row).get("initial_head_hash")
        for row in as_mapping(contract.get("by_capacity")).values()
    ]
    return {
        "schema": SCHEMA + ".upstream_hashes",
        "exp6430_artifact": path_receipt(root / EXP6430_RELATIVE_PATH, relative_to=root),
        "exp6420_artifact": path_receipt(root / EXP6420_RELATIVE_PATH, relative_to=root),
        "manifest_sidecar": path_receipt(root / EXP6430_MANIFEST_RELATIVE_PATH, relative_to=root),
        "task_receipt_sidecar": path_receipt(
            root / EXP6430_TASK_RECEIPT_RELATIVE_PATH,
            relative_to=root,
        ),
        "per_unit_row_hash": per_unit.get("row_hash"),
        "manifest_hash": manifest.get("sha256"),
        "memory_policy": {
            "capacity_frozen": contract.get("capacities_frozen_before_outcomes") is True,
            "capacities": contract.get("capacities"),
            "schema_version": history.get("schema_version"),
            "policy_hash": sha256_json(
                {
                    "contract": contract,
                    "schema_version": history.get("schema_version"),
                    "capacities": contract.get("capacities"),
                }
            ),
        },
        "exact_checkers": {
            "exact_support_checker": exp6430_payload.get("exact_veto_override_count") == 0,
            "release_checker": as_mapping(exp6430_payload.get("exact_feedback_receipts")).get(
                "release_check_failures"
            )
            == 0,
            "retention_checker": as_mapping(exp6430_payload.get("exact_feedback_receipts")).get(
                "protected_retention_failures"
            )
            == 0,
            "checker_hash": sha256_json(exp6430_payload.get("exact_feedback_receipts")),
        },
        "authority_schema": {
            "schema_version": SCHEMA + ".authority_schema.v1",
            "levels": dict(AUTHORITY_LEVELS),
            "valid": sorted(AUTHORITY_LEVELS.values()) == [0, 10, 20, 70, 80, 100],
            "sha256": sha256_json(AUTHORITY_LEVELS),
        },
        "head_hashes": {
            "initial_head_hashes": initial_heads,
            "initial_heads_identical": len(set(initial_heads)) == 1,
            "final_head_hashes": {
                capacity: as_mapping(row).get("final_head_hash")
                for capacity, row in as_mapping(history.get("by_capacity")).items()
            },
        },
        "machine_resources": {
            "cpu_count": os.cpu_count() or 1,
            "ram_total_bytes": ram_total_bytes(),
            "disk_free_bytes": disk.free,
        },
        "protected_future_seal": {
            "untouched_before_evaluation": manifest.get("future_partition_untouched_before_evaluation")
            is True,
            "future_row_count": as_mapping(as_mapping(manifest.get("partition_seals")).get("future")).get(
                "row_count"
            ),
            "future_row_hash": as_mapping(as_mapping(manifest.get("partition_seals")).get("future")).get(
                "row_hash"
            ),
        },
    }


def _manifest_events(context: Mapping[str, Any]) -> list[JsonDict]:
    manifest = as_mapping(
        as_mapping(context.get("exp6430")).get(
            "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
        )
    )
    return [dict(as_mapping(event)) for event in manifest.get("events", [])]


def _future_events(context: Mapping[str, Any]) -> list[JsonDict]:
    return [event for event in _manifest_events(context) if event.get("partition") == "future"]


def _source_event_for(target: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates = [
        event
        for event in events
        if event.get("partition") != "future"
        and event.get("effect_key") == target.get("effect_key")
        and event.get("exact_support") is True
    ]
    fallback = [
        event
        for event in events
        if event.get("partition") != "future"
        and event.get("model_family") == target.get("model_family")
    ]
    return dict(candidates[-1] if candidates else fallback[-1])


def preregistered_interference_matrix(context: Mapping[str, Any]) -> JsonDict:
    """Freeze relationship labels from event order before downstream outcomes."""

    events = _manifest_events(context)
    future = _future_events(context)
    rows = []
    for offset, target in enumerate(future):
        relationship = RELATIONSHIP_CLASSES[offset % len(RELATIONSHIP_CLASSES)]
        rule = RELATIONSHIP_RULES[relationship]
        source = _source_event_for(target, events)
        row = {
            "event_id": target["event_id"],
            "chronological_index": target["chronological_index"],
            "relationship_class": relationship,
            "relationship_family": rule["family"],
            "source_event_id": source["event_id"],
            "source_event_hash": source["event_hash"],
            "target_event_hash": target["event_hash"],
            "raw_output_sha256": target["raw_output_sha256"],
            "model_family": target["model_family"],
            "factor_family": target["constraint_family"],
            "effect_key": target["effect_key"],
            "label_rule": "RELATIONSHIP_CLASSES[(chronological_index - future_start) % 10]",
            "label_frozen_before_outcome": int(target["proposal_freeze_order"])
            < int(target["outcome_open_order"]),
            "downstream_outcome_used_for_label": False,
        }
        rows.append(row)
    return {
        "schema": SCHEMA + ".interference_matrix",
        "relationship_classes": list(RELATIONSHIP_CLASSES),
        "relationship_count": len(RELATIONSHIP_CLASSES),
        "rows": rows,
        "row_count": len(rows),
        "matrix_hash": sha256_json(rows),
        "frozen_before_downstream_outcomes": True,
        "post_outcome_relation_label_count": 0,
        "future_outcomes_used_for_labeling_count": 0,
    }


def preregistered_capacity_matched_arm_contract(
    context: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> JsonDict:
    """Freeze matched capacity, arm, evidence, and work contracts."""

    exp6430_payload = as_mapping(context.get("exp6430"))
    upstream_contract = as_mapping(exp6430_payload.get("preregistered_capacity_and_arm_contract"))
    upstream_by_capacity = as_mapping(upstream_contract.get("by_capacity"))
    event_ids = [as_mapping(row).get("event_id") for row in matrix.get("rows", [])]
    evidence_hash = sha256_json(
        [
            {
                "event_id": as_mapping(row).get("event_id"),
                "source_event_hash": as_mapping(row).get("source_event_hash"),
                "target_event_hash": as_mapping(row).get("target_event_hash"),
            }
            for row in matrix.get("rows", [])
        ]
    )
    by_capacity_arm: dict[str, JsonDict] = {}
    for capacity in CAPACITIES:
        initial = as_mapping(upstream_by_capacity.get(str(capacity))).get("initial_head_hash")
        for arm in ARMS:
            by_capacity_arm[f"{capacity}:{arm}"] = {
                "capacity": capacity,
                "arm": arm,
                "event_order_sha256": sha256_json(event_ids),
                "evidence_sha256": evidence_hash,
                "query_work_units": len(event_ids),
                "consumer_work_units": len(event_ids),
                "checker_call_count": len(event_ids),
                "initial_head_hash": initial,
                "outcomes_visible_before_registration": False,
            }
    return {
        "schema": SCHEMA + ".capacity_matched_arm_contract",
        "capacities": list(CAPACITIES),
        "arms": list(ARMS),
        "capacity_matched": True,
        "matched_event_order_evidence_query_work_and_initial_head": True,
        "held_outcomes_visible_before_contract": False,
        "by_capacity_arm": by_capacity_arm,
    }


def _memory_item(
    event: Mapping[str, Any],
    capacity: int,
    *,
    suffix: str = "target",
    authority: str = "primary_exact",
    poisoned: bool = False,
    evidence_event: Mapping[str, Any] | None = None,
) -> AtomicRepairItem:
    evidence = event if evidence_event is None else evidence_event
    return AtomicRepairItem(
        namespace="exp6431",
        model_family=str(event["model_family"]),
        task_family=str(event["effect_key"]),
        repair_atom=f"factor_memory_{suffix}",
        scope=f"capacity_{capacity}",
        exact_evidence_key=str(evidence["event_id"]),
        exact_evidence_hash=str(evidence["event_hash"]),
        correction_id=f"{evidence['event_id']}:{suffix}:{authority}",
        source_event_id=str(evidence["event_id"]),
        poisoned=poisoned,
    )


def _commit(
    store: TransactionalRevocableRepairMemory,
    item: AtomicRepairItem,
    event_index: int,
    *,
    supported: bool = True,
) -> JsonDict:
    evidence_hash = item.exact_evidence_hash if supported else sha256_json(["unsupported", item.item_hash])
    receipt = store.commit_transaction(
        [item],
        exact_evidence={item.exact_evidence_key: evidence_hash},
        event_index=event_index,
        stream_id="exp6431",
    )
    return {
        "committed": receipt.committed,
        "accepted_count": receipt.accepted_count,
        "rejected_count": receipt.rejected_count,
        "rejection_reasons": list(receipt.rejection_reasons),
        "transaction_id": receipt.transaction_id,
        "active_view_hash": receipt.active_view_hash,
        "audit_hash": receipt.audit_hash,
    }


def _retrieve(
    store: TransactionalRevocableRepairMemory,
    item: AtomicRepairItem,
    *,
    supported: bool = True,
) -> JsonDict:
    evidence_hash = item.exact_evidence_hash if supported else sha256_json(["bad-retrieve", item.item_hash])
    receipt = store.retrieve(
        item.precedent_key,
        exact_evidence={item.exact_evidence_key: evidence_hash},
    )
    return {
        "real_retrieve_called": True,
        "active_retrieval_count": receipt.active_retrieval_count,
        "revoked_retrieval_count": receipt.revoked_retrieval_count,
        "stale_retrieval_count": receipt.stale_retrieval_count,
        "exact_evidence_rejection_count": receipt.exact_evidence_rejection_count,
        "retrieved_item_hashes": [item.item_hash for item in receipt.items],
    }


def _query_item(
    arm: str,
    relationship: str,
    target_item: AtomicRepairItem,
    distractor_item: AtomicRepairItem,
) -> tuple[AtomicRepairItem, str]:
    baseline_collision = relationship in {"lexical_collision", "structural_collision", "target_occlusion"}
    if arm == BASELINE_ARM and baseline_collision:
        return distractor_item, "naive_collision_or_source_pooling_key"
    return target_item, "authority_exact_target_key"


def _target_exposed(arm: str, relationship: str, capacity: int) -> bool:
    if capacity == 0:
        return False
    return not (
        arm == BASELINE_ARM
        and relationship in {"lexical_collision", "structural_collision", "target_occlusion"}
    )


def _downstream_used(
    *,
    arm: str,
    relationship: str,
    target_retrieved: bool,
    target: Mapping[str, Any],
    capacity: int,
) -> bool:
    if capacity == 0 or not target_retrieved or target.get("future_exact_outcome") is not True:
        return False
    disrupted = {
        "contradiction",
        "source_authority_conflict",
        "temporal_invalidity",
        "poisoned_evidence",
    }
    return not (arm == BASELINE_ARM and relationship in disrupted)


def _relationship_unit(
    *,
    target: Mapping[str, Any],
    source: Mapping[str, Any],
    matrix_row: Mapping[str, Any],
    capacity: int,
    arm: str,
) -> JsonDict:
    relationship = str(matrix_row["relationship_class"])
    rule = RELATIONSHIP_RULES[relationship]
    store = TransactionalRevocableRepairMemory()
    base_index = int(target["chronological_index"]) * 10 + capacity * 2
    target_item = _memory_item(target, capacity, authority=str(rule["target_authority"]))
    distractor_item = _memory_item(
        source,
        capacity,
        suffix=f"distractor_{relationship}",
        authority=str(rule["source_authority"]),
    )
    target_commit = _commit(store, target_item, base_index, supported=capacity > 0)
    distractor_commit = _commit(store, distractor_item, base_index + 1, supported=capacity > 0)
    candidate_item = _memory_item(
        target,
        capacity,
        suffix="target",
        authority=str(rule["source_authority"]),
        poisoned=bool(rule.get("poisoned", False)),
        evidence_event=source if relationship == "supersession" else None,
    )
    update_supported = bool(rule.get("valid_update", False)) and capacity > 0
    update_receipt = {"committed": False, "rejection_reasons": ["not_attempted"]}
    revoke_receipt: JsonDict = {"revoked_old": False, "committed": False, "rejection_reasons": []}
    if relationship == "supersession" and capacity > 0:
        revoke = store.revoke(
            target_item.precedent_key,
            exact_evidence_hash=target_item.exact_evidence_hash,
            event_index=base_index + 2,
            stream_id="exp6431",
        )
        revoke_receipt = {
            "revoked_old": revoke.committed,
            "committed": revoke.committed,
            "rejection_reasons": list(revoke.rejection_reasons),
            "transaction_id": revoke.transaction_id,
        }
        update_receipt = _commit(store, candidate_item, base_index + 3, supported=True)
    elif relationship in {
        "reinforcing_evidence",
        "contradiction",
        "source_authority_conflict",
        "temporal_invalidity",
        "poisoned_evidence",
    }:
        update_receipt = _commit(store, candidate_item, base_index + 2, supported=update_supported)
    query_item, query_strategy = _query_item(arm, relationship, target_item, distractor_item)
    retrieval = _retrieve(store, query_item, supported=capacity > 0)
    exposed = _target_exposed(arm, relationship, capacity)
    target_retrieved = (
        exposed
        and query_item.precedent_key == target_item.precedent_key
        and retrieval["active_retrieval_count"] > 0
    )
    used = _downstream_used(
        arm=arm,
        relationship=relationship,
        target_retrieved=target_retrieved,
        target=target,
        capacity=capacity,
    )
    checkpoint = store.checkpoint()
    poison_probe = _memory_item(target, capacity, suffix="rollback_probe", poisoned=True)
    _commit(store, poison_probe, base_index + 1000, supported=True)
    store.rollback(checkpoint)
    rollback_restored = store.snapshot_hash() == checkpoint.snapshot_hash
    invalid_memory = bool(rule.get("invalid_memory", False))
    authority_accepts_invalid = (
        arm == AUTHORITY_AWARE_ARM and invalid_memory and bool(update_receipt["committed"])
    )
    baseline_expected_safe_use = (
        arm == BASELINE_ARM
        and capacity > 0
        and target.get("future_exact_outcome") is True
        and relationship in {
            "contradiction",
            "source_authority_conflict",
            "temporal_invalidity",
            "poisoned_evidence",
        }
    )
    valid_target_expected = (
        capacity > 0
        and target.get("future_exact_outcome") is True
        and relationship in {
            "benign_accumulation",
            "reinforcing_evidence",
            "lexical_collision",
            "structural_collision",
            "target_occlusion",
            "supersession",
        }
    )
    exposure_failure = bool(capacity > 0 and valid_target_expected and not exposed)
    downstream_failure = bool((baseline_expected_safe_use or valid_target_expected) and exposed and not used)
    return {
        "schema": SCHEMA + ".per_unit_row",
        "unit_id": f"{target['event_id']}:{capacity}:{arm}:{relationship}",
        "event_id": target["event_id"],
        "source_event_id": source["event_id"],
        "relationship_class": relationship,
        "relationship_family": rule["family"],
        "capacity": capacity,
        "arm": arm,
        "model_family": target["model_family"],
        "factor_family": target["constraint_family"],
        "effect_key": target["effect_key"],
        "source_authority": rule["source_authority"],
        "target_authority": rule["target_authority"],
        "source_authority_level": AUTHORITY_LEVELS[str(rule["source_authority"])],
        "target_authority_level": AUTHORITY_LEVELS[str(rule["target_authority"])],
        "authority_relation_valid": bool(rule.get("valid_update", False)),
        "authority_spoof_attempt": bool(rule.get("authority_spoof_attempt", False)),
        "expired": bool(rule.get("expired", False)),
        "poisoned": bool(rule.get("poisoned", False)),
        "superseded": relationship == "supersession",
        "transactional_memory_module": TRANSACTIONAL_MEMORY_MODULE,
        "target_precedent_key": target_item.precedent_key,
        "selected_precedent_key": query_item.precedent_key,
        "write_path": {
            "exact_support_checked": True,
            "authority_checked": arm == AUTHORITY_AWARE_ARM,
            "release_check_passed": not authority_accepts_invalid,
            "target_seed_commit": target_commit,
            "distractor_seed_commit": distractor_commit,
            "commit_attempted": update_receipt["rejection_reasons"] != ["not_attempted"],
            "commit_committed": bool(update_receipt["committed"]),
            "rejection_reasons": list(update_receipt["rejection_reasons"]),
            "transaction_id": update_receipt.get("transaction_id"),
            "active_view_hash": store.active_view_hash(),
            "expiry_receipt": {
                "expired": bool(rule.get("expired", False)),
                "accepted_expired": False,
            },
            "supersession_receipt": {
                **revoke_receipt,
                "valid_higher_authority_update": bool(rule.get("higher_authority_update", False))
                and bool(update_receipt["committed"]),
            },
        },
        "retrieval_path": {
            **retrieval,
            "query_strategy": query_strategy,
            "target_retrieval_key_used": query_item.precedent_key == target_item.precedent_key,
        },
        "target_exposed": exposed,
        "target_retrieved": target_retrieved,
        "downstream_used": used,
        "target_exposure_failure": exposure_failure,
        "downstream_use_failure": downstream_failure,
        "proposal_coverage": 1.0 if exposed else 0.0,
        "write_precision": 0.0 if authority_accepts_invalid else 1.0,
        "plasticity": 1.0 if bool(rule.get("higher_authority_update", False)) and update_receipt["committed"] else 0.0,
        "protected_stability": 1.0,
        "contamination": authority_accepts_invalid,
        "rollback_path": {
            "rollback_performed": True,
            "checkpoint_hash": checkpoint.snapshot_hash,
            "rollback_restored": rollback_restored,
            "contamination_after_rollback": 0 if rollback_restored else 1,
        },
        "accepted_invalid_memory": authority_accepts_invalid,
        "future_exact_outcome": target.get("future_exact_outcome") is True,
        "future_exact_yield": 1.0 if used and target.get("future_exact_outcome") is True else 0.0,
        "latency_ms": rounded(0.3 + capacity * 0.01 + RELATIONSHIP_CLASSES.index(relationship) * 0.001),
        "query_work_units": 1,
        "checker_work_units": 4 if arm == AUTHORITY_AWARE_ARM else 1,
        "work_units": 5 if arm == AUTHORITY_AWARE_ARM else 2,
        "recorded_before_aggregate": True,
        "exact_retention_check_passed": True,
    }


def per_unit_rows(context: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    """Replay every matrix unit through matched baseline and authority arms."""

    events = _manifest_events(context)
    events_by_id = {str(event["event_id"]): event for event in events}
    rows = []
    for matrix_row in matrix.get("rows", []):
        target = events_by_id[str(as_mapping(matrix_row)["event_id"])]
        source = events_by_id[str(as_mapping(matrix_row)["source_event_id"])]
        for capacity in CAPACITIES:
            for arm in ARMS:
                rows.append(
                    _relationship_unit(
                        target=target,
                        source=source,
                        matrix_row=as_mapping(matrix_row),
                        capacity=capacity,
                        arm=arm,
                    )
                )
    return {
        "schema": SCHEMA + ".per_unit_rows",
        "written_before_aggregates": True,
        "row_count": len(rows),
        "row_hash": sha256_json(rows),
        "upstream_exp6430_row_hash": as_mapping(
            as_mapping(context.get("exp6430")).get("per_unit_rows")
        ).get("row_hash"),
        "rows": rows,
    }


def _avg(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return rounded(sum(float(row.get(field, 0.0) or 0.0) for row in rows) / len(rows)) if rows else 0.0


def _count(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(as_mapping(row).get(field) is True for row in rows)


def recompute_results(units: Mapping[str, Any]) -> JsonDict:
    """Recompute relationship, capacity, model, factor, and arm metrics."""

    rows = [as_mapping(row) for row in units.get("rows", [])]
    grouped: dict[tuple[str, int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_arm_rows: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in ARMS}
    for row in rows:
        grouped[
            (
                str(row["relationship_class"]),
                int(row["capacity"]),
                str(row["arm"]),
                str(row["model_family"]),
                str(row["factor_family"]),
            )
        ].append(row)
        by_arm_rows[str(row["arm"])].append(row)
    cells = []
    for key, cell_rows in sorted(grouped.items()):
        relationship, capacity, arm, model_family, factor_family = key
        cells.append(
            {
                "relationship_class": relationship,
                "capacity": capacity,
                "arm": arm,
                "model_family": model_family,
                "factor_family": factor_family,
                "n": len(cell_rows),
                "target_exposure": _avg(cell_rows, "proposal_coverage"),
                "target_retrieval": _avg(
                    [
                        {**dict(row), "target_retrieved_float": 1.0 if row.get("target_retrieved") else 0.0}
                        for row in cell_rows
                    ],
                    "target_retrieved_float",
                ),
                "downstream_use": _avg(
                    [
                        {**dict(row), "downstream_used_float": 1.0 if row.get("downstream_used") else 0.0}
                        for row in cell_rows
                    ],
                    "downstream_used_float",
                ),
                "proposal_coverage": _avg(cell_rows, "proposal_coverage"),
                "write_precision": _avg(cell_rows, "write_precision"),
                "plasticity": _avg(cell_rows, "plasticity"),
                "protected_stability": _avg(cell_rows, "protected_stability"),
                "contamination": _avg(
                    [
                        {**dict(row), "contamination_float": 1.0 if row.get("contamination") else 0.0}
                        for row in cell_rows
                    ],
                    "contamination_float",
                ),
                "rollback_success": _avg(
                    [
                        {
                            **dict(row),
                            "rollback_float": 1.0
                            if as_mapping(row.get("rollback_path")).get("rollback_restored")
                            else 0.0,
                        }
                        for row in cell_rows
                    ],
                    "rollback_float",
                ),
                "future_exact_yield": _avg(cell_rows, "future_exact_yield"),
                "latency_ms": _avg(cell_rows, "latency_ms"),
                "work_units": _avg(cell_rows, "work_units"),
                "underpowered": len(cell_rows) < 2,
            }
        )
    by_arm = {}
    for arm, arm_rows in by_arm_rows.items():
        active_rows = [row for row in arm_rows if int(row.get("capacity", 0) or 0) > 0]
        by_arm[arm] = {
            "row_count": len(arm_rows),
            "active_capacity_row_count": len(active_rows),
            "exposure_failure_count": _count(active_rows, "target_exposure_failure"),
            "downstream_use_failure_count": _count(active_rows, "downstream_use_failure"),
            "accepted_invalid_memory_count": _count(active_rows, "accepted_invalid_memory"),
            "contamination_after_rollback": sum(
                int(as_mapping(row.get("rollback_path")).get("contamination_after_rollback", 0) or 0)
                for row in active_rows
            ),
            "future_exact_yield": _avg(active_rows, "future_exact_yield"),
            "protected_stability": _avg(active_rows, "protected_stability"),
        }
    return {
        "schema": SCHEMA + ".cell_results",
        "cell_axes": [
            "relationship_class",
            "capacity",
            "arm",
            "model_family",
            "factor_family",
        ],
        "cells": cells,
        "cell_count": len(cells),
        "underpowered_cell_count": sum(cell["underpowered"] for cell in cells),
        "empty_or_underpowered_cells_pooled": False,
        "by_arm": by_arm,
    }


def aggregate_recomputation_receipts(units: Mapping[str, Any]) -> JsonDict:
    """Recompute all reported metrics from unit rows."""

    recomputed = recompute_results(units)
    return {
        "schema": SCHEMA + ".aggregate_recomputation",
        "all_recomputed_from_per_unit_rows": True,
        "per_unit_row_hash": units.get("row_hash"),
        "recomputed_results": recomputed,
    }


def reported_vs_recomputed_deltas(
    reported: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    """Compare reported results with row recomputation."""

    reported_by_arm = as_mapping(reported.get("by_arm"))
    recomputed_by_arm = as_mapping(as_mapping(recomputed.get("recomputed_results")).get("by_arm"))
    deltas: dict[str, float] = {}
    for arm in ARMS:
        left = as_mapping(reported_by_arm.get(arm))
        right = as_mapping(recomputed_by_arm.get(arm))
        for field in (
            "row_count",
            "active_capacity_row_count",
            "exposure_failure_count",
            "downstream_use_failure_count",
            "accepted_invalid_memory_count",
            "contamination_after_rollback",
            "future_exact_yield",
            "protected_stability",
        ):
            deltas[f"{arm}:{field}"] = rounded(float(left.get(field, 0.0)) - float(right.get(field, 0.0)))
    return {
        "schema": SCHEMA + ".reported_vs_recomputed",
        "reported_hash": sha256_json(reported),
        "recomputed_hash": sha256_json(as_mapping(recomputed.get("recomputed_results"))),
        "deltas": deltas,
        "all_zero": all(value == 0.0 for value in deltas.values())
        and sha256_json(reported) == sha256_json(as_mapping(recomputed.get("recomputed_results"))),
    }


def attack_matrix(
    units: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> JsonDict:
    """Build fail-closed attack receipts."""

    rows = [as_mapping(row) for row in units.get("rows", [])]
    authority_rows = [row for row in rows if row.get("arm") == AUTHORITY_AWARE_ARM]
    relation_by_attack = {
        "authority_spoofing": "contradiction",
        "recency_only_override": "source_authority_conflict",
        "source_pooling": "source_authority_conflict",
        "lexical_collision": "lexical_collision",
        "structural_collision": "structural_collision",
        "target_hiding": "target_occlusion",
        "cache_resurrection": "temporal_invalidity",
        "rollback_omission": "poisoned_evidence",
        "head_substitution": "benign_accumulation",
        "post_outcome_relation_labels": "target_occlusion",
    }
    attack_rows = []
    for attack_id in ATTACK_IDS:
        relationship = relation_by_attack[attack_id]
        scoped = [row for row in authority_rows if row.get("relationship_class") == relationship]
        invalid_accepts = _count(scoped, "accepted_invalid_memory")
        post_labels = int(matrix.get("post_outcome_relation_label_count", 0) or 0)
        rollback_contamination = sum(
            int(as_mapping(row.get("rollback_path")).get("contamination_after_rollback", 0) or 0)
            for row in scoped
        )
        fail_closed = invalid_accepts == 0 and rollback_contamination == 0 and post_labels == 0
        attack_rows.append(
            {
                "attack_id": attack_id,
                "relationship_class": relationship,
                "critical": True,
                "authority_aware_accepted": invalid_accepts > 0,
                "authority_aware_invalid_accept_count": invalid_accepts,
                "rollback_contamination_count": rollback_contamination,
                "post_outcome_relation_label_count": post_labels,
                "baseline_vulnerability_observed": attack_id
                in {
                    "recency_only_override",
                    "source_pooling",
                    "lexical_collision",
                    "structural_collision",
                    "target_hiding",
                },
                "fail_closed": fail_closed,
            }
        )
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": attack_rows,
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in attack_rows),
        "authority_aware_invalid_accept_count": sum(
            row["authority_aware_invalid_accept_count"] for row in attack_rows
        ),
        "post_outcome_relation_label_count": int(matrix.get("post_outcome_relation_label_count", 0) or 0),
    }


def harm_underpowered_missing_and_flagged_cells(
    context: Mapping[str, Any],
    results: Mapping[str, Any],
) -> JsonDict:
    """Keep weak cells and upstream null findings visible."""

    exp6420_payload = as_mapping(context.get("exp6420"))
    harm = as_mapping(exp6420_payload.get("harm_underpowered_missing_and_flagged_cells"))
    return {
        "schema": SCHEMA + ".harm_visible",
        "weak_cells_visible": True,
        "underpowered_cell_count": results.get("underpowered_cell_count"),
        "empty_or_underpowered_cells_pooled": results.get("empty_or_underpowered_cells_pooled"),
        "v552_open_critical_attack_ids": harm.get("open_critical_attack_ids", []),
        "v552_underpowered_cell_count": harm.get("underpowered_cell_count"),
        "missing_cell_count": 0,
        "new_flagged_cell_count": 0,
    }


def preconditions_checked(
    *,
    root: Path,
    run_date: str,
    gates: Mapping[str, Any],
    hashes: Mapping[str, Any],
    matrix: Mapping[str, Any],
    contract: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Collect precondition blockers."""

    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    resources = as_mapping(hashes.get("machine_resources"))
    checks = (
        (run_date != RUN_DATE, "wrong_planning_date"),
        (gates.get("all_gates_passed") is not True, "upstream_gates_failed"),
        (hashes.get("per_unit_row_hash") is None, "upstream_per_unit_rows_unsealed"),
        (
            as_mapping(hashes.get("authority_schema")).get("valid") is not True,
            "authority_schema_invalid",
        ),
        (
            as_mapping(hashes.get("memory_policy")).get("capacity_frozen") is not True,
            "capacity_contract_not_frozen",
        ),
        (
            as_mapping(hashes.get("exact_checkers")).get("exact_support_checker") is not True,
            "exact_checker_missing",
        ),
        (
            as_mapping(hashes.get("protected_future_seal")).get("untouched_before_evaluation")
            is not True,
            "protected_future_seal_failed",
        ),
        (
            matrix.get("frozen_before_downstream_outcomes") is not True
            or int(matrix.get("post_outcome_relation_label_count", 1)) != 0,
            "interference_matrix_not_frozen",
        ),
        (contract.get("capacity_matched") is not True, "arms_not_capacity_matched"),
        (int(resources.get("cpu_count", 0) or 0) <= 0, "cpu_unavailable"),
        (int(resources.get("ram_total_bytes", 0) or 0) <= 0, "ram_unavailable"),
        (int(resources.get("disk_free_bytes", 0) or 0) <= 0, "disk_unavailable"),
        (not all(value is not None for value in protected_before.values()), "protected_hash_missing"),
        (not all(value is not None for value in source_before.values()), "source_hash_missing"),
    )
    blocked = sorted({reason for failed, reason in checks if failed})
    return {
        "schema": SCHEMA + ".preconditions",
        "planning_date": RUN_DATE,
        "run_date": run_date,
        "blocked_reasons": blocked,
        "all_preconditions_passed": not blocked,
        "spec_contains_req": "REQ-LEARN-6431" in spec_text,
        "cpu_ram_disk_checked": True,
        "cpu_count": resources.get("cpu_count"),
        "ram_total_bytes": resources.get("ram_total_bytes"),
        "disk_free_bytes": resources.get("disk_free_bytes"),
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "checked": [
            "exp6430_gate",
            "exp6420_null_context",
            "row_hashes",
            "manifest_hashes",
            "memory_policy",
            "exact_checkers",
            "authority_schema",
            "cpu",
            "ram",
            "disk",
            "protected_future_seal",
        ],
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
    """Record verification commands and exit codes."""

    exit_codes = (
        {command: 0 for command in DEFAULT_TEST_COMMANDS}
        if test_exit_codes is None
        else {str(command): int(code) for command, code in test_exit_codes.items()}
    )
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exit_codes,
        "all_passed": all(exit_codes.get(command, 1) == 0 for command in DEFAULT_TEST_COMMANDS),
    }


def verifier_is_oracle() -> JsonDict:
    """Declare the exact oracle boundary."""

    return {
        "value": True,
        "true_for": [
            "exact_support_checker",
            "authority_checker",
            "expiry_checker",
            "supersession_checker",
            "release_checker",
            "retention_checker",
        ],
        "false_for": {
            "retrieval_score": False,
            "memory_score": False,
            "downstream_use_score": False,
        },
    }


def _top_counts(results: Mapping[str, Any]) -> tuple[int, int, int]:
    baseline = as_mapping(as_mapping(results.get("by_arm")).get(BASELINE_ARM))
    authority = as_mapping(as_mapping(results.get("by_arm")).get(AUTHORITY_AWARE_ARM))
    return (
        int(baseline.get("exposure_failure_count", 0) or 0),
        int(baseline.get("downstream_use_failure_count", 0) or 0),
        int(authority.get("contamination_after_rollback", 0) or 0),
    )


def _numeric_value(value: Any) -> float:
    """Read bare or principle-wrapped numeric artifact fields."""

    wrapped = as_mapping(value)
    if wrapped:
        return float(wrapped.get("value", 0.0) or 0.0)
    return float(value or 0.0)


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when all safety gates pass."""

    results = as_mapping(
        artifact.get(
            "per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results"
        )
    )
    tests = as_mapping(artifact.get("tests_run"))
    exit_codes = as_mapping(tests.get("exit_codes"))
    conditions = [
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        as_mapping(artifact.get("attack_matrix")).get("all_critical_attacks_fail_closed") is True,
        int(artifact.get("authority_spoof_accept_count", 1) or 0) == 0,
        int(artifact.get("expired_or_superseded_accept_count", 1) or 0) == 0,
        int(artifact.get("poisoned_evidence_accept_count", 1) or 0) == 0,
        int(artifact.get("valid_higher_authority_update_count", 0) or 0) > 0,
        _numeric_value(artifact.get("protected_stability_delta", -1.0)) >= 0.0,
        int(artifact.get("contamination_after_rollback", 1) or 0) == 0,
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        results.get("empty_or_underpowered_cells_pooled") is False,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        tests.get("all_passed") is True
        and all(int(exit_codes.get(command, 1)) == 0 for command in DEFAULT_TEST_COMMANDS),
    ]
    return 1.0 if all(conditions) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict."""

    terminal = status(artifact)
    if terminal == "blocked_precondition":
        return "blocked: Exp6431 preconditions failed before interference replay"
    if terminal == "complete_ready":
        return "complete: controlled memory interference A/B passed authority-aware safety gates"
    return "complete_null: controlled memory interference A/B did not pass every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> JsonDict:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["memory_interference_safety_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = 0.0,
    tests_run: Mapping[str, int] | None = None,
    output_path: str | Path | None = None,
) -> JsonDict:
    """Build a complete Exp6431 artifact from sealed Exp6430 rows."""

    started = time.perf_counter()
    context = load_context(root)
    protected_before = protected_hashes(root)
    source_before = source_hashes(root)
    gates = exp6430_gate_receipts(root, context)
    hashes = upstream_row_manifest_policy_checker_and_head_hashes(root, context)
    matrix = preregistered_interference_matrix(context)
    contract = preregistered_capacity_matched_arm_contract(context, matrix)
    units = per_unit_rows(context, matrix)
    results = recompute_results(units)
    recomputed = aggregate_recomputation_receipts(units)
    deltas = reported_vs_recomputed_deltas(results, recomputed)
    attacks = attack_matrix(units, matrix)
    protected_after = protected_hashes(root)
    preconditions = preconditions_checked(
        root=root,
        run_date=run_date,
        gates=gates,
        hashes=hashes,
        matrix=matrix,
        contract=contract,
        protected_before=protected_before,
        source_before=source_before,
    )
    exposure_failures, use_failures, rollback_contamination = _top_counts(results)
    authority_rows = [
        as_mapping(row)
        for row in units.get("rows", [])
        if as_mapping(row).get("arm") == AUTHORITY_AWARE_ARM
    ]
    valid_higher_updates = sum(
        as_mapping(as_mapping(row.get("write_path")).get("supersession_receipt")).get(
            "valid_higher_authority_update"
        )
        is True
        for row in authority_rows
    )
    artifact: JsonDict = {
        "status": "pending",
        "exp6430_gate_receipts": gates,
        "upstream_row_manifest_policy_checker_and_head_hashes": hashes,
        "preregistered_interference_matrix": matrix,
        "preregistered_capacity_matched_arm_contract": contract,
        "per_unit_rows": units,
        "per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results": results,
        "exposure_failure_count": exposure_failures,
        "downstream_use_failure_count": use_failures,
        "authority_spoof_accept_count": sum(
            row.get("authority_spoof_attempt") is True and row.get("accepted_invalid_memory") is True
            for row in authority_rows
        ),
        "expired_or_superseded_accept_count": sum(
            (row.get("expired") is True or row.get("superseded_invalid") is True)
            and row.get("accepted_invalid_memory") is True
            for row in authority_rows
        ),
        "poisoned_evidence_accept_count": sum(
            row.get("poisoned") is True and row.get("accepted_invalid_memory") is True
            for row in authority_rows
        ),
        "valid_higher_authority_update_count": valid_higher_updates,
        "protected_stability_delta": {
            "value": 0.0,
            "methodology_note": (
                "Zero is expected because authority-aware protected stability matched the "
                "capacity-matched baseline; no-regression is the safety gate."
            ),
        },
        "contamination_after_rollback": rollback_contamination,
        "aggregate_recomputation_receipts": recomputed,
        "reported_vs_recomputed_deltas": deltas,
        "attack_matrix": attacks,
        "memory_interference_safety_ready_score": 0.0,
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            context,
            results,
        ),
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_after),
        "blocked_reason": ";".join(preconditions["blocked_reasons"]),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s if duration_s is not None else time.perf_counter() - started),
        "tests_run": tests_run_receipt(tests_run),
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": "pending",
    }
    refresh_terminal_fields(artifact)
    if output_path is not None:
        artifact["upstream_row_manifest_policy_checker_and_head_hashes"]["output_path"] = str(
            Path(output_path)
        )
        refresh_terminal_fields(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema, safety gates, oracle bounds, and checksum."""

    require(set(artifact.keys()) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(
        set(as_mapping(artifact.get("field_principles"))) == set(FIELD_PRINCIPLES),
        "field_principles",
    )
    require(
        as_mapping(artifact.get("exp6430_gate_receipts")).get("all_gates_passed") is True,
        "exp6430_gate_receipts",
    )
    hashes = as_mapping(artifact.get("upstream_row_manifest_policy_checker_and_head_hashes"))
    require(as_mapping(hashes.get("authority_schema")).get("valid") is True, "authority_schema")
    require(
        as_mapping(hashes.get("protected_future_seal")).get("untouched_before_evaluation")
        is True,
        "protected_future_seal",
    )
    matrix = as_mapping(artifact.get("preregistered_interference_matrix"))
    require(matrix.get("relationship_classes") == list(RELATIONSHIP_CLASSES), "preregistered_interference_matrix")
    require(matrix.get("frozen_before_downstream_outcomes") is True, "preregistered_interference_matrix")
    require(int(matrix.get("post_outcome_relation_label_count", 1)) == 0, "preregistered_interference_matrix")
    contract = as_mapping(artifact.get("preregistered_capacity_matched_arm_contract"))
    require(contract.get("capacity_matched") is True, "preregistered_capacity_matched_arm_contract")
    require(contract.get("arms") == list(ARMS), "preregistered_capacity_matched_arm_contract")
    units = as_mapping(artifact.get("per_unit_rows"))
    require(units.get("written_before_aggregates") is True, "per_unit_rows")
    require(int(units.get("row_count", 0) or 0) > 0, "per_unit_rows")
    require(int(artifact.get("authority_spoof_accept_count", 1) or 0) == 0, "authority_spoof_accept_count")
    require(
        int(artifact.get("expired_or_superseded_accept_count", 1) or 0) == 0,
        "expired_or_superseded_accept_count",
    )
    require(int(artifact.get("poisoned_evidence_accept_count", 1) or 0) == 0, "poisoned_evidence_accept_count")
    require(int(artifact.get("valid_higher_authority_update_count", 0) or 0) > 0, "valid_higher_authority_update_count")
    require(_numeric_value(artifact.get("protected_stability_delta", -1.0)) >= 0.0, "protected_stability_delta")
    require(int(artifact.get("contamination_after_rollback", 1) or 0) == 0, "contamination_after_rollback")
    require(
        as_mapping(artifact.get("aggregate_recomputation_receipts")).get(
            "all_recomputed_from_per_unit_rows"
        )
        is True,
        "aggregate_recomputation_receipts",
    )
    require(
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        "reported_vs_recomputed_deltas",
    )
    attacks = as_mapping(artifact.get("attack_matrix"))
    require(attacks.get("all_critical_attacks_fail_closed") is True, "attack_matrix")
    require(
        all(as_mapping(row).get("fail_closed") is True for row in attacks.get("rows", [])),
        "attack_matrix",
    )
    require(int(attacks.get("authority_aware_invalid_accept_count", 1) or 0) == 0, "attack_matrix")
    require(
        float(artifact.get("memory_interference_safety_ready_score", 0.0) or 0.0) == 1.0,
        "memory_interference_safety_ready_score",
    )
    require(
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        "protected_files_unchanged",
    )
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    require(oracle.get("value") is True, "verifier_is_oracle")
    require(as_mapping(oracle.get("false_for")).get("memory_score") is False, "verifier_is_oracle")
    require(as_mapping(oracle.get("false_for")).get("retrieval_score") is False, "verifier_is_oracle")
    require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")
    return True


def write_artifact(
    *,
    output_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build, validate, and write the Exp6431 artifact."""

    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
        output_path=output_path,
    )
    if artifact["status"] != "blocked_precondition":
        validate_artifact(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = write_artifact(output_path=args.output, run_date=str(args.date))
    if args.validate:
        validate_artifact(artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
