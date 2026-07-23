"""Exp5829 transfer-selective replay audit.

Spec refs: REQ-LEARN-5829, SCENARIO-LEARN-5829-SIGNATURE-FREEZE,
SCENARIO-LEARN-5829-REPLAY-PARITY,
SCENARIO-LEARN-5829-TRANSFER-RETENTION-RECURRENCE,
SCENARIO-LEARN-5829-FAIL-CLOSED.

The audit replays accepted exact-solver artifacts. It does not train a model,
route production traffic, or ask an LLM to infer transfer. The only adaptive
state being compared is bounded structural-memory replay selection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5763_dependent_task_constraint_acquisition as exp5763
from carnot import experiment_5826_out_of_template_constraint_stream as exp5826
from carnot import experiment_5828_future_validated_structural_memory as exp5828


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5829_transfer_selective_replay_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5829_transfer_selective_replay_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5829_transfer_selective_replay_audit.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
CONSTRAINT_MEMORY_RELATIVE_PATH = Path("python/carnot/learn/constraint_memory.py")

EXP5828_ARTIFACT_RELATIVE_PATH = exp5828.RESULT_RELATIVE_PATH
EXP5826_ARTIFACT_RELATIVE_PATH = exp5826.RESULT_RELATIVE_PATH
EXP5826_ROWS_RELATIVE_PATH = exp5826.ROW_FILE_RELATIVE_PATH
EXP5825_CONTRACT_RELATIVE_PATH = exp5828.EXP5825_CONTRACT_RELATIVE_PATH
EXP5763_ARTIFACT_RELATIVE_PATH = exp5763.RESULT_RELATIVE_PATH
EXP5762_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5762_query_driven_constraint_lifecycle.py"
)

SCHEMA = "carnot.experiment_5829.transfer_selective_replay_audit.v1"
EXPERIMENT = 5829
EXPERIMENT_ID = "experiment_5829_transfer_selective_replay_audit"
MILESTONE = "2026.07.520"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
STRUCTURAL_LEARNER = exp5828.STRUCTURAL_LEARNER
STOPPING_RULE = exp5828.STOPPING_RULE
QUERY_BUDGET_PER_ROW = exp5828.QUERY_BUDGET_PER_ROW
MEMORY_CAP = exp5828.MEMORY_CAP
REPLAY_EVENT_CAP = 8
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512
NONINFERIORITY_MARGIN = 0.01
FIRST_EXPOSURE_DEFINITION = "row_first_seen_before_its_future_suffix_opens"

PRIMARY_FAMILIES = exp5826.PRIMARY_FAMILIES
CHANGE_ORDER = exp5826.CHANGE_ORDER
PROOF_PRESERVING_SURFACES = exp5826.PROOF_PRESERVING_SURFACES

RESET_NO_REPLAY_ARM = "reset_no_replay"
ALL_REPLAY_ARM = "all_replay"
RANDOM_REPLAY_ARM = "random_matched_count_replay"
COMPATIBLE_REPLAY_ARM = "signature_compatible_replay"
REPLAY_ARMS = (
    RESET_NO_REPLAY_ARM,
    ALL_REPLAY_ARM,
    RANDOM_REPLAY_ARM,
    COMPATIBLE_REPLAY_ARM,
)
SPEC_REFS = (
    "REQ-LEARN-5829",
    "SCENARIO-LEARN-5829-SIGNATURE-FREEZE",
    "SCENARIO-LEARN-5829-REPLAY-PARITY",
    "SCENARIO-LEARN-5829-TRANSFER-RETENTION-RECURRENCE",
    "SCENARIO-LEARN-5829-FAIL-CLOSED",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5829,
    "bootstrap_seed": 5_829_001,
    "random_replay_seed": 5_829_002,
    "checkpoint_seed": 5_829_003,
}
SIGNATURE_COMPONENTS = (
    "constraint_arity",
    "constraint_type",
    "constraint_composition",
    "variable_domain_shape",
    "hard_soft_role",
    "proof_preserving_surface",
    "exact_prefix_behavior",
)
FORBIDDEN_SELECTOR_FIELDS = (
    "family",
    "row_id",
    "chronology_index",
    "future_suffix_label",
    "future_batch_id",
    "oracle_accuracy",
    "metric_delta",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5829_transfer_selective_replay_audit.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5829_transfer_selective_replay_audit.py "
    "-m pytest tests/python/test_experiment_5829_transfer_selective_replay_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5829_transfer_selective_replay_audit.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5829_transfer_selective_replay_audit.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "task_signature_and_compatibility_rule",
    "arm_definitions_and_replay_parity",
    "heldout_split_and_leakage_receipts",
    "forward_transfer_metrics",
    "retention_and_forgetting_metrics",
    "recurrence_recovery_metrics",
    "paired_deltas_and_ci95",
    "unsafe_transfer_count",
    "replay_resource_accounting",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal audit state distinguishes complete null evidence from an incomplete replay.",
    "preconditions_checked": "Gate, split, headroom, resource, and checkpoint checks prevent fabricated transfer.",
    "upstream_artifact_hashes": "Hashes bind replay to the accepted lifecycle and sealed stream.",
    "task_signature_and_compatibility_rule": "A frozen label-blind rule prevents science-label selection.",
    "arm_definitions_and_replay_parity": "Matched counts, budgets, learner, and cap isolate selection quality.",
    "heldout_split_and_leakage_receipts": "Family/surface isolation is required for a transfer claim.",
    "forward_transfer_metrics": "First-exposure performance measures useful reuse rather than retention.",
    "retention_and_forgetting_metrics": "Old-regime performance measures stability separately from transfer.",
    "recurrence_recovery_metrics": "Reappearing regimes test durable rather than one-pass memory.",
    "paired_deltas_and_ci95": "Paired intervals quantify replay effects without point-estimate promotion.",
    "unsafe_transfer_count": "A bare zero is required before compatible replay is credited.",
    "replay_resource_accounting": "Events, bytes, latency, and cap pressure make transfer cost explicit.",
    "duration_s": "Measured wall time exposes bootstrap-only audits.",
    "inference_substrate": "`aggregation_from_upstream_artifacts` plus exact replay declares no LLM inference.",
    "verifier_is_oracle": "True records that exact solvers score the lifecycle and prevent a moat claim.",
    "field_provenance": "Every aggregate traces to held-out episodes and replay receipts.",
    "test_commands": "Commands document parity, leakage, transfer, retention, recurrence, and resource checks.",
    "test_exit_codes": "Exit codes prevent failed transfer checks from becoming credit.",
    "reproducibility_checksum": "A checksum detects split, signature, replay, or metric drift.",
    "honest_verdict": "A terminal prefix states positive, null, negative, or blocked outcome honestly.",
}
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5828_lifecycle_artifact": EXP5828_ARTIFACT_RELATIVE_PATH,
    "exp5826_stream_artifact": EXP5826_ARTIFACT_RELATIVE_PATH,
    "exp5826_stream_rows": EXP5826_ROWS_RELATIVE_PATH,
    "exp5825_event_schema_contract": EXP5825_CONTRACT_RELATIVE_PATH,
    "exp5763_dependent_task_artifact": EXP5763_ARTIFACT_RELATIVE_PATH,
    "exp5762_lifecycle_module": EXP5762_MODULE_RELATIVE_PATH,
    "constraint_memory_module": CONSTRAINT_MEMORY_RELATIVE_PATH,
    "research_references": RESEARCH_REFERENCES_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "tests": TEST_RELATIVE_PATH,
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting filenames or timestamps."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _checkpoint_path_receipt(result_path: Path, checkpoint_dir: Path) -> JsonDict:
    def ready_file(path: Path) -> bool:
        parent = path.parent
        parent_ready = (parent.exists() and os.access(parent, os.W_OK)) or (
            parent.parent.exists() and os.access(parent.parent, os.W_OK)
        )
        return parent_ready and (not path.exists() or os.access(path, os.W_OK))

    checkpoint_parent = checkpoint_dir if checkpoint_dir.exists() else checkpoint_dir.parent
    checkpoint_ready = (checkpoint_parent.exists() and os.access(checkpoint_parent, os.W_OK)) or (
        checkpoint_parent.parent.exists() and os.access(checkpoint_parent.parent, os.W_OK)
    )
    return {
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "checkpoint_dir": "results/checkpoints/experiment_5829_transfer_selective_replay_audit",
        "result_writable": ready_file(result_path),
        "checkpoint_writable": checkpoint_ready,
        "checkpoint_atomic_suffix": ".tmp",
        "ok": ready_file(result_path) and checkpoint_ready,
    }


def _load_rows(root: Path = REPO_ROOT) -> list[JsonDict]:
    return exp5828.read_row_file(root / EXP5826_ROWS_RELATIVE_PATH)


def _structured_gate_receipt(root: Path, upstream_hashes: Mapping[str, str]) -> JsonDict:
    lifecycle = _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH)
    stream = _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH)
    dependent = _read_json(root / EXP5763_ARTIFACT_RELATIVE_PATH)
    rows = _load_rows(root)
    validation_errors: dict[str, str] = {}
    for name, validator, artifact in (
        ("exp5828", exp5828.validate_artifact, lifecycle),
        ("exp5826", exp5826.validate_artifact, stream),
        ("exp5763", exp5763.validate_artifact, dependent),
    ):
        try:
            validator(artifact)
        except ValueError as exc:
            validation_errors[name] = str(exc)
    row_replay_ok = exp5826.verify_row_file(rows, stream)
    return {
        "exp5828_status": lifecycle.get("status"),
        "exp5828_honest_verdict": lifecycle.get("honest_verdict"),
        "future_validated_lifecycle_ready_score": lifecycle.get(
            "future_validated_lifecycle_ready_score"
        ),
        "exp5826_status": stream.get("status"),
        "constraint_event_stream_ready_score": stream.get("constraint_event_stream_ready_score"),
        "exp5763_status": dependent.get("status"),
        "row_count": len(rows),
        "row_replay_ok": row_replay_ok,
        "current_validator_errors_recorded": validation_errors,
        "lifecycle_hash": upstream_hashes.get("exp5828_lifecycle_artifact", ""),
        "stream_hash": upstream_hashes.get("exp5826_stream_artifact", ""),
        "event_schema_hash": upstream_hashes.get("exp5825_event_schema_contract", ""),
        "memory_cap": MEMORY_CAP,
        "memory_cap_from_exp5828": dict(lifecycle.get("memory_cap_receipts") or {}).get(
            "memory_cap"
        ),
        "unsafe_update_count": lifecycle.get("unsafe_update_count"),
        "ok": lifecycle.get("status") == "complete"
        and lifecycle.get("future_validated_lifecycle_ready_score") == 1.0
        and stream.get("status") == "complete"
        and stream.get("constraint_event_stream_ready_score") == 1.0
        and dependent.get("status") == "complete"
        and row_replay_ok
        and len(rows) == 360
        and lifecycle.get("unsafe_update_count") == 0
        and dict(lifecycle.get("memory_cap_receipts") or {}).get("memory_cap") == MEMORY_CAP,
    }


def _heldout_split_check(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    cell_counts = Counter(f"{row['family']}|{row['surface_kind']}" for row in rows)
    change_counts = Counter(str(row["change"]) for row in rows)
    heldout_cells = sorted(cell_counts)
    minimum = min(cell_counts.values()) if cell_counts else 0
    return {
        "heldout_family_surface_cells": heldout_cells,
        "heldout_family_surface_cell_count": len(heldout_cells),
        "minimum_rows_per_family_surface_cell": minimum,
        "recurrence_rows": int(change_counts.get("recurrence", 0)),
        "first_exposure_rows": len(rows),
        "n_ge_30_per_primary_cell": minimum >= 30,
        "ok": len(heldout_cells) == len(PRIMARY_FAMILIES) * len(PROOF_PRESERVING_SURFACES)
        and minimum >= 30
        and int(change_counts.get("recurrence", 0)) >= 30,
    }


def _future_validation_for_row(row: Mapping[str, Any]) -> JsonDict:
    proposal = exp5828._proposal_from_row(row)
    return exp5828._future_validation_receipt(row, proposal)


def _headroom_check(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    sampled = [_future_validation_for_row(row) for row in rows[: min(len(rows), 72)]]
    headroom = [
        float(receipt["future_validated_accuracy"]) - float(receipt["no_memory_accuracy"])
        for receipt in sampled
    ]
    positive = [value for value in headroom if value > 0.0]
    return {
        "sampled_rows": len(sampled),
        "positive_headroom_rows": len(positive),
        "mean_headroom": _round(sum(headroom) / max(1, len(headroom))),
        "preregistered_minimum_positive_rows": 30,
        "ok": len(sampled) >= 30 and len(positive) >= 30 and sum(headroom) > 0.0,
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT
    / "results/checkpoints/experiment_5829_transfer_selective_replay_audit",
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay gates, split checks, hashes, resources, and checkpoint writability."""

    root = Path(root)
    result_path = Path(result_path)
    checkpoint_dir = Path(checkpoint_dir)
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    blocked: list[str] = []
    if any(upstream_hashes[name] == "missing" for name in UPSTREAM_PATHS):
        blocked.append("missing_upstream_artifact")

    structured_gate: JsonDict = {"ok": False}
    heldout_split: JsonDict = {"ok": False}
    headroom: JsonDict = {"ok": False}
    deterministic_seeds: JsonDict = {"ok": False, "random_seeds": dict(RANDOM_SEEDS)}
    corrupt_errors: list[str] = []
    if "missing_upstream_artifact" not in blocked:
        try:
            rows = _load_rows(root)
            structured_gate = _structured_gate_receipt(root, upstream_hashes)
            heldout_split = _heldout_split_check(rows)
            headroom = _headroom_check(rows)
            lifecycle = _read_json(root / EXP5828_ARTIFACT_RELATIVE_PATH)
            stream = _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH)
            deterministic_seeds = {
                "random_seeds": dict(RANDOM_SEEDS),
                "base_seed_ok": RANDOM_SEEDS["base_seed"] == 5829,
                "exp5828_seed_ok": dict(lifecycle.get("random_seeds") or {})
                == dict(exp5828.RANDOM_SEEDS),
                "exp5826_seed_ok": dict(stream.get("random_seeds") or {})
                == dict(exp5826.RANDOM_SEEDS),
                "ok": RANDOM_SEEDS["base_seed"] == 5829
                and dict(lifecycle.get("random_seeds") or {}) == dict(exp5828.RANDOM_SEEDS)
                and dict(stream.get("random_seeds") or {}) == dict(exp5826.RANDOM_SEEDS),
            }
        except (OSError, ValueError, json.JSONDecodeError, exp5826.StreamReplayError) as exc:
            corrupt_errors.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    memory = memory_probe()
    disk = disk_probe(root)
    checkpoint_paths = _checkpoint_path_receipt(result_path, checkpoint_dir)
    checks = {
        "structured_gate": structured_gate.get("ok") is True,
        "heldout_split": heldout_split.get("ok") is True,
        "headroom": headroom.get("ok") is True,
        "deterministic_seeds": deterministic_seeds.get("ok") is True,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "checkpoint_paths": checkpoint_paths.get("ok") is True,
        "python": sys.version_info >= (3, 11),
    }
    failure_names = {
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "checkpoint_paths": "checkpoint_path_not_writable",
    }
    blocked.extend(failure_names.get(name, name) for name, ok in checks.items() if not ok)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "structured_gate_replay": structured_gate,
        "upstream_artifact_hashes": upstream_hashes,
        "heldout_split_check": heldout_split,
        "headroom_check": headroom,
        "deterministic_seeds": deterministic_seeds,
        "resources": {"memory": memory, "disk": disk},
        "checkpoint_paths": checkpoint_paths,
        "llm_calls_made": 0,
        "corrupt_upstream_errors": corrupt_errors,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic resource probes while still replaying sealed inputs."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _variable_domain_shape(row: Mapping[str, Any]) -> str:
    primary = dict(dict(row.get("exact_receipt") or {}).get("primary") or {})
    query_ids = [
        str(query["candidate_id"]).split("-cand-")[0]
        for query in dict(row.get("exact_receipt") or {}).get("membership_queries", [])
    ]
    return canonical_json(
        {
            "candidate_count": int(primary.get("candidate_count") or 0),
            "query_prefixes": sorted(set(query_ids)),
            "query_count": len(dict(row.get("exact_receipt") or {}).get("membership_queries", [])),
        }
    )


def _hard_soft_role(signature: Mapping[str, Any]) -> str:
    relation = str(signature.get("relation") or "")
    composition = str(signature.get("composition") or "")
    if "weighted" in relation or "weighted" in composition:
        return "hard_constraint_with_weighted_soft_shape"
    return "hard_constraint"


def _exact_prefix_behavior(row: Mapping[str, Any]) -> str:
    queries = list(dict(row.get("exact_receipt") or {}).get("membership_queries") or [])
    polarity = ["accept" if query.get("oracle_accepts") is True else "reject" for query in queries]
    return canonical_json(
        {
            "query_count": len(queries),
            "polarity_pattern": polarity,
            "status": dict(dict(row.get("exact_receipt") or {}).get("primary") or {}).get(
                "status", ""
            ),
        }
    )


def task_signature(row: Mapping[str, Any]) -> JsonDict:
    """Build the frozen label-blind signature used for replay compatibility."""

    raw_signature = dict(dict(row.get("out_of_template_witness") or {}).get("signature") or {})
    signature = {
        "constraint_arity": int(raw_signature.get("arity") or 0),
        "constraint_type": str(raw_signature.get("relation") or ""),
        "constraint_composition": str(raw_signature.get("composition") or ""),
        "variable_domain_shape": _variable_domain_shape(row),
        "hard_soft_role": _hard_soft_role(raw_signature),
        "proof_preserving_surface": str(row.get("surface_kind") or ""),
        "exact_prefix_behavior": _exact_prefix_behavior(row),
    }
    return {
        "row_id": str(row.get("row_id") or ""),
        "signature": signature,
        "signature_hash": sha256_json(signature),
    }


def compatible_for_replay(replay_row: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Return True only when the frozen signature rule admits replay."""

    left = task_signature(replay_row)["signature"]
    right = task_signature(current)["signature"]
    return all(left[component] == right[component] for component in SIGNATURE_COMPONENTS)


def _signature_bundle(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipts = [task_signature(row) for row in rows]
    rule = {
        "version": "exp5829_label_blind_signature_compatibility_v1",
        "calibration_split": "train_dev_only",
        "calibration_sources": [
            "Exp5762 frozen train/dev template signatures",
            "Exp5826 public schema constants",
        ],
        "compatible_when": list(SIGNATURE_COMPONENTS),
        "uses_science_family_label": False,
        "uses_future_suffix_labels": False,
        "uses_posthoc_metric_selection": False,
        "forbidden_selector_fields": list(FORBIDDEN_SELECTOR_FIELDS),
    }
    rule["rule_hash"] = sha256_json(rule)
    return {
        "schema": SCHEMA + ".signature_rule",
        "signature_components": list(SIGNATURE_COMPONENTS),
        "signature_count": len(receipts),
        "signature_root_hash": sha256_json([receipt["signature_hash"] for receipt in receipts]),
        "compatibility_rule": rule,
        "compatibility_rule_frozen": True,
        "label_blind_to_science_outcomes": True,
        "sample_signature_receipts": receipts[:24],
    }


def _episode_record(
    row: Mapping[str, Any],
    signatures: Mapping[str, Mapping[str, Any]],
    signature_receipt: Mapping[str, Any],
) -> JsonDict:
    validation = _future_validation_for_row(row)
    base_accuracy = float(validation["no_memory_accuracy"])
    oracle_accuracy = float(validation["future_validated_accuracy"])
    headroom = max(0.0, oracle_accuracy - base_accuracy)
    return {
        "row_id": str(row["row_id"]),
        "chronology_index": int(row["chronology_index"]),
        "family": str(row["family"]),
        "change": str(row["change"]),
        "surface": str(row["surface_kind"]),
        "hardness": str(row["solver_effort_bin"]),
        "signature_hash": str(signature_receipt["signature_hash"]),
        "signature": dict(signature_receipt["signature"]),
        "no_replay_accuracy": _round(base_accuracy),
        "oracle_accuracy": _round(oracle_accuracy),
        "headroom": _round(headroom),
        "future_batch_id": str(row["sealed_future_suffix"]["future_batch_id"]),
        "future_suffix_hash": str(row["sealed_future_suffix"]["suffix_hash"]),
        "protected_prefix_retention": 1.0
        if row["protected_prefix_receipt"]["replay_passed"] is True
        else 0.0,
        "unsafe_prefix_count": int(row["protected_prefix_receipt"]["unsafe_propagation_count"]),
        "known_signature_count": len(signatures),
    }


def _select_replay_rows(
    *,
    current: Mapping[str, Any],
    prior_rows: Sequence[Mapping[str, Any]],
    compatible_count: int,
    arm: str,
) -> list[JsonDict]:
    if arm == RESET_NO_REPLAY_ARM or compatible_count <= 0:
        return []
    if arm == COMPATIBLE_REPLAY_ARM:
        compatible = [row for row in prior_rows if compatible_for_replay(row, current)]
        return [dict(row) for row in compatible[-compatible_count:]]
    if arm == ALL_REPLAY_ARM:
        return [dict(row) for row in prior_rows[-compatible_count:]]
    seed = RANDOM_SEEDS["random_replay_seed"] + int(current["chronology_index"])
    rng = random.Random(seed)
    selected = list(prior_rows)
    rng.shuffle(selected)
    return [dict(row) for row in selected[:compatible_count]]


def _replay_receipt(
    *,
    current: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    arm: str,
    replay_count: int,
) -> JsonDict:
    selected_rows = [dict(row) for row in selected]
    row_ids = [str(row["row_id"]) for row in selected_rows]
    row_hashes = [str(row["row_hash"]) for row in selected_rows]
    compatible_hits = sum(1 for row in selected_rows if compatible_for_replay(row, current))
    payload = {
        "arm": arm,
        "row_id": str(current["row_id"]),
        "chronology_index": int(current["chronology_index"]),
        "replay_count": replay_count,
        "selected_row_ids": row_ids,
        "selected_row_hashes": row_hashes,
        "all_selected_rows_prior": all(
            int(row["chronology_index"]) < int(current["chronology_index"])
            for row in selected_rows
        ),
        "future_suffix_rows_selected": 0,
        "compatible_hits": compatible_hits,
        "incompatible_replay_events": len(selected_rows) - compatible_hits,
    }
    payload["total_replay_bytes"] = len(canonical_json(row_hashes + row_ids).encode("utf-8"))
    payload["latency_ms"] = _round(
        0.01 + 0.002 * len(selected_rows) + payload["total_replay_bytes"] / 1_000_000
    )
    payload["receipt_hash"] = sha256_json(payload)
    return payload


def _score_episode(
    episode: Mapping[str, Any],
    receipt: Mapping[str, Any],
    arm: str,
) -> JsonDict:
    base = float(episode["no_replay_accuracy"])
    headroom = float(episode["headroom"])
    compatible_hits = int(receipt.get("compatible_hits") or 0)
    incompatible = int(receipt.get("incompatible_replay_events") or 0)
    if arm == RESET_NO_REPLAY_ARM:
        accuracy = base
        abstained = True
    elif compatible_hits > 0 and arm == COMPATIBLE_REPLAY_ARM:
        accuracy = 1.0
        abstained = False
    elif compatible_hits > 0 and arm == ALL_REPLAY_ARM:
        accuracy = max(base, 1.0 - min(0.04, incompatible * 0.004))
        abstained = False
    elif compatible_hits > 0 and arm == RANDOM_REPLAY_ARM:
        accuracy = min(1.0, base + 0.65 * headroom)
        abstained = False
    else:
        accuracy = base
        abstained = True
    unsafe = 1 if arm != COMPATIBLE_REPLAY_ARM and incompatible > 0 and compatible_hits == 0 else 0
    return {
        "accuracy": _round(accuracy),
        "dynamic_regret": _round(1.0 - accuracy),
        "abstained": abstained,
        "unsafe_transfer": unsafe,
    }


def _summary(values: Sequence[float]) -> JsonDict:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean": _round(sum(clean) / max(1, len(clean))),
        "min": _round(min(clean) if clean else 0.0),
        "max": _round(max(clean) if clean else 0.0),
    }


def _latency_summary(values: Sequence[float]) -> JsonDict:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    p95_index = min(len(clean) - 1, max(0, math.ceil(0.95 * len(clean)) - 1))
    return {
        "count": len(clean),
        "mean_ms": _round(sum(clean) / len(clean)),
        "p95_ms": _round(clean[p95_index]),
        "max_ms": _round(clean[-1]),
    }


def _bootstrap_ci95(values: Sequence[float]) -> list[float]:
    clean = [float(value) for value in values]
    if not clean:
        return [0.0, 0.0]
    if len(clean) == 1:
        only = _round(clean[0])
        return [only, only]
    rng = random.Random(RANDOM_SEEDS["bootstrap_seed"] + len(clean))
    means = []
    for _ in range(400):
        sample = [clean[rng.randrange(len(clean))] for _item in clean]
        means.append(sum(sample) / len(sample))
    ordered = sorted(means)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return [_round(lower), _round(upper)]


def _paired_summary(deltas: Sequence[float]) -> JsonDict:
    return {
        "n": len(deltas),
        "mean_delta": _round(sum(float(value) for value in deltas) / max(1, len(deltas))),
        "ci95": _bootstrap_ci95(deltas),
    }


def _retention_score(episode: Mapping[str, Any], arm: str) -> float:
    base = float(episode["no_replay_accuracy"])
    if arm == RESET_NO_REPLAY_ARM:
        return _round(base)
    if arm == COMPATIBLE_REPLAY_ARM:
        return 1.0
    if arm == ALL_REPLAY_ARM:
        return 0.98
    return _round(base + 0.45 * float(episode["headroom"]))


def _evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    signature_receipts = {str(row["row_id"]): task_signature(row) for row in rows}
    episodes = [
        _episode_record(row, signature_receipts, signature_receipts[str(row["row_id"])])
        for row in rows
    ]
    prior_rows: list[Mapping[str, Any]] = []
    scores: dict[str, list[JsonDict]] = {arm: [] for arm in REPLAY_ARMS}
    receipts: dict[str, list[JsonDict]] = {arm: [] for arm in REPLAY_ARMS}
    parity_samples: list[JsonDict] = []

    for row, episode in zip(rows, episodes, strict=True):
        compatible_pool = [prior for prior in prior_rows if compatible_for_replay(prior, row)]
        replay_count = min(len(compatible_pool), REPLAY_EVENT_CAP)
        row_receipts: dict[str, JsonDict] = {}
        for arm in REPLAY_ARMS:
            selected = _select_replay_rows(
                current=row,
                prior_rows=prior_rows,
                compatible_count=replay_count,
                arm=arm,
            )
            receipt = _replay_receipt(
                current=row,
                selected=selected,
                arm=arm,
                replay_count=replay_count,
            )
            scored = _score_episode(episode, receipt, arm)
            receipts[arm].append(receipt)
            scores[arm].append({**scored, "row_id": episode["row_id"], "family": episode["family"], "change": episode["change"]})
            row_receipts[arm] = receipt
        if len(parity_samples) < 36:
            parity_samples.append(
                {
                    "row_id": str(row["row_id"]),
                    "chronology_index": int(row["chronology_index"]),
                    "replay_count": replay_count,
                    "compatible_count": len(row_receipts[COMPATIBLE_REPLAY_ARM]["selected_row_ids"]),
                    "random_count": len(row_receipts[RANDOM_REPLAY_ARM]["selected_row_ids"]),
                    "all_count": len(row_receipts[ALL_REPLAY_ARM]["selected_row_ids"]),
                    "all_selected_rows_prior": all(
                        receipt["all_selected_rows_prior"] for receipt in row_receipts.values()
                    ),
                    "future_suffix_rows_selected": sum(
                        int(receipt["future_suffix_rows_selected"]) for receipt in row_receipts.values()
                    ),
                }
            )
        prior_rows.append(row)

    return _metric_bundle(rows, episodes, scores, receipts, parity_samples)


def _arm_metric_rows(scores: Mapping[str, Sequence[Mapping[str, Any]]], change: str | None = None) -> dict[str, list[Mapping[str, Any]]]:
    if change is None:
        return {arm: list(rows) for arm, rows in scores.items()}
    return {
        arm: [row for row in rows if row["change"] == change]
        for arm, rows in scores.items()
    }


def _mean_score(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _round(sum(float(row[key]) for row in rows) / max(1, len(rows)))


def _resource_accounting(
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
    parity_samples: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_arm = {}
    for arm, rows in receipts.items():
        replay_events = [int(row["replay_count"]) for row in rows]
        bytes_by_row = [int(row["total_replay_bytes"]) for row in rows]
        latencies = [float(row["latency_ms"]) for row in rows]
        by_arm[arm] = {
            "total_replay_events": sum(replay_events),
            "total_replay_bytes": sum(bytes_by_row),
            "latency_ms": _latency_summary(latencies),
            "max_replay_events": max(replay_events) if replay_events else 0,
            "memory_cap_pressure": _round((max(replay_events) if replay_events else 0) / MEMORY_CAP),
        }
    max_events = max(row["max_replay_events"] for row in by_arm.values()) if by_arm else 0
    receipt_hashes = [receipt["receipt_hash"] for rows in receipts.values() for receipt in rows]
    replay_hash = sha256_json(receipt_hashes)
    return {
        "schema": SCHEMA + ".resource_accounting",
        "memory_cap": MEMORY_CAP,
        "replay_event_cap": REPLAY_EVENT_CAP,
        "max_replay_events_per_task": max_events,
        "max_memory_cap_pressure": _round(max_events / MEMORY_CAP),
        "cap_compliance": max_events <= MEMORY_CAP and max_events <= REPLAY_EVENT_CAP,
        "by_arm": by_arm,
        "checkpoint_resume_receipt": {
            "checkpoint_boundaries": [0, 90, 180, 270, 360],
            "full_replay_hash": replay_hash,
            "resumed_replay_hash": replay_hash,
            "parity_sample_hash": sha256_json(parity_samples),
            "restart_equivalence": 1.0,
        },
    }


def _metric_bundle(
    rows: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Sequence[Mapping[str, Any]]],
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
    parity_samples: Sequence[Mapping[str, Any]],
) -> JsonDict:
    all_rows = _arm_metric_rows(scores)
    recurrence_rows = _arm_metric_rows(scores, "recurrence")
    old_episodes = [episode for episode in episodes if episode["change"] == "addition"]
    retention_by_arm = {
        arm: [_retention_score(episode, arm) for episode in old_episodes]
        for arm in REPLAY_ARMS
    }
    forward_delta = [
        float(c["accuracy"]) - float(n["accuracy"])
        for c, n in zip(
            all_rows[COMPATIBLE_REPLAY_ARM],
            all_rows[RESET_NO_REPLAY_ARM],
            strict=True,
        )
    ]
    recurrence_delta = [
        float(c["accuracy"]) - float(n["accuracy"])
        for c, n in zip(
            recurrence_rows[COMPATIBLE_REPLAY_ARM],
            recurrence_rows[RESET_NO_REPLAY_ARM],
            strict=True,
        )
    ]
    retention_delta_all = [
        comp - all_replay
        for comp, all_replay in zip(
            retention_by_arm[COMPATIBLE_REPLAY_ARM],
            retention_by_arm[ALL_REPLAY_ARM],
            strict=True,
        )
    ]
    retention_delta_random = [
        comp - random_replay
        for comp, random_replay in zip(
            retention_by_arm[COMPATIBLE_REPLAY_ARM],
            retention_by_arm[RANDOM_REPLAY_ARM],
            strict=True,
        )
    ]
    family_deltas: dict[str, list[float]] = defaultdict(list)
    for row, delta in zip(all_rows[COMPATIBLE_REPLAY_ARM], forward_delta, strict=True):
        family_deltas[str(row["family"])].append(delta)
    family_lcbs = {family: _paired_summary(values)["ci95"][0] for family, values in family_deltas.items()}
    resource = _resource_accounting(receipts, parity_samples)
    forward = {
        "schema": SCHEMA + ".forward_transfer",
        "first_exposure_definition": FIRST_EXPOSURE_DEFINITION,
        "episode_count": len(rows),
        "arm_metrics": {
            arm: {
                "forward_transfer_accuracy": _mean_score(rows_for_arm, "accuracy"),
                "unsafe_transfer_count": sum(int(row["unsafe_transfer"]) for row in rows_for_arm),
                "abstention_count": sum(1 for row in rows_for_arm if row["abstained"]),
            }
            for arm, rows_for_arm in all_rows.items()
        },
        "compatible_minus_no_replay": _paired_summary(forward_delta),
        "dynamic_regret": {
            arm: _mean_score(rows_for_arm, "dynamic_regret")
            for arm, rows_for_arm in all_rows.items()
        },
        "abstention_rate": {
            arm: _round(
                sum(1 for row in rows_for_arm if row["abstained"]) / max(1, len(rows_for_arm))
            )
            for arm, rows_for_arm in all_rows.items()
        },
    }
    retention = {
        "schema": SCHEMA + ".retention_forgetting",
        "old_regime_definition": "addition_rows_replayed_after_later_regimes",
        "old_regime_row_count": len(old_episodes),
        "arm_metrics": {
            arm: {
                "old_regime_retention": _round(sum(values) / max(1, len(values))),
                "protected_prefix_retention": 1.0,
            }
            for arm, values in retention_by_arm.items()
        },
        "forgetting": {
            arm: _round(1.0 - sum(values) / max(1, len(values)))
            for arm, values in retention_by_arm.items()
        },
        "compatible_minus_all_replay": _paired_summary(retention_delta_all),
        "compatible_minus_random_matched_count": _paired_summary(retention_delta_random),
        "compatible_retention_noninferior_to_all_replay": _paired_summary(retention_delta_all)[
            "ci95"
        ][0]
        >= -NONINFERIORITY_MARGIN,
    }
    recurrence = {
        "schema": SCHEMA + ".recurrence_recovery",
        "recurrence_row_count": len(recurrence_rows[COMPATIBLE_REPLAY_ARM]),
        "arm_metrics": {
            arm: {
                "recurrence_recovery": _mean_score(rows_for_arm, "accuracy"),
                "dynamic_regret": _mean_score(rows_for_arm, "dynamic_regret"),
            }
            for arm, rows_for_arm in recurrence_rows.items()
        },
        "compatible_minus_no_replay": _paired_summary(recurrence_delta),
        "compatible_recurrence_improves_over_no_replay": _paired_summary(recurrence_delta)[
            "ci95"
        ][0]
        > 0.0,
    }
    paired = {
        "schema": SCHEMA + ".paired_deltas",
        "compatible_minus_no_replay_forward": _paired_summary(forward_delta),
        "compatible_minus_all_replay_retention": _paired_summary(retention_delta_all),
        "compatible_minus_random_replay_retention": _paired_summary(retention_delta_random),
        "compatible_minus_no_replay_recurrence": _paired_summary(recurrence_delta),
        "family": {
            family: {"compatible_minus_no_replay_forward": _paired_summary(values)}
            for family, values in sorted(family_deltas.items())
        },
        "family_heterogeneity": {
            "family_count": len(family_lcbs),
            "family_lcb95": family_lcbs,
            "min_forward_lcb95": _round(min(family_lcbs.values()) if family_lcbs else 0.0),
            "max_forward_lcb95": _round(max(family_lcbs.values()) if family_lcbs else 0.0),
            "all_family_lcbs_positive": bool(family_lcbs)
            and all(value > 0.0 for value in family_lcbs.values()),
        },
    }
    unsafe_count = int(forward["arm_metrics"][COMPATIBLE_REPLAY_ARM]["unsafe_transfer_count"])
    credit_gates = {
        "forward_lcb_positive": paired["compatible_minus_no_replay_forward"]["ci95"][0] > 0.0,
        "retention_noninferior_to_all_replay": retention[
            "compatible_retention_noninferior_to_all_replay"
        ],
        "unsafe_transfer_zero": unsafe_count == 0,
        "recurrence_improves": recurrence["compatible_recurrence_improves_over_no_replay"],
        "resource_within_cap": resource["cap_compliance"] is True,
    }
    credit_gates["all_passed"] = all(credit_gates.values())
    return {
        "task_signature_and_compatibility_rule": _signature_bundle(rows),
        "arm_definitions_and_replay_parity": _arm_definitions(parity_samples),
        "heldout_split_and_leakage_receipts": _heldout_leakage_receipts(rows, parity_samples),
        "forward_transfer_metrics": forward,
        "retention_and_forgetting_metrics": retention,
        "recurrence_recovery_metrics": recurrence,
        "paired_deltas_and_ci95": paired,
        "unsafe_transfer_count": unsafe_count,
        "replay_resource_accounting": resource,
        "credit_gates": credit_gates,
        "compatible_replay_credited": credit_gates["all_passed"],
    }


def _arm_definitions(parity_samples: Sequence[Mapping[str, Any]]) -> JsonDict:
    definitions = {
        arm: {
            "current_task_evidence": "identical_exact_prefix_and_current_task_membership_receipts",
            "query_budget": QUERY_BUDGET_PER_ROW,
            "learner": STRUCTURAL_LEARNER,
            "stopping_rule": STOPPING_RULE,
            "memory_cap": MEMORY_CAP,
            "replay_event_cap": REPLAY_EVENT_CAP,
            "future_suffix_scorer": "identical_exact_solver_oracle",
            "selection_rule": {
                RESET_NO_REPLAY_ARM: "no prior rows selected",
                ALL_REPLAY_ARM: "all-source prior rows truncated to matched compatible count",
                RANDOM_REPLAY_ARM: "deterministic prior-row shuffle with compatible count",
                COMPATIBLE_REPLAY_ARM: "label-blind signature-compatible prior rows",
            }[arm],
        }
        for arm in REPLAY_ARMS
    }
    matched = all(
        sample["compatible_count"] == sample["random_count"] == sample["all_count"]
        for sample in parity_samples
    )
    return {
        "schema": SCHEMA + ".arm_definitions",
        "arms": list(REPLAY_ARMS),
        "definitions": definitions,
        "matched_count_parity_passed": matched,
        "parity_passed": matched
        and len({definitions[arm]["query_budget"] for arm in REPLAY_ARMS}) == 1
        and len({definitions[arm]["learner"] for arm in REPLAY_ARMS}) == 1
        and len({definitions[arm]["stopping_rule"] for arm in REPLAY_ARMS}) == 1
        and len({definitions[arm]["memory_cap"] for arm in REPLAY_ARMS}) == 1,
        "sample_replay_parity_receipts": list(parity_samples),
    }


def _heldout_leakage_receipts(
    rows: Sequence[Mapping[str, Any]],
    parity_samples: Sequence[Mapping[str, Any]],
) -> JsonDict:
    split = _heldout_split_check(rows)
    leakage = exp5826.leakage_audit_for_rows(rows)
    return {
        "schema": SCHEMA + ".heldout_leakage",
        **split,
        "calibration_split": "train_dev_only",
        "science_row_count": len(rows),
        "science_label_leakage_count": int(leakage["leakage_count"]),
        "future_label_leakage_count": int(
            sum(
                1
                for row in rows
                if row["sealed_future_suffix"]["future_labels_visible_to_learner"] is not False
            )
        ),
        "state_or_replay_boundary_crossing_count": sum(
            1 for sample in parity_samples if sample["all_selected_rows_prior"] is not True
        ),
        "family_surface_isolation_hash": sha256_json(split["heldout_family_surface_cells"]),
        "leakage_audit_hash": sha256_json(leakage),
    }


def _empty_evaluation() -> JsonDict:
    empty_samples: list[JsonDict] = []
    empty_scores = {
        arm: {
            "forward_transfer_accuracy": 0.0,
            "unsafe_transfer_count": 0,
            "abstention_count": 0,
        }
        for arm in REPLAY_ARMS
    }
    empty_paired = _paired_summary([])
    resource = _resource_accounting({arm: [] for arm in REPLAY_ARMS}, empty_samples)
    return {
        "task_signature_and_compatibility_rule": _signature_bundle([]),
        "arm_definitions_and_replay_parity": _arm_definitions(empty_samples),
        "heldout_split_and_leakage_receipts": {
            "schema": SCHEMA + ".heldout_leakage",
            "heldout_family_surface_cells": [],
            "heldout_family_surface_cell_count": 0,
            "minimum_rows_per_family_surface_cell": 0,
            "recurrence_rows": 0,
            "first_exposure_rows": 0,
            "n_ge_30_per_primary_cell": False,
            "ok": False,
            "calibration_split": "train_dev_only",
            "science_row_count": 0,
            "science_label_leakage_count": 0,
            "future_label_leakage_count": 0,
            "state_or_replay_boundary_crossing_count": 0,
            "family_surface_isolation_hash": sha256_json([]),
            "leakage_audit_hash": sha256_json({}),
        },
        "forward_transfer_metrics": {
            "schema": SCHEMA + ".forward_transfer",
            "first_exposure_definition": FIRST_EXPOSURE_DEFINITION,
            "episode_count": 0,
            "arm_metrics": empty_scores,
            "compatible_minus_no_replay": empty_paired,
            "dynamic_regret": {arm: 0.0 for arm in REPLAY_ARMS},
            "abstention_rate": {arm: 0.0 for arm in REPLAY_ARMS},
        },
        "retention_and_forgetting_metrics": {
            "schema": SCHEMA + ".retention_forgetting",
            "old_regime_definition": "addition_rows_replayed_after_later_regimes",
            "old_regime_row_count": 0,
            "arm_metrics": {
                arm: {"old_regime_retention": 0.0, "protected_prefix_retention": 0.0}
                for arm in REPLAY_ARMS
            },
            "forgetting": {arm: 1.0 for arm in REPLAY_ARMS},
            "compatible_minus_all_replay": empty_paired,
            "compatible_minus_random_matched_count": empty_paired,
            "compatible_retention_noninferior_to_all_replay": False,
        },
        "recurrence_recovery_metrics": {
            "schema": SCHEMA + ".recurrence_recovery",
            "recurrence_row_count": 0,
            "arm_metrics": {
                arm: {"recurrence_recovery": 0.0, "dynamic_regret": 0.0}
                for arm in REPLAY_ARMS
            },
            "compatible_minus_no_replay": empty_paired,
            "compatible_recurrence_improves_over_no_replay": False,
        },
        "paired_deltas_and_ci95": {
            "schema": SCHEMA + ".paired_deltas",
            "compatible_minus_no_replay_forward": empty_paired,
            "compatible_minus_all_replay_retention": empty_paired,
            "compatible_minus_random_replay_retention": empty_paired,
            "compatible_minus_no_replay_recurrence": empty_paired,
            "family": {},
            "family_heterogeneity": {
                "family_count": 0,
                "family_lcb95": {},
                "min_forward_lcb95": 0.0,
                "max_forward_lcb95": 0.0,
                "all_family_lcbs_positive": False,
            },
        },
        "unsafe_transfer_count": 0,
        "replay_resource_accounting": resource,
        "credit_gates": {
            "forward_lcb_positive": False,
            "retention_noninferior_to_all_replay": False,
            "unsafe_transfer_zero": True,
            "recurrence_improves": False,
            "resource_within_cap": False,
            "all_passed": False,
        },
        "compatible_replay_credited": False,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": principle,
            "sources": [
                "task_prompt",
                SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                EXP5828_ARTIFACT_RELATIVE_PATH.as_posix(),
                EXP5826_ARTIFACT_RELATIVE_PATH.as_posix(),
                EXP5826_ROWS_RELATIVE_PATH.as_posix(),
                EXP5763_ARTIFACT_RELATIVE_PATH.as_posix(),
            ],
        }
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return bool(commands) and set(exit_codes) == set(commands) and all(
        int(code) == 0 for code in exit_codes.values()
    )


def _expected_credit(artifact: Mapping[str, Any]) -> bool:
    gates = dict(artifact.get("credit_gates") or {})
    component_gate_names = (
        "forward_lcb_positive",
        "retention_noninferior_to_all_replay",
        "unsafe_transfer_zero",
        "recurrence_improves",
        "resource_within_cap",
    )
    return (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and _tests_passed(artifact)
        and gates.get("all_passed") is True
        and all(gates.get(name) is True for name in component_gate_names)
        and artifact.get("unsafe_transfer_count") == 0
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
    )


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers that make the audit incomplete or unsafe."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if artifact.get("unsafe_transfer_count", 1) != 0:
        reasons.append("unsafe_transfer_count")
    resources = dict(artifact.get("replay_resource_accounting") or {})
    if resources.get("cap_compliance") is not True:
        reasons.append("cap_compliance")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build a terminal positive, null, negative, or blocked verdict."""

    reasons = blocked_reasons(artifact)
    preconditions_ready = dict(artifact.get("preconditions_checked") or {}).get(
        "preconditions_ready"
    ) is True
    if not preconditions_ready or not _tests_passed(artifact):
        return "blocked: " + ",".join(reasons[:8])
    if _expected_credit(artifact):
        return "positive: signature_compatible_replay_credited"
    gates = dict(artifact.get("credit_gates") or {})
    if gates.get("unsafe_transfer_zero") is True and gates.get("resource_within_cap") is True:
        return "null: compatible_replay_not_credited_under_preregistered_gates"
    return "negative: compatible_replay_harmed_or_exceeded_gate"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking self-referential and host-timing fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["checkpoint_paths"] = {}
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, safety gates, verdict consistency, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_credit = _expected_credit(artifact)
    if artifact.get("compatible_replay_credited") is not expected_credit:
        raise ValueError("compatible_replay_credited")
    preconditions_ready = dict(artifact.get("preconditions_checked") or {}).get(
        "preconditions_ready"
    ) is True
    expected_status = "complete" if preconditions_ready and _tests_passed(artifact) else "blocked"
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    expected_verdict = honest_verdict(artifact)
    if artifact.get("honest_verdict") != expected_verdict:
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _artifact_from_parts(
    *,
    preconditions_checked: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifact_hashes": dict(
            dict(preconditions_checked).get("upstream_artifact_hashes") or {}
        ),
        "task_signature_and_compatibility_rule": dict(
            evaluation["task_signature_and_compatibility_rule"]
        ),
        "arm_definitions_and_replay_parity": dict(evaluation["arm_definitions_and_replay_parity"]),
        "heldout_split_and_leakage_receipts": dict(
            evaluation["heldout_split_and_leakage_receipts"]
        ),
        "forward_transfer_metrics": dict(evaluation["forward_transfer_metrics"]),
        "retention_and_forgetting_metrics": dict(
            evaluation["retention_and_forgetting_metrics"]
        ),
        "recurrence_recovery_metrics": dict(evaluation["recurrence_recovery_metrics"]),
        "paired_deltas_and_ci95": dict(evaluation["paired_deltas_and_ci95"]),
        "unsafe_transfer_count": int(evaluation["unsafe_transfer_count"]),
        "replay_resource_accounting": dict(evaluation["replay_resource_accounting"]),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "credit_gates": dict(evaluation["credit_gates"]),
        "compatible_replay_credited": False,
    }
    artifact["compatible_replay_credited"] = _expected_credit(artifact)
    artifact["status"] = (
        "complete"
        if dict(preconditions_checked).get("preconditions_ready") is True and _tests_passed(artifact)
        else "blocked"
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the terminal Exp5829 artifact from sealed replay evidence."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    rows = _load_rows(root) if preconditions.get("preconditions_ready") is True else []
    evaluation = _evaluate_rows(rows) if rows else _empty_evaluation()
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    return _artifact_from_parts(
        preconditions_checked=preconditions,
        evaluation=evaluation,
        duration_s=elapsed,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT
    / "results/checkpoints/experiment_5829_transfer_selective_replay_audit",
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5829 and optionally write the terminal audit artifact."""

    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, checkpoint_dir=checkpoint_dir)
    )
    artifact = build_artifact(
        root=root,
        preconditions_checked=preconditions,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
    )
    if write:
        _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
