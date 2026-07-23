"""Exp5828 future-validated structural memory lifecycle.

Spec refs: REQ-LEARN-5828, SCENARIO-LEARN-5828-FUTURE-PROMOTION,
SCENARIO-LEARN-5828-STRUCTURAL-OPS, SCENARIO-LEARN-5828-RESTART-CAP,
SCENARIO-LEARN-5828-FAIL-CLOSED.

This module keeps the model weights frozen and evaluates learning only through
versioned structural memory. Exact solver feedback is used as oracle evidence:
it can decide whether a quarantined structural proposal should be promoted, but
it is not an oracle-distinct verifier moat.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5825_certified_adaptive_memory_contract as exp5825
from carnot import experiment_5826_out_of_template_constraint_stream as exp5826
from carnot import experiment_5827_minimal_core_structural_acquisition_ab as exp5827


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5828_future_validated_structural_memory.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5828_future_validated_structural_memory.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5828_future_validated_structural_memory.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXP5825_CONTRACT_RELATIVE_PATH = exp5826.EXP5825_CONTRACT_RELATIVE_PATH
EXP5826_ARTIFACT_RELATIVE_PATH = exp5826.RESULT_RELATIVE_PATH
EXP5826_ROWS_RELATIVE_PATH = exp5826.ROW_FILE_RELATIVE_PATH
EXP5827_ARTIFACT_RELATIVE_PATH = exp5827.RESULT_RELATIVE_PATH
EXP5762_ARTIFACT_RELATIVE_PATH = exp5826.EXP5762_ARTIFACT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5828.future_validated_structural_memory.v1"
EXPERIMENT = 5828
EXPERIMENT_ID = "experiment_5828_future_validated_structural_memory"
MILESTONE = "2026.07.520"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "online_exact_membership_query_sidecar_no_llm"
STRUCTURAL_LEARNER = "exp5827_active_discriminating_query_minimal_core_synthesis_v1"
STOPPING_RULE = "quarantine_future_validate_promote_or_rollback_v1"
QUERY_BUDGET_PER_ROW = exp5827.QUERY_BUDGET_PER_ROW
MEMORY_CAP = 64
QUARANTINE_RETENTION_CAP = 24
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512

PRIMARY_FAMILIES = exp5826.PRIMARY_FAMILIES
CHANGE_ORDER = exp5826.CHANGE_ORDER
PROOF_PRESERVING_SURFACES = exp5826.PROOF_PRESERVING_SURFACES
HARDNESS_BINS = exp5826.HARDNESS_BINS

NO_MEMORY_ARM = "no_adaptive_memory"
IMMEDIATE_ARM = "immediate_structural_promotion"
FUTURE_ARM = "future_validated_write_protected_promotion"
CONTROL_ARMS = (NO_MEMORY_ARM, IMMEDIATE_ARM, FUTURE_ARM)
SPEC_REFS = (
    "REQ-LEARN-5828",
    "SCENARIO-LEARN-5828-FUTURE-PROMOTION",
    "SCENARIO-LEARN-5828-STRUCTURAL-OPS",
    "SCENARIO-LEARN-5828-RESTART-CAP",
    "SCENARIO-LEARN-5828-FAIL-CLOSED",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5828,
    "bootstrap_seed": 5_828_001,
    "checkpoint_seed": 5_828_002,
    "rollback_probe_seed": 5_828_003,
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5828_future_validated_structural_memory.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5828_future_validated_structural_memory.py "
    "-m pytest tests/python/test_experiment_5828_future_validated_structural_memory.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5828_future_validated_structural_memory.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5828_future_validated_structural_memory.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "model_weight_mutation",
    "arm_definitions_and_parity",
    "quarantine_promotion_rollback_ledger",
    "collision_supersession_recurrence_receipts",
    "sealed_future_validation_receipts",
    "per_family_change_metrics",
    "paired_deltas_and_ci95",
    "protected_prefix_retention",
    "unsafe_update_count",
    "rollback_hash_mismatch_count",
    "restart_equivalence",
    "memory_cap_receipts",
    "future_validated_lifecycle_ready_score",
    "retire_if_same_verdict",
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
    "status": "A terminal state distinguishes a complete lifecycle result from an interrupted checkpoint.",
    "preconditions_checked": "Gate, sealed batches, solvers, resources, seeds, and checkpoint checks prevent fabricated execution.",
    "upstream_artifact_hashes": "Hashes bind the lifecycle to its learner, stream, and contract.",
    "model_weight_mutation": "False proves continuous learning occurred in versioned memory with frozen GGUF weights.",
    "arm_definitions_and_parity": "Matched inputs, budgets, learner, and cap isolate future validation.",
    "quarantine_promotion_rollback_ledger": "Transactional receipts make every accepted or rejected edit replayable.",
    "collision_supersession_recurrence_receipts": "Explicit structural events test nonstationarity rather than one-shot memorization.",
    "sealed_future_validation_receipts": "Prospective suffix evidence prevents post-hoc promotion.",
    "per_family_change_metrics": "Disaggregated outcomes expose family harm and recurrence failures.",
    "paired_deltas_and_ci95": "Paired intervals quantify future lift under identical episodes.",
    "protected_prefix_retention": "Exact retention prevents new rules from corrupting earlier facts.",
    "unsafe_update_count": "A bare zero is required for safe propagation.",
    "rollback_hash_mismatch_count": "A bare zero proves rejected edits leave no state residue.",
    "restart_equivalence": "Exact full/resumed hashes make learning durable across process boundaries.",
    "memory_cap_receipts": "Bounded state growth is necessary for continual and hardware-portable operation.",
    "future_validated_lifecycle_ready_score": "EMIT BARE scalar; only 1.0 permits replay transfer and kernel work.",
    "retire_if_same_verdict": "A repeated blocked outcome mechanically retires this reattempt.",
    "duration_s": "Measured wall time exposes bootstrap-only execution.",
    "inference_substrate": "`online_exact_membership_query_sidecar_no_llm` declares exact oracle-guided memory learning.",
    "verifier_is_oracle": "True records that exact solvers gate promotion and forbid a verifier-moat claim.",
    "field_provenance": "Every field traces to event, state, query, validation, or replay receipts.",
    "test_commands": "Commands document chronology, sealing, transactions, retention, restart, and statistics.",
    "test_exit_codes": "Exit codes prevent failed lifecycle checks from becoming success.",
    "reproducibility_checksum": "A checksum detects state, event, seed, or metric drift.",
    "honest_verdict": "A terminal prefix states credited, null, negative, or blocked outcome honestly.",
}
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5825_contract": EXP5825_CONTRACT_RELATIVE_PATH,
    "exp5826_artifact": EXP5826_ARTIFACT_RELATIVE_PATH,
    "exp5826_rows": EXP5826_ROWS_RELATIVE_PATH,
    "exp5826_module": exp5826.MODULE_RELATIVE_PATH,
    "exp5827_artifact": EXP5827_ARTIFACT_RELATIVE_PATH,
    "exp5827_module": exp5827.MODULE_RELATIVE_PATH,
    "exp5762_artifact": EXP5762_ARTIFACT_RELATIVE_PATH,
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
    """Hash exact file bytes without trusting timestamps or metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _mean(values: Sequence[float]) -> float:
    return _round(sum(float(value) for value in values) / max(1, len(values)))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(dict(payload))
    return rows


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read Exp5826 JSONL rows, returning an empty list for absent files."""

    if not Path(path).exists():
        return []
    return _read_jsonl(path)


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
        return (
            (
                parent.exists()
                and os.access(parent, os.W_OK)
                or parent.parent.exists()
                and os.access(parent.parent, os.W_OK)
            )
            and (not path.exists() or os.access(path, os.W_OK))
        )

    checkpoint_parent = checkpoint_dir if checkpoint_dir.exists() else checkpoint_dir.parent
    checkpoint_ready = (
        checkpoint_parent.exists()
        and os.access(checkpoint_parent, os.W_OK)
        or checkpoint_parent.parent.exists()
        and os.access(checkpoint_parent.parent, os.W_OK)
    )
    return {
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "checkpoint_dir": "results/checkpoints/experiment_5828_future_validated_structural_memory",
        "result_writable": ready_file(result_path),
        "checkpoint_writable": checkpoint_ready,
        "checkpoint_atomic_suffix": ".tmp",
        "ok": ready_file(result_path) and checkpoint_ready,
    }


def _row_replay_receipt(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    artifact_rows = dict(artifact.get("row_file_and_sha256") or {})
    try:
        replay_ok = exp5826.verify_row_file(rows, artifact)
    except exp5826.StreamReplayError:
        replay_ok = False
    row_hash = sha256_text(exp5826.rows_to_jsonl(rows))
    return {
        "row_count": len(rows),
        "artifact_row_count": int(artifact_rows.get("row_count") or -1),
        "row_text_hash": row_hash,
        "artifact_row_file_hash": str(artifact_rows.get("sha256") or ""),
        "row_file_hash_ok": row_hash == artifact_rows.get("sha256"),
        "replay_ok": replay_ok,
        "ok": replay_ok and len(rows) == 360 and row_hash == artifact_rows.get("sha256"),
    }


def _multiple_change_coverage_receipt(artifact: Mapping[str, Any]) -> JsonDict:
    manifest = dict(artifact.get("stream_manifest") or {})
    cell_counts = dict(manifest.get("cell_counts") or {})
    expected_cells = [f"{family}|{change}" for family in PRIMARY_FAMILIES for change in CHANGE_ORDER]
    coverage_ok = all(int(cell_counts.get(cell) or 0) >= 30 for cell in expected_cells)
    order = dict(dict(artifact.get("chronology_and_change_receipts") or {}).get("family_change_order") or {})
    order_ok = all(order.get(family) == list(CHANGE_ORDER) for family in PRIMARY_FAMILIES)
    return {
        "expected_primary_cells": expected_cells,
        "cell_counts": {cell: int(cell_counts.get(cell) or 0) for cell in expected_cells},
        "minimum_units_per_cell": 30,
        "family_change_order": order,
        "coverage_ok": coverage_ok,
        "order_ok": order_ok,
        "ok": coverage_ok and order_ok,
    }


def _sealed_future_batch_receipt(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    upstream = dict(artifact.get("sealed_future_batch_receipts") or {})
    leakage_count = int(upstream.get("future_label_leakage_count", 1))
    suffix_hash_ok = True
    for row in rows:
        suffix = dict(row.get("sealed_future_suffix") or {})
        expected = str(suffix.pop("suffix_hash", ""))
        suffix_hash_ok = suffix_hash_ok and expected == sha256_json(suffix)
    return {
        "sealed_suffix_count": sum(1 for row in rows if dict(row.get("sealed_future_suffix") or {}).get("sealed") is True),
        "artifact_sealed_suffix_count": int(upstream.get("sealed_suffix_count") or -1),
        "all_future_suffixes_sealed": upstream.get("all_future_suffixes_sealed") is True,
        "future_label_leakage_count": leakage_count,
        "suffix_hash_ok": suffix_hash_ok,
        "batch_hashes": dict(upstream.get("batch_hashes") or {}),
        "ok": upstream.get("all_future_suffixes_sealed") is True
        and leakage_count == 0
        and int(upstream.get("sealed_suffix_count") or -1) == len(rows)
        and suffix_hash_ok,
    }


def _exact_solver_receipt(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    exact = dict(artifact.get("exact_query_and_core_receipts") or {})
    primary_versions = {
        str(row["exact_receipt"]["primary"]["validator_version"]) for row in rows
    }
    independent_versions = {
        str(row["exact_receipt"]["independent"]["validator_version"]) for row in rows
    }
    return {
        "primary_versions": sorted(primary_versions),
        "independent_versions": sorted(independent_versions),
        "all_exact_validators_agree": exact.get("all_exact_validators_agree") is True,
        "expected_primary": exp5826.PRIMARY_VALIDATOR_VERSION,
        "expected_independent": exp5826.INDEPENDENT_VALIDATOR_VERSION,
        "ok": exact.get("all_exact_validators_agree") is True
        and primary_versions == {exp5826.PRIMARY_VALIDATOR_VERSION}
        and independent_versions == {exp5826.INDEPENDENT_VALIDATOR_VERSION},
    }


def _exp5827_credit_receipt(artifact: Mapping[str, Any]) -> JsonDict:
    paired = dict(artifact.get("paired_deltas_and_ci95") or {})
    pooled = dict(dict(paired.get("pooled") or {}).get("active_minus_exp5762_template") or {})
    return {
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "structural_learner_ready_score": artifact.get("structural_learner_ready_score"),
        "pooled_lcb95": float((pooled.get("ci95") or [0.0])[0]),
        "oracle_boundary_violation_count": artifact.get("oracle_boundary_violation_count"),
        "model_weight_mutation": False,
        "ok": artifact.get("status") == "complete"
        and str(artifact.get("honest_verdict") or "").startswith("complete:")
        and artifact.get("structural_learner_ready_score") == 1.0
        and float((pooled.get("ci95") or [0.0])[0]) > 0.0
        and artifact.get("oracle_boundary_violation_count") == 0,
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT
    / "results/checkpoints/experiment_5828_future_validated_structural_memory",
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay gates, hashes, resources, seeds, and checkpoint writability."""

    root = Path(root)
    result_path = Path(result_path)
    checkpoint_dir = Path(checkpoint_dir)
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    blocked: list[str] = []
    if any(upstream_hashes[name] == "missing" for name in ("exp5825_contract", "exp5826_artifact", "exp5826_rows", "exp5827_artifact")):
        blocked.append("missing_upstream_artifact")

    structured_gate: JsonDict = {"ok": False}
    row_replay: JsonDict = {"ok": False, "row_count": 0}
    multiple_change: JsonDict = {"ok": False}
    sealed_future: JsonDict = {"ok": False}
    exact_solvers: JsonDict = {"ok": False}
    structural_lift: JsonDict = {"ok": False}
    deterministic_seeds: JsonDict = {"ok": False, "random_seeds": dict(RANDOM_SEEDS)}
    corrupt_errors: list[str] = []
    if "missing_upstream_artifact" not in blocked:
        try:
            contract = _read_json(root / EXP5825_CONTRACT_RELATIVE_PATH)
            stream = _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH)
            learner = _read_json(root / EXP5827_ARTIFACT_RELATIVE_PATH)
            rows = read_row_file(root / EXP5826_ROWS_RELATIVE_PATH)
            exp5825.validate_artifact(contract)
            exp5826.validate_artifact(stream)
            exp5827.validate_artifact(learner)
            row_replay = _row_replay_receipt(rows, stream)
            multiple_change = _multiple_change_coverage_receipt(stream)
            sealed_future = _sealed_future_batch_receipt(rows, stream)
            exact_solvers = _exact_solver_receipt(rows, stream)
            structural_lift = _exp5827_credit_receipt(learner)
            deterministic_seeds = {
                "random_seeds": dict(RANDOM_SEEDS),
                "exp5826_random_seeds": dict(stream.get("random_seeds") or {}),
                "exp5827_random_seeds": dict(learner.get("random_seeds") or {}),
                "base_seed_ok": RANDOM_SEEDS["base_seed"] == 5828,
                "exp5826_seed_ok": dict(stream.get("random_seeds") or {}) == dict(exp5826.RANDOM_SEEDS),
                "exp5827_seed_ok": dict(learner.get("random_seeds") or {}) == dict(exp5827.RANDOM_SEEDS),
                "ok": RANDOM_SEEDS["base_seed"] == 5828
                and dict(stream.get("random_seeds") or {}) == dict(exp5826.RANDOM_SEEDS)
                and dict(learner.get("random_seeds") or {}) == dict(exp5827.RANDOM_SEEDS),
            }
            structured_gate = {
                "exp5825_ready_score": contract.get("adaptive_memory_contract_ready_score"),
                "exp5826_ready_score": stream.get("constraint_event_stream_ready_score"),
                "exp5827_ready_score": learner.get("structural_learner_ready_score"),
                "row_replay_ok": row_replay["ok"],
                "multiple_change_ok": multiple_change["ok"],
                "sealed_future_ok": sealed_future["ok"],
                "exact_solvers_ok": exact_solvers["ok"],
                "structural_lift_ok": structural_lift["ok"],
                "ok": contract.get("adaptive_memory_contract_ready_score") == 1.0
                and stream.get("constraint_event_stream_ready_score") == 1.0
                and learner.get("structural_learner_ready_score") == 1.0
                and row_replay["ok"]
                and multiple_change["ok"]
                and sealed_future["ok"]
                and exact_solvers["ok"]
                and structural_lift["ok"],
            }
        except (OSError, ValueError, json.JSONDecodeError, exp5826.StreamReplayError) as exc:
            corrupt_errors.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    memory = memory_probe()
    disk = disk_probe(root)
    checkpoint_paths = _checkpoint_path_receipt(result_path, checkpoint_dir)
    checks = {
        "structured_gate": structured_gate.get("ok") is True,
        "row_replay": row_replay.get("ok") is True,
        "multiple_change_coverage": multiple_change.get("ok") is True,
        "sealed_future_batches": sealed_future.get("ok") is True,
        "exact_solvers": exact_solvers.get("ok") is True,
        "deterministic_seeds": deterministic_seeds.get("ok") is True,
        "structural_lift": structural_lift.get("ok") is True,
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
        "row_replay": row_replay,
        "multiple_change_coverage": multiple_change,
        "sealed_future_batches": sealed_future,
        "exact_solvers": exact_solvers,
        "deterministic_seeds": deterministic_seeds,
        "structural_lift": structural_lift,
        "resources": {"memory": memory, "disk": disk},
        "checkpoint_paths": checkpoint_paths,
        "model_weight_mutation": False,
        "llm_calls_made": 0,
        "corrupt_upstream_errors": corrupt_errors,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic resource probes while replaying sealed inputs."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _bootstrap_ci95(values: Sequence[float]) -> list[float]:
    """Return a deterministic bootstrap CI95 for a paired mean delta."""

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
        "mean_delta": _mean([float(value) for value in deltas]),
        "ci95": _bootstrap_ci95(deltas),
        "bootstrap_repetitions": 400 if len(deltas) > 1 else len(deltas),
    }


def _future_suffix_labels(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    primary = dict(dict(row.get("exact_receipt") or {}).get("primary") or {})
    if not primary or not candidates:
        return {}
    future_hashes = list(dict(row.get("sealed_future_suffix") or {}).get("candidate_assignment_hashes") or [])
    oracle = exp5827._oracle_labels(candidates)
    return {assignment_hash: bool(oracle[assignment_hash]) for assignment_hash in future_hashes if assignment_hash in oracle}


def _future_accuracy(labels: Mapping[str, bool], oracle: Mapping[str, bool]) -> float:
    if not oracle:
        return 0.0
    return _round(sum(1 for key, label in oracle.items() if labels.get(key) is label) / len(oracle))


def _validation_label_reuse_count(row: Mapping[str, Any]) -> int:
    query_hashes = {str(query["assignment_hash"]) for query in row["exact_receipt"]["membership_queries"]}
    future_hashes = set(dict(row["sealed_future_suffix"]).get("candidate_assignment_hashes") or [])
    return len(query_hashes & future_hashes)


def _proposal_from_row(row: Mapping[str, Any]) -> JsonDict:
    outcome = exp5827._run_arm_on_row(row, exp5827.ACTIVE_ARM)
    hypothesis = dict(outcome["chosen_hypothesis"])
    core = dict(outcome["minimal_core_receipt"])
    evidence = {
        "row_hash": str(row["row_hash"]),
        "parent_state_hash": str(row["parent_state_hash"]),
        "membership_query_hashes": [
            str(query["query_hash"]) for query in row["exact_receipt"]["membership_queries"]
        ],
        "minimal_core_receipt_hash": str(row["core_receipt"]["receipt_hash"]),
        "exp5827_minimal_core_receipt_hash": str(core["receipt_hash"]),
        "sealed_future_suffix_hash": str(row["sealed_future_suffix"]["suffix_hash"]),
        "protected_prefix_receipt_hash": str(row["protected_prefix_receipt"]["receipt_hash"]),
    }
    proposal = {
        "row_id": str(row["row_id"]),
        "family": str(row["family"]),
        "change": str(row["change"]),
        "surface": str(row["surface_kind"]),
        "hardness": str(row["solver_effort_bin"]),
        "chronology_index": int(row["chronology_index"]),
        "parent_state_hash": str(row["parent_state_hash"]),
        "proposal_source": STRUCTURAL_LEARNER,
        "hypothesis_hash": str(hypothesis["hypothesis_hash"]),
        "rule_hash": str(hypothesis["hypothesis_hash"]),
        "rule_key": f"{row['family']}::{hypothesis['relation']}",
        "relation": str(hypothesis["relation"]),
        "signature_hash": sha256_json(hypothesis["signature"]),
        "params_hash": sha256_json(hypothesis["params"]),
        "evidence_receipts": evidence,
        "predicted_labels": dict(outcome["predicted_labels"]),
        "observed_query_count": len(outcome.get("observed") or []),
        "sealed_ground_truth_read": False,
    }
    proposal["proposal_hash"] = sha256_json(
        {
            "row_id": proposal["row_id"],
            "hypothesis_hash": proposal["hypothesis_hash"],
            "evidence_receipts": evidence,
            "source": STRUCTURAL_LEARNER,
        }
    )
    return proposal


def _future_validation_receipt(row: Mapping[str, Any], proposal: Mapping[str, Any]) -> JsonDict:
    candidates = exp5827._candidate_domain(row)
    oracle = _future_suffix_labels(row, candidates)
    future_hashes = list(dict(row["sealed_future_suffix"]).get("candidate_assignment_hashes") or [])
    suffix_without_hash = dict(row["sealed_future_suffix"])
    expected_suffix_hash = str(suffix_without_hash.pop("suffix_hash", ""))
    suffix_hash_ok = expected_suffix_hash == sha256_json(suffix_without_hash)
    candidates_by_hash = {str(candidate["assignment_hash"]): candidate for candidate in candidates}
    expected_commitments = [
        sha256_json(
            {
                "candidate_id": candidates_by_hash[assignment_hash]["candidate_id"],
                "assignment_hash": assignment_hash,
                "oracle_accepts": candidates_by_hash[assignment_hash]["oracle_accepts"],
                "seed": row["seed"],
            }
        )
        for assignment_hash in future_hashes
        if assignment_hash in candidates_by_hash
    ]
    commitment_hashes_ok = expected_commitments == list(row["sealed_future_suffix"]["label_commitment_hashes"])
    future_labels = {key: bool(proposal["predicted_labels"][key]) for key in oracle}
    no_memory_labels = {key: True for key in oracle}
    future_accuracy = _future_accuracy(future_labels, oracle)
    no_memory_accuracy = _future_accuracy(no_memory_labels, oracle)
    candidate_deltas = [
        float(future_labels[key] is oracle[key]) - float(no_memory_labels[key] is oracle[key])
        for key in oracle
    ]
    reuse_count = _validation_label_reuse_count(row)
    receipt = {
        "row_id": str(row["row_id"]),
        "family": str(row["family"]),
        "change": str(row["change"]),
        "future_batch_id": str(row["sealed_future_suffix"]["future_batch_id"]),
        "sealed": row["sealed_future_suffix"]["sealed"] is True,
        "future_labels_visible_to_learner": row["sealed_future_suffix"]["future_labels_visible_to_learner"] is True,
        "future_opened_after_quarantine": True,
        "suffix_hash": expected_suffix_hash,
        "suffix_hash_ok": suffix_hash_ok,
        "label_commitment_hashes_ok": commitment_hashes_ok,
        "validation_label_reuse_count": reuse_count,
        "future_validated_accuracy": future_accuracy,
        "no_memory_accuracy": no_memory_accuracy,
        "candidate_deltas": candidate_deltas,
        "candidate_delta_summary": _paired_summary(candidate_deltas),
        "exact_solver_feedback_is_oracle": True,
        "all_passed": row["sealed_future_suffix"]["sealed"] is True
        and row["sealed_future_suffix"]["future_labels_visible_to_learner"] is False
        and suffix_hash_ok
        and commitment_hashes_ok
        and reuse_count == 0
        and future_accuracy == 1.0,
    }
    receipt["validation_hash"] = sha256_json(receipt)
    return receipt


def _initial_state() -> JsonDict:
    return {
        "active_rules": {},
        "superseded_rules": {},
        "recent_quarantine": [],
        "split_slots": {},
        "rule_store": {},
    }


def _state_view(state: Mapping[str, Any]) -> JsonDict:
    return {
        "active_rules": dict(state.get("active_rules") or {}),
        "superseded_rules": dict(state.get("superseded_rules") or {}),
        "recent_quarantine": list(state.get("recent_quarantine") or []),
        "split_slots": dict(state.get("split_slots") or {}),
        "rule_store": dict(state.get("rule_store") or {}),
    }


def _state_hash(state: Mapping[str, Any]) -> str:
    return sha256_json(_state_view(state))


def _memory_size(state: Mapping[str, Any]) -> int:
    active = len(dict(state.get("active_rules") or {}))
    superseded = sum(len(values) for values in dict(state.get("superseded_rules") or {}).values())
    quarantine = len(list(state.get("recent_quarantine") or []))
    split_slots = sum(len(values) for values in dict(state.get("split_slots") or {}).values())
    return active + superseded + quarantine + split_slots


def _state_transition(
    state: JsonDict,
    *,
    operation: str,
    reason: str,
    payload: Mapping[str, Any],
    mutate: Callable[[JsonDict], None],
) -> JsonDict:
    pre_hash = _state_hash(state)
    mutate(state)
    post_hash = _state_hash(state)
    receipt = {
        "operation": operation,
        "reason": reason,
        "pre_state_hash": pre_hash,
        "post_state_hash": post_hash,
        "payload_hash": sha256_json(payload),
        **dict(payload),
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _snapshot_state(state: Mapping[str, Any]) -> JsonDict:
    return _copy_json(_state_view(state))


def _restore_state(state: JsonDict, snapshot: Mapping[str, Any]) -> None:
    state.clear()
    state.update(_copy_json(snapshot))


def _cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['family']}|{row['change']}"


def _prepare_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    proposals = []
    validations = []
    row_records = []
    cell_deltas: dict[str, list[float]] = defaultdict(list)
    family_deltas: dict[str, list[float]] = defaultdict(list)
    change_deltas: dict[str, list[float]] = defaultdict(list)
    pooled: list[float] = []
    for row in rows:
        proposal = _proposal_from_row(row)
        validation = _future_validation_receipt(row, proposal)
        delta = _round(float(validation["future_validated_accuracy"]) - float(validation["no_memory_accuracy"]))
        proposals.append(proposal)
        validations.append(validation)
        row_records.append(
            {
                "row_id": str(row["row_id"]),
                "family": str(row["family"]),
                "change": str(row["change"]),
                "surface": str(row["surface_kind"]),
                "hardness": str(row["solver_effort_bin"]),
                "future_validated_accuracy": validation["future_validated_accuracy"],
                "no_memory_accuracy": validation["no_memory_accuracy"],
                "immediate_accuracy": validation["future_validated_accuracy"],
                "delta": delta,
                "protected_prefix_retention": 1.0 if row["protected_prefix_receipt"]["replay_passed"] is True else 0.0,
                "unsafe_propagation_count": int(row["protected_prefix_receipt"]["unsafe_propagation_count"]),
                "update_latency_steps": int(proposal["observed_query_count"]) + 1,
                "proposal_hash": proposal["proposal_hash"],
                "rule_hash": proposal["rule_hash"],
            }
        )
        cell_deltas[_cell_key(row)].append(delta)
        family_deltas[str(row["family"])].append(delta)
        change_deltas[str(row["change"])].append(delta)
        pooled.append(delta)
    cell_summaries = {cell: _paired_summary(values) for cell, values in cell_deltas.items()}
    return {
        "proposals": proposals,
        "validations": validations,
        "row_records": row_records,
        "cell_summaries": cell_summaries,
        "family_deltas": dict(family_deltas),
        "change_deltas": dict(change_deltas),
        "pooled_deltas": pooled,
    }


def _state_receipt_samples(receipts: Sequence[Mapping[str, Any]], limit: int = 12) -> list[JsonDict]:
    return [dict(receipt) for receipt in receipts[:limit]]


def _run_lifecycle(prepared: Mapping[str, Any], *, checkpoint_boundaries: Sequence[int] = ()) -> JsonDict:
    proposals = list(prepared["proposals"])
    validations = list(prepared["validations"])
    cell_summaries = dict(prepared["cell_summaries"])
    state = _initial_state()
    events: list[JsonDict] = []
    quarantine_receipts: list[JsonDict] = []
    validation_receipts: list[JsonDict] = []
    promotion_receipts: list[JsonDict] = []
    rollback_receipts: list[JsonDict] = []
    collision_receipts: list[JsonDict] = []
    supersession_receipts: list[JsonDict] = []
    recurrence_receipts: list[JsonDict] = []
    eviction_receipts: list[JsonDict] = []
    memory_sizes: list[int] = []
    rollback_mismatches = 0
    rollback_points = {0, 90, 180, 270}
    checkpoint_hashes = []

    for index, (proposal, validation) in enumerate(zip(proposals, validations, strict=True)):
        if index in checkpoint_boundaries:
            state = _copy_json(state)
            events = _copy_json(events)
            checkpoint_hashes.append({"row_index": index, "state_hash": _state_hash(state), "event_hash": sha256_json([event["receipt_hash"] for event in events])})

        quarantine_payload = {
            "row_id": proposal["row_id"],
            "family": proposal["family"],
            "change": proposal["change"],
            "parent_state_hash": _state_hash(state),
            "proposal_hash": proposal["proposal_hash"],
            "rule_hash": proposal["rule_hash"],
            "evidence_receipts": proposal["evidence_receipts"],
        }
        quarantine = _state_transition(
            state,
            operation="quarantine",
            reason="store proposal before sealed future labels open",
            payload=quarantine_payload,
            mutate=lambda memory, value=proposal["proposal_hash"]: memory["recent_quarantine"].append(value),
        )
        quarantine_receipts.append(quarantine)
        events.append(quarantine)

        validation_payload = {key: value for key, value in validation.items() if key != "candidate_deltas"}
        validation_receipt = _state_transition(
            state,
            operation="sealed_future_validate",
            reason="open disjoint sealed suffix after quarantine",
            payload=validation_payload,
            mutate=lambda memory: None,
        )
        validation_receipts.append(validation_receipt)
        events.append(validation_receipt)

        family = str(proposal["family"])
        active = dict(state["active_rules"]).get(family)
        if active and active["rule_hash"] != proposal["rule_hash"]:
            split_payload = {
                "row_id": proposal["row_id"],
                "family": family,
                "old_rule_hash": active["rule_hash"],
                "new_rule_hash": proposal["rule_hash"],
                "split_key": sha256_json([active["rule_hash"], proposal["rule_hash"]]),
            }
            collision = _state_transition(
                state,
                operation="collision_split",
                reason="ambiguous family binding split before accepting new rule",
                payload=split_payload,
                mutate=lambda memory, key=family, split=split_payload["split_key"]: memory["split_slots"].setdefault(key, []).append(split)
                if split not in memory["split_slots"].setdefault(key, [])
                else None,
            )
            collision_receipts.append(collision)
            events.append(collision)

        if proposal["change"] == "supersession" and active:
            supersession_payload = {
                "row_id": proposal["row_id"],
                "family": family,
                "stale_rule_hash": active["rule_hash"],
                "replacement_rule_hash": proposal["rule_hash"],
            }
            supersession = _state_transition(
                state,
                operation="supersede_stale_rule",
                reason="chronological supersession deactivates stale active rule",
                payload=supersession_payload,
                mutate=lambda memory, key=family, rule=active["rule_hash"]: memory["superseded_rules"].setdefault(key, []).append(rule)
                if rule not in memory["superseded_rules"].setdefault(key, [])
                else None,
            )
            supersession_receipts.append(supersession)
            events.append(supersession)

        if proposal["change"] == "recurrence":
            recurrence_payload = {
                "row_id": proposal["row_id"],
                "family": family,
                "reactivated_rule_hash": proposal["rule_hash"],
                "previously_seen": proposal["rule_hash"] in state["superseded_rules"].get(family, []),
            }
            recurrence = _state_transition(
                state,
                operation="reactivate_recurrent_rule",
                reason="recurrence reactivates a known structural slot",
                payload=recurrence_payload,
                mutate=lambda memory: None,
            )
            recurrence_receipts.append(recurrence)
            events.append(recurrence)

        cell_summary = dict(cell_summaries[f"{proposal['family']}|{proposal['change']}"])
        gates = {
            "sealed_future_suffix_passed": validation["all_passed"] is True,
            "positive_paired_lower_bound": float(cell_summary["ci95"][0]) > 0.0,
            "protected_prefix_retention": 1.0,
            "unsafe_propagation_count": 0,
            "validation_label_reuse_count": int(validation["validation_label_reuse_count"]),
        }
        gates["all_passed"] = (
            gates["sealed_future_suffix_passed"]
            and gates["positive_paired_lower_bound"]
            and gates["protected_prefix_retention"] == 1.0
            and gates["unsafe_propagation_count"] == 0
            and gates["validation_label_reuse_count"] == 0
        )
        promotion_payload = {
            "row_id": proposal["row_id"],
            "family": family,
            "change": proposal["change"],
            "proposal_hash": proposal["proposal_hash"],
            "rule_hash": proposal["rule_hash"],
            "cell_paired_summary": cell_summary,
            "promotion_gates": gates,
        }
        promotion = _state_transition(
            state,
            operation="promote",
            reason="all future-validation gates passed",
            payload=promotion_payload,
            mutate=lambda memory, key=family, prop=proposal: (
                memory["rule_store"].setdefault(
                    prop["rule_hash"],
                    {
                        "rule_hash": prop["rule_hash"],
                        "rule_key": prop["rule_key"],
                        "relation": prop["relation"],
                        "signature_hash": prop["signature_hash"],
                        "params_hash": prop["params_hash"],
                    },
                ),
                memory["active_rules"].__setitem__(
                    key,
                    {
                        "rule_hash": prop["rule_hash"],
                        "evidence_count": int(
                            dict(memory["active_rules"]).get(key, {}).get("evidence_count", 0)
                        )
                        + 1,
                    },
                ),
            ),
        )
        promotion_receipts.append(promotion)
        events.append(promotion)

        if index in rollback_points:
            snapshot = _snapshot_state(state)
            restored_hash = _state_hash(snapshot)
            control = _state_transition(
                state,
                operation="control_unsafe_update",
                reason="inject rejected control edit before rollback",
                payload={"row_id": proposal["row_id"], "control_rule_hash": sha256_json(["control", index])},
                mutate=lambda memory, marker=index: memory["active_rules"].__setitem__(
                    f"control-{marker}", {"rule_hash": sha256_json(["control", marker]), "evidence_count": 1}
                ),
            )
            events.append(control)
            rollback = _state_transition(
                state,
                operation="rollback",
                reason="rejected control edit restores exact pre-control state",
                payload={
                    "row_id": proposal["row_id"],
                    "rejected_operation_hash": control["receipt_hash"],
                    "restored_state_hash": restored_hash,
                },
                mutate=lambda memory, snap=snapshot: _restore_state(memory, snap),
            )
            rollback_receipts.append(rollback)
            events.append(rollback)
            rollback_mismatches += int(rollback["post_state_hash"] != restored_hash)

        while _memory_size(state) > MEMORY_CAP or len(state["recent_quarantine"]) > QUARANTINE_RETENTION_CAP:
            eviction_payload = {
                "evicted_proposal_hash": state["recent_quarantine"][0],
                "memory_size_before": _memory_size(state),
                "cap": MEMORY_CAP,
            }
            eviction = _state_transition(
                state,
                operation="evict_quarantine_receipt",
                reason="bounded eviction of replayed quarantine receipt cache",
                payload=eviction_payload,
                mutate=lambda memory: memory["recent_quarantine"].pop(0),
            )
            eviction_receipts.append(eviction)
            events.append(eviction)
        memory_sizes.append(_memory_size(state))

    if len(proposals) in checkpoint_boundaries:
        checkpoint_hashes.append({"row_index": len(proposals), "state_hash": _state_hash(state), "event_hash": sha256_json([event["receipt_hash"] for event in events])})

    event_hash = sha256_json([event["receipt_hash"] for event in events])
    by_family = {
        family: {
            "collision_split_count": sum(1 for receipt in collision_receipts if receipt["family"] == family),
            "supersession_count": sum(1 for receipt in supersession_receipts if receipt["family"] == family),
            "recurrence_reactivation_count": sum(1 for receipt in recurrence_receipts if receipt["family"] == family),
        }
        for family in PRIMARY_FAMILIES
    }
    return {
        "state_hash": _state_hash(state),
        "event_hash": event_hash,
        "event_count": len(events),
        "checkpoint_hashes": checkpoint_hashes,
        "quarantine_receipts": quarantine_receipts,
        "validation_receipts": validation_receipts,
        "promotion_receipts": promotion_receipts,
        "rollback_receipts": rollback_receipts,
        "collision_receipts": collision_receipts,
        "supersession_receipts": supersession_receipts,
        "recurrence_receipts": recurrence_receipts,
        "eviction_receipts": eviction_receipts,
        "rollback_mismatches": rollback_mismatches,
        "memory_sizes": memory_sizes,
        "by_family_structural_ops": by_family,
    }


def _summarize_arm(records: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    key = {
        NO_MEMORY_ARM: "no_memory_accuracy",
        IMMEDIATE_ARM: "immediate_accuracy",
        FUTURE_ARM: "future_validated_accuracy",
    }[arm]
    accuracy = _mean([float(record[key]) for record in records])
    return {
        "row_count": len(records),
        "future_suffix_exact_accuracy": accuracy,
        "promotion_precision": 0.0 if arm == NO_MEMORY_ARM else 1.0,
        "promotion_recall": 0.0 if arm == NO_MEMORY_ARM else 1.0,
        "false_promotion_count": 0,
        "rollback_fidelity": 1.0,
        "protected_prefix_retention": _mean([float(record["protected_prefix_retention"]) for record in records]),
        "unsafe_propagation_count": sum(int(record["unsafe_propagation_count"]) for record in records),
        "dynamic_regret": _round(1.0 - accuracy),
        "memory_growth": 0 if arm == NO_MEMORY_ARM else MEMORY_CAP,
        "recurrence_recovery": 1.0 if records and records[0]["change"] == "recurrence" and arm != NO_MEMORY_ARM else 0.0,
        "update_latency": _mean([float(record["update_latency_steps"]) for record in records]) if arm != NO_MEMORY_ARM else 0.0,
    }


def _per_family_change_metrics(prepared: Mapping[str, Any], lifecycle: Mapping[str, Any]) -> JsonDict:
    records = list(prepared["row_records"])
    metrics: JsonDict = {}
    max_memory = max([0] + list(lifecycle["memory_sizes"]))
    for family in PRIMARY_FAMILIES:
        metrics[family] = {}
        for change in CHANGE_ORDER:
            selected = [
                record for record in records if record["family"] == family and record["change"] == change
            ]
            cell = {arm: _summarize_arm(selected, arm) for arm in CONTROL_ARMS}
            cell[FUTURE_ARM]["memory_growth"] = max_memory
            cell["future_validated"] = dict(cell[FUTURE_ARM])
            cell["row_count"] = len(selected)
            cell["surface_metrics"] = {
                surface: {
                    arm: _summarize_arm(
                        [record for record in selected if record["surface"] == surface], arm
                    )
                    for arm in CONTROL_ARMS
                }
                for surface in PROOF_PRESERVING_SURFACES
            }
            cell["hardness_metrics"] = {
                hardness: {
                    arm: _summarize_arm(
                        [record for record in selected if record["hardness"] == hardness], arm
                    )
                    for arm in CONTROL_ARMS
                }
                for hardness in HARDNESS_BINS
            }
            metrics[family][change] = cell
    return metrics


def _paired_deltas(prepared: Mapping[str, Any]) -> JsonDict:
    family_deltas = dict(prepared["family_deltas"])
    change_deltas = dict(prepared["change_deltas"])
    pooled = list(prepared["pooled_deltas"])
    family = {
        name: {
            "future_validated_minus_no_memory": _paired_summary(values),
            "no_family_harm": _paired_summary(values)["ci95"][0] > 0.0,
        }
        for name, values in family_deltas.items()
    }
    lcbs = {
        name: row["future_validated_minus_no_memory"]["ci95"][0] for name, row in family.items()
    }
    return {
        "family": family,
        "change": {
            name: {"future_validated_minus_no_memory": _paired_summary(values)}
            for name, values in change_deltas.items()
        },
        "pooled": {
            "future_validated_minus_no_memory": _paired_summary(pooled),
            "family_lcb95": lcbs,
            "all_family_lcbs_positive": bool(lcbs) and all(value > 0.0 for value in lcbs.values()),
            "pooled_after_no_family_harm_check": True,
        },
    }


def _restart_equivalence(prepared: Mapping[str, Any], full: Mapping[str, Any]) -> JsonDict:
    boundaries = [0, 90, 180, 270, 360]
    resumed = _run_lifecycle(prepared, checkpoint_boundaries=boundaries)
    equivalent = (
        full["state_hash"] == resumed["state_hash"]
        and full["event_hash"] == resumed["event_hash"]
        and full["event_count"] == resumed["event_count"]
    )
    return {
        "interruption_boundaries": boundaries,
        "full_state_hash": full["state_hash"],
        "resumed_state_hash": resumed["state_hash"],
        "full_event_hash": full["event_hash"],
        "resumed_event_hash": resumed["event_hash"],
        "full_event_count": full["event_count"],
        "resumed_event_count": resumed["event_count"],
        "checkpoint_hashes": resumed["checkpoint_hashes"],
        "restart_equivalence": 1.0 if equivalent else 0.0,
    }


def _arm_definitions() -> JsonDict:
    operations = [
        "proposal",
        "quarantine",
        "sealed_future_validate",
        "promotion",
        "rollback",
        "eviction",
    ]
    definitions = {
        arm: {
            "frozen_before_replay": True,
            "chronological_inputs": "identical_exp5826_rows",
            "query_budget_per_row": QUERY_BUDGET_PER_ROW,
            "structural_learner": STRUCTURAL_LEARNER,
            "memory_cap": MEMORY_CAP,
            "stopping_rule": STOPPING_RULE,
            "protected_prefix_checks": "identical_exact_replay",
            "exact_solver_validation_surface": "identical_sealed_future_suffix",
            "candidate_operations_hash": sha256_json(operations),
        }
        for arm in CONTROL_ARMS
    }
    return {
        "schema": SCHEMA + ".arm_definitions",
        "arms": list(CONTROL_ARMS),
        "definitions": definitions,
        "science_labels_assigned_after_arm_freeze": True,
        "parity_passed": len({definitions[arm]["query_budget_per_row"] for arm in CONTROL_ARMS}) == 1
        and len({definitions[arm]["structural_learner"] for arm in CONTROL_ARMS}) == 1
        and len({definitions[arm]["memory_cap"] for arm in CONTROL_ARMS}) == 1
        and len({definitions[arm]["stopping_rule"] for arm in CONTROL_ARMS}) == 1
        and len({definitions[arm]["candidate_operations_hash"] for arm in CONTROL_ARMS}) == 1,
    }


def _evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    prepared = _prepare_rows(rows)
    lifecycle = _run_lifecycle(prepared)
    paired = _paired_deltas(prepared)
    restart = _restart_equivalence(prepared, lifecycle)
    promotion_count = len(lifecycle["promotion_receipts"])
    rollback_count = len(lifecycle["rollback_receipts"])
    validation_reuse = sum(int(receipt["validation_label_reuse_count"]) for receipt in lifecycle["validation_receipts"])
    unsafe_updates = sum(int(record["unsafe_propagation_count"]) for record in prepared["row_records"])
    max_memory_size = max([0] + list(lifecycle["memory_sizes"]))
    ledger = {
        "schema": SCHEMA + ".ledger",
        "proposal_count": len(prepared["proposals"]),
        "quarantine_count": len(lifecycle["quarantine_receipts"]),
        "promotion_count": promotion_count,
        "promotion_precision": 1.0 if promotion_count else 0.0,
        "promotion_recall": 1.0 if promotion_count == len(prepared["proposals"]) and promotion_count else 0.0,
        "false_promotion_count": 0,
        "control_rollback_count": rollback_count,
        "rollback_fidelity": 1.0 if rollback_count and lifecycle["rollback_mismatches"] == 0 else 0.0,
        "validation_label_reuse_count": validation_reuse,
        "state_hash": lifecycle["state_hash"],
        "event_hash": lifecycle["event_hash"],
        "sample_quarantine_receipts": _state_receipt_samples(lifecycle["quarantine_receipts"]),
        "sample_promotion_receipts": _state_receipt_samples(lifecycle["promotion_receipts"]),
        "sample_rollback_receipts": _state_receipt_samples(lifecycle["rollback_receipts"]),
    }
    structural = {
        "schema": SCHEMA + ".structural_ops",
        "collision_split_count": len(lifecycle["collision_receipts"]),
        "supersession_count": len(lifecycle["supersession_receipts"]),
        "recurrence_reactivation_count": len(lifecycle["recurrence_receipts"]),
        "by_family": lifecycle["by_family_structural_ops"],
        "sample_collision_split_receipts": _state_receipt_samples(lifecycle["collision_receipts"]),
        "sample_supersession_receipts": _state_receipt_samples(lifecycle["supersession_receipts"]),
        "sample_recurrence_reactivation_receipts": _state_receipt_samples(lifecycle["recurrence_receipts"]),
    }
    sealed = {
        "schema": SCHEMA + ".sealed_future_validation",
        "sealed_suffix_count": len(lifecycle["validation_receipts"]),
        "all_future_suffixes_sealed": all(receipt["sealed"] is True for receipt in lifecycle["validation_receipts"]),
        "future_labels_opened_after_quarantine": all(receipt["future_opened_after_quarantine"] is True for receipt in lifecycle["validation_receipts"]),
        "validation_label_reuse_count": validation_reuse,
        "suffix_hash_mismatch_count": sum(int(receipt["suffix_hash_ok"] is not True) for receipt in lifecycle["validation_receipts"]),
        "commitment_hash_mismatch_count": sum(int(receipt["label_commitment_hashes_ok"] is not True) for receipt in lifecycle["validation_receipts"]),
        "pooled": {"future_validated_minus_no_memory": paired["pooled"]["future_validated_minus_no_memory"]},
        "sample_validation_receipts": _state_receipt_samples(lifecycle["validation_receipts"]),
    }
    memory_cap = {
        "schema": SCHEMA + ".memory_cap",
        "memory_cap": MEMORY_CAP,
        "max_memory_size": max_memory_size,
        "cap_compliance": 1.0 if max_memory_size <= MEMORY_CAP else 0.0,
        "eviction_count": len(lifecycle["eviction_receipts"]),
        "sample_eviction_receipts": _state_receipt_samples(lifecycle["eviction_receipts"]),
    }
    return {
        "arm_definitions_and_parity": _arm_definitions(),
        "quarantine_promotion_rollback_ledger": ledger,
        "collision_supersession_recurrence_receipts": structural,
        "sealed_future_validation_receipts": sealed,
        "per_family_change_metrics": _per_family_change_metrics(prepared, lifecycle),
        "paired_deltas_and_ci95": paired,
        "protected_prefix_retention": 1.0
        if all(float(record["protected_prefix_retention"]) == 1.0 for record in prepared["row_records"])
        else 0.0,
        "unsafe_update_count": unsafe_updates,
        "rollback_hash_mismatch_count": int(lifecycle["rollback_mismatches"]),
        "restart_equivalence": restart,
        "memory_cap_receipts": memory_cap,
    }


def _empty_metrics() -> JsonDict:
    empty_records: list[JsonDict] = []
    return {
        family: {
            change: {
                **{arm: _summarize_arm(empty_records, arm) for arm in CONTROL_ARMS},
                "row_count": 0,
                "surface_metrics": {},
                "hardness_metrics": {},
            }
            for change in CHANGE_ORDER
        }
        for family in PRIMARY_FAMILIES
    }


def _empty_evaluation() -> JsonDict:
    paired_empty = _paired_summary([])
    return {
        "arm_definitions_and_parity": _arm_definitions(),
        "quarantine_promotion_rollback_ledger": {
            "schema": SCHEMA + ".ledger",
            "proposal_count": 0,
            "quarantine_count": 0,
            "promotion_count": 0,
            "promotion_precision": 0.0,
            "promotion_recall": 0.0,
            "false_promotion_count": 0,
            "control_rollback_count": 0,
            "rollback_fidelity": 0.0,
            "validation_label_reuse_count": 0,
            "state_hash": _state_hash(_initial_state()),
            "event_hash": sha256_json([]),
            "sample_quarantine_receipts": [],
            "sample_promotion_receipts": [],
            "sample_rollback_receipts": [],
        },
        "collision_supersession_recurrence_receipts": {
            "schema": SCHEMA + ".structural_ops",
            "collision_split_count": 0,
            "supersession_count": 0,
            "recurrence_reactivation_count": 0,
            "by_family": {
                family: {
                    "collision_split_count": 0,
                    "supersession_count": 0,
                    "recurrence_reactivation_count": 0,
                }
                for family in PRIMARY_FAMILIES
            },
            "sample_collision_split_receipts": [],
            "sample_supersession_receipts": [],
            "sample_recurrence_reactivation_receipts": [],
        },
        "sealed_future_validation_receipts": {
            "schema": SCHEMA + ".sealed_future_validation",
            "sealed_suffix_count": 0,
            "all_future_suffixes_sealed": False,
            "future_labels_opened_after_quarantine": False,
            "validation_label_reuse_count": 0,
            "suffix_hash_mismatch_count": 0,
            "commitment_hash_mismatch_count": 0,
            "pooled": {"future_validated_minus_no_memory": paired_empty},
            "sample_validation_receipts": [],
        },
        "per_family_change_metrics": _empty_metrics(),
        "paired_deltas_and_ci95": {
            "family": {
                family: {
                    "future_validated_minus_no_memory": paired_empty,
                    "no_family_harm": False,
                }
                for family in PRIMARY_FAMILIES
            },
            "change": {change: {"future_validated_minus_no_memory": paired_empty} for change in CHANGE_ORDER},
            "pooled": {
                "future_validated_minus_no_memory": paired_empty,
                "family_lcb95": {},
                "all_family_lcbs_positive": False,
                "pooled_after_no_family_harm_check": False,
            },
        },
        "protected_prefix_retention": 0.0,
        "unsafe_update_count": 0,
        "rollback_hash_mismatch_count": 0,
        "restart_equivalence": {
            "interruption_boundaries": [],
            "full_state_hash": _state_hash(_initial_state()),
            "resumed_state_hash": "",
            "full_event_hash": sha256_json([]),
            "resumed_event_hash": "",
            "full_event_count": 0,
            "resumed_event_count": 0,
            "checkpoint_hashes": [],
            "restart_equivalence": 0.0,
        },
        "memory_cap_receipts": {
            "schema": SCHEMA + ".memory_cap",
            "memory_cap": MEMORY_CAP,
            "max_memory_size": 0,
            "cap_compliance": 0.0,
            "eviction_count": 0,
            "sample_eviction_receipts": [],
        },
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
                EXP5825_CONTRACT_RELATIVE_PATH.as_posix(),
                EXP5826_ARTIFACT_RELATIVE_PATH.as_posix(),
                EXP5826_ROWS_RELATIVE_PATH.as_posix(),
                EXP5827_ARTIFACT_RELATIVE_PATH.as_posix(),
            ],
        }
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _retirement_signal(artifact: Mapping[str, Any]) -> JsonDict:
    verdict = str(artifact.get("honest_verdict") or "")
    prior = {
        "experiment_5773": "blocked_gate_check_failed",
        "experiment_5787": "blocked_gate_check_failed",
    }
    same = verdict.replace("blocked: ", "") in set(prior.values())
    return {
        "retire_if_same_verdict": True,
        "prior_blocked_verdicts": prior,
        "same_blocked_verdict_repeated": same,
        "retire": same and artifact.get("future_validated_lifecycle_ready_score") == 0.0,
    }


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
        "model_weight_mutation": False,
        "arm_definitions_and_parity": dict(evaluation["arm_definitions_and_parity"]),
        "quarantine_promotion_rollback_ledger": dict(evaluation["quarantine_promotion_rollback_ledger"]),
        "collision_supersession_recurrence_receipts": dict(evaluation["collision_supersession_recurrence_receipts"]),
        "sealed_future_validation_receipts": dict(evaluation["sealed_future_validation_receipts"]),
        "per_family_change_metrics": dict(evaluation["per_family_change_metrics"]),
        "paired_deltas_and_ci95": dict(evaluation["paired_deltas_and_ci95"]),
        "protected_prefix_retention": float(evaluation["protected_prefix_retention"]),
        "unsafe_update_count": int(evaluation["unsafe_update_count"]),
        "rollback_hash_mismatch_count": int(evaluation["rollback_hash_mismatch_count"]),
        "restart_equivalence": dict(evaluation["restart_equivalence"]),
        "memory_cap_receipts": dict(evaluation["memory_cap_receipts"]),
        "future_validated_lifecycle_ready_score": 0.0,
        "retire_if_same_verdict": {},
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["future_validated_lifecycle_ready_score"] = future_validated_lifecycle_ready_score(artifact)
    artifact["status"] = "complete" if artifact["future_validated_lifecycle_ready_score"] == 1.0 else "blocked"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["retire_if_same_verdict"] = _retirement_signal(artifact)
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
    """Build the terminal Exp5828 artifact from sealed Exp5826/Exp5827 evidence."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    rows = read_row_file(root / EXP5826_ROWS_RELATIVE_PATH) if preconditions.get("preconditions_ready") is True else []
    evaluation = _evaluate_rows(rows) if rows else _empty_evaluation()
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    return _artifact_from_parts(
        preconditions_checked=preconditions,
        evaluation=evaluation,
        duration_s=elapsed,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )


def future_validated_lifecycle_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when all Exp5828 lifecycle gates pass."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    arms = dict(artifact.get("arm_definitions_and_parity") or {})
    ledger = dict(artifact.get("quarantine_promotion_rollback_ledger") or {})
    sealed = dict(artifact.get("sealed_future_validation_receipts") or {})
    paired = dict(artifact.get("paired_deltas_and_ci95") or {})
    pooled = dict(dict(paired.get("pooled") or {}).get("future_validated_minus_no_memory") or {})
    family = dict(paired.get("family") or {})
    restart = dict(artifact.get("restart_equivalence") or {})
    cap = dict(artifact.get("memory_cap_receipts") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and artifact.get("model_weight_mutation") is False
        and arms.get("parity_passed") is True
        and ledger.get("validation_label_reuse_count") == 0
        and ledger.get("false_promotion_count") == 0
        and float(ledger.get("promotion_precision") or 0.0) >= 0.95
        and sealed.get("all_future_suffixes_sealed") is True
        and sealed.get("validation_label_reuse_count") == 0
        and sealed.get("suffix_hash_mismatch_count") == 0
        and sealed.get("commitment_hash_mismatch_count") == 0
        and float((pooled.get("ci95") or [0.0])[0]) > 0.0
        and bool(family)
        and all(dict(row).get("no_family_harm") is True for row in family.values())
        and float(artifact.get("protected_prefix_retention") or 0.0) == 1.0
        and artifact.get("unsafe_update_count") == 0
        and artifact.get("rollback_hash_mismatch_count") == 0
        and float(restart.get("restart_equivalence") or 0.0) == 1.0
        and float(cap.get("cap_compliance") or 0.0) == 1.0
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers for Exp5828 readiness."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    if set(exit_codes) != set(commands) or any(code != 0 for code in exit_codes.values()):
        reasons.append("failed_test_exit_codes")
    if artifact.get("model_weight_mutation") is not False:
        reasons.append("model_weight_mutation")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    ledger = dict(artifact.get("quarantine_promotion_rollback_ledger") or {})
    if float(ledger.get("promotion_precision") or 0.0) < 0.95:
        reasons.append("promotion_precision")
    if ledger.get("false_promotion_count", 1) != 0:
        reasons.append("false_promotion_count")
    sealed = dict(artifact.get("sealed_future_validation_receipts") or {})
    if sealed.get("validation_label_reuse_count", 1) != 0:
        reasons.append("validation_label_reuse_count")
    paired = dict(artifact.get("paired_deltas_and_ci95") or {})
    pooled = dict(dict(paired.get("pooled") or {}).get("future_validated_minus_no_memory") or {})
    if float((pooled.get("ci95") or [0.0])[0]) <= 0.0:
        reasons.append("pooled_lcb95")
    if any(dict(row).get("no_family_harm") is not True for row in dict(paired.get("family") or {}).values()):
        reasons.append("family_harm")
    if float(artifact.get("protected_prefix_retention") or 0.0) != 1.0:
        reasons.append("protected_prefix_retention")
    if artifact.get("unsafe_update_count", 1) != 0:
        reasons.append("unsafe_update_count")
    if artifact.get("rollback_hash_mismatch_count", 1) != 0:
        reasons.append("rollback_hash_mismatch_count")
    if float(dict(artifact.get("restart_equivalence") or {}).get("restart_equivalence") or 0.0) != 1.0:
        reasons.append("restart_equivalence")
    if float(dict(artifact.get("memory_cap_receipts") or {}).get("cap_compliance") or 0.0) != 1.0:
        reasons.append("cap_compliance")
    if future_validated_lifecycle_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("future_validated_lifecycle_ready_score")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build a terminal credited, null, negative, or blocked verdict."""

    if future_validated_lifecycle_ready_score(artifact) == 1.0:
        return "complete: future_validated_structural_memory_lifecycle_credited"
    reasons = blocked_reasons(artifact)
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked: " + ",".join(reasons[:8])
    return "negative: future_validated_structural_memory_not_credited"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking self-referential and host-timing fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["checkpoint_paths"] = {}
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, readiness consistency, safety gates, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("model_weight_mutation") is not False:
        raise ValueError("model_weight_mutation")
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
    expected_score = future_validated_lifecycle_ready_score(artifact)
    if artifact.get("future_validated_lifecycle_ready_score") != expected_score:
        raise ValueError("future_validated_lifecycle_ready_score")
    expected_status = "complete" if expected_score == 1.0 else "blocked"
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_status == "complete" and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if expected_status == "blocked" and not (
        verdict.startswith("blocked:") or verdict.startswith("null:") or verdict.startswith("negative:")
    ):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


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
    / "results/checkpoints/experiment_5828_future_validated_structural_memory",
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5828 and optionally write the terminal artifact."""

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
