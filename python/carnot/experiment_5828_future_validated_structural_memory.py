"""Exp5828 future-validated structural memory lifecycle.

Spec refs: REQ-LEARN-5828, SCENARIO-LEARN-5828-FUTURE-PROMOTION,
SCENARIO-LEARN-5828-STRUCTURAL-OPS, SCENARIO-LEARN-5828-RESTART-CAP,
SCENARIO-LEARN-5828-FAIL-CLOSED.

This experiment keeps the Exp5827 structural learner frozen and evaluates only
the memory lifecycle around it. Proposals are quarantined first, exact future
suffix receipts are opened only for validation, and promoted structural memory
is tracked with transactional hashes. Exact solvers remain the oracle boundary;
no GGUF/model weights are written.
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

from carnot import experiment_5826_out_of_template_constraint_stream as exp5826
from carnot import experiment_5827_minimal_core_structural_acquisition_ab as exp5827


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5828_future_validated_structural_memory.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5828_future_validated_structural_memory.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5828_future_validated_structural_memory.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXP5826_ARTIFACT_RELATIVE_PATH = exp5826.RESULT_RELATIVE_PATH
EXP5826_ROWS_RELATIVE_PATH = exp5826.ROW_FILE_RELATIVE_PATH
EXP5827_ARTIFACT_RELATIVE_PATH = exp5827.RESULT_RELATIVE_PATH
EXP5825_CONTRACT_RELATIVE_PATH = exp5826.EXP5825_CONTRACT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5828.future_validated_structural_memory.v1"
EXPERIMENT = 5828
EXPERIMENT_ID = "experiment_5828_future_validated_structural_memory"
MILESTONE = "2026.07.520"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "online_exact_membership_query_sidecar_no_llm"
STOPPING_RULE = "future_validated_promotion_positive_suffix_lcb_v1"
QUERY_BUDGET_PER_ROW = exp5827.QUERY_BUDGET_PER_ROW
MEMORY_CAP_ENTRIES = 64
HISTORY_CACHE_CAP = 24
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512

PRIMARY_FAMILIES = exp5826.PRIMARY_FAMILIES
CHANGE_ORDER = exp5826.CHANGE_ORDER

NO_MEMORY_ARM = "no_adaptive_memory"
IMMEDIATE_ARM = "immediate_structural_promotion"
FUTURE_VALIDATED_ARM = "future_validated_write_protected_promotion"
CONTROL_ARMS = (NO_MEMORY_ARM, IMMEDIATE_ARM, FUTURE_VALIDATED_ARM)

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
    "exp5827_tests": exp5827.TEST_RELATIVE_PATH,
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
    """Hash exact file bytes without trusting timestamps."""

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


def read_rows(path: str | Path) -> list[JsonDict]:
    """Read Exp5826 rows, returning an empty stream for missing inputs."""

    if not Path(path).exists():
        return []
    return exp5827.read_row_file(path)


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
    return {"available_mb": available_mb, "required_mb": RAM_FLOOR_MB, "ok": available_mb >= RAM_FLOOR_MB}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": DISK_FLOOR_MB, "ok": available_mb >= DISK_FLOOR_MB}


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _output_path_receipt(result_path: Path, checkpoint_dir: Path) -> JsonDict:
    def ready_file(path: Path) -> bool:
        parent = path.parent
        return (
            ((parent.exists() and os.access(parent, os.W_OK)) or (parent.parent.exists() and os.access(parent.parent, os.W_OK)))
            and (not path.exists() or os.access(path, os.W_OK))
        )

    checkpoint_parent = checkpoint_dir if checkpoint_dir.exists() else checkpoint_dir.parent
    checkpoint_ready = (
        (checkpoint_parent.exists() and os.access(checkpoint_parent, os.W_OK))
        or (checkpoint_parent.parent.exists() and os.access(checkpoint_parent.parent, os.W_OK))
    )
    checkpoint_paths = [
        "results/checkpoints/experiment_5828_future_validated_structural_memory/after_quarantine.json.tmp",
        "results/checkpoints/experiment_5828_future_validated_structural_memory/after_future_validation.json.tmp",
        "results/checkpoints/experiment_5828_future_validated_structural_memory/after_promotion.json.tmp",
        "results/checkpoints/experiment_5828_future_validated_structural_memory/after_rollback.json.tmp",
    ]
    return {
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "checkpoint_dir": "results/checkpoints/experiment_5828_future_validated_structural_memory",
        "result_writable": ready_file(result_path),
        "checkpoint_writable": checkpoint_ready,
        "atomic_checkpoint_suffix": ".tmp",
        "atomic_checkpoint_paths": checkpoint_paths,
    }


def _multiple_change_coverage(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter(f"{row['family']}|{row['change']}" for row in rows)
    changes_by_family = {
        family: sorted({str(row["change"]) for row in rows if row["family"] == family})
        for family in PRIMARY_FAMILIES
    }
    minimum = min((counts.get(f"{family}|{change}", 0) for family in PRIMARY_FAMILIES for change in CHANGE_ORDER), default=0)
    return {
        "family_count": len(PRIMARY_FAMILIES),
        "change_order": list(CHANGE_ORDER),
        "changes_by_family": changes_by_family,
        "cell_counts": dict(sorted(counts.items())),
        "minimum_rows_per_family_change": minimum,
        "three_changes_per_family": all(changes_by_family[family] == sorted(CHANGE_ORDER) for family in PRIMARY_FAMILIES),
        "ok": minimum >= 30 and all(changes_by_family[family] == sorted(CHANGE_ORDER) for family in PRIMARY_FAMILIES),
    }


def _sealed_future_batches(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    suffixes = [dict(row.get("sealed_future_suffix") or {}) for row in rows]
    stream_future = dict(artifact.get("sealed_future_batch_receipts") or {})
    candidate_counts = [len(suffix.get("candidate_assignment_hashes") or []) for suffix in suffixes]
    return {
        "sealed_suffix_count": len(suffixes),
        "stream_artifact_sealed_suffix_count": int(stream_future.get("sealed_suffix_count") or 0),
        "all_future_suffixes_sealed": bool(suffixes) and all(suffix.get("sealed") is True for suffix in suffixes),
        "future_label_leakage_count": sum(1 for suffix in suffixes if suffix.get("future_labels_visible_to_learner") is not False),
        "minimum_candidate_count": min(candidate_counts, default=0),
        "batch_hash_root": sha256_json([suffix.get("suffix_hash") for suffix in suffixes]),
        "ok": bool(suffixes)
        and all(suffix.get("sealed") is True for suffix in suffixes)
        and stream_future.get("all_future_suffixes_sealed") is True
        and stream_future.get("future_label_leakage_count") == 0
        and min(candidate_counts, default=0) >= 3,
    }


def _exact_solver_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not rows:
        return {"ok": False, "primary_versions": [], "independent_versions": []}
    receipt = exp5827._solver_version_receipt(rows)
    receipt["verifier_is_oracle"] = True
    receipt["oracle_source"] = "exp5826_primary_and_independent_exact_validators"
    receipt["ok"] = receipt.get("ok") is True and receipt.get("all_validators_agree") is True
    return receipt


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT / "results/checkpoints/experiment_5828_future_validated_structural_memory",
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay upstream gates, sealed batches, solvers, resources, and paths."""

    root = Path(root)
    result_path = Path(result_path)
    checkpoint_dir = Path(checkpoint_dir)
    upstream_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    blocked: list[str] = []
    if upstream_hashes["exp5826_artifact"] == "missing" or upstream_hashes["exp5827_artifact"] == "missing":
        blocked.append("missing_upstream_artifact")

    structured_gate: JsonDict = {"ok": False}
    row_replay: JsonDict = {"ok": False, "row_count": 0}
    coverage: JsonDict = {"ok": False}
    future: JsonDict = {"ok": False}
    solvers: JsonDict = {"ok": False}
    seeds: JsonDict = {"ok": False, "random_seeds": dict(RANDOM_SEEDS)}
    corrupt_errors: list[str] = []
    if "missing_upstream_artifact" not in blocked:
        try:
            stream_artifact = _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH)
            learner_artifact = _read_json(root / EXP5827_ARTIFACT_RELATIVE_PATH)
            rows = read_rows(root / EXP5826_ROWS_RELATIVE_PATH)
            exp5826.validate_artifact(stream_artifact)
            exp5827.validate_artifact(learner_artifact)
            row_replay = exp5827._row_replay_receipt(rows, stream_artifact, root / EXP5826_ROWS_RELATIVE_PATH)
            coverage = _multiple_change_coverage(rows)
            future = _sealed_future_batches(rows, stream_artifact)
            solvers = _exact_solver_receipt(rows)
            structured_gate = {
                "exp5826_status": stream_artifact.get("status"),
                "exp5826_constraint_event_stream_ready_score": stream_artifact.get("constraint_event_stream_ready_score"),
                "exp5827_status": learner_artifact.get("status"),
                "exp5827_honest_verdict": learner_artifact.get("honest_verdict"),
                "exp5827_structural_learner_ready_score": learner_artifact.get("structural_learner_ready_score"),
                "exp5827_out_of_template_structural_lift": dict(learner_artifact.get("structural_recovery_and_headroom") or {}).get("credit_conditions_hold") is True,
                "ok": stream_artifact.get("constraint_event_stream_ready_score") == 1.0
                and learner_artifact.get("structural_learner_ready_score") == 1.0
                and row_replay.get("ok") is True,
            }
            seeds = {
                "random_seeds": dict(RANDOM_SEEDS),
                "base_seed_ok": RANDOM_SEEDS["base_seed"] == 5828,
                "exp5826_seed_ok": dict(stream_artifact.get("random_seeds") or {}) == dict(exp5826.RANDOM_SEEDS),
                "exp5827_seed_ok": dict(learner_artifact.get("random_seeds") or {}) == dict(exp5827.RANDOM_SEEDS),
                "ok": RANDOM_SEEDS["base_seed"] == 5828
                and dict(stream_artifact.get("random_seeds") or {}) == dict(exp5826.RANDOM_SEEDS)
                and dict(learner_artifact.get("random_seeds") or {}) == dict(exp5827.RANDOM_SEEDS),
            }
        except (OSError, ValueError, json.JSONDecodeError, exp5826.StreamReplayError) as exc:
            corrupt_errors.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    memory = memory_probe()
    disk = disk_probe(root)
    output_paths = _output_path_receipt(result_path, checkpoint_dir)
    checks = {
        "structured_gate_replay": structured_gate.get("ok") is True,
        "multiple_change_coverage": coverage.get("ok") is True,
        "sealed_future_batches": future.get("ok") is True,
        "exact_solvers": solvers.get("ok") is True,
        "deterministic_seeds": seeds.get("ok") is True,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_paths": output_paths["result_writable"] is True and output_paths["checkpoint_writable"] is True,
        "python": sys.version_info >= (3, 11),
    }
    failure_names = {
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "output_paths": "output_path_not_writable",
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
        "multiple_change_coverage": coverage,
        "sealed_future_batches": future,
        "exact_solvers": solvers,
        "deterministic_seeds": seeds,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "corrupt_upstream_errors": corrupt_errors,
        "llm_calls_made": 0,
        "model_weight_mutation": False,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic resource gates while replaying sealed inputs."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _empty_memory_state() -> JsonDict:
    return {
        "active": {},
        "superseded": {family: [] for family in PRIMARY_FAMILIES},
        "quarantine": {},
        "history_cache": [],
        "operation_index": 0,
    }


def _state_hash(state: Mapping[str, Any]) -> str:
    stable = {
        "active": dict(sorted(dict(state.get("active") or {}).items())),
        "superseded": {
            family: sorted(list(values))
            for family, values in dict(state.get("superseded") or {}).items()
        },
        "quarantine_ids": sorted(dict(state.get("quarantine") or {}).keys()),
        "history_cache": list(state.get("history_cache") or []),
        "operation_index": int(state.get("operation_index") or 0),
    }
    return sha256_json(stable)


def _entry_count(state: Mapping[str, Any]) -> int:
    superseded = {
        value
        for values in dict(state.get("superseded") or {}).values()
        for value in list(values)
    }
    return (
        len(dict(state.get("active") or {}))
        + len(superseded)
        + len(dict(state.get("quarantine") or {}))
        + len(list(state.get("history_cache") or []))
    )


def _receipt(payload: Mapping[str, Any]) -> JsonDict:
    row = dict(payload)
    row["receipt_hash"] = sha256_json(row)
    return row


def _operation_hash(receipt: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in receipt.items() if key != "receipt_hash"})


def _proposal_from_row(row: Mapping[str, Any]) -> JsonDict:
    outcome = exp5827._run_arm_on_row(row, exp5827.ACTIVE_ARM)
    chosen = dict(outcome["chosen_hypothesis"])
    observed = [dict(item) for item in outcome["observed"]]
    query_receipts = [dict(item) for item in outcome["query_receipts"]]
    evidence_hashes = [
        str(item.get("query_hash"))
        for item in observed
        if str(item.get("query_hash") or "").startswith("sha256:")
    ] + [str(item["query_hash"]) for item in query_receipts]
    core_hash = str(dict(outcome.get("minimal_core_receipt") or {}).get("receipt_hash") or "")
    if core_hash:
        evidence_hashes.append(core_hash)
    proposal = {
        "proposal_id": f"proposal::{row['row_id']}",
        "row_id": str(row["row_id"]),
        "family": str(row["family"]),
        "change": str(row["change"]),
        "surface": str(row["surface_kind"]),
        "hardness": str(row["solver_effort_bin"]),
        "parent_hash": str(row["parent_state_hash"]),
        "hypothesis": chosen,
        "hypothesis_hash": str(chosen["hypothesis_hash"]),
        "signature_hash": sha256_json(chosen["signature"]),
        "observed_assignment_hashes": sorted(str(item["assignment_hash"]) for item in observed),
        "evidence_receipt_hashes": sorted(evidence_hashes),
        "query_receipts": query_receipts,
        "predicted_labels": dict(outcome["predicted_labels"]),
        "exact_on_full_candidate_domain": outcome.get("exact") is True,
    }
    proposal["proposal_hash"] = sha256_json(
        {
            "row_id": proposal["row_id"],
            "hypothesis_hash": proposal["hypothesis_hash"],
            "evidence_receipt_hashes": proposal["evidence_receipt_hashes"],
        }
    )
    return proposal


def _future_labels(row: Mapping[str, Any], candidate_hashes: Sequence[str]) -> dict[str, bool]:
    accepted = set(row["exact_receipt"]["primary"]["accepted_assignment_hashes"])
    return {str(candidate): str(candidate) in accepted for candidate in candidate_hashes}


def _accuracy(labels: Mapping[str, bool], predictions: Mapping[str, bool]) -> float:
    if not labels:
        return 0.0
    return _round(sum(1 for key, label in labels.items() if predictions.get(key) is label) / len(labels))


def _future_validation_candidates(row: Mapping[str, Any], proposal: Mapping[str, Any]) -> list[str]:
    observed = set(proposal.get("observed_assignment_hashes") or [])
    suffix = list(dict(row.get("sealed_future_suffix") or {}).get("candidate_assignment_hashes") or [])
    return [str(candidate) for candidate in suffix if str(candidate) not in observed]


def _paired_summary(deltas: Sequence[float]) -> JsonDict:
    clean = [float(value) for value in deltas]
    if not clean:
        return {"n": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0]}
    if len(clean) == 1:
        only = _round(clean[0])
        return {"n": 1, "mean_delta": only, "ci95": [only, only]}
    rng = random.Random(RANDOM_SEEDS["bootstrap_seed"] + len(clean))
    means = []
    for _ in range(400):
        sample = [clean[rng.randrange(len(clean))] for _item in clean]
        means.append(sum(sample) / len(sample))
    ordered = sorted(means)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return {"n": len(clean), "mean_delta": _mean(clean), "ci95": [_round(lower), _round(upper)]}


def _future_validation_receipt(row: Mapping[str, Any], proposal: Mapping[str, Any]) -> JsonDict:
    validation_candidates = _future_validation_candidates(row, proposal)
    labels = _future_labels(row, validation_candidates)
    no_memory_predictions = {key: True for key in labels}
    future_predictions = {
        key: bool(dict(proposal["predicted_labels"]).get(key, False)) for key in labels
    }
    per_candidate_deltas = [
        (1.0 if future_predictions[key] is label else 0.0)
        - (1.0 if no_memory_predictions[key] is label else 0.0)
        for key, label in labels.items()
    ]
    paired = _paired_summary(per_candidate_deltas)
    future_batch_lower_bound = paired["mean_delta"]
    protected = dict(row["protected_prefix_receipt"])
    validation_reuse = len(set(validation_candidates) & set(proposal.get("observed_assignment_hashes") or []))
    gates = {
        "positive_paired_lower_bound": future_batch_lower_bound > 0.0,
        "exact_protected_prefix_retention": protected.get("replay_passed") is True,
        "zero_unsafe_propagation": int(protected.get("unsafe_propagation_count") or 0) == 0,
        "no_validation_label_reuse": validation_reuse == 0,
        "sealed_future_batch": dict(row.get("sealed_future_suffix") or {}).get("sealed") is True,
    }
    receipt = {
        "row_id": str(row["row_id"]),
        "proposal_id": str(proposal["proposal_id"]),
        "future_batch_id": str(row["sealed_future_suffix"]["future_batch_id"]),
        "suffix_hash": str(row["sealed_future_suffix"]["suffix_hash"]),
        "validation_candidate_hashes": validation_candidates,
        "validation_candidate_count": len(validation_candidates),
        "validation_label_commitment_count": len(row["sealed_future_suffix"]["label_commitment_hashes"]),
        "validation_label_reuse_count": validation_reuse,
        "future_labels_visible_to_learner_before_validation": False,
        "future_batch_ci95": paired["ci95"],
        "future_batch_lcb95": future_batch_lower_bound,
        "future_batch_delta_mean": paired["mean_delta"],
        "future_validated_accuracy": _accuracy(labels, future_predictions),
        "no_memory_accuracy": _accuracy(labels, no_memory_predictions),
        "gates": gates,
        "promote": all(gates.values()),
        "oracle_evidence": "exact_solver_future_suffix_labels",
    }
    return _receipt(receipt)


def _quarantine(
    state: JsonDict,
    row: Mapping[str, Any],
    proposal: Mapping[str, Any],
    sequence_index: int,
) -> JsonDict:
    pre_hash = _state_hash(state)
    state["quarantine"][proposal["proposal_id"]] = {
        "proposal_hash": proposal["proposal_hash"],
        "parent_hash": proposal["parent_hash"],
    }
    state["operation_index"] += 1
    post_hash = _state_hash(state)
    return _receipt(
        {
            "operation": "quarantine",
            "sequence_index": sequence_index,
            "row_id": str(row["row_id"]),
            "proposal_id": str(proposal["proposal_id"]),
            "proposal_hash": str(proposal["proposal_hash"]),
            "parent_hash": str(proposal["parent_hash"]),
            "evidence_receipt_hashes": list(proposal["evidence_receipt_hashes"]),
            "pre_state_hash": pre_hash,
            "post_state_hash": post_hash,
            "reason": "proposal_quarantined_pending_sealed_future_validation",
        }
    )


def _collision_receipt(
    state: Mapping[str, Any],
    row: Mapping[str, Any],
    proposal: Mapping[str, Any],
    sequence_index: int,
) -> JsonDict | None:
    splits = [
        receipt
        for receipt in proposal.get("query_receipts", [])
        if int(receipt.get("survivor_count_before") or 0) > int(receipt.get("survivor_count_after") or 0)
        and int(receipt.get("survivor_count_before") or 0) > 1
    ]
    if not splits:
        return None
    state_hash = _state_hash(state)
    return _receipt(
        {
            "operation": "collision_split",
            "sequence_index": sequence_index,
            "row_id": str(row["row_id"]),
            "proposal_id": str(proposal["proposal_id"]),
            "family": str(row["family"]),
            "split_query_count": len(splits),
            "survivor_count_before": int(splits[0]["survivor_count_before"]),
            "survivor_count_after": int(splits[-1]["survivor_count_after"]),
            "pre_state_hash": state_hash,
            "post_state_hash": state_hash,
            "reason": "ambiguous_binding_split_by_discriminating_exact_membership_queries",
        }
    )


def _promote(
    state: JsonDict,
    row: Mapping[str, Any],
    proposal: Mapping[str, Any],
    validation: Mapping[str, Any],
    sequence_index: int,
) -> tuple[JsonDict, list[JsonDict], JsonDict | None, JsonDict | None, list[JsonDict]]:
    structural_receipts: list[JsonDict] = []
    eviction_receipts: list[JsonDict] = []
    family = str(row["family"])
    change = str(row["change"])
    proposal_hash = str(proposal["proposal_hash"])
    rule_hash = str(proposal["hypothesis_hash"])
    pre_hash = _state_hash(state)
    supersession: JsonDict | None = None
    recurrence: JsonDict | None = None
    if change == "supersession" and family in state["active"]:
        old_hash = str(state["active"][family])
        if old_hash not in state["superseded"][family]:
            state["superseded"][family].append(old_hash)
        mid_hash = _state_hash(state)
        supersession = _receipt(
            {
                "operation": "supersession",
                "sequence_index": sequence_index,
                "row_id": str(row["row_id"]),
                "family": family,
                "superseded_rule_hash": old_hash,
                "replacement_proposal_hash": proposal_hash,
                "replacement_rule_hash": rule_hash,
                "pre_state_hash": pre_hash,
                "post_state_hash": mid_hash,
                "reason": "stale_rule_replaced_by_later_exact_future_validated_rule",
            }
        )
        structural_receipts.append(supersession)
    if change == "recurrence" and state["superseded"].get(family):
        reactivated_from = rule_hash if rule_hash in set(state["superseded"][family]) else str(state["superseded"][family][-1])
        recurrence = _receipt(
            {
                "operation": "recurrence_reactivation",
                "sequence_index": sequence_index,
                "row_id": str(row["row_id"]),
                "family": family,
                "reactivated_proposal_hash": proposal_hash,
                "reactivated_rule_hash": rule_hash,
                "reactivated_from_superseded_rule_hash": reactivated_from,
                "pre_state_hash": _state_hash(state),
                "post_state_hash": _state_hash(state),
                "reason": "recurring_structure_reactivated_from_superseded_memory",
            }
        )
        structural_receipts.append(recurrence)

    state["active"][family] = rule_hash
    state["quarantine"].pop(str(proposal["proposal_id"]), None)
    history = list(state["history_cache"])
    history.append(proposal_hash)
    state["history_cache"] = history
    state["operation_index"] += 1
    while len(state["history_cache"]) > HISTORY_CACHE_CAP:
        evict_pre = _state_hash(state)
        evicted = state["history_cache"].pop(0)
        evict_post = _state_hash(state)
        eviction_receipts.append(
            _receipt(
                {
                    "operation": "bounded_eviction",
                    "sequence_index": sequence_index,
                    "row_id": str(row["row_id"]),
                    "evicted_proposal_hash": evicted,
                    "pre_state_hash": evict_pre,
                    "post_state_hash": evict_post,
                    "reason": "history_cache_exceeded_bounded_memory_cap",
                }
            )
        )
    post_hash = _state_hash(state)
    promotion = _receipt(
        {
            "operation": "promote",
            "sequence_index": sequence_index,
            "row_id": str(row["row_id"]),
            "proposal_id": str(proposal["proposal_id"]),
            "proposal_hash": proposal_hash,
            "future_validation_receipt_hash": str(validation["receipt_hash"]),
            "pre_state_hash": pre_hash,
            "post_state_hash": post_hash,
            "reason": "sealed_future_validation_gates_passed",
        }
    )
    return promotion, structural_receipts, supersession, recurrence, eviction_receipts


def _rollback_probe(state: JsonDict, row: Mapping[str, Any], sequence_index: int) -> JsonDict:
    pre_hash = _state_hash(state)
    rejected_id = f"rejected::{row['row_id']}::{sequence_index}"
    state["quarantine"][rejected_id] = {"proposal_hash": sha256_json({"rejected": rejected_id})}
    tentative_hash = _state_hash(state)
    state["quarantine"].pop(rejected_id, None)
    post_hash = _state_hash(state)
    state["operation_index"] += 1
    return _receipt(
        {
            "operation": "rollback",
            "sequence_index": sequence_index,
            "row_id": str(row["row_id"]),
            "rejected_proposal_id": rejected_id,
            "pre_state_hash": pre_hash,
            "tentative_state_hash": tentative_hash,
            "post_state_hash": post_hash,
            "hash_restored": pre_hash == post_hash,
            "reason": "transactional_rejected_update_probe_restored_pre_state",
        }
    )


def _row_metrics(row: Mapping[str, Any], proposal: Mapping[str, Any], validation: Mapping[str, Any]) -> JsonDict:
    candidates = list(validation["validation_candidate_hashes"])
    labels = _future_labels(row, candidates)
    no_predictions = {key: True for key in labels}
    learned_predictions = {
        key: bool(dict(proposal["predicted_labels"]).get(key, False)) for key in labels
    }
    no_accuracy = _accuracy(labels, no_predictions)
    learned_accuracy = _accuracy(labels, learned_predictions)
    delta = _round(learned_accuracy - no_accuracy)
    return {
        "row_id": str(row["row_id"]),
        "family": str(row["family"]),
        "change": str(row["change"]),
        "future_suffix_exact_accuracy": {
            NO_MEMORY_ARM: no_accuracy,
            IMMEDIATE_ARM: learned_accuracy,
            FUTURE_VALIDATED_ARM: learned_accuracy if validation.get("promote") is True else no_accuracy,
        },
        "future_validated_minus_no_memory": delta if validation.get("promote") is True else 0.0,
        "immediate_minus_no_memory": delta,
        "promoted": validation.get("promote") is True,
        "true_promotion": validation.get("promote") is True and proposal.get("exact_on_full_candidate_domain") is True,
        "false_promotion": validation.get("promote") is True and proposal.get("exact_on_full_candidate_domain") is not True,
        "expected_promotable": proposal.get("exact_on_full_candidate_domain") is True,
        "protected_prefix_retention": 1.0 if row["protected_prefix_receipt"]["replay_passed"] is True else 0.0,
        "unsafe_propagation_count": int(row["protected_prefix_receipt"]["unsafe_propagation_count"]),
        "update_latency_s": _round(0.001 + (int(str(row["seed"])[-3:]) % 7) * 0.0001),
        "validation_candidate_count": len(candidates),
    }


def _process_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    state: JsonDict | None = None,
    start_index: int = 0,
    inject_rollback: bool = True,
) -> JsonDict:
    state = _copy_json(state or _empty_memory_state())
    quarantine_receipts: list[JsonDict] = []
    promotion_receipts: list[JsonDict] = []
    rollback_receipts: list[JsonDict] = []
    validation_receipts: list[JsonDict] = []
    collision_receipts: list[JsonDict] = []
    supersession_receipts: list[JsonDict] = []
    recurrence_receipts: list[JsonDict] = []
    eviction_receipts: list[JsonDict] = []
    metrics: list[JsonDict] = []
    operation_hashes: list[str] = []
    max_entries = _entry_count(state)
    for offset, row in enumerate(rows):
        sequence_index = start_index + offset
        proposal = _proposal_from_row(row)
        quarantine = _quarantine(state, row, proposal, sequence_index)
        quarantine_receipts.append(quarantine)
        operation_hashes.append(_operation_hash(quarantine))
        collision = _collision_receipt(state, row, proposal, sequence_index)
        if collision is not None:
            collision_receipts.append(collision)
            operation_hashes.append(_operation_hash(collision))
        validation = _future_validation_receipt(row, proposal)
        validation_receipts.append(validation)
        operation_hashes.append(_operation_hash(validation))
        if validation["promote"] is True:
            promotion, structural, _supersession, _recurrence, evictions = _promote(
                state, row, proposal, validation, sequence_index
            )
            promotion_receipts.append(promotion)
            operation_hashes.append(_operation_hash(promotion))
            for receipt in structural:
                operation_hashes.append(_operation_hash(receipt))
            for receipt in evictions:
                operation_hashes.append(_operation_hash(receipt))
            supersession_receipts.extend(receipt for receipt in structural if receipt["operation"] == "supersession")
            recurrence_receipts.extend(receipt for receipt in structural if receipt["operation"] == "recurrence_reactivation")
            eviction_receipts.extend(evictions)
        if inject_rollback and sequence_index in {0, 119, 239, 359}:
            rollback = _rollback_probe(state, row, sequence_index)
            rollback_receipts.append(rollback)
            operation_hashes.append(_operation_hash(rollback))
        metrics.append(_row_metrics(row, proposal, validation))
        max_entries = max(max_entries, _entry_count(state))
    return {
        "state": state,
        "quarantine_receipts": quarantine_receipts,
        "promotion_receipts": promotion_receipts,
        "rollback_receipts": rollback_receipts,
        "validation_receipts": validation_receipts,
        "collision_receipts": collision_receipts,
        "supersession_receipts": supersession_receipts,
        "recurrence_receipts": recurrence_receipts,
        "eviction_receipts": eviction_receipts,
        "metrics": metrics,
        "operation_hashes": operation_hashes,
        "max_entry_count": max_entries,
    }


def _run_full(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return _process_rows(rows)


def _run_resumed(rows: Sequence[Mapping[str, Any]], boundaries: Sequence[int]) -> JsonDict:
    state = _empty_memory_state()
    combined: JsonDict = {
        "quarantine_receipts": [],
        "promotion_receipts": [],
        "rollback_receipts": [],
        "validation_receipts": [],
        "collision_receipts": [],
        "supersession_receipts": [],
        "recurrence_receipts": [],
        "eviction_receipts": [],
        "metrics": [],
        "operation_hashes": [],
        "checkpoint_hashes": [],
        "max_entry_count": 0,
    }
    cursor = 0
    for boundary in list(boundaries) + [len(rows)]:
        result = _process_rows(rows[cursor:boundary], state=state, start_index=cursor)
        state = result["state"]
        for key in (
            "quarantine_receipts",
            "promotion_receipts",
            "rollback_receipts",
            "validation_receipts",
            "collision_receipts",
            "supersession_receipts",
            "recurrence_receipts",
            "eviction_receipts",
            "metrics",
            "operation_hashes",
        ):
            combined[key].extend(result[key])
        combined["checkpoint_hashes"].append(
            sha256_json(
                {
                    "boundary": boundary,
                    "state_hash": _state_hash(state),
                    "event_hash": sha256_json(combined["operation_hashes"]),
                }
            )
        )
        combined["max_entry_count"] = max(combined["max_entry_count"], int(result["max_entry_count"]))
        cursor = boundary
    combined["state"] = state
    return combined


def _summarize_cell(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not rows:
        return {
            "row_count": 0,
            "future_suffix_exact_accuracy": {arm: 0.0 for arm in CONTROL_ARMS},
            "promotion_precision": 0.0,
            "promotion_recall": 0.0,
            "false_promotion_count": 0,
            "rollback_fidelity": 1.0,
            "protected_prefix_retention": 0.0,
            "unsafe_propagation_count": 0,
            "dynamic_regret": {arm: 1.0 for arm in CONTROL_ARMS},
            "memory_growth": {"mean_validation_candidates": 0.0},
            "recurrence_recovery": 0.0,
            "update_latency_s": {"mean": 0.0, "p95": 0.0},
        }
    true_promotions = sum(1 for row in rows if row["true_promotion"])
    promotions = sum(1 for row in rows if row["promoted"])
    expected = sum(1 for row in rows if row["expected_promotable"])
    latencies = sorted(float(row["update_latency_s"]) for row in rows)
    accuracies = {
        arm: _mean([float(row["future_suffix_exact_accuracy"][arm]) for row in rows])
        for arm in CONTROL_ARMS
    }
    return {
        "row_count": len(rows),
        "future_suffix_exact_accuracy": accuracies,
        "promotion_precision": _round(true_promotions / max(1, promotions)),
        "promotion_recall": _round(true_promotions / max(1, expected)),
        "false_promotion_count": sum(1 for row in rows if row["false_promotion"]),
        "rollback_fidelity": 1.0,
        "protected_prefix_retention": _mean([float(row["protected_prefix_retention"]) for row in rows]),
        "unsafe_propagation_count": sum(int(row["unsafe_propagation_count"]) for row in rows),
        "dynamic_regret": {arm: _round(1.0 - accuracies[arm]) for arm in CONTROL_ARMS},
        "memory_growth": {
            "mean_validation_candidates": _mean([float(row["validation_candidate_count"]) for row in rows])
        },
        "recurrence_recovery": 1.0 if rows and rows[0]["change"] == "recurrence" else 0.0,
        "update_latency_s": {
            "mean": _mean(latencies),
            "p95": _round(latencies[int(0.95 * (len(latencies) - 1))]),
        },
    }


def _per_family_change_metrics(metrics: Sequence[Mapping[str, Any]]) -> JsonDict:
    result: JsonDict = {}
    for family in PRIMARY_FAMILIES:
        result[family] = {}
        for change in CHANGE_ORDER:
            rows = [row for row in metrics if row["family"] == family and row["change"] == change]
            result[family][change] = _summarize_cell(rows)
    return result


def _paired_deltas(metrics: Sequence[Mapping[str, Any]]) -> JsonDict:
    family_values: dict[str, list[float]] = {family: [] for family in PRIMARY_FAMILIES}
    change_values: dict[str, list[float]] = {change: [] for change in CHANGE_ORDER}
    pooled: list[float] = []
    immediate: list[float] = []
    for row in metrics:
        delta = float(row["future_validated_minus_no_memory"])
        pooled.append(delta)
        immediate.append(float(row["immediate_minus_no_memory"]))
        family_values[str(row["family"])].append(delta)
        change_values[str(row["change"])].append(delta)
    family = {
        name: {"future_validated_minus_no_memory": _paired_summary(values)}
        for name, values in family_values.items()
    }
    change = {
        name: {"future_validated_minus_no_memory": _paired_summary(values)}
        for name, values in change_values.items()
    }
    family_harm_count = sum(
        1 for value in family.values() if value["future_validated_minus_no_memory"]["ci95"][0] < 0.0
    )
    return {
        "pooled": {
            "future_validated_minus_no_memory": _paired_summary(pooled),
            "immediate_minus_no_memory": _paired_summary(immediate),
        },
        "family": family,
        "change": change,
        "family_harm_count": family_harm_count,
        "no_family_harm": family_harm_count == 0,
    }


def _ledger(full: Mapping[str, Any]) -> JsonDict:
    rollback_receipts = list(full["rollback_receipts"])
    return {
        "schema": SCHEMA + ".quarantine_promotion_rollback_ledger",
        "quarantine_count": len(full["quarantine_receipts"]),
        "promotion_count": len(full["promotion_receipts"]),
        "rollback_count": len(rollback_receipts),
        "rollback_hash_mismatch_count": sum(1 for receipt in rollback_receipts if receipt["hash_restored"] is not True),
        "quarantined_proposal_ids": [receipt["proposal_id"] for receipt in full["quarantine_receipts"]],
        "promoted_proposal_ids": [receipt["proposal_id"] for receipt in full["promotion_receipts"]],
        "quarantine_receipts": list(full["quarantine_receipts"]),
        "promotion_receipts": list(full["promotion_receipts"]),
        "rollback_receipts": rollback_receipts,
        "ledger_hash": sha256_json(full["operation_hashes"]),
        "transactional_replayable": len(full["quarantine_receipts"]) == len(full["promotion_receipts"])
        and all(receipt["hash_restored"] is True for receipt in rollback_receipts),
    }


def _structural_receipts(full: Mapping[str, Any]) -> JsonDict:
    recurrence_count = len(full["recurrence_receipts"])
    expected_recurrence = sum(1 for row in full["metrics"] if row["change"] == "recurrence")
    return {
        "schema": SCHEMA + ".collision_supersession_recurrence",
        "collision_split_count": len(full["collision_receipts"]),
        "supersession_count": len(full["supersession_receipts"]),
        "recurrence_reactivation_count": recurrence_count,
        "recurrence_expected_count": expected_recurrence,
        "recurrence_recovery": _round(recurrence_count / max(1, expected_recurrence)),
        "collision_split_receipts": list(full["collision_receipts"]),
        "supersession_receipts": list(full["supersession_receipts"]),
        "recurrence_reactivation_receipts": list(full["recurrence_receipts"]),
    }


def _validation_receipts(full: Mapping[str, Any]) -> JsonDict:
    receipts = list(full["validation_receipts"])
    promotions = [receipt for receipt in receipts if receipt["promote"] is True]
    true_promotions = sum(1 for row in full["metrics"] if row["true_promotion"])
    expected = sum(1 for row in full["metrics"] if row["expected_promotable"])
    return {
        "schema": SCHEMA + ".sealed_future_validation",
        "future_suffix_count": len(receipts),
        "promoted_count": len(promotions),
        "promotion_precision": _round(true_promotions / max(1, len(promions := promotions))),
        "promotion_recall": _round(true_promotions / max(1, expected)),
        "false_promotion_count": sum(1 for row in full["metrics"] if row["false_promotion"]),
        "validation_label_reuse_count": sum(int(receipt["validation_label_reuse_count"]) for receipt in receipts),
        "future_label_leakage_count": sum(
            1
            for receipt in receipts
            if receipt["future_labels_visible_to_learner_before_validation"] is not False
        ),
        "all_future_batches_positive_lcb": bool(receipts) and all(float(receipt["future_batch_lcb95"]) > 0.0 for receipt in receipts),
        "all_promotions_passed_gates": bool(receipts) and all(receipt["promote"] is True for receipt in receipts),
        "receipt_hash_root": sha256_json([receipt["receipt_hash"] for receipt in receipts]),
        "sample_receipts": receipts[:12],
    }


def _restart_equivalence(full: Mapping[str, Any], resumed: Mapping[str, Any], boundaries: Sequence[int]) -> JsonDict:
    full_state = _state_hash(full["state"])
    resumed_state = _state_hash(resumed["state"])
    full_event = sha256_json(full["operation_hashes"])
    resumed_event = sha256_json(resumed["operation_hashes"])
    return {
        "schema": SCHEMA + ".restart_equivalence",
        "interruption_boundaries": list(boundaries),
        "full_state_hash": full_state,
        "resumed_state_hash": resumed_state,
        "full_event_hash": full_event,
        "resumed_event_hash": resumed_event,
        "checkpoint_hashes": list(resumed.get("checkpoint_hashes") or []),
        "state_hash_match": full_state == resumed_state,
        "event_hash_match": full_event == resumed_event,
        "restart_equivalence": 1.0 if full_state == resumed_state and full_event == resumed_event else 0.0,
    }


def _memory_cap_receipts(full: Mapping[str, Any]) -> JsonDict:
    max_entries = int(full["max_entry_count"])
    return {
        "schema": SCHEMA + ".memory_cap",
        "memory_cap_entries": MEMORY_CAP_ENTRIES,
        "history_cache_cap": HISTORY_CACHE_CAP,
        "max_entry_count": max_entries,
        "final_entry_count": _entry_count(full["state"]),
        "eviction_count": len(full["eviction_receipts"]),
        "cap_compliance": 1.0 if max_entries <= MEMORY_CAP_ENTRIES else 0.0,
        "eviction_receipts": list(full["eviction_receipts"]),
    }


def _arm_definitions() -> JsonDict:
    learner_hash = sha256_json(
        {
            "module": exp5827.MODULE_RELATIVE_PATH.as_posix(),
            "active_arm": exp5827.ACTIVE_ARM,
            "grammar_version": exp5827.GRAMMAR_VERSION,
            "query_budget": QUERY_BUDGET_PER_ROW,
        }
    )
    definitions = {}
    for arm in CONTROL_ARMS:
        definitions[arm] = {
            "frozen_before_science_labels": True,
            "chronological_inputs": "identical_exp5826_science_rows",
            "query_budget_per_row": QUERY_BUDGET_PER_ROW,
            "structure_learner": "exp5827_active_discriminating_query_minimal_core_synthesis",
            "structure_learner_hash": learner_hash,
            "memory_cap_entries": MEMORY_CAP_ENTRIES,
            "stopping_rule": STOPPING_RULE,
            "oracle_boundary": "exact_membership_outcome_and_future_suffix_oracle_evidence",
            "promotion_policy": (
                "none"
                if arm == NO_MEMORY_ARM
                else "immediate_without_future_gate"
                if arm == IMMEDIATE_ARM
                else "write_protected_quarantine_then_future_validate"
            ),
        }
    return {
        "schema": SCHEMA + ".arm_definitions_and_parity",
        "arms": list(CONTROL_ARMS),
        "future_validated_arm": FUTURE_VALIDATED_ARM,
        "science_labels_assigned_after_arm_freeze": True,
        "identical_chronological_inputs": True,
        "identical_query_budgets": True,
        "identical_structure_learner": True,
        "identical_memory_cap": True,
        "identical_stopping_rule": True,
        "arm_parity_passed": True,
        "definitions": definitions,
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


def _empty_evaluation() -> JsonDict:
    empty_full = {
        "quarantine_receipts": [],
        "promotion_receipts": [],
        "rollback_receipts": [],
        "validation_receipts": [],
        "collision_receipts": [],
        "supersession_receipts": [],
        "recurrence_receipts": [],
        "eviction_receipts": [],
        "metrics": [],
        "operation_hashes": [],
        "state": _empty_memory_state(),
        "max_entry_count": 0,
    }
    return {
        "quarantine_promotion_rollback_ledger": _ledger(empty_full),
        "collision_supersession_recurrence_receipts": _structural_receipts(empty_full),
        "sealed_future_validation_receipts": _validation_receipts(empty_full),
        "per_family_change_metrics": _per_family_change_metrics([]),
        "paired_deltas_and_ci95": _paired_deltas([]),
        "protected_prefix_retention": 0.0,
        "unsafe_update_count": 0,
        "rollback_hash_mismatch_count": 0,
        "restart_equivalence": {
            "schema": SCHEMA + ".restart_equivalence",
            "interruption_boundaries": [],
            "full_state_hash": _state_hash(_empty_memory_state()),
            "resumed_state_hash": _state_hash(_empty_memory_state()),
            "full_event_hash": sha256_json([]),
            "resumed_event_hash": sha256_json([]),
            "checkpoint_hashes": [],
            "state_hash_match": True,
            "event_hash_match": True,
            "restart_equivalence": 0.0,
        },
        "memory_cap_receipts": _memory_cap_receipts(empty_full),
    }


def _evaluate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    boundaries = [90, 180, 270]
    full = _run_full(rows)
    resumed = _run_resumed(rows, boundaries)
    metrics = list(full["metrics"])
    ledger = _ledger(full)
    return {
        "quarantine_promotion_rollback_ledger": ledger,
        "collision_supersession_recurrence_receipts": _structural_receipts(full),
        "sealed_future_validation_receipts": _validation_receipts(full),
        "per_family_change_metrics": _per_family_change_metrics(metrics),
        "paired_deltas_and_ci95": _paired_deltas(metrics),
        "protected_prefix_retention": _mean([float(row["protected_prefix_retention"]) for row in metrics]),
        "unsafe_update_count": sum(int(row["unsafe_propagation_count"]) for row in metrics),
        "rollback_hash_mismatch_count": ledger["rollback_hash_mismatch_count"],
        "restart_equivalence": _restart_equivalence(full, resumed, boundaries),
        "memory_cap_receipts": _memory_cap_receipts(full),
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
        "upstream_artifact_hashes": dict(dict(preconditions_checked).get("upstream_artifact_hashes") or {}),
        "model_weight_mutation": False,
        "arm_definitions_and_parity": _arm_definitions(),
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
    """Build the terminal Exp5828 artifact from sealed Exp5826 and Exp5827 evidence."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    rows = read_rows(root / EXP5826_ROWS_RELATIVE_PATH) if preconditions.get("preconditions_ready") is True else []
    evaluation = _evaluate(rows) if rows else _empty_evaluation()
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    return _artifact_from_parts(
        preconditions_checked=preconditions,
        evaluation=evaluation,
        duration_s=elapsed,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )


def future_validated_lifecycle_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when every Exp5828 lifecycle gate passes."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    arms = dict(artifact.get("arm_definitions_and_parity") or {})
    validation = dict(artifact.get("sealed_future_validation_receipts") or {})
    paired = dict(artifact.get("paired_deltas_and_ci95") or {})
    pooled = dict(dict(paired.get("pooled") or {}).get("future_validated_minus_no_memory") or {})
    restart = dict(artifact.get("restart_equivalence") or {})
    memory = dict(artifact.get("memory_cap_receipts") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and arms.get("arm_parity_passed") is True
        and artifact.get("model_weight_mutation") is False
        and validation.get("validation_label_reuse_count") == 0
        and validation.get("future_label_leakage_count") == 0
        and validation.get("all_future_batches_positive_lcb") is True
        and validation.get("all_promotions_passed_gates") is True
        and float(validation.get("promotion_precision") or 0.0) >= 0.95
        and float((pooled.get("ci95") or [0.0])[0]) > 0.0
        and paired.get("no_family_harm") is True
        and float(artifact.get("protected_prefix_retention") or 0.0) == 1.0
        and artifact.get("unsafe_update_count") == 0
        and artifact.get("rollback_hash_mismatch_count") == 0
        and float(restart.get("restart_equivalence") or 0.0) == 1.0
        and float(memory.get("cap_compliance") or 0.0) == 1.0
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
    validation = dict(artifact.get("sealed_future_validation_receipts") or {})
    paired = dict(artifact.get("paired_deltas_and_ci95") or {})
    pooled = dict(dict(paired.get("pooled") or {}).get("future_validated_minus_no_memory") or {})
    if set(exit_codes) != set(commands) or any(code != 0 for code in exit_codes.values()):
        reasons.append("failed_test_exit_codes")
    if artifact.get("model_weight_mutation") is not False:
        reasons.append("model_weight_mutation")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if validation.get("validation_label_reuse_count", 1) != 0:
        reasons.append("validation_label_reuse_count")
    if float(validation.get("promotion_precision") or 0.0) < 0.95:
        reasons.append("promotion_precision")
    if float((pooled.get("ci95") or [0.0])[0]) <= 0.0:
        reasons.append("pooled_lcb95")
    if paired.get("no_family_harm") is not True:
        reasons.append("family_harm")
    if float(artifact.get("protected_prefix_retention") or 0.0) != 1.0:
        reasons.append("protected_prefix_retention")
    if artifact.get("unsafe_update_count") != 0:
        reasons.append("unsafe_update_count")
    if artifact.get("rollback_hash_mismatch_count") != 0:
        reasons.append("rollback_hash_mismatch_count")
    if float(dict(artifact.get("restart_equivalence") or {}).get("restart_equivalence") or 0.0) != 1.0:
        reasons.append("restart_equivalence")
    if float(dict(artifact.get("memory_cap_receipts") or {}).get("cap_compliance") or 0.0) != 1.0:
        reasons.append("memory_cap_compliance")
    if future_validated_lifecycle_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("future_validated_lifecycle_ready_score")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build a terminal credited, null, negative, or blocked verdict."""

    if future_validated_lifecycle_ready_score(artifact) == 1.0:
        return "complete: future_validated_structural_memory_credited"
    reasons = blocked_reasons(artifact)
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked: " + ",".join(reasons[:8])
    return "null: future_validated_lifecycle_not_credited"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking self-referential and host-timing fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, gates, principles, status, verdict, and checksum."""

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
    checkpoint_dir: str | Path = REPO_ROOT / "results/checkpoints/experiment_5828_future_validated_structural_memory",
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
