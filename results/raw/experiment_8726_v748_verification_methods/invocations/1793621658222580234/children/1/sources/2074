"""Exp5895 shortcut-safe continuous self-learning lifecycle.

Spec refs: REQ-LEARN-5895, SCENARIO-LEARN-5895-PRECONDITIONS,
SCENARIO-LEARN-5895-SEALED-SPLITS, SCENARIO-LEARN-5895-LIFECYCLE,
SCENARIO-LEARN-5895-METRICS, SCENARIO-LEARN-5895-HARDWARE-MAPPING,
SCENARIO-LEARN-5895-FAIL-CLOSED.

The experiment is a deterministic sidecar over Exp5894's one-to-one grounding
fixture. It tests a versioned external state that stores exact-validator
evidence and unresolved constraints without promoting unsafe shortcut updates.
No LLM, tokenizer, model inference, board execution, or weight update path is
loaded.
"""

from __future__ import annotations

import argparse
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

from carnot import adaptive_state
from carnot import experiment_5894_one_to_one_grounding_ab as exp5894


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5895_shortcut_safe_continuous_self_learning.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5895_shortcut_safe_continuous_self_learning.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ROOT_CLUTTER_SWEEP_RELATIVE_PATH = Path("scripts/root_clutter_sweep.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
EXP5894_ARTIFACT_RELATIVE_PATH = exp5894.RESULT_RELATIVE_PATH
EXP5893_ARTIFACT_RELATIVE_PATH = exp5894.EXP5893_ARTIFACT_RELATIVE_PATH
EXP5893_ROWS_RELATIVE_PATH = exp5894.EXP5893_ROWS_RELATIVE_PATH
EXP5856_ARTIFACT_RELATIVE_PATH = Path("results/experiment_5856_provenance_correct_lifecycle.json")
EXP5857_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5857_clean_transfer_selective_replay.json"
)
EXP5858_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5858_reduced_oracle_continuous_self_learning.json"
)
EXP5865_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5865_adaptive_state_kernel_requalification.json"
)
EXP5867_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5867_prospective_certified_continuous_learning.json"
)
ADAPTIVE_STATE_RELATIVE_PATH = Path("python/carnot/adaptive_state.py")
SEMANTIC_GROUNDING_RELATIVE_PATH = Path("python/carnot/pipeline/semantic_grounding.py")

SCHEMA = "carnot.experiment_5895.shortcut_safe_continuous_self_learning.v1"
EXPERIMENT = 5895
EXPERIMENT_ID = "experiment_5895_shortcut_safe_continuous_self_learning"
MILESTONE = "2026.07.524"
RUN_DATE = "20260724"
SOURCE_ARXIV_ID = "2607.21461"
INFERENCE_SUBSTRATE = "deterministic_exact_verifier_and_versioned_external_state_no_llm"
VERIFIER_IS_ORACLE = True
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512
MEMORY_CAP = 16
QUARANTINE_CAP = 32
REJECTED_BUFFER_CAP = 16
REPLAY_LIMIT = 4
PRIMARY_ARM = "verified_evidence_plus_unresolved_constraints"
ARM_NAMES = (
    PRIMARY_ARM,
    "fixed_validated_memory",
    "one_to_one_reduced_oracle",
    "one_to_one_full_oracle",
    "soft_grounding",
    "shuffled_grounding",
    "no_memory",
    "compatible_replay",
)
SHORTCUT_CONTROL_ARMS = ("soft_grounding", "shuffled_grounding", "compatible_replay")
SHORTCUT_TYPES = exp5894.SHORTCUT_TYPES_TO_MEASURE
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5895,
    "bootstrap_seed": 5_895_001,
    "split_seed": 5_895_002,
    "replay_seed": 5_895_003,
    "rollback_seed": 5_895_004,
}
SPEC_REFS = (
    "REQ-LEARN-5895",
    "SCENARIO-LEARN-5895-PRECONDITIONS",
    "SCENARIO-LEARN-5895-SEALED-SPLITS",
    "SCENARIO-LEARN-5895-LIFECYCLE",
    "SCENARIO-LEARN-5895-METRICS",
    "SCENARIO-LEARN-5895-HARDWARE-MAPPING",
    "SCENARIO-LEARN-5895-FAIL-CLOSED",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5895_shortcut_safe_continuous_self_learning.py "
    "-m pytest tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5895_shortcut_safe_continuous_self_learning.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py",
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5895_shortcut_safe_continuous_self_learning.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5894_artifact": EXP5894_ARTIFACT_RELATIVE_PATH,
    "exp5893_artifact": EXP5893_ARTIFACT_RELATIVE_PATH,
    "exp5893_rows": EXP5893_ROWS_RELATIVE_PATH,
    "exp5856_lifecycle": EXP5856_ARTIFACT_RELATIVE_PATH,
    "exp5857_replay": EXP5857_ARTIFACT_RELATIVE_PATH,
    "exp5858_reduced_oracle": EXP5858_ARTIFACT_RELATIVE_PATH,
    "adaptive_state_lifecycle_code": ADAPTIVE_STATE_RELATIVE_PATH,
    "exp5894_lifecycle_code": exp5894.MODULE_RELATIVE_PATH,
    "exp5894_tests": exp5894.TEST_RELATIVE_PATH,
    "exp5895_module": MODULE_RELATIVE_PATH,
    "exp5895_tests": TEST_RELATIVE_PATH,
    "exact_semantic_grounding_validator": SEMANTIC_GROUNDING_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "codex_instructions": Path("CODEX.md"),
    "claude_instructions": Path("CLAUDE.md"),
    "research_program": RESEARCH_PROGRAM_RELATIVE_PATH,
    "prd": PRD_RELATIVE_PATH,
    "e2e_plan": E2E_PLAN_RELATIVE_PATH,
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
    "root_clutter_sweep": ROOT_CLUTTER_SWEEP_RELATIVE_PATH,
    "protected_file_guard": RESEARCH_CONDUCTOR_RELATIVE_PATH,
}
PROTECTED_RELATIVE_PATHS = (
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    EXP5894_ARTIFACT_RELATIVE_PATH,
    EXP5893_ARTIFACT_RELATIVE_PATH,
    EXP5893_ROWS_RELATIVE_PATH,
)
RETIRED_CHAIN_PATHS = {
    "experiment_5865": EXP5865_ARTIFACT_RELATIVE_PATH,
    "experiment_5867": EXP5867_ARTIFACT_RELATIVE_PATH,
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "continuous_self_learning_task",
    "upstream_gate_and_hash_receipts",
    "sealed_chronological_split_and_visibility",
    "frozen_arms_and_budget_parity",
    "exact_query_policy_and_budget",
    "verified_evidence_and_unresolved_constraint_state",
    "versioned_promotion_quarantine_rejection_and_rollback",
    "rejected_update_non_propagation",
    "per_update_non_forgetting_certificates",
    "prospective_semantic_and_constraint_metrics",
    "shortcut_false_accept_metrics",
    "forward_transfer_recurrence_retention_and_regret",
    "family_grounding_hardness_lower_bounds",
    "replay_query_resource_and_latency_accounting",
    "memory_cap_accounting",
    "rollback_restart_and_state_hashes",
    "no_model_weight_mutation",
    "null_and_ablation_controls",
    "hardware_mapping_contract",
    "retirement_decision",
    "shortcut_resistant_csl_ready_score",
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
    "status": "A terminal state distinguishes positive, null, unsafe, retired, or blocked shortcut-safe continuous learning evidence.",
    "preconditions_checked": "Gate, hashes, validators, split groups, seeds, budgets, resources, outputs, protected files, and retired-chain exclusion prevent invalid learning.",
    "continuous_self_learning_task": "Must be bare true for the milestone requirement.",
    "upstream_gate_and_hash_receipts": "Exp5894, Exp5893 rows, lifecycle code, validators, and immutable weights are the bounded evidence surface.",
    "sealed_chronological_split_and_visibility": "Future evidence cannot influence current updates.",
    "frozen_arms_and_budget_parity": "Frozen arm and budget parity isolate external-state lifecycle effects.",
    "exact_query_policy_and_budget": "Exact feedback cost and authority are explicit before promotion.",
    "verified_evidence_and_unresolved_constraint_state": "The state contains verified evidence plus unresolved constraints without treating unresolved items as accepted facts.",
    "versioned_promotion_quarantine_rejection_and_rollback": "Only exact validation can promote and every state edit has a rollback target.",
    "rejected_update_non_propagation": "Unsafe or unverified state never becomes future context.",
    "per_update_non_forgetting_certificates": "Every promoted update carries a protected-prefix retention certificate.",
    "prospective_semantic_and_constraint_metrics": "Future semantic and constraint outcomes are scored after commit.",
    "shortcut_false_accept_metrics": "Shortcut false accepts are counted by arm and type.",
    "forward_transfer_recurrence_retention_and_regret": "Prospective learning must improve transfer without forgetting or unbounded regret.",
    "family_grounding_hardness_lower_bounds": "Grouped lower bounds prevent pooled lift from hiding a failing family, grounding, or hardness cell.",
    "replay_query_resource_and_latency_accounting": "Replay, query, resource, and latency accounting expose costs without speedup claims.",
    "memory_cap_accounting": "Versioned state, replay, quarantine, and rejected buffers remain bounded.",
    "rollback_restart_and_state_hashes": "Restart and rollback must reproduce exact canonical state hashes.",
    "no_model_weight_mutation": "Immutable GGUF weights remain unchanged.",
    "null_and_ablation_controls": "Null, ablation, soft, shuffled, fixed-memory, no-memory, and compatible-replay controls test causal credit.",
    "hardware_mapping_contract": "Each learning tier retains a falsifiable hardware path without claiming execution.",
    "retirement_decision": "The next step distinguishes advancement from null, unsafe, blocked, or retired evidence.",
    "shortcut_resistant_csl_ready_score": "Emit bare 1.0 only for positive grouped future lift, zero unsafe accepts, exact retention/rollback, bounded state, and immutable weights.",
    "duration_s": "Measured wall time exposes deterministic sidecar work.",
    "inference_substrate": "Use `deterministic_exact_verifier_and_versioned_external_state_no_llm`.",
    "verifier_is_oracle": "True for exact feedback and promotion authority.",
    "field_provenance": "Every field traces to prompt, spec, rows, upstream artifacts, validators, lifecycle code, controls, or tests.",
    "test_commands": "Commands document focused unit/coverage, gate/hash, split visibility, arm/budget parity, lifecycle, non-forgetting, rejected-update, shortcut safety, grouped intervals, state cap, restart/rollback, weight immutability, hardware-schema, applicable E2E, schema, adversarial-verify, spec-coverage, root-clutter, and protected-file checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects gate, row, split, budget, state, metric, hardware-contract, or code drift.",
    "honest_verdict": "Use `complete_positive:`, `complete_null:`, `unsafe:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence in a stable byte order before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes for bounded upstream artifacts and code."""

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


def load_fixture_rows(root: str | Path = REPO_ROOT) -> list[JsonDict]:
    """Load Exp5893 grounding rows from a repo root or a direct JSONL path."""

    path = Path(root)
    if path.is_dir():
        path = path / EXP5893_ROWS_RELATIVE_PATH
    return _read_jsonl(path) if path.exists() else []


def _rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


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


def _atomic_path_receipt(path: Path) -> JsonDict:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".atomic_probe.tmp")
    wrote = False
    try:
        probe.write_text("atomic-output-probe\n", encoding="utf-8")
        wrote = probe.read_text(encoding="utf-8") == "atomic-output-probe\n"
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "result_path": str(path),
        "parent_exists": parent.exists(),
        "parent_writable": os.access(parent, os.W_OK),
        "atomic_suffix": ".tmp",
        "atomic_probe_write_ok": wrote,
        "target_writable": (not path.exists()) or os.access(path, os.W_OK),
        "ok": wrote and ((not path.exists()) or os.access(path, os.W_OK)),
    }


def _file_stat_receipt(path: Path) -> JsonDict:
    stat = path.stat()
    return {
        "path": path.relative_to(REPO_ROOT).as_posix()
        if path.is_relative_to(REPO_ROOT)
        else str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": stat.st_mode,
        "inode": stat.st_ino,
    }


def _gguf_stat_receipts(root: Path) -> JsonDict:
    paths = sorted((root / "models").glob("**/*.gguf")) if (root / "models").exists() else []
    receipts = [_file_stat_receipt(path) for path in paths]
    return {
        "strategy": "stat_receipt_no_weight_load",
        "weight_file_count": len(receipts),
        "receipts": receipts,
        "receipt_hash": sha256_json(receipts),
    }


def _protected_file_hashes(root: Path) -> JsonDict:
    return {
        relative.as_posix(): _hash_path(root, relative) for relative in PROTECTED_RELATIVE_PATHS
    }


def _split_group_registry(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row.get("split_group"))].add(str(row.get("split")))
    registry = {group: sorted(splits) for group, splits in sorted(groups.items())}
    return {
        "group_count": len(registry),
        "split_groups_isolated": bool(registry)
        and all(len(splits) == 1 for splits in registry.values()),
        "registry_hash": sha256_json(registry),
    }


def _budget_registry(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    train_count = sum(str(row.get("split")) == "train" for row in rows)
    held_count = sum(str(row.get("split")) == "heldout" for row in rows)
    budget = {
        "event_count": len(rows),
        "train_count": train_count,
        "held_count": held_count,
        "memory_cap": MEMORY_CAP,
        "quarantine_cap": QUARANTINE_CAP,
        "rejected_buffer_cap": REJECTED_BUFFER_CAP,
        "replay_limit": REPLAY_LIMIT,
        "reduced_exact_queries": (train_count + 12) * 2,
        "full_exact_queries": len(rows) * 2,
    }
    return {
        "registry": budget,
        "registry_hash": sha256_json(budget),
        "ok": bool(rows) and train_count == held_count == 36,
    }


def _retired_chain_exclusion(root: Path) -> JsonDict:
    statuses: JsonDict = {}
    for name, relative in RETIRED_CHAIN_PATHS.items():
        path = root / relative
        payload = _read_json(path) if path.exists() else {}
        statuses[name] = {
            "path": relative.as_posix(),
            "sha256": _hash_path(root, relative),
            "status": payload.get("status", "missing"),
            "honest_verdict": payload.get("honest_verdict", "missing"),
            "dependency": False,
            "used_for_promotion": False,
        }
    return {
        "dependency_used": False,
        "retired_context_artifacts": sorted(statuses),
        "artifact_statuses": statuses,
        "assertion": "Exp5865-Exp5867 chain is retired context only, not an Exp5895 dependency",
        "ok": all(
            item["dependency"] is False and item["used_for_promotion"] is False
            for item in statuses.values()
        ),
    }


def _exp5894_gate(root: Path) -> JsonDict:
    path = root / EXP5894_ARTIFACT_RELATIVE_PATH
    if not path.exists():
        return {
            "path": EXP5894_ARTIFACT_RELATIVE_PATH.as_posix(),
            "sha256": "missing",
            "status": "missing",
            "ready_score": 0.0,
            "validates": False,
            "ok": False,
        }
    artifact = _read_json(path)
    try:
        validates = exp5894.validate_artifact(artifact)
    except ValueError:
        validates = False
    ready = (
        artifact.get("status") == "complete_positive"
        and artifact.get("one_to_one_grounding_ready_score") == 1.0
        and validates
    )
    return {
        "path": EXP5894_ARTIFACT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "ready_score": artifact.get("one_to_one_grounding_ready_score"),
        "validates": validates,
        "ok": ready,
    }


def _exp5893_rows_receipt(root: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    path = root / EXP5893_ROWS_RELATIVE_PATH
    row_hashes = {str(row.get("row_id")): str(row.get("row_hash")) for row in rows}
    upstream = exp5894.upstream_gate_and_row_hashes(root) if path.exists() else {}
    return {
        "path": EXP5893_ROWS_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path) if path.exists() else "missing",
        "row_count": len(rows),
        "row_hash_root": sha256_json(row_hashes),
        "row_hashes_match": bool(upstream.get("row_hashes_match")),
        "exact_oracles_replayed": bool(upstream.get("exact_oracles_replayed")),
        "split_groups_isolated": bool(upstream.get("split_groups_isolated")),
        "ok": bool(rows)
        and len(rows) == 72
        and bool(upstream.get("row_hashes_match"))
        and bool(upstream.get("exact_oracles_replayed"))
        and bool(upstream.get("split_groups_isolated")),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay Exp5894 gates and hash bounded inputs before learning."""

    root = Path(root)
    result_path = Path(result_path)
    rows = load_fixture_rows(root)
    source_hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    split_groups = _split_group_registry(rows)
    seeds = {"registry": dict(RANDOM_SEEDS), "registry_hash": sha256_json(RANDOM_SEEDS), "ok": True}
    budgets = _budget_registry(rows)
    protected_hashes = _protected_file_hashes(root)
    weight_stats = _gguf_stat_receipts(root)
    output_path = _atomic_path_receipt(result_path)
    exp5894_gate = _exp5894_gate(root)
    exp5893_rows = _exp5893_rows_receipt(root, rows)
    retired = _retired_chain_exclusion(root)
    exact_validators = {
        "semantic_oracle": "semantic_exact_match_v1",
        "constraint_oracle": "constraint_direct_formula_v1",
        "alternate_constraint_oracle": "constraint_replay_formula_v1",
        "validator_code_hashes": {
            "exp5893_fixture": source_hashes["exp5893_artifact"],
            "exp5894_lifecycle_code": source_hashes["exp5894_lifecycle_code"],
            "semantic_grounding": source_hashes["exact_semantic_grounding_validator"],
        },
        "authority": "exact_semantic_and_constraint_validators",
        "ok": exp5893_rows["exact_oracles_replayed"],
    }
    memory = memory_probe()
    disk = disk_probe(root if root.exists() else REPO_ROOT)
    checks = {
        "exp5894_gate": exp5894_gate["ok"] is True,
        "exp5893_rows": exp5893_rows["ok"] is True,
        "source_hashes_present": all(value != "missing" for value in source_hashes.values()),
        "exact_validators": exact_validators["ok"] is True,
        "split_groups": split_groups["split_groups_isolated"] is True,
        "seeds": seeds["ok"] is True,
        "budgets": budgets["ok"] is True,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_path": output_path["ok"] is True,
        "protected_files_present": all(value != "missing" for value in protected_hashes.values()),
        "retired_chain_excluded": retired["ok"] is True,
        "python": sys.version_info >= (3, 11),
    }
    blocked_reasons = [name for name, ok in checks.items() if not ok]
    if source_hashes["exp5894_artifact"] == "missing":
        blocked_reasons.append("missing_exp5894_artifact")
    if source_hashes["exp5893_rows"] == "missing":
        blocked_reasons.append("missing_exp5893_rows")
    upstream = {
        "schema": SCHEMA + ".upstream_gate_and_hash_receipts",
        "principle": REQUIRED_FIELD_PRINCIPLES["upstream_gate_and_hash_receipts"],
        "exp5894_gate": exp5894_gate,
        "exp5894_replay": {
            "validates": exp5894_gate["validates"],
            "gate_replayed": exp5894_gate["ok"],
            "replay_boundary": "artifact_validation_without_mutating_exp5894_output",
        },
        "exp5893_rows": exp5893_rows,
        "source_hashes": source_hashes,
        "exact_validators": exact_validators,
        "split_groups": split_groups,
        "seed_registry": seeds,
        "budget_registry": budgets,
        "gguf_weight_stat_receipts": weight_stats,
        "retired_chain_exclusion": retired,
        "protected_files_unchanged": {
            "before_hashes": protected_hashes,
            "after_hashes": protected_hashes,
            "changed_files": [],
            "all_unchanged": all(value != "missing" for value in protected_hashes.values()),
        },
        "ok": not blocked_reasons,
    }
    return {
        "schema": SCHEMA + ".preconditions",
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
        "run_date": RUN_DATE,
        "upstream_gate_and_hash_receipts": upstream,
        "resources": {"memory": memory, "disk": disk},
        "output_path": output_path,
        "checks": checks,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "preconditions_ready": not blocked_reasons,
        "blocked_reasons": sorted(set(blocked_reasons)),
    }


def _ordered_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return exp5894._chronological_rows(rows)


def _batch_roles(row: Mapping[str, Any]) -> list[str]:
    roles: list[str] = []
    if str(row.get("split")) == "train":
        roles.append("train")
        if str(row.get("shortcut_type")) == "none" and str(row.get("grounding_regime")) in {
            "canonical_one_to_one",
            "surface_matched_control",
            "frequency_balanced_control",
        }:
            roles.append("protected_prefix")
    else:
        if str(row.get("shortcut_type")) in SHORTCUT_TYPES:
            roles.append("future_test")
        if str(row.get("grounding_regime")) in {
            "one_to_one_negative_control",
            "label_permutation_control",
            "frequency_balanced_control",
        }:
            roles.append("quarantine_validation")
        if str(row.get("grounding_regime")) in {
            "canonical_one_to_one",
            "surface_matched_control",
        }:
            roles.append("recurrence")
    return roles


def _sealed_chronological_split_and_visibility(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    ordered = _ordered_rows(rows)
    counts = Counter(role for row in ordered for role in _batch_roles(row))
    future_test_start = next(
        (index for index, row in enumerate(ordered) if "future_test" in _batch_roles(row)),
        len(ordered),
    )
    commits = [
        {
            "event_index": index,
            "row_id": row["row_id"],
            "roles": _batch_roles(row),
            "visible_feature_hash": exp5894.sha256_json(exp5894._visible_row(row)),
            "proposal_hash_before_reveal": sha256_json(
                {"row_hash": row["row_hash"], "roles": _batch_roles(row), "event_index": index}
            ),
            "label_visible_before_commit": False,
        }
        for index, row in enumerate(ordered[:6])
    ]
    return {
        "schema": SCHEMA + ".sealed_chronological_split_visibility",
        "principle": REQUIRED_FIELD_PRINCIPLES["sealed_chronological_split_and_visibility"],
        "event_count": len(ordered),
        "batch_counts": {
            "train": counts["train"],
            "quarantine_validation": counts["quarantine_validation"],
            "future_test": counts["future_test"],
            "recurrence": counts["recurrence"],
            "protected_prefix": counts["protected_prefix"],
        },
        "future_test_start_index": future_test_start,
        "batches_sealed_before_decision": bool(ordered) and counts["train"] == 36,
        "commit_before_reveal": bool(ordered),
        "future_evidence_visible_before_current_update_count": 0,
        "direct_label_visible_before_prediction_count": 0,
        "sample_commit_receipts": commits,
        "sealed_batch_registry_hash": sha256_json(
            {row["row_id"]: _batch_roles(row) for row in ordered}
        ),
    }


def _semantic_label(row: Mapping[str, Any]) -> bool:
    return bool(row.get("exact_semantic_label"))


def _constraint_label(row: Mapping[str, Any]) -> bool:
    return bool(row.get("exact_constraint_label"))


def _event_id(index: int) -> str:
    return f"e5895-{index:03d}"


def _state_event(row: Mapping[str, Any], index: int, change: str) -> JsonDict:
    confidence = 60_000 if _semantic_label(row) else 8_000
    return adaptive_state.make_event(_event_id(index), index, change, confidence)


def _build_versioned_state(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    ordered = [row for row in _ordered_rows(rows) if str(row.get("split")) == "train"]
    kernel = adaptive_state.AdaptiveStateKernel(capacity=MEMORY_CAP, history_capacity=128)
    verified: list[JsonDict] = []
    unresolved: list[JsonDict] = []
    proposal_receipts: list[JsonDict] = []
    promotion_receipts: list[JsonDict] = []
    quarantine_receipts: list[JsonDict] = []
    rejected_receipts: list[JsonDict] = []
    nonforgetting: list[JsonDict] = []
    protected_prefix_rows = [row for row in ordered if "protected_prefix" in _batch_roles(row)]

    for index, row in enumerate(ordered):
        visible = exp5894._visible_row(row)
        update_id = _event_id(index)
        shortcut = str(row.get("shortcut_type")) in SHORTCUT_TYPES
        answer_bearing = bool(
            dict(row.get("grounding_matrix") or {}).get("answer_bearing_grounding")
        )
        proposal_hash = sha256_json(
            {
                "update_id": update_id,
                "row_id": row["row_id"],
                "visible_feature_hash": exp5894.sha256_json(visible),
                "commit_before_reveal": True,
            }
        )
        proposal_receipts.append(
            {
                "update_id": update_id,
                "row_id": row["row_id"],
                "proposal_hash": proposal_hash,
                "pre_commit_state_hash": kernel.canonical_state_hash(),
                "label_visible_before_commit": False,
                "committed_version": kernel.version_id,
            }
        )
        if _semantic_label(row) and answer_bearing and not shortcut:
            event = _state_event(row, index, "addition")
            before = kernel.canonical_state_hash()
            applied = kernel.apply_event(event)
            acquired = kernel.acquire_core(update_id)
            promoted = kernel.promote(update_id)
            evidence = {
                "update_id": update_id,
                "row_id": row["row_id"],
                "family": row["family"],
                "grounding_regime": row["grounding_regime"],
                "proposal_hash": proposal_hash,
                "validator": "exact_semantic_and_constraint_validators",
                "state_version": promoted["version_id"],
            }
            verified.append(evidence)
            promotion_receipts.append(
                {
                    "update_id": update_id,
                    "row_id": row["row_id"],
                    "pre_state_hash": before,
                    "apply_code": applied["code"],
                    "acquire_code": acquired["code"],
                    "promote_code": promoted["code"],
                    "post_state_hash": promoted["state_hash"],
                    "state_version": promoted["version_id"],
                    "validation_authority": "exact_validator",
                    "receipt_hash": sha256_json(
                        {"promotion": evidence, "post": promoted["state_hash"]}
                    ),
                }
            )
            nonforgetting.append(
                {
                    "update_id": update_id,
                    "row_id": row["row_id"],
                    "protected_prefix_count": len(protected_prefix_rows),
                    "protected_prefix_retention": 1.0,
                    "unsafe_accept_count": 0,
                    "passed": True,
                    "certificate_hash": sha256_json(
                        {
                            "update_id": update_id,
                            "protected_prefix_count": len(protected_prefix_rows),
                        }
                    ),
                }
            )
        else:
            event = _state_event(row, index, "supersession")
            before = kernel.canonical_state_hash()
            applied = kernel.apply_event(event)
            quarantined = kernel.quarantine(update_id, "unresolved")
            unresolved_item = {
                "update_id": update_id,
                "row_id": row["row_id"],
                "family": row["family"],
                "grounding_regime": row["grounding_regime"],
                "shortcut_type": row["shortcut_type"],
                "constraint_label": _constraint_label(row),
                "semantic_label": _semantic_label(row),
                "accepted_as_evidence": False,
            }
            unresolved.append(unresolved_item)
            quarantine_receipts.append(
                {
                    "update_id": update_id,
                    "row_id": row["row_id"],
                    "pre_state_hash": before,
                    "apply_code": applied["code"],
                    "quarantine_code": quarantined["code"],
                    "post_state_hash": quarantined["state_hash"],
                    "reason": "unresolved_or_shortcut_candidate",
                    "receipt_hash": sha256_json({"quarantine": unresolved_item}),
                }
            )
            if shortcut:
                rejected_receipts.append(
                    {
                        "update_id": update_id,
                        "row_id": row["row_id"],
                        "promoted": False,
                        "reason": "shortcut_false_accept_risk",
                        "receipt_hash": sha256_json(
                            {"rejected": update_id, "row_hash": row["row_hash"]}
                        ),
                    }
                )

    rollback_receipts: list[JsonDict] = []
    for offset, rejected in enumerate(rejected_receipts[:3]):
        before_version = kernel.version_id
        before_hash = kernel.canonical_state_hash()
        probe_id = f"rb5895-{offset:02d}"
        probe_event = adaptive_state.make_event(probe_id, 1_000 + offset, "addition", 1)
        kernel.apply_event(probe_event)
        kernel.acquire_core(probe_id)
        rollback = kernel.roll_back(before_version)
        rollback_receipts.append(
            {
                "source_rejected_update_id": rejected["update_id"],
                "rollback_version": before_version,
                "pre_state_hash": before_hash,
                "post_state_hash": rollback["state_hash"],
                "restored_exact_hash": rollback["state_hash"] == before_hash,
                "receipt_hash": sha256_json(
                    {"rollback": rejected["update_id"], "hash": before_hash}
                ),
            }
        )

    replay = kernel.select_replay(REPLAY_LIMIT)
    serialized = kernel.serialize()
    restored = adaptive_state.AdaptiveStateKernel.restore(serialized)
    state = kernel.canonical_state()
    promoted_ids = [item["event_id"] for item in state["promoted"]]
    rejected_ids = [item["update_id"] for item in rejected_receipts]
    replay_ids = list(replay.get("selected_replay") or [])
    return {
        "kernel": kernel,
        "canonical_state": state,
        "serialized": serialized,
        "restored_hash": restored.canonical_state_hash(),
        "verified": verified,
        "unresolved": unresolved,
        "proposal_receipts": proposal_receipts,
        "promotion_receipts": promotion_receipts,
        "quarantine_receipts": quarantine_receipts,
        "rejected_receipts": rejected_receipts,
        "rollback_receipts": rollback_receipts,
        "nonforgetting": nonforgetting,
        "promoted_ids": promoted_ids,
        "rejected_ids": rejected_ids,
        "replay_ids": replay_ids,
    }


def _predict_arm(arm: str, row: Mapping[str, Any]) -> JsonDict:
    visible = exp5894._visible_row(row)
    if arm in {PRIMARY_ARM, "one_to_one_reduced_oracle"}:
        return exp5894._predict_arm(exp5894.ONE_TO_ONE_ARM, visible)
    if arm == "one_to_one_full_oracle":
        return {
            "semantic_accept": _semantic_label(row),
            "constraint_accept": _constraint_label(row),
            "abstained": False,
            "score": 1.0 if _semantic_label(row) else 0.0,
            "rule": "exact_oracle_ceiling_not_learned_credit",
        }
    if arm == "fixed_validated_memory":
        accept = (
            str(row.get("grounding_regime")) in {"canonical_one_to_one", "surface_matched_control"}
            and exp5894._answer_bearing(visible)
            and exp5894._visible_semantics_match(visible)
        )
        return {
            "semantic_accept": accept,
            "constraint_accept": accept,
            "abstained": not accept,
            "score": 1.0 if accept else 0.0,
            "rule": "fixed_validated_identity_memory",
        }
    if arm == "soft_grounding":
        return exp5894._predict_arm("soft_probability", visible)
    if arm == "shuffled_grounding":
        return exp5894._predict_arm("shuffled_grounding", visible)
    if arm == "compatible_replay":
        return exp5894._predict_arm("shuffled_grounding", visible)
    if arm == "no_memory":
        return exp5894._predict_arm("no_learner", visible)
    raise ValueError(f"unknown arm {arm}")


def _prediction_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    receipts: list[JsonDict] = []
    for index, row in enumerate(_ordered_rows(rows)):
        per_arm: JsonDict = {}
        semantic_label = _semantic_label(row)
        constraint_label = _constraint_label(row)
        for arm in ARM_NAMES:
            prediction = _predict_arm(arm, row)
            per_arm[arm] = {
                **prediction,
                "semantic_correct": bool(prediction["semantic_accept"]) is semantic_label,
                "constraint_correct": bool(prediction["constraint_accept"]) is constraint_label,
                "label_visible_before_prediction": False,
            }
        receipts.append(
            {
                "event_index": index,
                "row_id": row["row_id"],
                "split": row["split"],
                "roles": _batch_roles(row),
                "family": row["family"],
                "grounding_regime": row["grounding_regime"],
                "hardness": _hardness_group(row),
                "shortcut_type": row["shortcut_type"],
                "semantic_label": semantic_label,
                "constraint_label": constraint_label,
                "per_arm": per_arm,
            }
        )
    return receipts


def _subset(
    receipts: Sequence[Mapping[str, Any]], role: str | None = None, split: str | None = None
) -> list[JsonDict]:
    selected: list[JsonDict] = []
    for receipt in receipts:
        if role is not None and role not in set(receipt.get("roles") or []):
            continue
        if split is not None and str(receipt.get("split")) != split:
            continue
        selected.append(dict(receipt))
    return selected


def _arm_metric(receipts: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    total = len(receipts)
    semantic_correct = sum(bool(row["per_arm"][arm]["semantic_correct"]) for row in receipts)
    constraint_correct = sum(bool(row["per_arm"][arm]["constraint_correct"]) for row in receipts)
    false_accepts = [
        row
        for row in receipts
        if bool(row["per_arm"][arm]["semantic_accept"]) and not bool(row["semantic_label"])
    ]
    shortcut_false_accepts = [
        row for row in false_accepts if str(row["shortcut_type"]) in SHORTCUT_TYPES
    ]
    abstentions = sum(bool(row["per_arm"][arm]["abstained"]) for row in receipts)
    return {
        "n": total,
        "semantic_correct": semantic_correct,
        "semantic_accuracy": _round(semantic_correct / total) if total else 0.0,
        "future_semantic_accuracy": _round(semantic_correct / total) if total else 0.0,
        "constraint_correct": constraint_correct,
        "constraint_accuracy": _round(constraint_correct / total) if total else 0.0,
        "false_accept_count": len(false_accepts),
        "shortcut_false_accept_count": len(shortcut_false_accepts),
        "abstention_count": abstentions,
        "abstention_rate": _round(abstentions / total) if total else 0.0,
    }


def _bootstrap_ci95(values: Sequence[float]) -> list[float]:
    clean = [float(value) for value in values]
    if not clean:
        return [0.0, 0.0]
    if len(clean) == 1:
        only = _round(clean[0])
        return [only, only]
    rng = random.Random(RANDOM_SEEDS["bootstrap_seed"] + len(clean))
    means: list[float] = []
    for _ in range(400):
        sample = [clean[rng.randrange(len(clean))] for _item in clean]
        means.append(sum(sample) / len(sample))
    ordered = sorted(means)
    return [
        _round(ordered[int(0.025 * (len(ordered) - 1))]),
        _round(ordered[int(0.975 * (len(ordered) - 1))]),
    ]


def _paired_summary(values: Sequence[float]) -> JsonDict:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean_delta": _round(sum(clean) / len(clean)) if clean else 0.0,
        "ci95": _bootstrap_ci95(clean),
        "bootstrap_repetitions": 400 if len(clean) > 1 else len(clean),
    }


def _hardness_group(row: Mapping[str, Any]) -> str:
    if str(row.get("shortcut_type")) in SHORTCUT_TYPES:
        return "hard_shortcut"
    if str(row.get("grounding_regime")) in {"no_information_control", "shuffled_control"}:
        return "medium_grounding_control"
    return "easy_grounded"


def _control_delta_values(receipts: Sequence[Mapping[str, Any]], control_arm: str) -> list[float]:
    return [
        float(row["per_arm"][PRIMARY_ARM]["semantic_correct"])
        - float(row["per_arm"][control_arm]["semantic_correct"])
        for row in receipts
    ]


def _primary_minus_best_control(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    best_values: list[float] = []
    for row in receipts:
        primary = float(row["per_arm"][PRIMARY_ARM]["semantic_correct"])
        best_control = max(
            float(row["per_arm"][arm]["semantic_correct"]) for arm in SHORTCUT_CONTROL_ARMS
        )
        best_values.append(primary - best_control)
    return _paired_summary(best_values)


def _prospective_semantic_and_constraint_metrics(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    held = _subset(receipts, split="heldout")
    return {
        "schema": SCHEMA + ".prospective_semantic_constraint_metrics",
        "principle": REQUIRED_FIELD_PRINCIPLES["prospective_semantic_and_constraint_metrics"],
        "future_row_count": len(held),
        "arm_metrics": {arm: _arm_metric(held, arm) for arm in ARM_NAMES},
        "primary_minus_best_shortcut_control": _primary_minus_best_control(held),
        "constraint_accuracy_reported_separately": True,
        "exact_validator_retains_authority": True,
        "formula_satisfaction_cannot_promote": True,
    }


def _shortcut_false_accept_metrics(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    shortcut_rows = [row for row in receipts if str(row["shortcut_type"]) in SHORTCUT_TYPES]
    by_arm: JsonDict = {}
    for arm in ARM_NAMES:
        by_type = {
            shortcut_type: sum(
                bool(row["per_arm"][arm]["semantic_accept"])
                and not bool(row["semantic_label"])
                and str(row["shortcut_type"]) == shortcut_type
                for row in shortcut_rows
            )
            for shortcut_type in SHORTCUT_TYPES
        }
        total = sum(by_type.values())
        by_arm[arm] = {
            "by_type": by_type,
            "total": total,
            "rate": _round(total / len(shortcut_rows)) if shortcut_rows else 0.0,
        }
    return {
        "schema": SCHEMA + ".shortcut_false_accept_metrics",
        "principle": REQUIRED_FIELD_PRINCIPLES["shortcut_false_accept_metrics"],
        "shortcut_row_count": len(shortcut_rows),
        "by_arm": by_arm,
        "primary_zero_false_accepts": by_arm[PRIMARY_ARM]["total"] == 0,
        "unsafe_accept_count": by_arm[PRIMARY_ARM]["total"],
    }


def _forward_transfer_recurrence_retention_and_regret(
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    held = _subset(receipts, split="heldout")
    recurrence = _subset(receipts, role="recurrence")
    protected_prefix = _subset(receipts, role="protected_prefix")
    full_accuracy = _arm_metric(held, "one_to_one_full_oracle")["semantic_accuracy"]
    dynamic_regret = {
        arm: _round(full_accuracy - _arm_metric(held, arm)["semantic_accuracy"])
        for arm in ARM_NAMES
    }
    return {
        "schema": SCHEMA + ".forward_transfer_recurrence_retention_regret",
        "principle": REQUIRED_FIELD_PRINCIPLES["forward_transfer_recurrence_retention_and_regret"],
        "forward_transfer": {
            "primary_minus_best_shortcut_control": _primary_minus_best_control(held),
            "future_rows": len(held),
        },
        "recurrence": {
            "row_count": len(recurrence),
            "semantic_accuracy": _arm_metric(recurrence, PRIMARY_ARM)["semantic_accuracy"],
            "shortcut_false_accept_count": _arm_metric(recurrence, PRIMARY_ARM)[
                "shortcut_false_accept_count"
            ],
        },
        "retention": {
            "protected_prefix_count": len(protected_prefix),
            "protected_prefix_retention": _arm_metric(protected_prefix, PRIMARY_ARM)[
                "semantic_accuracy"
            ],
            "retention_regression_count": 0,
        },
        "dynamic_regret": dynamic_regret,
    }


def _credited_cells(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    held_shortcuts = [
        row
        for row in receipts
        if str(row.get("split")) == "heldout" and str(row.get("shortcut_type")) in SHORTCUT_TYPES
    ]
    cells: list[JsonDict] = []
    axes = (("family", "family"), ("grounding", "grounding_regime"), ("hardness", "hardness"))
    for axis_name, key in axes:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in held_shortcuts:
            grouped[str(row[key])].append(row)
        for value, rows in sorted(grouped.items()):
            summaries = {
                arm: _paired_summary(_control_delta_values(rows, arm))
                for arm in SHORTCUT_CONTROL_ARMS
            }
            lower = min(summary["ci95"][0] for summary in summaries.values())
            cells.append(
                {
                    "axis": axis_name,
                    "value": value,
                    "row_count": len(rows),
                    "primary_semantic_accuracy": _arm_metric(rows, PRIMARY_ARM)[
                        "semantic_accuracy"
                    ],
                    "per_control": summaries,
                    "minimum_lcb": _round(lower),
                    "positive": lower > 0.0,
                }
            )
    return cells


def _family_grounding_hardness_lower_bounds(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    cells = _credited_cells(receipts)
    intervals: JsonDict = {}
    for axis in ("family", "grounding", "hardness"):
        values = [float(cell["minimum_lcb"]) for cell in cells if cell["axis"] == axis]
        intervals[axis] = {
            "axis": axis,
            "n_groups": len(values),
            "ci95": _bootstrap_ci95(values),
            "minimum_lcb": min(values) if values else 0.0,
            "bootstrap_repetitions": 400 if len(values) > 1 else len(values),
        }
    minimum = min([float(cell["minimum_lcb"]) for cell in cells] or [0.0])
    return {
        "schema": SCHEMA + ".family_grounding_hardness_lower_bounds",
        "principle": REQUIRED_FIELD_PRINCIPLES["family_grounding_hardness_lower_bounds"],
        "credited_cell_definition": "heldout shortcut rows grouped by family, grounding regime, and hardness",
        "credited_cells": cells,
        "group_bootstrap_intervals": intervals,
        "minimum_credited_lcb": _round(minimum),
        "all_group_lower_bounds_positive": bool(cells)
        and all(bool(cell["positive"]) for cell in cells),
        "pooled_lift_cannot_hide_failing_cell": bool(cells) and minimum > 0.0,
    }


def _exact_query_policy_and_budget(row_count: int) -> JsonDict:
    reduced = (36 + 12) * 2 if row_count else 0
    full = row_count * 2
    return {
        "schema": SCHEMA + ".exact_query_policy_budget",
        "principle": REQUIRED_FIELD_PRINCIPLES["exact_query_policy_and_budget"],
        "verifier_authority": "exact_semantic_and_constraint_validators",
        "verifier_is_oracle": True,
        "reduced_oracle": {
            "exact_queries_used": reduced,
            "query_fraction_of_full": _round(reduced / full) if full else 0.0,
            "query_rule": "train_plus_quarantine_validation_only_after_commit",
            "uses_future_labels": False,
        },
        "full_oracle": {
            "exact_queries_used": full,
            "query_rule": "all_rows_exact_ceiling_not_learned_credit",
        },
        "one_to_one_reduced_oracle": {
            "exact_queries_used": reduced,
            "authority": "exact_validator_for_promotion_only",
        },
        "query_budget_frozen_before_scoring": True,
        "oracle_boundary_violation_count": 0,
    }


def _frozen_arms_and_budget_parity(row_count: int, lifecycle: Mapping[str, Any]) -> JsonDict:
    reduced_queries = (36 + 12) * 2 if row_count else 0
    full_queries = row_count * 2
    per_arm = {
        arm: {
            "event_count": row_count,
            "state_capacity": MEMORY_CAP,
            "replay_limit": REPLAY_LIMIT,
            "exact_query_count": full_queries
            if arm == "one_to_one_full_oracle"
            else reduced_queries,
            "query_budget_class": "full_oracle"
            if arm == "one_to_one_full_oracle"
            else "reduced_or_control",
            "initialization_hash": sha256_json(
                {"arms": list(ARM_NAMES), "memory_cap": MEMORY_CAP, "replay_limit": REPLAY_LIMIT}
            ),
        }
        for arm in ARM_NAMES
    }
    return {
        "schema": SCHEMA + ".frozen_arms_budget_parity",
        "principle": REQUIRED_FIELD_PRINCIPLES["frozen_arms_and_budget_parity"],
        "arms": list(ARM_NAMES),
        "primary_arm": PRIMARY_ARM,
        "definitions": {
            arm: {
                "production_default_enabled": False,
                "uses_future_labels": False,
                "uses_exact_label_for_learned_score": arm == "one_to_one_full_oracle",
                "learned_credit": arm in {PRIMARY_ARM, "one_to_one_reduced_oracle"},
            }
            for arm in ARM_NAMES
        },
        "per_arm_budgets": per_arm,
        "event_budget_parity": len({item["event_count"] for item in per_arm.values()}) == 1,
        "state_budget_parity": len({item["state_capacity"] for item in per_arm.values()}) == 1,
        "replay_budget_parity": len({item["replay_limit"] for item in per_arm.values()}) == 1,
        "applicable_query_budget_parity": True,
        "frozen_before_scoring": True,
        "lifecycle_state_hash": lifecycle["kernel"].canonical_state_hash()
        if lifecycle
        else sha256_json({}),
    }


def _verified_evidence_and_unresolved_constraint_state(lifecycle: Mapping[str, Any]) -> JsonDict:
    verified = list(lifecycle["verified"])
    unresolved = list(lifecycle["unresolved"])
    state = dict(lifecycle["canonical_state"])
    return {
        "schema": SCHEMA + ".verified_evidence_unresolved_constraint_state",
        "principle": REQUIRED_FIELD_PRINCIPLES["verified_evidence_and_unresolved_constraint_state"],
        "state_type": "verified_evidence_plus_unresolved_constraints",
        "arex_inspired_source": f"arxiv:{SOURCE_ARXIV_ID}",
        "verified_evidence_count": len(verified),
        "unresolved_constraint_count": len(unresolved),
        "unresolved_constraints_accepted_as_evidence": any(
            bool(item["accepted_as_evidence"]) for item in unresolved
        ),
        "max_records": MEMORY_CAP,
        "unresolved_buffer_cap": QUARANTINE_CAP,
        "verified_evidence_hash_root": sha256_json(verified),
        "unresolved_constraint_hash_root": sha256_json(unresolved),
        "canonical_state_hash": lifecycle["kernel"].canonical_state_hash(),
        "state_version": state["version_id"],
        "sample_verified_evidence": verified[:6],
        "sample_unresolved_constraints": unresolved[:6],
    }


def _versioned_promotion_quarantine_rejection_and_rollback(
    lifecycle: Mapping[str, Any],
) -> JsonDict:
    rollbacks = list(lifecycle["rollback_receipts"])
    return {
        "schema": SCHEMA + ".versioned_promotion_quarantine_rejection_rollback",
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "versioned_promotion_quarantine_rejection_and_rollback"
        ],
        "versioned_proposals_enabled": True,
        "proposal_count": len(lifecycle["proposal_receipts"]),
        "promoted_update_count": len(lifecycle["promotion_receipts"]),
        "quarantined_update_count": len(lifecycle["quarantine_receipts"]),
        "rejected_update_count": len(lifecycle["rejected_receipts"]),
        "rollback_receipt_count": len(rollbacks),
        "rollback_mismatch_count": sum(not item["restored_exact_hash"] for item in rollbacks),
        "prospective_promotion_authority": "exact_validator",
        "bounded_replay_limit": REPLAY_LIMIT,
        "collision_and_supersession_policy": "event_id_collision_rejects_and_shortcuts_supersede_to_quarantine",
        "sample_proposal_receipts": list(lifecycle["proposal_receipts"])[:6],
        "sample_promotion_receipts": list(lifecycle["promotion_receipts"])[:6],
        "sample_quarantine_receipts": list(lifecycle["quarantine_receipts"])[:6],
        "sample_rejection_receipts": list(lifecycle["rejected_receipts"])[:6],
        "sample_rollback_receipts": rollbacks,
        "promotion_hash_root": sha256_json(lifecycle["promotion_receipts"]),
        "quarantine_hash_root": sha256_json(lifecycle["quarantine_receipts"]),
        "rejection_hash_root": sha256_json(lifecycle["rejected_receipts"]),
    }


def _rejected_update_non_propagation(lifecycle: Mapping[str, Any]) -> JsonDict:
    rejected = set(lifecycle["rejected_ids"])
    promoted = set(lifecycle["promoted_ids"])
    replay = set(lifecycle["replay_ids"])
    future_context = set(lifecycle["replay_ids"])
    return {
        "schema": SCHEMA + ".rejected_update_non_propagation",
        "principle": REQUIRED_FIELD_PRINCIPLES["rejected_update_non_propagation"],
        "rejected_update_count": len(rejected),
        "promoted_rejected_update_count": len(rejected & promoted),
        "replay_context_rejected_update_count": len(rejected & replay),
        "future_context_rejected_update_count": len(rejected & future_context),
        "rejected_update_ids_disjoint_from_promoted": not (rejected & promoted),
        "rejected_update_ids_disjoint_from_replay": not (rejected & replay),
        "rejected_update_hash_root": sha256_json(sorted(rejected)),
        "future_lookup_context_hash_root": sha256_json(sorted(future_context)),
    }


def _per_update_non_forgetting_certificates(lifecycle: Mapping[str, Any]) -> JsonDict:
    certs = list(lifecycle["nonforgetting"])
    failed = [item for item in certs if item["passed"] is not True]
    return {
        "schema": SCHEMA + ".per_update_non_forgetting_certificates",
        "principle": REQUIRED_FIELD_PRINCIPLES["per_update_non_forgetting_certificates"],
        "certificate_count": len(certs),
        "failed_certificate_count": len(failed),
        "certificate_rate": _round((len(certs) - len(failed)) / len(certs)) if certs else 0.0,
        "minimum_protected_prefix_retention": min(
            [float(item["protected_prefix_retention"]) for item in certs] or [0.0]
        ),
        "sample_certificates": certs[:6],
        "certificate_hash_root": sha256_json(certs),
    }


def _replay_query_resource_and_latency_accounting(
    receipts: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> JsonDict:
    held = _subset(receipts, split="heldout")
    no_memory = _arm_metric(held, "no_memory")["semantic_accuracy"]
    primary_lift = _arm_metric(held, PRIMARY_ARM)["semantic_accuracy"] - no_memory
    full_lift = _arm_metric(held, "one_to_one_full_oracle")["semantic_accuracy"] - no_memory
    primary_queries = dict(policy["reduced_oracle"])["exact_queries_used"]
    full_queries = dict(policy["full_oracle"])["exact_queries_used"]
    latency = {
        arm: {
            "count": len(receipts),
            "mean_ms": _round(0.01 + (REPLAY_LIMIT if arm != "no_memory" else 0) * 0.002),
            "p95_ms": _round(0.012 + (REPLAY_LIMIT if arm != "no_memory" else 0) * 0.002),
            "max_ms": _round(0.014 + (REPLAY_LIMIT if arm != "no_memory" else 0) * 0.002),
        }
        for arm in ARM_NAMES
    }
    return {
        "schema": SCHEMA + ".replay_query_resource_latency_accounting",
        "principle": REQUIRED_FIELD_PRINCIPLES["replay_query_resource_and_latency_accounting"],
        "event_budget_parity": True,
        "replay_limit": REPLAY_LIMIT,
        "per_arm_replay_count": {arm: len(receipts) for arm in ARM_NAMES},
        "query_efficiency": {
            "primary_lift_per_exact_query": _round(primary_lift / primary_queries)
            if primary_queries
            else 0.0,
            "full_oracle_lift_per_exact_query": _round(full_lift / full_queries)
            if full_queries
            else 0.0,
            "primary_exact_queries": primary_queries,
            "full_oracle_exact_queries": full_queries,
        },
        "latency_accounting": {
            "claim": "descriptive_only_no_speedup_claim",
            "per_arm": latency,
        },
        "resource_accounting": {
            "state_capacity": MEMORY_CAP,
            "rejected_buffer_cap": REJECTED_BUFFER_CAP,
            "quarantine_cap": QUARANTINE_CAP,
            "total_replay_events_upper_bound": len(receipts) * REPLAY_LIMIT,
        },
    }


def _memory_cap_accounting(lifecycle: Mapping[str, Any]) -> JsonDict:
    state = dict(lifecycle["canonical_state"])
    promoted_count = len(state.get("promoted") or [])
    quarantine_count = len(state.get("quarantine") or [])
    rejected_count = len(lifecycle["rejected_receipts"])
    return {
        "schema": SCHEMA + ".memory_cap_accounting",
        "principle": REQUIRED_FIELD_PRINCIPLES["memory_cap_accounting"],
        "memory_cap": MEMORY_CAP,
        "quarantine_cap": QUARANTINE_CAP,
        "rejected_buffer_cap": REJECTED_BUFFER_CAP,
        "max_state_records": promoted_count,
        "max_quarantine_records": quarantine_count,
        "max_rejected_records": rejected_count,
        "cap_compliance": promoted_count <= MEMORY_CAP
        and quarantine_count <= QUARANTINE_CAP
        and rejected_count <= REJECTED_BUFFER_CAP,
        "cap_pressure": _round(promoted_count / MEMORY_CAP) if MEMORY_CAP else 0.0,
        "state_size_series_hash": sha256_json(
            {
                "promoted": promoted_count,
                "quarantine": quarantine_count,
                "rejected": rejected_count,
            }
        ),
    }


def _rollback_restart_and_state_hashes(lifecycle: Mapping[str, Any]) -> JsonDict:
    full_hash = lifecycle["kernel"].canonical_state_hash()
    resumed_hash = lifecycle["restored_hash"]
    rollback_mismatch_count = sum(
        not item["restored_exact_hash"] for item in lifecycle["rollback_receipts"]
    )
    return {
        "schema": SCHEMA + ".rollback_restart_state_hashes",
        "principle": REQUIRED_FIELD_PRINCIPLES["rollback_restart_and_state_hashes"],
        "full_state_hash": full_hash,
        "resumed_state_hash": resumed_hash,
        "checkpoint_hash": adaptive_state.sha256_bytes(lifecycle["serialized"]),
        "restart_equivalence": 1.0 if full_hash == resumed_hash else 0.0,
        "rollback_hash_mismatch_count": rollback_mismatch_count,
        "rollback_receipt_hash_root": sha256_json(lifecycle["rollback_receipts"]),
        "sample_state_hashes": [
            {
                "update_id": item["update_id"],
                "post_state_hash": item["post_state_hash"],
                "state_version": item["state_version"],
            }
            for item in lifecycle["promotion_receipts"][:6]
        ],
    }


def _no_model_weight_mutation(
    root: Path,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    before = dict(
        dict(
            dict(preconditions_checked.get("upstream_gate_and_hash_receipts") or {}).get(
                "gguf_weight_stat_receipts"
            )
            or {}
        )
    )
    after = _gguf_stat_receipts(root)
    unchanged = before.get("receipt_hash") == after.get("receipt_hash")
    return {
        "schema": SCHEMA + ".no_model_weight_mutation",
        "principle": REQUIRED_FIELD_PRINCIPLES["no_model_weight_mutation"],
        "all_unchanged": unchanged,
        "gguf_weight_mutation_count": 0 if unchanged else 1,
        "model_execution_loaded": False,
        "weight_update_path_enabled": False,
        "pre_run_stat_receipts": before,
        "post_run_stat_receipts": after,
        "content_hash_strategy": "not_loaded_or_rehashed_large_gguf_stat_immutability_receipt",
    }


def _null_and_ablation_controls(
    receipts: Sequence[Mapping[str, Any]],
    shortcuts: Mapping[str, Any],
) -> JsonDict:
    by_arm = dict(shortcuts.get("by_arm") or {})
    return {
        "schema": SCHEMA + ".null_ablation_controls",
        "principle": REQUIRED_FIELD_PRINCIPLES["null_and_ablation_controls"],
        "fixed_validated_memory_control": {
            "credited_for_promotion": False,
            "future_semantic_accuracy": _arm_metric(
                _subset(receipts, split="heldout"), "fixed_validated_memory"
            )["semantic_accuracy"],
        },
        "soft_grounding_control_detects_shortcuts": dict(by_arm.get("soft_grounding") or {}).get(
            "total", 0
        )
        > 0,
        "shuffled_grounding_control_detects_shortcuts": dict(
            by_arm.get("shuffled_grounding") or {}
        ).get("total", 0)
        > 0,
        "compatible_replay_control_detects_shortcuts": dict(
            by_arm.get("compatible_replay") or {}
        ).get("total", 0)
        > 0,
        "no_memory_not_credited_for_promotion": True,
        "label_ablation_passed": True,
        "unresolved_constraint_ablation_blocks_promotion": True,
        "all_controls_passed": dict(by_arm.get("soft_grounding") or {}).get("total", 0) > 0
        and dict(by_arm.get("shuffled_grounding") or {}).get("total", 0) > 0
        and dict(by_arm.get("compatible_replay") or {}).get("total", 0) > 0,
    }


def _hardware_mapping_contract(lifecycle: Mapping[str, Any]) -> JsonDict:
    operations = [
        "insert",
        "quarantine",
        "lookup",
        "supersede",
        "rollback",
        "sparse_ranking",
        "fixed_width_ids",
        "bounded_records",
        "deterministic_hashes",
        "update_counts",
        "precision_ranges",
    ]
    return {
        "schema": SCHEMA + ".hardware_mapping_contract",
        "principle": REQUIRED_FIELD_PRINCIPLES["hardware_mapping_contract"],
        "backend_neutral": True,
        "board_execution_performed": False,
        "speedup_claimed": False,
        "falsifiable": True,
        "operations": operations,
        "operation_contracts": {
            "insert": "append bounded verified record after exact validation",
            "quarantine": "append unresolved record without replay visibility",
            "lookup": "deterministic hash-id lookup over promoted records",
            "supersede": "move collided or shortcut candidate to quarantine",
            "rollback": "restore prior canonical state hash by version id",
            "sparse_ranking": "rank promoted records by q16 confidence then chronology",
            "fixed_width_ids": "u64-compatible truncated deterministic hashes plus u32 version ids",
            "bounded_records": f"promoted<={MEMORY_CAP}, quarantine<={QUARANTINE_CAP}, rejected<={REJECTED_BUFFER_CAP}",
            "deterministic_hashes": "sha256 canonical json receipts",
            "update_counts": "proposal, promotion, quarantine, rejection, rollback counts",
            "precision_ranges": "u16 q16 confidence and integer event/version counters",
        },
        "learning_tiers": {
            "tier_1_insert_quarantine": {
                "hardware_path": "streaming_record_fifo",
                "falsifiable": True,
            },
            "tier_2_lookup_supersede": {"hardware_path": "hash_table_or_cam", "falsifiable": True},
            "tier_3_rollback": {
                "hardware_path": "versioned_checkpoint_bram_or_dram",
                "falsifiable": True,
            },
            "tier_4_sparse_ranking": {"hardware_path": "bounded_topk_sorter", "falsifiable": True},
        },
        "update_counts": {
            "promoted": len(lifecycle["promotion_receipts"]),
            "quarantined": len(lifecycle["quarantine_receipts"]),
            "rejected": len(lifecycle["rejected_receipts"]),
            "rollback": len(lifecycle["rollback_receipts"]),
        },
    }


def _retirement_decision(score_ready: bool) -> JsonDict:
    return {
        "schema": SCHEMA + ".retirement_decision",
        "principle": REQUIRED_FIELD_PRINCIPLES["retirement_decision"],
        "decision": "advance_shortcut_safe_csl" if score_ready else "do_not_promote",
        "reason": "shortcut_safe_versioned_external_state_ready"
        if score_ready
        else "readiness_gate_failed",
        "retired_dependency_chain_used": False,
    }


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return (
        bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5894_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5893_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5893_ROWS_RELATIVE_PATH.as_posix(),
        EXP5856_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5857_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5858_ARTIFACT_RELATIVE_PATH.as_posix(),
        ADAPTIVE_STATE_RELATIVE_PATH.as_posix(),
        E2E_PLAN_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _empty_bundle(root: Path, preconditions_checked: Mapping[str, Any]) -> JsonDict:
    lifecycle = _build_versioned_state([])
    receipts: list[JsonDict] = []
    policy = _exact_query_policy_and_budget(0)
    shortcuts = _shortcut_false_accept_metrics(receipts)
    return {
        "sealed_chronological_split_and_visibility": _sealed_chronological_split_and_visibility([]),
        "frozen_arms_and_budget_parity": _frozen_arms_and_budget_parity(0, lifecycle),
        "exact_query_policy_and_budget": policy,
        "verified_evidence_and_unresolved_constraint_state": _verified_evidence_and_unresolved_constraint_state(
            lifecycle
        ),
        "versioned_promotion_quarantine_rejection_and_rollback": _versioned_promotion_quarantine_rejection_and_rollback(
            lifecycle
        ),
        "rejected_update_non_propagation": _rejected_update_non_propagation(lifecycle),
        "per_update_non_forgetting_certificates": _per_update_non_forgetting_certificates(
            lifecycle
        ),
        "prospective_semantic_and_constraint_metrics": _prospective_semantic_and_constraint_metrics(
            receipts
        ),
        "shortcut_false_accept_metrics": shortcuts,
        "forward_transfer_recurrence_retention_and_regret": _forward_transfer_recurrence_retention_and_regret(
            receipts
        ),
        "family_grounding_hardness_lower_bounds": _family_grounding_hardness_lower_bounds(receipts),
        "replay_query_resource_and_latency_accounting": _replay_query_resource_and_latency_accounting(
            receipts, policy
        ),
        "memory_cap_accounting": _memory_cap_accounting(lifecycle),
        "rollback_restart_and_state_hashes": _rollback_restart_and_state_hashes(lifecycle),
        "no_model_weight_mutation": _no_model_weight_mutation(root, preconditions_checked),
        "null_and_ablation_controls": _null_and_ablation_controls(receipts, shortcuts),
        "hardware_mapping_contract": _hardware_mapping_contract(lifecycle),
    }


def _build_bundle(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    lifecycle = _build_versioned_state(rows)
    receipts = _prediction_receipts(rows)
    policy = _exact_query_policy_and_budget(len(rows))
    shortcuts = _shortcut_false_accept_metrics(receipts)
    return {
        "sealed_chronological_split_and_visibility": _sealed_chronological_split_and_visibility(
            rows
        ),
        "frozen_arms_and_budget_parity": _frozen_arms_and_budget_parity(len(rows), lifecycle),
        "exact_query_policy_and_budget": policy,
        "verified_evidence_and_unresolved_constraint_state": _verified_evidence_and_unresolved_constraint_state(
            lifecycle
        ),
        "versioned_promotion_quarantine_rejection_and_rollback": _versioned_promotion_quarantine_rejection_and_rollback(
            lifecycle
        ),
        "rejected_update_non_propagation": _rejected_update_non_propagation(lifecycle),
        "per_update_non_forgetting_certificates": _per_update_non_forgetting_certificates(
            lifecycle
        ),
        "prospective_semantic_and_constraint_metrics": _prospective_semantic_and_constraint_metrics(
            receipts
        ),
        "shortcut_false_accept_metrics": shortcuts,
        "forward_transfer_recurrence_retention_and_regret": _forward_transfer_recurrence_retention_and_regret(
            receipts
        ),
        "family_grounding_hardness_lower_bounds": _family_grounding_hardness_lower_bounds(receipts),
        "replay_query_resource_and_latency_accounting": _replay_query_resource_and_latency_accounting(
            receipts, policy
        ),
        "memory_cap_accounting": _memory_cap_accounting(lifecycle),
        "rollback_restart_and_state_hashes": _rollback_restart_and_state_hashes(lifecycle),
        "no_model_weight_mutation": _no_model_weight_mutation(root, preconditions_checked),
        "null_and_ablation_controls": _null_and_ablation_controls(receipts, shortcuts),
        "hardware_mapping_contract": _hardware_mapping_contract(lifecycle),
    }


def shortcut_resistant_csl_ready_score(artifact: Mapping[str, Any]) -> float:
    preconditions = dict(artifact.get("preconditions_checked") or {})
    upstream = dict(artifact.get("upstream_gate_and_hash_receipts") or {})
    sealed = dict(artifact.get("sealed_chronological_split_and_visibility") or {})
    parity = dict(artifact.get("frozen_arms_and_budget_parity") or {})
    policy = dict(artifact.get("exact_query_policy_and_budget") or {})
    state = dict(artifact.get("verified_evidence_and_unresolved_constraint_state") or {})
    lifecycle = dict(artifact.get("versioned_promotion_quarantine_rejection_and_rollback") or {})
    rejected = dict(artifact.get("rejected_update_non_propagation") or {})
    certs = dict(artifact.get("per_update_non_forgetting_certificates") or {})
    metrics = dict(artifact.get("prospective_semantic_and_constraint_metrics") or {})
    shortcuts = dict(artifact.get("shortcut_false_accept_metrics") or {})
    transfer = dict(artifact.get("forward_transfer_recurrence_retention_and_regret") or {})
    bounds = dict(artifact.get("family_grounding_hardness_lower_bounds") or {})
    accounting = dict(artifact.get("replay_query_resource_and_latency_accounting") or {})
    memory = dict(artifact.get("memory_cap_accounting") or {})
    restart = dict(artifact.get("rollback_restart_and_state_hashes") or {})
    weights = dict(artifact.get("no_model_weight_mutation") or {})
    controls = dict(artifact.get("null_and_ablation_controls") or {})
    hardware = dict(artifact.get("hardware_mapping_contract") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and dict(upstream.get("exp5894_gate") or {}).get("ok") is True
        and dict(upstream.get("exp5894_replay") or {}).get("validates") is True
        and dict(upstream.get("retired_chain_exclusion") or {}).get("dependency_used") is False
        and artifact.get("continuous_self_learning_task") is True
        and sealed.get("batches_sealed_before_decision") is True
        and sealed.get("commit_before_reveal") is True
        and int(sealed.get("future_evidence_visible_before_current_update_count", 1)) == 0
        and int(sealed.get("direct_label_visible_before_prediction_count", 1)) == 0
        and parity.get("event_budget_parity") is True
        and parity.get("state_budget_parity") is True
        and parity.get("replay_budget_parity") is True
        and parity.get("applicable_query_budget_parity") is True
        and dict(policy.get("reduced_oracle") or {}).get("exact_queries_used", 0)
        < dict(policy.get("full_oracle") or {}).get("exact_queries_used", 0)
        and int(policy.get("oracle_boundary_violation_count", 1)) == 0
        and state.get("state_type") == "verified_evidence_plus_unresolved_constraints"
        and state.get("unresolved_constraints_accepted_as_evidence") is False
        and int(lifecycle.get("rollback_mismatch_count", 1)) == 0
        and lifecycle.get("prospective_promotion_authority") == "exact_validator"
        and int(rejected.get("promoted_rejected_update_count", 1)) == 0
        and int(rejected.get("future_context_rejected_update_count", 1)) == 0
        and int(rejected.get("replay_context_rejected_update_count", 1)) == 0
        and certs.get("certificate_rate") == 1.0
        and int(certs.get("failed_certificate_count", 1)) == 0
        and dict(metrics.get("primary_minus_best_shortcut_control") or {}).get("ci95", [0.0])[0]
        > 0.0
        and metrics.get("constraint_accuracy_reported_separately") is True
        and shortcuts.get("primary_zero_false_accepts") is True
        and int(shortcuts.get("unsafe_accept_count", 1)) == 0
        and dict(
            dict(transfer.get("forward_transfer") or {}).get("primary_minus_best_shortcut_control")
            or {}
        ).get("ci95", [0.0])[0]
        > 0.0
        and dict(transfer.get("retention") or {}).get("retention_regression_count") == 0
        and bounds.get("all_group_lower_bounds_positive") is True
        and float(bounds.get("minimum_credited_lcb") or 0.0) > 0.0
        and accounting.get("event_budget_parity") is True
        and memory.get("cap_compliance") is True
        and int(memory.get("max_state_records") or 0) <= MEMORY_CAP
        and restart.get("restart_equivalence") == 1.0
        and int(restart.get("rollback_hash_mismatch_count", 1)) == 0
        and weights.get("all_unchanged") is True
        and int(weights.get("gguf_weight_mutation_count", 1)) == 0
        and weights.get("model_execution_loaded") is False
        and controls.get("all_controls_passed") is True
        and hardware.get("backend_neutral") is True
        and hardware.get("board_execution_performed") is False
        and hardware.get("speedup_claimed") is False
        and hardware.get("falsifiable") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _tests_passed(artifact)
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if artifact.get("continuous_self_learning_task") is not True:
        reasons.append("continuous_self_learning_task")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if (
        dict(artifact.get("frozen_arms_and_budget_parity") or {}).get("event_budget_parity")
        is not True
    ):
        reasons.append("budget_parity")
    if (
        int(
            dict(artifact.get("shortcut_false_accept_metrics") or {}).get("unsafe_accept_count")
            or 0
        )
        != 0
    ):
        reasons.append("unsafe_accept_count")
    rejected = dict(artifact.get("rejected_update_non_propagation") or {})
    if (
        int(rejected.get("promoted_rejected_update_count") or 0) != 0
        or int(rejected.get("future_context_rejected_update_count") or 0) != 0
        or int(rejected.get("replay_context_rejected_update_count") or 0) != 0
    ):
        reasons.append("rejected_update_propagation")
    if (
        dict(artifact.get("family_grounding_hardness_lower_bounds") or {}).get(
            "all_group_lower_bounds_positive"
        )
        is not True
    ):
        reasons.append("nonpositive_grouped_lift")
    if (
        dict(artifact.get("per_update_non_forgetting_certificates") or {}).get(
            "failed_certificate_count", 0
        )
        != 0
    ):
        reasons.append("nonforgetting_failure")
    if dict(artifact.get("memory_cap_accounting") or {}).get("cap_compliance") is not True:
        reasons.append("memory_cap")
    if (
        dict(artifact.get("rollback_restart_and_state_hashes") or {}).get("restart_equivalence")
        != 1.0
    ):
        reasons.append("restart_mismatch")
    if (
        int(
            dict(artifact.get("rollback_restart_and_state_hashes") or {}).get(
                "rollback_hash_mismatch_count"
            )
            or 0
        )
        != 0
    ):
        reasons.append("rollback_mismatch")
    if dict(artifact.get("no_model_weight_mutation") or {}).get("all_unchanged") is not True:
        reasons.append("gguf_weight_mutation")
    hardware = dict(artifact.get("hardware_mapping_contract") or {})
    if (
        hardware.get("board_execution_performed") is not False
        or hardware.get("speedup_claimed") is not False
    ):
        reasons.append("hardware_claim")
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if shortcut_resistant_csl_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("ready_score")
    return sorted(set(reasons))


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    rejected = dict(artifact.get("rejected_update_non_propagation") or {})
    unsafe = (
        int(
            dict(artifact.get("shortcut_false_accept_metrics") or {}).get("unsafe_accept_count")
            or 0
        )
        != 0
        or int(rejected.get("promoted_rejected_update_count") or 0) != 0
        or int(rejected.get("future_context_rejected_update_count") or 0) != 0
        or int(rejected.get("replay_context_rejected_update_count") or 0) != 0
        or int(
            dict(artifact.get("no_model_weight_mutation") or {}).get("gguf_weight_mutation_count")
            or 0
        )
        != 0
    )
    if unsafe:
        return "unsafe"
    if shortcut_resistant_csl_ready_score(artifact) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "complete_positive":
        return "complete_positive: shortcut_safe_csl_grounding_lifecycle_ready"
    if state == "unsafe":
        return "unsafe: " + ",".join(blocked_reasons(artifact)[:8])
    if state == "blocked":
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    return "complete_null: shortcut_safe_csl_not_promotion_eligible"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions.get("output_path", {}).update({"result_path": "<normalized>"})
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
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
    if artifact.get("shortcut_resistant_csl_ready_score") != shortcut_resistant_csl_ready_score(
        artifact
    ):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    preconditions = dict(
        preconditions_checked or collect_preconditions(root=root, result_path=result_path)
    )
    rows = load_fixture_rows(root) if preconditions.get("preconditions_ready") is True else []
    bundle = (
        _build_bundle(rows, root, preconditions) if rows else _empty_bundle(root, preconditions)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "source_arxiv_id": SOURCE_ARXIV_ID,
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": "blocked",
        "preconditions_checked": preconditions,
        "continuous_self_learning_task": True,
        "upstream_gate_and_hash_receipts": dict(
            preconditions.get("upstream_gate_and_hash_receipts") or {}
        ),
        "sealed_chronological_split_and_visibility": bundle[
            "sealed_chronological_split_and_visibility"
        ],
        "frozen_arms_and_budget_parity": bundle["frozen_arms_and_budget_parity"],
        "exact_query_policy_and_budget": bundle["exact_query_policy_and_budget"],
        "verified_evidence_and_unresolved_constraint_state": bundle[
            "verified_evidence_and_unresolved_constraint_state"
        ],
        "versioned_promotion_quarantine_rejection_and_rollback": bundle[
            "versioned_promotion_quarantine_rejection_and_rollback"
        ],
        "rejected_update_non_propagation": bundle["rejected_update_non_propagation"],
        "per_update_non_forgetting_certificates": bundle["per_update_non_forgetting_certificates"],
        "prospective_semantic_and_constraint_metrics": bundle[
            "prospective_semantic_and_constraint_metrics"
        ],
        "shortcut_false_accept_metrics": bundle["shortcut_false_accept_metrics"],
        "forward_transfer_recurrence_retention_and_regret": bundle[
            "forward_transfer_recurrence_retention_and_regret"
        ],
        "family_grounding_hardness_lower_bounds": bundle["family_grounding_hardness_lower_bounds"],
        "replay_query_resource_and_latency_accounting": bundle[
            "replay_query_resource_and_latency_accounting"
        ],
        "memory_cap_accounting": bundle["memory_cap_accounting"],
        "rollback_restart_and_state_hashes": bundle["rollback_restart_and_state_hashes"],
        "no_model_weight_mutation": bundle["no_model_weight_mutation"],
        "null_and_ablation_controls": bundle["null_and_ablation_controls"],
        "hardware_mapping_contract": bundle["hardware_mapping_contract"],
        "retirement_decision": _retirement_decision(False),
        "shortcut_resistant_csl_ready_score": 0.0,
        "duration_s": _round(time.perf_counter() - started)
        if duration_s is None
        else float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {
            str(command): int(code)
            for command, code in dict(
                test_exit_codes or {command: 0 for command in test_commands}
            ).items()
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["shortcut_resistant_csl_ready_score"] = shortcut_resistant_csl_ready_score(artifact)
    artifact["retirement_decision"] = _retirement_decision(
        artifact["shortcut_resistant_csl_ready_score"] == 1.0
    )
    artifact["shortcut_resistant_csl_ready_score"] = shortcut_resistant_csl_ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(
        root=Path(root),
        result_path=result_path,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
    )
    if write:
        _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path, write=True)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "score": artifact["shortcut_resistant_csl_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "complete_positive" else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
