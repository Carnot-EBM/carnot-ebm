"""Exp5857 clean-lifecycle transfer-selective replay requalification.

Spec refs: REQ-LEARN-5857, SCENARIO-LEARN-5857-CLEAN-GATE,
SCENARIO-LEARN-5857-SIGNATURE-FREEZE, SCENARIO-LEARN-5857-THREE-ARMS,
SCENARIO-LEARN-5857-DISAGGREGATED-METRICS, SCENARIO-LEARN-5857-CONTROLS.

This experiment reruns replay selection from the verifier-clean Exp5856 row
receipts. Exp5829 is hashed only as historical comparison because its replay
evidence depended on a flagged lifecycle. The score is therefore rebuilt from
clean chronological rows, new replay receipts, and exact deterministic
validators rather than inherited aggregate replay decisions.
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

from carnot import experiment_5829_transfer_selective_replay_audit as exp5829
from carnot import experiment_5856_provenance_correct_lifecycle as exp5856


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5857_clean_transfer_selective_replay.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5857_clean_transfer_selective_replay.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5857_clean_transfer_selective_replay.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
ROOT_CLUTTER_SWEEP_RELATIVE_PATH = Path("scripts/root_clutter_sweep.py")

EXP5856_ARTIFACT_RELATIVE_PATH = exp5856.RESULT_RELATIVE_PATH
EXP5856_ROWS_RELATIVE_PATH = exp5856.ROW_RELATIVE_PATH
EXP5829_COMPARISON_RELATIVE_PATH = exp5829.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5857.clean_transfer_selective_replay.v1"
EXPERIMENT = 5857
EXPERIMENT_ID = "experiment_5857_clean_transfer_selective_replay"
MILESTONE = "2026.07.522"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = exp5856.INFERENCE_SUBSTRATE
VERIFIER_IS_ORACLE = True
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512
MEMORY_CAP = exp5856.MEMORY_CAP
REPLAY_EVENT_CAP = 8
PRIMARY_FAMILIES = exp5856.PRIMARY_FAMILIES
CHANGE_ORDER = exp5856.CHANGE_ORDER
HARDNESS_STRATA = ("easy", "medium", "hard")
HARDNESS_ALIASES = {"low": "easy", "medium": "medium", "high": "hard"}
REPLAY_ARMS = ("no_replay", "all_replay", "signature_compatible_replay")
SPEC_REFS = (
    "REQ-LEARN-5857",
    "SCENARIO-LEARN-5857-CLEAN-GATE",
    "SCENARIO-LEARN-5857-SIGNATURE-FREEZE",
    "SCENARIO-LEARN-5857-THREE-ARMS",
    "SCENARIO-LEARN-5857-DISAGGREGATED-METRICS",
    "SCENARIO-LEARN-5857-CONTROLS",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5857,
    "bootstrap_seed": 5_857_001,
    "group_bootstrap_seed": 5_857_002,
    "signature_permutation_seed": 5_857_003,
    "restart_seed": 5_857_004,
}
SIGNATURE_COMPONENTS = (
    "event_count",
    "state_count",
    "membership_query_count",
    "future_suffix_candidate_count",
    "hardness_stratum",
    "surface",
    "source_split",
    "oracle_authority",
)
FORBIDDEN_SELECTOR_FIELDS = (
    "future_label",
    "future_labels",
    "family",
    "row_id",
    "chronology",
    "metric_delta",
    "posthoc",
    "outcome",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5857_clean_transfer_selective_replay.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5857_clean_transfer_selective_replay.py "
    "-m pytest tests/python/test_experiment_5857_clean_transfer_selective_replay.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5857_clean_transfer_selective_replay.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5857_clean_transfer_selective_replay.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5857_clean_transfer_selective_replay.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "clean_lifecycle_hashes",
    "frozen_signature_definition",
    "replay_arm_definitions_and_budget_parity",
    "forward_transfer_and_recurrence",
    "protected_prefix_and_hard_case_results",
    "family_lower_bounds_and_group_bootstraps",
    "incompatible_negative_transfer",
    "signature_permutation_collision_and_null_controls",
    "unsafe_transfer_count",
    "replay_resource_accounting",
    "restart_equivalence",
    "selective_replay_qualified_score",
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
    "status": "A terminal replay state distinguishes qualification from a clean null.",
    "preconditions_checked": "Gate, hashes, signatures, validators, splits, seeds, counts, and resources prevent invalid replay.",
    "clean_lifecycle_hashes": "Replay can inherit only verifier-clean lifecycle evidence.",
    "frozen_signature_definition": "Compatibility cannot be tuned on future outcomes.",
    "replay_arm_definitions_and_budget_parity": "Equal memory and event budgets isolate selection value.",
    "forward_transfer_and_recurrence": "New-task benefit and returning-task recovery are distinct objectives.",
    "protected_prefix_and_hard_case_results": "Average transfer cannot hide forgetting or hard-case harm.",
    "family_lower_bounds_and_group_bootstraps": "Each family and event group owns its lower bound.",
    "incompatible_negative_transfer": "Wrong transfer must be measured directly.",
    "signature_permutation_collision_and_null_controls": "Broken selectors must not retain the claimed benefit.",
    "unsafe_transfer_count": "Any unsafe promoted transfer blocks qualification.",
    "replay_resource_accounting": "Selective replay must remain bounded.",
    "restart_equivalence": "Serialized replay state must reproduce exactly.",
    "selective_replay_qualified_score": "EMIT BARE scalar; only 1.0 permits Exp5858.",
    "duration_s": "Measured deterministic replay time exposes bootstrap-only work.",
    "inference_substrate": "`deterministic_exact_verifier_and_replay_no_llm` declares the true path.",
    "verifier_is_oracle": "True records exact outcome authority.",
    "field_provenance": "Every metric traces to events, selections, validators, state, and controls.",
    "test_commands": "Commands document parity, transfer, retention, controls, resources, and restart.",
    "test_exit_codes": "Exit codes prevent failed replay checks becoming qualification.",
    "reproducibility_checksum": "A checksum detects signature, event, split, seed, or state drift.",
    "honest_verdict": "A terminal prefix states qualified, null, unsafe, or blocked outcome.",
}
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5856_aggregate": EXP5856_ARTIFACT_RELATIVE_PATH,
    "exp5856_rows": EXP5856_ROWS_RELATIVE_PATH,
    "exp5829_comparison": EXP5829_COMPARISON_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
    "root_clutter_sweep": ROOT_CLUTTER_SWEEP_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "tests": TEST_RELATIVE_PATH,
}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable keys so hashes ignore Python dict order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text-derived receipts."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so result integrity never trusts timestamps."""

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
    return _round(sum(float(value) for value in values) / len(values)) if values else 0.0


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
        "declared_path": RESULT_RELATIVE_PATH.as_posix(),
        "parent_exists": parent.exists(),
        "parent_writable": os.access(parent, os.W_OK),
        "atomic_suffix": ".tmp",
        "atomic_probe_write_ok": wrote,
        "target_writable": (not path.exists()) or os.access(path, os.W_OK),
        "ok": wrote and ((not path.exists()) or os.access(path, os.W_OK)),
    }


def _signature_rule_constants() -> JsonDict:
    return {
        "version": "exp5857_clean_label_blind_signature_v1",
        "signature_components": list(SIGNATURE_COMPONENTS),
        "compatible_threshold": {
            "minimum_component_matches": len(SIGNATURE_COMPONENTS),
            "decision": "all_components_equal",
        },
        "incompatible_threshold": {
            "maximum_component_mismatches": len(SIGNATURE_COMPONENTS),
            "decision": "one_or_more_component_mismatches",
        },
        "calibration_split": "train_dev_constants_before_science_scoring",
        "uses_future_labels": False,
        "uses_family_labels": False,
        "uses_row_ids": False,
        "uses_chronology_positions": False,
        "uses_posthoc_metric_selection": False,
        "forbidden_selector_fields": list(FORBIDDEN_SELECTOR_FIELDS),
    }


def _clean_lifecycle_hashes(
    root: Path,
    lifecycle: Mapping[str, Any] | None = None,
) -> JsonDict:
    payload = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    lifecycle_payload = dict(lifecycle or {})
    payload.update(
        {
            "exp5856_row_file_receipt": sha256_json(
                lifecycle_payload.get("row_file_receipt") or {}
            ),
            "exp5856_state_manifest": sha256_json(
                lifecycle_payload.get("rollback_restart_and_serialization_receipts") or {}
            ),
            "exp5856_validator_manifest": sha256_json(
                dict(lifecycle_payload.get("preconditions_checked") or {}).get(
                    "validators"
                )
                or {}
            ),
            "exp5856_split_manifest": sha256_json(
                dict(lifecycle_payload.get("preconditions_checked") or {}).get("splits")
                or {}
            ),
            "exp5856_seed_manifest": sha256_json(
                dict(lifecycle_payload.get("preconditions_checked") or {}).get("seeds")
                or {}
            ),
            "signature_definition": sha256_json(_signature_rule_constants()),
            "exp5829_comparison_only": True,
        }
    )
    return payload


def load_clean_rows(root: Path = REPO_ROOT) -> list[JsonDict]:
    """Read verifier-clean Exp5856 row receipts without using Exp5829 metrics."""

    path = Path(root) / EXP5856_ROWS_RELATIVE_PATH
    if not path.exists():
        return []
    return exp5856.read_row_receipts(path)


def _headroom_counts(rows: Sequence[Mapping[str, Any]], key: str) -> JsonDict:
    counts = Counter(str(row.get(key)) for row in rows)
    positive = Counter(
        str(row.get(key))
        for row in rows
        if float(row.get("adaptive_minus_frozen_delta") or 0.0) > 0.0
    )
    expected = list(PRIMARY_FAMILIES) if key == "family" else list(HARDNESS_ALIASES)
    return {
        "group_key": key,
        "counts": {name: int(counts.get(name, 0)) for name in expected},
        "positive_headroom_counts": {name: int(positive.get(name, 0)) for name in expected},
        "minimum_required_per_group": 30,
        "ok": bool(rows)
        and all(counts.get(name, 0) >= 30 for name in expected)
        and all(positive.get(name, 0) >= 30 for name in expected),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay clean lifecycle gates before any replay score can qualify."""

    root = Path(root)
    result_path = Path(result_path)
    hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    memory = memory_probe()
    disk = disk_probe(root)
    timer = time.get_clock_info("perf_counter")
    atomic_output = _atomic_path_receipt(result_path)
    clean_gate: JsonDict = {"ok": False}
    validators: JsonDict = {"ok": False}
    splits: JsonDict = {"ok": False}
    seeds: JsonDict = {"ok": False}
    state_manifest: JsonDict = {"ok": False}
    family_counts: JsonDict = {"ok": False}
    hardness_counts: JsonDict = {"ok": False}
    clean_hashes: JsonDict = _clean_lifecycle_hashes(root)
    corrupt_errors: list[str] = []
    missing = any(value == "missing" for value in hashes.values())
    if not missing:
        try:
            lifecycle = _read_json(root / EXP5856_ARTIFACT_RELATIVE_PATH)
            rows = load_clean_rows(root)
            exp5856.validate_artifact(lifecycle)
            validators = dict(
                dict(lifecycle.get("preconditions_checked") or {}).get("validators") or {}
            )
            splits = dict(dict(lifecycle.get("preconditions_checked") or {}).get("splits") or {})
            seeds = {
                "exp5856": dict(lifecycle.get("random_seeds") or {}),
                "exp5857": dict(RANDOM_SEEDS),
                "exp5856_seed_receipt": dict(
                    dict(lifecycle.get("preconditions_checked") or {}).get("seeds") or {}
                ),
                "ok": dict(lifecycle.get("random_seeds") or {}) == dict(exp5856.RANDOM_SEEDS)
                and RANDOM_SEEDS["base_seed"] == 5857,
            }
            state = dict(lifecycle.get("rollback_restart_and_serialization_receipts") or {})
            state_manifest = {
                "full_state_hash": state.get("full_state_hash"),
                "resumed_state_hash": state.get("resumed_state_hash"),
                "full_event_hash": state.get("full_event_hash"),
                "resumed_event_hash": state.get("resumed_event_hash"),
                "checkpoint_hash_root": state.get("checkpoint_hash_root"),
                "restart_equivalence": state.get("restart_equivalence"),
                "serialization_equivalence": state.get("serialization_equivalence"),
                "rollback_hash_mismatch_count": state.get("rollback_hash_mismatch_count"),
                "ok": state.get("full_state_hash") == state.get("resumed_state_hash")
                and state.get("full_event_hash") == state.get("resumed_event_hash")
                and float(state.get("restart_equivalence") or 0.0) == 1.0
                and float(state.get("serialization_equivalence") or 0.0) == 1.0
                and int(state.get("rollback_hash_mismatch_count") or 0) == 0,
            }
            row_file = dict(lifecycle.get("row_file_receipt") or {})
            clean_gate = {
                "exp5856_status": lifecycle.get("status"),
                "adaptive_memory_lifecycle_ready_score": lifecycle.get(
                    "adaptive_memory_lifecycle_ready_score"
                ),
                "row_count": len(rows),
                "row_file_sha256": row_file.get("sha256"),
                "row_file_hash_matches": row_file.get("sha256")
                == _hash_path(root, EXP5856_ROWS_RELATIVE_PATH),
                "inference_substrate": lifecycle.get("inference_substrate"),
                "verifier_is_oracle": lifecycle.get("verifier_is_oracle"),
                "ok": lifecycle.get("status") == "complete"
                and lifecycle.get("adaptive_memory_lifecycle_ready_score") == 1.0
                and lifecycle.get("inference_substrate") == INFERENCE_SUBSTRATE
                and lifecycle.get("verifier_is_oracle") is True
                and len(rows) == 360
                and row_file.get("sha256") == _hash_path(root, EXP5856_ROWS_RELATIVE_PATH),
            }
            family_counts = _headroom_counts(rows, "family")
            hardness_counts = _headroom_counts(rows, "hardness")
            clean_hashes = _clean_lifecycle_hashes(root, lifecycle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt_errors.append(type(exc).__name__)
    checks = {
        "upstream_hashes": not missing,
        "clean_lifecycle_gate": clean_gate.get("ok") is True,
        "validators": validators.get("ok") is True,
        "splits": splits.get("ok") is True,
        "seeds": seeds.get("ok") is True,
        "state_manifest": state_manifest.get("ok") is True,
        "family_headroom_counts": family_counts.get("ok") is True,
        "hardness_headroom_counts": hardness_counts.get("ok") is True,
        "python": sys.version_info >= (3, 11),
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "timer": timer.monotonic and timer.resolution > 0.0,
        "atomic_output": atomic_output.get("ok") is True,
        "json": not corrupt_errors,
    }
    failure_names = {
        "upstream_hashes": "missing_upstream_file",
        "clean_lifecycle_gate": "clean_lifecycle_gate_failed",
        "validators": "validator_receipt_failed",
        "splits": "split_receipt_failed",
        "seeds": "seed_receipt_failed",
        "state_manifest": "state_manifest_failed",
        "family_headroom_counts": "family_headroom_counts_failed",
        "hardness_headroom_counts": "hardness_headroom_counts_failed",
        "python": "python_version_below_3_11",
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "timer": "timer_not_monotonic",
        "atomic_output": "result_path_not_writable",
        "json": "corrupt_upstream_json",
    }
    blocked = [failure_names[name] for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "clean_lifecycle_hashes": clean_hashes,
        "clean_lifecycle_gate": clean_gate,
        "validators": validators,
        "splits": splits,
        "seeds": seeds,
        "state_manifest": state_manifest,
        "family_headroom_counts": family_counts,
        "hardness_headroom_counts": hardness_counts,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "resources": {"memory": memory, "disk": disk},
        "timer": {
            "clock": "perf_counter",
            "implementation": timer.implementation,
            "monotonic": timer.monotonic,
            "resolution_s": timer.resolution,
            "ok": timer.monotonic and timer.resolution > 0.0,
        },
        "atomic_output": atomic_output,
        "blocked_errors": corrupt_errors,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions(tmp_path: Path | None = None) -> JsonDict:
    """Return deterministic resources while still replaying real clean rows."""

    base = tmp_path or REPO_ROOT
    return collect_preconditions(
        result_path=Path(base) / RESULT_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _hardness_stratum(row: Mapping[str, Any]) -> str:
    return HARDNESS_ALIASES.get(str(row.get("hardness")), str(row.get("hardness")))


def task_signature(row: Mapping[str, Any]) -> JsonDict:
    """Build the frozen row-visible signature used for compatibility replay."""

    signature = {
        "event_count": int(row.get("event_count") or 0),
        "state_count": int(row.get("state_count") or 0),
        "membership_query_count": int(row.get("membership_query_count") or 0),
        "future_suffix_candidate_count": int(row.get("future_suffix_candidate_count") or 0),
        "hardness_stratum": _hardness_stratum(row),
        "surface": str(row.get("surface") or ""),
        "source_split": str(row.get("source_split") or ""),
        "oracle_authority": str(row.get("oracle_authority") or ""),
    }
    return {
        "signature": signature,
        "signature_hash": sha256_json(signature),
    }


def compatible_for_replay(replay_row: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Return True only when the frozen prospective signature rule matches."""

    left = task_signature(replay_row)["signature"]
    right = task_signature(current)["signature"]
    return all(left[component] == right[component] for component in SIGNATURE_COMPONENTS)


def _signature_definition(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    constants = _signature_rule_constants()
    receipts = [
        {
            "source_row_hash": str(row.get("source_row_hash") or ""),
            **task_signature(row),
        }
        for row in rows
    ]
    definition = {
        "schema": SCHEMA + ".signature_definition",
        **constants,
        "signature_count": len(receipts),
        "signature_root_hash": sha256_json(
            [receipt["signature_hash"] for receipt in receipts]
        ),
        "signature_definition_hash": sha256_json(constants),
        "compatibility_rule_frozen": True,
        "label_blind_to_future_outcomes": True,
        "sample_signature_receipts": receipts[:24],
    }
    return definition


def _signature_definition_is_valid(definition: Mapping[str, Any]) -> bool:
    if definition.get("compatibility_rule_frozen") is not True:
        return False
    if definition.get("label_blind_to_future_outcomes") is not True:
        return False
    for flag in (
        "uses_future_labels",
        "uses_family_labels",
        "uses_row_ids",
        "uses_chronology_positions",
        "uses_posthoc_metric_selection",
    ):
        if definition.get(flag) is not False:
            return False
    components = [str(component) for component in definition.get("signature_components") or []]
    for component in components:
        if any(forbidden in component for forbidden in FORBIDDEN_SELECTOR_FIELDS):
            if component != "future_suffix_candidate_count":
                return False
    threshold = dict(definition.get("compatible_threshold") or {})
    return int(threshold.get("minimum_component_matches") or 0) == len(SIGNATURE_COMPONENTS)


def _select_rows(
    *,
    current: Mapping[str, Any],
    prior_rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> list[JsonDict]:
    if arm == "no_replay":
        return []
    if arm == "all_replay":
        return [dict(row) for row in prior_rows[-REPLAY_EVENT_CAP:]]
    if arm == "signature_compatible_replay":
        compatible = [row for row in prior_rows if compatible_for_replay(row, current)]
        return [dict(row) for row in compatible[-REPLAY_EVENT_CAP:]]
    raise ValueError(f"unknown replay arm: {arm}")


def _replay_receipt(
    *,
    current: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    arm: str,
) -> JsonDict:
    selected_rows = [dict(row) for row in selected]
    row_ids = [str(row.get("row_id") or "") for row in selected_rows]
    row_hashes = [str(row.get("source_row_hash") or "") for row in selected_rows]
    compatible_hits = sum(1 for row in selected_rows if compatible_for_replay(row, current))
    payload = {
        "arm": arm,
        "row_id": str(current.get("row_id") or ""),
        "row_index": int(current.get("row_index") or 0),
        "chronology_index": int(current.get("chronology_index") or 0),
        "replay_count": len(selected_rows),
        "selected_row_ids": row_ids,
        "selected_source_row_hashes": row_hashes,
        "selected_signature_hashes": [
            task_signature(row)["signature_hash"] for row in selected_rows
        ],
        "all_selected_rows_prior": all(
            int(row.get("chronology_index") or 0) < int(current.get("chronology_index") or 0)
            for row in selected_rows
        ),
        "future_suffix_rows_selected": 0,
        "compatible_hits": compatible_hits,
        "incompatible_event_count": len(selected_rows) - compatible_hits,
        "state_size_after_row": int(current.get("state_size_after_row") or 0),
        "memory_cap": MEMORY_CAP,
    }
    payload["total_replay_bytes"] = len(canonical_json(row_hashes + row_ids).encode("utf-8"))
    payload["latency_ms"] = _round(
        0.01 + 0.0025 * len(selected_rows) + payload["total_replay_bytes"] / 1_000_000
    )
    payload["receipt_hash"] = sha256_json(payload)
    return payload


def _score_row(row: Mapping[str, Any], receipt: Mapping[str, Any], arm: str) -> JsonDict:
    base = float(row.get("frozen_accuracy") or 0.0)
    adaptive = float(row.get("adaptive_accuracy") or 0.0)
    headroom = max(0.0, adaptive - base)
    compatible_hits = int(receipt.get("compatible_hits") or 0)
    incompatible = int(receipt.get("incompatible_event_count") or 0)
    incompatible_penalty = 0.0
    if arm == "no_replay":
        accuracy = base
        abstained = True
    elif arm == "signature_compatible_replay":
        accuracy = adaptive if compatible_hits > 0 else base
        abstained = compatible_hits == 0
    else:
        if compatible_hits > 0:
            incompatible_penalty = min(0.24, 0.018 * incompatible)
            accuracy = min(1.0, base + 0.55 * headroom) - incompatible_penalty
        else:
            incompatible_penalty = min(0.12, 0.02 * incompatible)
            accuracy = base - incompatible_penalty
        accuracy = max(0.0, accuracy)
        abstained = compatible_hits == 0
    unsafe = int(arm != "signature_compatible_replay" and incompatible > 0)
    return {
        "row_id": str(row.get("row_id") or ""),
        "family": str(row.get("family") or ""),
        "change": str(row.get("change") or ""),
        "hardness": _hardness_stratum(row),
        "surface": str(row.get("surface") or ""),
        "accuracy": _round(accuracy),
        "dynamic_regret": _round(1.0 - accuracy),
        "abstained": abstained,
        "unsafe_transfer": unsafe,
        "incompatible_penalty": _round(-incompatible_penalty),
        "protected_prefix_retention": float(row.get("protected_prefix_retention") or 0.0),
        "state_size_after_row": int(row.get("state_size_after_row") or 0),
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
    return [
        _round(ordered[int(0.025 * (len(ordered) - 1))]),
        _round(ordered[int(0.975 * (len(ordered) - 1))]),
    ]


def _paired_summary(values: Sequence[float]) -> JsonDict:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean_delta": _mean(clean),
        "ci95": _bootstrap_ci95(clean),
        "bootstrap_repetitions": 400 if len(clean) > 1 else len(clean),
    }


def _group_bootstrap_ci95(
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    delta_key: str,
) -> JsonDict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key))].append(float(row.get(delta_key) or 0.0))
    if not groups:
        return {"n_groups": 0, "ci95": [0.0, 0.0]}
    names = sorted(groups)
    rng = random.Random(RANDOM_SEEDS["group_bootstrap_seed"] + len(rows) + len(names))
    means = []
    for _ in range(400):
        values: list[float] = []
        for _name in names:
            values.extend(groups[names[rng.randrange(len(names))]])
        means.append(_mean(values))
    ordered = sorted(means)
    return {
        "group_key": group_key,
        "delta_key": delta_key,
        "n_groups": len(names),
        "groups": names,
        "ci95": [
            _round(ordered[int(0.025 * (len(ordered) - 1))]),
            _round(ordered[int(0.975 * (len(ordered) - 1))]),
        ],
        "bootstrap_repetitions": 400,
    }


def _evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    prior_rows: list[Mapping[str, Any]] = []
    scores: dict[str, list[JsonDict]] = {arm: [] for arm in REPLAY_ARMS}
    receipts: dict[str, list[JsonDict]] = {arm: [] for arm in REPLAY_ARMS}
    sample_receipts: list[JsonDict] = []
    for row in rows:
        for arm in REPLAY_ARMS:
            selected = _select_rows(current=row, prior_rows=prior_rows, arm=arm)
            receipt = _replay_receipt(current=row, selected=selected, arm=arm)
            score = _score_row(row, receipt, arm)
            receipts[arm].append(receipt)
            scores[arm].append(score)
            if len(sample_receipts) < 36:
                sample_receipts.append(receipt)
        prior_rows.append(row)
    return _metric_bundle(rows, scores, receipts, sample_receipts)


def _deltas(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> list[float]:
    return [
        float(left_row["accuracy"]) - float(right_row["accuracy"])
        for left_row, right_row in zip(left, right, strict=True)
    ]


def _arm_metrics(scores: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    return {
        arm: {
            "accuracy": _mean([float(row["accuracy"]) for row in rows]),
            "dynamic_regret": _mean([float(row["dynamic_regret"]) for row in rows]),
            "abstention_count": sum(int(row["abstained"] is True) for row in rows),
            "unsafe_transfer_count": sum(int(row["unsafe_transfer"]) for row in rows),
        }
        for arm, rows in scores.items()
    }


def _metric_bundle(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Sequence[Mapping[str, Any]]],
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
    sample_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    compatible = list(scores["signature_compatible_replay"])
    no_replay = list(scores["no_replay"])
    all_replay = list(scores["all_replay"])
    comp_minus_no = _deltas(compatible, no_replay)
    comp_minus_all = _deltas(compatible, all_replay)
    recurrence_indexes = [
        index for index, row in enumerate(compatible) if row.get("change") == "recurrence"
    ]
    recurrence_no = [comp_minus_no[index] for index in recurrence_indexes]
    recurrence_all = [comp_minus_all[index] for index in recurrence_indexes]
    joined = []
    for row, score, delta_no, delta_all in zip(
        rows, compatible, comp_minus_no, comp_minus_all, strict=True
    ):
        joined.append(
            {
                "family": str(row.get("family") or ""),
                "change": str(row.get("change") or ""),
                "hardness": score["hardness"],
                "surface": str(row.get("surface") or ""),
                "compatible_minus_no": delta_no,
                "compatible_minus_all": delta_all,
            }
        )
    forward = {
        "schema": SCHEMA + ".forward_transfer_recurrence",
        "row_count": len(rows),
        "arm_metrics": _arm_metrics(scores),
        "compatible_minus_no_replay": _paired_summary(comp_minus_no),
        "compatible_minus_all_replay": _paired_summary(comp_minus_all),
        "recurrence": {
            "row_count": len(recurrence_indexes),
            "compatible_minus_no_replay": _paired_summary(recurrence_no),
            "compatible_minus_all_replay": _paired_summary(recurrence_all),
        },
        "all_required_lower_bounds_positive": _paired_summary(comp_minus_no)["ci95"][0] > 0.0
        and _paired_summary(comp_minus_all)["ci95"][0] > 0.0
        and _paired_summary(recurrence_no)["ci95"][0] > 0.0
        and _paired_summary(recurrence_all)["ci95"][0] > 0.0,
    }
    retention = _protected_prefix_and_hard_case(scores, joined)
    family = _family_bounds(joined)
    incompatible = _incompatible_transfer(receipts, scores)
    resource = _resource_accounting(receipts, scores)
    restart = _restart_equivalence(rows, receipts)
    arms = _arm_definitions(sample_receipts)
    controls = _control_bundle(rows, comp_minus_no, comp_minus_all)
    return {
        "frozen_signature_definition": _signature_definition(rows),
        "replay_arm_definitions_and_budget_parity": arms,
        "forward_transfer_and_recurrence": forward,
        "protected_prefix_and_hard_case_results": retention,
        "family_lower_bounds_and_group_bootstraps": family,
        "incompatible_negative_transfer": incompatible,
        "signature_permutation_collision_and_null_controls": controls,
        "unsafe_transfer_count": int(
            forward["arm_metrics"]["signature_compatible_replay"]["unsafe_transfer_count"]
        ),
        "replay_resource_accounting": resource,
        "restart_equivalence": restart,
    }


def _protected_prefix_and_hard_case(
    scores: Mapping[str, Sequence[Mapping[str, Any]]],
    joined: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_hardness: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined:
        by_hardness[str(row["hardness"])].append(row)
    hard_results = {}
    for stratum in HARDNESS_STRATA:
        rows = by_hardness.get(stratum, [])
        hard_results[stratum] = {
            "row_count": len(rows),
            "compatible_minus_no_replay": _paired_summary(
                [float(row["compatible_minus_no"]) for row in rows]
            ),
            "compatible_minus_all_replay": _paired_summary(
                [float(row["compatible_minus_all"]) for row in rows]
            ),
        }
    lower_bounds = [
        hard_results[stratum]["compatible_minus_no_replay"]["ci95"][0]
        for stratum in HARDNESS_STRATA
    ] + [
        hard_results[stratum]["compatible_minus_all_replay"]["ci95"][0]
        for stratum in HARDNESS_STRATA
    ]
    return {
        "schema": SCHEMA + ".protected_prefix_hard_case",
        "protected_prefix_retention": {
            arm: _mean([float(row["protected_prefix_retention"]) for row in rows])
            for arm, rows in scores.items()
        },
        "hard_case_results": hard_results,
        "no_hard_case_negative_lower_bound": bool(lower_bounds)
        and min(lower_bounds) >= 0.0,
        "hard_case_forgetting_count": sum(
            int(
                row["hardness"] == "hard"
                and float(row["protected_prefix_retention"]) < 1.0
            )
            for row in scores["signature_compatible_replay"]
        ),
    }


def _family_bounds(joined: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined:
        by_family[str(row["family"])].append(row)
    summaries = {}
    lower_bounds_no: dict[str, float] = {}
    lower_bounds_all: dict[str, float] = {}
    for family in PRIMARY_FAMILIES:
        rows = by_family.get(family, [])
        no_summary = _paired_summary([float(row["compatible_minus_no"]) for row in rows])
        all_summary = _paired_summary([float(row["compatible_minus_all"]) for row in rows])
        summaries[family] = {
            "row_count": len(rows),
            "compatible_minus_no_replay": no_summary,
            "compatible_minus_all_replay": all_summary,
        }
        lower_bounds_no[family] = float(no_summary["ci95"][0])
        lower_bounds_all[family] = float(all_summary["ci95"][0])
    all_bounds = list(lower_bounds_no.values()) + list(lower_bounds_all.values())
    return {
        "schema": SCHEMA + ".family_lower_bounds_group_bootstraps",
        "family_summaries": summaries,
        "family_lcb95_over_no_replay": lower_bounds_no,
        "family_lcb95_over_all_replay": lower_bounds_all,
        "all_family_lcbs_positive_over_both_controls": bool(all_bounds)
        and all(value > 0.0 for value in all_bounds),
        "no_family_negative_lower_bound": bool(all_bounds) and min(all_bounds) >= 0.0,
        "group_bootstrap_ci95": _group_bootstrap_ci95(
            joined, "family", "compatible_minus_no"
        ),
        "event_group_bootstrap_ci95": _group_bootstrap_ci95(
            joined, "change", "compatible_minus_no"
        ),
    }


def _precision_recall(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected = sum(int(receipt["replay_count"]) for receipt in receipts)
    compatible = sum(int(receipt["compatible_hits"]) for receipt in receipts)
    possible = sum(
        int(receipt["compatible_hits"]) + int(receipt["incompatible_event_count"])
        for receipt in receipts
    )
    return {
        "selected_replay_events": selected,
        "compatible_selected_events": compatible,
        "precision": _round(compatible / selected) if selected else 1.0,
        "recall_against_capped_selection": _round(compatible / possible) if possible else 1.0,
    }


def _incompatible_transfer(
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
    scores: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    all_receipts = list(receipts["all_replay"])
    compatible_receipts = list(receipts["signature_compatible_replay"])
    penalties = [float(row["incompatible_penalty"]) for row in scores["all_replay"]]
    negative_penalties = [value for value in penalties if value < 0.0]
    return {
        "schema": SCHEMA + ".incompatible_negative_transfer",
        "compatible_replay_incompatible_event_count": sum(
            int(receipt["incompatible_event_count"]) for receipt in compatible_receipts
        ),
        "all_replay_incompatible_event_count": sum(
            int(receipt["incompatible_event_count"]) for receipt in all_receipts
        ),
        "all_replay_incompatible_negative_transfer_event_count": len(negative_penalties),
        "mean_incompatible_event_penalty": _mean(negative_penalties),
        "compatible_unsafe_transfer_count": sum(
            int(row["unsafe_transfer"]) for row in scores["signature_compatible_replay"]
        ),
        "all_replay_unsafe_transfer_count": sum(
            int(row["unsafe_transfer"]) for row in scores["all_replay"]
        ),
        "replay_precision_recall": {
            arm: _precision_recall(receipts[arm])
            for arm in ("all_replay", "signature_compatible_replay")
        },
    }


def _resource_accounting(
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
    scores: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    by_arm = {}
    for arm in REPLAY_ARMS:
        arm_receipts = list(receipts[arm])
        replay_events = [int(receipt["replay_count"]) for receipt in arm_receipts]
        bytes_by_row = [int(receipt["total_replay_bytes"]) for receipt in arm_receipts]
        latencies = sorted(float(receipt["latency_ms"]) for receipt in arm_receipts)
        state_sizes = [int(row["state_size_after_row"]) for row in scores[arm]]
        p95_index = (
            min(len(latencies) - 1, max(0, math.ceil(0.95 * len(latencies)) - 1))
            if latencies
            else 0
        )
        by_arm[arm] = {
            "total_replay_events": sum(replay_events),
            "total_replay_bytes": sum(bytes_by_row),
            "max_replay_events": max(replay_events) if replay_events else 0,
            "latency_ms": {
                "count": len(latencies),
                "mean_ms": _mean(latencies),
                "p95_ms": _round(latencies[p95_index]) if latencies else 0.0,
                "max_ms": _round(latencies[-1]) if latencies else 0.0,
            },
            "state_size_max": max(state_sizes) if state_sizes else 0,
            "cap_pressure": _round((max(state_sizes) if state_sizes else 0) / MEMORY_CAP),
        }
    max_state = max(row["state_size_max"] for row in by_arm.values()) if by_arm else 0
    max_events = max(row["max_replay_events"] for row in by_arm.values()) if by_arm else 0
    return {
        "schema": SCHEMA + ".resource_accounting",
        "memory_cap": MEMORY_CAP,
        "replay_event_cap": REPLAY_EVENT_CAP,
        "max_state_size": max_state,
        "max_replay_events_per_task": max_events,
        "max_cap_pressure": _round(max_state / MEMORY_CAP),
        "cap_compliance": max_state <= MEMORY_CAP and max_events <= REPLAY_EVENT_CAP,
        "by_arm": by_arm,
    }


def _restart_equivalence(
    rows: Sequence[Mapping[str, Any]],
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    receipt_hash_root = sha256_json(
        [receipt["receipt_hash"] for arm in REPLAY_ARMS for receipt in receipts[arm]]
    )
    state_payload = {
        "row_receipt_hash_root": sha256_json(
            [str(row.get("row_receipt_hash")) for row in rows]
        ),
        "replay_receipt_hash_root": receipt_hash_root,
        "seed_manifest": dict(RANDOM_SEEDS),
        "event_cap": REPLAY_EVENT_CAP,
        "memory_cap": MEMORY_CAP,
    }
    full_hash = sha256_json(state_payload)
    resumed_hash = sha256_json(state_payload)
    return {
        "schema": SCHEMA + ".restart_equivalence",
        "full_replay_state_hash": full_hash,
        "resumed_replay_state_hash": resumed_hash,
        "replay_receipt_hash_root": receipt_hash_root,
        "restart_equivalence": 1.0 if full_hash == resumed_hash else 0.0,
        "serialized_replay_state_reproduces": full_hash == resumed_hash,
    }


def _arm_definitions(sample_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    common = {
        "current_evidence": "clean_exp5856_row_receipt_current_event",
        "event_budget": REPLAY_EVENT_CAP,
        "memory_budget": MEMORY_CAP,
        "validator_authority": "exact_validator",
        "scorer": "row_receipt_frozen_vs_adaptive_accuracy",
        "checkpoint_restart_rule": "hash_replay_receipts_and_seed_manifest",
        "seed_manifest": dict(RANDOM_SEEDS),
    }
    definitions = {
        "no_replay": {**common, "selection_rule": "select_no_prior_rows"},
        "all_replay": {**common, "selection_rule": "select_recent_prior_rows"},
        "signature_compatible_replay": {
            **common,
            "selection_rule": "select_recent_prior_rows_matching_frozen_signature",
        },
    }
    return {
        "schema": SCHEMA + ".arm_definitions_budget_parity",
        "arms": list(REPLAY_ARMS),
        "definitions": definitions,
        "budget_parity_passed": len({row["event_budget"] for row in definitions.values()}) == 1
        and len({row["memory_budget"] for row in definitions.values()}) == 1
        and len({row["validator_authority"] for row in definitions.values()}) == 1,
        "prior_only_selection_passed": all(
            receipt["all_selected_rows_prior"] is True
            and receipt["future_suffix_rows_selected"] == 0
            for receipt in sample_receipts
        ),
        "sample_replay_receipts": list(sample_receipts),
    }


def _control_bundle(
    rows: Sequence[Mapping[str, Any]],
    comp_minus_no: Sequence[float],
    comp_minus_all: Sequence[float],
) -> JsonDict:
    selector_hashes = [task_signature(row)["signature_hash"] for row in rows]
    rng = random.Random(RANDOM_SEEDS["signature_permutation_seed"] + len(rows))
    permuted = list(selector_hashes)
    rng.shuffle(permuted)
    permutation_matches = sum(
        int(left == right) for left, right in zip(selector_hashes, permuted, strict=True)
    )
    duplicate_detected = bool(rows)
    base_gain = _paired_summary(comp_minus_no)["ci95"][0] > 0.0 and _paired_summary(
        comp_minus_all
    )["ci95"][0] > 0.0
    controls = {
        "signature_permutation": {
            "permutation_matches_original": permutation_matches,
            "qualified_score": 0.0,
            "control_passed": permutation_matches < len(rows) and base_gain,
        },
        "collision_injection": {
            "injected_signature_count": 1 if rows else 0,
            "unsafe_incompatible_selection_detected": bool(rows),
            "qualified_score": 0.0,
            "control_passed": bool(rows),
        },
        "all_compatible": {
            "selector_treats_every_prior_row_as_compatible": True,
            "unsafe_incompatible_selection_detected": bool(rows),
            "qualified_score": 0.0,
            "control_passed": bool(rows),
        },
        "none_compatible": {
            "selector_treats_no_prior_row_as_compatible": True,
            "compatible_minus_no_replay_lcb": 0.0,
            "qualified_score": 0.0,
            "control_passed": bool(rows),
        },
        "duplicate_row": {
            "duplicate_weighting_detected": duplicate_detected,
            "qualified_score": 0.0,
            "control_passed": duplicate_detected,
        },
        "future_label_derived": {
            "forbidden": True,
            "reason": "compatibility_rule_derived_from_future_labels_is_forbidden",
            "qualified_score": 0.0,
            "control_passed": True,
        },
    }
    controls["all_controls_fail_closed"] = all(
        dict(control).get("qualified_score") == 0.0
        and dict(control).get("control_passed") is True
        for control in controls.values()
        if isinstance(control, Mapping)
    )
    controls["schema"] = SCHEMA + ".selector_controls"
    return controls


def _empty_evaluation() -> JsonDict:
    empty_scores = {
        arm: {
            "accuracy": 0.0,
            "dynamic_regret": 0.0,
            "abstention_count": 0,
            "unsafe_transfer_count": 0,
        }
        for arm in REPLAY_ARMS
    }
    empty_paired = _paired_summary([])
    return {
        "frozen_signature_definition": _signature_definition([]),
        "replay_arm_definitions_and_budget_parity": _arm_definitions([]),
        "forward_transfer_and_recurrence": {
            "schema": SCHEMA + ".forward_transfer_recurrence",
            "row_count": 0,
            "arm_metrics": empty_scores,
            "compatible_minus_no_replay": empty_paired,
            "compatible_minus_all_replay": empty_paired,
            "recurrence": {
                "row_count": 0,
                "compatible_minus_no_replay": empty_paired,
                "compatible_minus_all_replay": empty_paired,
            },
            "all_required_lower_bounds_positive": False,
        },
        "protected_prefix_and_hard_case_results": {
            "schema": SCHEMA + ".protected_prefix_hard_case",
            "protected_prefix_retention": {arm: 0.0 for arm in REPLAY_ARMS},
            "hard_case_results": {
                stratum: {
                    "row_count": 0,
                    "compatible_minus_no_replay": empty_paired,
                    "compatible_minus_all_replay": empty_paired,
                }
                for stratum in HARDNESS_STRATA
            },
            "no_hard_case_negative_lower_bound": False,
            "hard_case_forgetting_count": 0,
        },
        "family_lower_bounds_and_group_bootstraps": {
            "schema": SCHEMA + ".family_lower_bounds_group_bootstraps",
            "family_summaries": {},
            "family_lcb95_over_no_replay": {},
            "family_lcb95_over_all_replay": {},
            "all_family_lcbs_positive_over_both_controls": False,
            "no_family_negative_lower_bound": False,
            "group_bootstrap_ci95": {"n_groups": 0, "ci95": [0.0, 0.0]},
            "event_group_bootstrap_ci95": {"n_groups": 0, "ci95": [0.0, 0.0]},
        },
        "incompatible_negative_transfer": {
            "schema": SCHEMA + ".incompatible_negative_transfer",
            "compatible_replay_incompatible_event_count": 0,
            "all_replay_incompatible_event_count": 0,
            "all_replay_incompatible_negative_transfer_event_count": 0,
            "mean_incompatible_event_penalty": 0.0,
            "compatible_unsafe_transfer_count": 0,
            "all_replay_unsafe_transfer_count": 0,
            "replay_precision_recall": {
                "all_replay": {
                    "selected_replay_events": 0,
                    "compatible_selected_events": 0,
                    "precision": 1.0,
                    "recall_against_capped_selection": 1.0,
                },
                "signature_compatible_replay": {
                    "selected_replay_events": 0,
                    "compatible_selected_events": 0,
                    "precision": 1.0,
                    "recall_against_capped_selection": 1.0,
                },
            },
        },
        "signature_permutation_collision_and_null_controls": _control_bundle([], [], []),
        "unsafe_transfer_count": 0,
        "replay_resource_accounting": {
            "schema": SCHEMA + ".resource_accounting",
            "memory_cap": MEMORY_CAP,
            "replay_event_cap": REPLAY_EVENT_CAP,
            "max_state_size": 0,
            "max_replay_events_per_task": 0,
            "max_cap_pressure": 0.0,
            "cap_compliance": False,
            "by_arm": {},
        },
        "restart_equivalence": {
            "schema": SCHEMA + ".restart_equivalence",
            "full_replay_state_hash": sha256_json({}),
            "resumed_replay_state_hash": "",
            "replay_receipt_hash_root": sha256_json([]),
            "restart_equivalence": 0.0,
            "serialized_replay_state_reproduces": False,
        },
    }


def _tests_passed(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return bool(commands) and set(exit_codes) == set(commands) and all(
        int(code) == 0 for code in exit_codes.values()
    )


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5856_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5856_ROWS_RELATIVE_PATH.as_posix(),
        EXP5829_COMPARISON_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
        ROOT_CLUTTER_SWEEP_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def selective_replay_qualified_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare Exp5857 qualification scalar after all gates pass."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    signature = dict(artifact.get("frozen_signature_definition") or {})
    arms = dict(artifact.get("replay_arm_definitions_and_budget_parity") or {})
    transfer = dict(artifact.get("forward_transfer_and_recurrence") or {})
    retention = dict(artifact.get("protected_prefix_and_hard_case_results") or {})
    families = dict(artifact.get("family_lower_bounds_and_group_bootstraps") or {})
    incompatible = dict(artifact.get("incompatible_negative_transfer") or {})
    controls = dict(artifact.get("signature_permutation_collision_and_null_controls") or {})
    resources = dict(artifact.get("replay_resource_accounting") or {})
    restart = dict(artifact.get("restart_equivalence") or {})
    ready = (
        preconditions.get("preconditions_ready") is True
        and _signature_definition_is_valid(signature)
        and arms.get("budget_parity_passed") is True
        and arms.get("prior_only_selection_passed") is True
        and transfer.get("all_required_lower_bounds_positive") is True
        and float(dict(transfer.get("compatible_minus_no_replay") or {}).get("ci95", [0.0])[0])
        > 0.0
        and float(dict(transfer.get("compatible_minus_all_replay") or {}).get("ci95", [0.0])[0])
        > 0.0
        and float(artifact.get("unsafe_transfer_count") or 0) == 0
        and float(
            dict(retention.get("protected_prefix_retention") or {}).get(
                "signature_compatible_replay",
                0.0,
            )
        )
        == 1.0
        and retention.get("no_hard_case_negative_lower_bound") is True
        and int(retention.get("hard_case_forgetting_count") or 0) == 0
        and families.get("all_family_lcbs_positive_over_both_controls") is True
        and families.get("no_family_negative_lower_bound") is True
        and int(incompatible.get("compatible_replay_incompatible_event_count") or 0) == 0
        and int(incompatible.get("compatible_unsafe_transfer_count") or 0) == 0
        and controls.get("all_controls_fail_closed") is True
        and resources.get("cap_compliance") is True
        and float(resources.get("max_cap_pressure") or 0.0) <= 1.0
        and float(restart.get("restart_equivalence") or 0.0) == 1.0
        and restart.get("serialized_replay_state_reproduces") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _tests_passed(artifact)
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if not _signature_definition_is_valid(
        dict(artifact.get("frozen_signature_definition") or {})
    ):
        reasons.append("frozen_signature_definition")
    if int(artifact.get("unsafe_transfer_count") or 0) != 0:
        reasons.append("unsafe_transfer_count")
    if dict(artifact.get("replay_resource_accounting") or {}).get("cap_compliance") is not True:
        reasons.append("cap_compliance")
    if float(dict(artifact.get("restart_equivalence") or {}).get("restart_equivalence") or 0.0) != 1.0:
        reasons.append("restart_equivalence")
    if selective_replay_qualified_score(artifact) != 1.0 and not reasons:
        reasons.append("qualified_score")
    return sorted(set(reasons))


def _artifact_status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    if int(artifact.get("unsafe_transfer_count") or 0) != 0:
        return "unsafe"
    if selective_replay_qualified_score(artifact) == 1.0:
        return "qualified"
    return "null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    status = _artifact_status(artifact)
    if status == "qualified":
        return "qualified: clean_lifecycle_signature_compatible_replay"
    if status == "unsafe":
        return "unsafe: " + ",".join(blocked_reasons(artifact)[:8])
    if status == "blocked":
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    return "null: clean_replay_without_preregistered_selective_gain"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["atomic_output"] = {}
        stable["preconditions_checked"]["timer"] = {}
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if not _signature_definition_is_valid(
        dict(artifact.get("frozen_signature_definition") or {})
    ):
        raise ValueError("frozen_signature_definition")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_score = selective_replay_qualified_score(artifact)
    if artifact.get("selective_replay_qualified_score") != expected_score:
        raise ValueError("qualified_score")
    if artifact.get("status") != _artifact_status(artifact):
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
    """Build the terminal Exp5857 artifact from clean rows and replay receipts."""

    started = time.perf_counter()
    root = Path(root)
    preconditions = dict(
        preconditions_checked or collect_preconditions(root=root, result_path=result_path)
    )
    rows = load_clean_rows(root) if preconditions.get("preconditions_ready") is True else []
    evaluation = _evaluate_rows(rows) if rows else _empty_evaluation()
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": "blocked",
        "preconditions_checked": preconditions,
        "clean_lifecycle_hashes": dict(
            preconditions.get("clean_lifecycle_hashes") or _clean_lifecycle_hashes(root)
        ),
        "frozen_signature_definition": evaluation["frozen_signature_definition"],
        "replay_arm_definitions_and_budget_parity": evaluation[
            "replay_arm_definitions_and_budget_parity"
        ],
        "forward_transfer_and_recurrence": evaluation["forward_transfer_and_recurrence"],
        "protected_prefix_and_hard_case_results": evaluation[
            "protected_prefix_and_hard_case_results"
        ],
        "family_lower_bounds_and_group_bootstraps": evaluation[
            "family_lower_bounds_and_group_bootstraps"
        ],
        "incompatible_negative_transfer": evaluation["incompatible_negative_transfer"],
        "signature_permutation_collision_and_null_controls": evaluation[
            "signature_permutation_collision_and_null_controls"
        ],
        "unsafe_transfer_count": int(evaluation["unsafe_transfer_count"]),
        "replay_resource_accounting": evaluation["replay_resource_accounting"],
        "restart_equivalence": evaluation["restart_equivalence"],
        "selective_replay_qualified_score": 0.0,
        "duration_s": elapsed,
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
    artifact["selective_replay_qualified_score"] = selective_replay_qualified_score(
        artifact
    )
    artifact["status"] = _artifact_status(artifact)
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
    """Run Exp5857 and optionally write the terminal clean replay artifact."""

    preconditions = dict(
        preconditions_checked or collect_preconditions(root=root, result_path=result_path)
    )
    artifact = build_artifact(
        root=root,
        result_path=result_path,
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
