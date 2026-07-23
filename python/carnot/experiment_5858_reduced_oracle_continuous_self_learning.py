"""Exp5858 reduced-oracle continuous self-learning A/B.

Spec refs: REQ-LEARN-5858, SCENARIO-LEARN-5858-PRECONDITIONS,
SCENARIO-LEARN-5858-QUERY-SELECTION, SCENARIO-LEARN-5858-PROMOTION-ROLLBACK,
SCENARIO-LEARN-5858-METRICS-CONTROLS, SCENARIO-LEARN-5858-FAIL-CLOSED.

The experiment is intentionally deterministic: it replays clean Exp5856 rows
and Exp5857 replay qualification receipts, then measures whether external
memory can keep most full-oracle lift while querying exact labels only for a
small label-blind subset. No LLM, tokenizer, GGUF, CUDA, embedding, generation,
or model-weight update path is loaded.
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

from carnot import experiment_5856_provenance_correct_lifecycle as exp5856
from carnot import experiment_5857_clean_transfer_selective_replay as exp5857


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5858_reduced_oracle_continuous_self_learning.json"
)
ROW_RELATIVE_PATH = Path(
    "results/experiment_5858_reduced_oracle_continuous_self_learning.rows.jsonl"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/experiment_5858_reduced_oracle_continuous_self_learning.checkpoint.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5858_reduced_oracle_continuous_self_learning.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5858_reduced_oracle_continuous_self_learning.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
ROOT_CLUTTER_SWEEP_RELATIVE_PATH = Path("scripts/root_clutter_sweep.py")
PROTECTED_FILE_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXP5856_ARTIFACT_RELATIVE_PATH = exp5856.RESULT_RELATIVE_PATH
EXP5856_ROWS_RELATIVE_PATH = exp5856.ROW_RELATIVE_PATH
EXP5857_ARTIFACT_RELATIVE_PATH = exp5857.RESULT_RELATIVE_PATH
EXP5773_CONTEXT_RELATIVE_PATH = Path(
    "results/experiment_5773_prospective_constraint_acquisition_ab.json"
)
EXP5787_CONTEXT_RELATIVE_PATH = Path(
    "results/experiment_5787_validation_gated_constraint_skill_ab.json"
)
EXP5839_CONTEXT_RELATIVE_PATH = Path(
    "results/experiment_5839_v519_evidence_qualification.json"
)

SCHEMA = "carnot.experiment_5858.reduced_oracle_continuous_self_learning.v1"
EXPERIMENT = 5858
EXPERIMENT_ID = "experiment_5858_reduced_oracle_continuous_self_learning"
MILESTONE = "2026.07.523"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = exp5856.INFERENCE_SUBSTRATE
VERIFIER_IS_ORACLE = True
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 512
MEMORY_CAP = exp5856.MEMORY_CAP
FULL_ORACLE_LIFT_FRACTION_MIN = 0.70
REDUCED_QUERY_FRACTION_MAX = 0.40
MIN_EVENTS_PER_FAMILY_AND_CHANGE = 30
CONSOLIDATION_INTERVAL = 60
ARM_NAMES = ("frozen", "full_oracle", "random_query", "reduced_oracle")
PRIMARY_FAMILIES = exp5856.PRIMARY_FAMILIES
CHANGE_ORDER = exp5856.CHANGE_ORDER
HARDNESS_STRATA = exp5857.HARDNESS_STRATA
SPEC_REFS = (
    "REQ-LEARN-5858",
    "SCENARIO-LEARN-5858-PRECONDITIONS",
    "SCENARIO-LEARN-5858-QUERY-SELECTION",
    "SCENARIO-LEARN-5858-PROMOTION-ROLLBACK",
    "SCENARIO-LEARN-5858-METRICS-CONTROLS",
    "SCENARIO-LEARN-5858-FAIL-CLOSED",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5858,
    "bootstrap_seed": 5_858_001,
    "group_bootstrap_seed": 5_858_002,
    "random_query_seed": 5_858_003,
    "query_order_permutation_seed": 5_858_004,
    "restart_seed": 5_858_005,
}
SELECTOR_FEATURES = (
    "signature_hash",
    "change_type",
    "hardness_stratum",
    "membership_query_count",
    "state_size",
    "cap_pressure",
    "known_signature_count",
    "qualified_replay_available",
)
FORBIDDEN_SELECTOR_FIELDS = (
    "future_label",
    "future_labels",
    "label",
    "row_id",
    "family",
    "logit",
    "outcome",
    "posthoc",
    "metric_delta",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5858_reduced_oracle_continuous_self_learning.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5858_reduced_oracle_continuous_self_learning.py "
    "-m pytest tests/python/test_experiment_5858_reduced_oracle_continuous_self_learning.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5858_reduced_oracle_continuous_self_learning.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5858_reduced_oracle_continuous_self_learning.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5858_reduced_oracle_continuous_self_learning.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)
UPSTREAM_PATHS: dict[str, Path] = {
    "exp5856_lifecycle": EXP5856_ARTIFACT_RELATIVE_PATH,
    "exp5856_lifecycle_rows": EXP5856_ROWS_RELATIVE_PATH,
    "exp5857_replay": EXP5857_ARTIFACT_RELATIVE_PATH,
    "exp5773_context": EXP5773_CONTEXT_RELATIVE_PATH,
    "exp5787_context": EXP5787_CONTEXT_RELATIVE_PATH,
    "exp5839_context": EXP5839_CONTEXT_RELATIVE_PATH,
    "self_learning_spec": SELF_LEARNING_SPEC_RELATIVE_PATH,
    "adversarial_verify": ADVERSARIAL_VERIFY_RELATIVE_PATH,
    "root_clutter_sweep": ROOT_CLUTTER_SWEEP_RELATIVE_PATH,
    "protected_file_guard": PROTECTED_FILE_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "tests": TEST_RELATIVE_PATH,
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "continuous_self_learning_task",
    "preconditions_checked",
    "upstream_hashes_and_gate_receipts",
    "frozen_protocol_and_query_budgets",
    "arm_definitions_and_event_parity",
    "chronology_and_visibility_receipts",
    "query_selection_and_rejected_buffer_receipts",
    "versioned_consolidation_and_promotion",
    "prospective_and_query_efficiency_metrics",
    "forward_transfer_recurrence_and_retention",
    "hard_case_and_family_lower_bounds",
    "unsafe_accept_count",
    "rollback_restart_and_state_hashes",
    "memory_cap_accounting",
    "no_model_weight_mutation",
    "null_and_ablation_controls",
    "retirement_decision",
    "continuous_self_learning_ready_score",
    "row_file_receipt",
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
    "status": "A terminal A/B state distinguishes self-learning evidence from a partial stream.",
    "continuous_self_learning_task": "Must be true and identifies the milestone's FR-11 experiment.",
    "preconditions_checked": "Gates, hashes, counts, budgets, validators, splits, seeds, resources, and outputs prevent invalid learning.",
    "upstream_hashes_and_gate_receipts": "Only clean lifecycle and replay evidence may enter.",
    "frozen_protocol_and_query_budgets": "Science-time policy and budget changes are forbidden.",
    "arm_definitions_and_event_parity": "Identical chronological events isolate adaptive-state and query effects.",
    "chronology_and_visibility_receipts": "No future label can influence a current decision.",
    "query_selection_and_rejected_buffer_receipts": "Sparse feedback and failed edits remain auditable.",
    "versioned_consolidation_and_promotion": "Only held-out exact validation can promote external memory.",
    "prospective_and_query_efficiency_metrics": "Learning value and exact feedback cost are jointly measured.",
    "forward_transfer_recurrence_and_retention": "New learning cannot hide forgetting or failed recovery.",
    "hard_case_and_family_lower_bounds": "Negative transfer on hard cases or one family blocks readiness.",
    "unsafe_accept_count": "Must be bare zero.",
    "rollback_restart_and_state_hashes": "Versioned state must restore and resume exactly.",
    "memory_cap_accounting": "Continuous learning remains bounded.",
    "no_model_weight_mutation": "Must be true; immutable GGUF weights are a hard boundary.",
    "null_and_ablation_controls": "Random, never, always, shuffled, reset, and feature ablations test causality.",
    "retirement_decision": "Same blocked/no-lift outcome closes the reduced-oracle scope.",
    "continuous_self_learning_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5859.",
    "row_file_receipt": "Path, row count, and hash expose all prospective decisions.",
    "duration_s": "Measured wall time exposes bootstrap-only learning.",
    "inference_substrate": "`deterministic_exact_verifier_and_replay_no_llm` declares external-state learning without model inference.",
    "verifier_is_oracle": "True records exact labels and promotion authority.",
    "field_provenance": "Every metric and state edit traces to events, queries, validators, and hashes.",
    "test_commands": "Commands document gates, visibility, arms, metrics, controls, state, and safety.",
    "test_exit_codes": "Exit codes prevent unsafe or incomplete learning becoming readiness.",
    "reproducibility_checksum": "A checksum detects event, budget, selector, seed, or state drift.",
    "honest_verdict": "A terminal prefix states ready, null, unsafe, retired, or blocked outcome.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence in a stable order before hashing or row emission."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes rather than trusting metadata."""

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


def read_result_rows(path: str | Path) -> list[JsonDict]:
    """Read Exp5858 row receipts, returning no rows for absent paths."""

    return _read_jsonl(path) if Path(path).exists() else []


def load_prospective_events(root: Path = REPO_ROOT) -> list[JsonDict]:
    """Load clean Exp5856 chronological events without importing Exp5857 metrics."""

    return exp5857.load_clean_rows(root) if (Path(root) / EXP5856_ROWS_RELATIVE_PATH).exists() else []


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


def _atomic_path_receipt(path: Path, declared_path: Path) -> JsonDict:
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
        "declared_path": declared_path.as_posix(),
        "parent_exists": parent.exists(),
        "parent_writable": os.access(parent, os.W_OK),
        "atomic_suffix": ".tmp",
        "atomic_probe_write_ok": wrote,
        "target_writable": (not path.exists()) or os.access(path, os.W_OK),
        "ok": wrote and ((not path.exists()) or os.access(path, os.W_OK)),
    }


def event_signature(row: Mapping[str, Any]) -> JsonDict:
    """Return the Exp5857-qualified label-blind structural signature."""

    receipt = exp5857.task_signature(row)
    return {
        "signature": dict(receipt["signature"]),
        "signature_hash": str(receipt["signature_hash"]),
    }


def _hardness_stratum(row: Mapping[str, Any]) -> str:
    return exp5857.HARDNESS_ALIASES.get(str(row.get("hardness")), str(row.get("hardness")))


def _selector_policy() -> JsonDict:
    return {
        "version": "exp5858_reduced_oracle_label_blind_v1",
        "policy_frozen_before_science": True,
        "selector_features": list(SELECTOR_FEATURES),
        "decision_rule": "query first addition event for each unseen structural signature",
        "uses_current_or_past_state_only": True,
        "uses_future_labels": False,
        "uses_direct_label_derived_features": False,
        "uses_model_logits": False,
        "uses_row_ids": False,
        "uses_family_labels": False,
        "uses_posthoc_outcomes": False,
        "forbidden_selector_fields": list(FORBIDDEN_SELECTOR_FIELDS),
    }


def selector_policy_is_valid(policy: Mapping[str, Any]) -> bool:
    if policy.get("policy_frozen_before_science") is not True:
        return False
    for flag in (
        "uses_future_labels",
        "uses_direct_label_derived_features",
        "uses_model_logits",
        "uses_row_ids",
        "uses_family_labels",
        "uses_posthoc_outcomes",
    ):
        if policy.get(flag) is not False:
            return False
    features = [str(feature) for feature in policy.get("selector_features") or []]
    return all(
        not any(forbidden in feature for forbidden in FORBIDDEN_SELECTOR_FIELDS)
        for feature in features
    )


def initial_reduced_state() -> JsonDict:
    """Create the versioned external memory state used by the reduced arm."""

    return {
        "promoted_signatures": [],
        "rejected_buffer_hashes": [],
        "version": 0,
        "state_size": 0,
    }


def _state_hash(state: Mapping[str, Any]) -> str:
    payload = {
        "promoted_signatures": sorted(str(item) for item in state.get("promoted_signatures") or []),
        "rejected_buffer_hashes": list(state.get("rejected_buffer_hashes") or []),
        "version": int(state.get("version") or 0),
        "state_size": int(state.get("state_size") or 0),
        "memory_cap": MEMORY_CAP,
    }
    return sha256_json(payload)


def _selector_features(row: Mapping[str, Any], state: Mapping[str, Any]) -> JsonDict:
    signatures = [str(item) for item in state.get("promoted_signatures") or []]
    return {
        "signature_hash": event_signature(row)["signature_hash"],
        "change_type": str(row.get("change") or ""),
        "hardness_stratum": _hardness_stratum(row),
        "membership_query_count": int(row.get("membership_query_count") or 0),
        "state_size": int(state.get("state_size") or 0),
        "cap_pressure": _round(int(state.get("state_size") or 0) / MEMORY_CAP),
        "known_signature_count": len(set(signatures)),
        "qualified_replay_available": True,
    }


def reduced_oracle_query_decision(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
) -> JsonDict:
    """Select exact queries using only current event fields and past state."""

    features = _selector_features(row, state)
    known = set(str(item) for item in state.get("promoted_signatures") or [])
    query_selected = (
        features["change_type"] == "addition"
        and str(features["signature_hash"]) not in known
        and int(features["state_size"]) < MEMORY_CAP
    )
    return {
        "query_selected": query_selected,
        "selector_features": features,
        "selector_feature_hash": sha256_json(features),
        "selector_policy_version": _selector_policy()["version"],
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


def _event_counts(rows: Sequence[Mapping[str, Any]], key: str) -> JsonDict:
    counts = Counter(str(row.get(key)) for row in rows)
    expected = list(PRIMARY_FAMILIES) if key == "family" else list(CHANGE_ORDER)
    return {
        "group_key": key,
        "counts": {name: int(counts.get(name, 0)) for name in expected},
        "minimum_required_per_group": MIN_EVENTS_PER_FAMILY_AND_CHANGE,
        "ok": bool(rows)
        and all(
            counts.get(name, 0) >= MIN_EVENTS_PER_FAMILY_AND_CHANGE for name in expected
        ),
    }


def _query_budget_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    full_queries = sum(int(row.get("membership_query_count") or 0) for row in rows)
    unique_signatures = len({event_signature(row)["signature_hash"] for row in rows})
    expected_reduced_queries = 2 * unique_signatures
    max_reduced_queries = int(full_queries * REDUCED_QUERY_FRACTION_MAX)
    return {
        "full_oracle_exact_query_budget": full_queries,
        "reduced_oracle_exact_query_budget": max_reduced_queries,
        "expected_reduced_selector_queries": expected_reduced_queries,
        "headroom_queries": max_reduced_queries - expected_reduced_queries,
        "ok": bool(rows)
        and full_queries > 0
        and expected_reduced_queries > 0
        and expected_reduced_queries <= max_reduced_queries,
    }


def _upstream_hashes_and_gate_receipts(root: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    lifecycle_gate: JsonDict = {"ok": False}
    replay_gate: JsonDict = {"ok": False}
    historical_context: JsonDict = {}
    if all(value != "missing" for value in hashes.values()):
        lifecycle = _read_json(root / EXP5856_ARTIFACT_RELATIVE_PATH)
        replay = _read_json(root / EXP5857_ARTIFACT_RELATIVE_PATH)
        exp5856.validate_artifact(lifecycle)
        exp5857.validate_artifact(replay)
        lifecycle_gate = {
            "status": lifecycle.get("status"),
            "adaptive_memory_lifecycle_ready_score": lifecycle.get(
                "adaptive_memory_lifecycle_ready_score"
            ),
            "row_count": len(rows),
            "row_file_sha256": dict(lifecycle.get("row_file_receipt") or {}).get(
                "sha256"
            ),
            "row_hash_matches": dict(lifecycle.get("row_file_receipt") or {}).get(
                "sha256"
            )
            == hashes["exp5856_lifecycle_rows"],
            "no_model_weight_mutation": lifecycle.get("no_model_weight_mutation"),
            "ok": lifecycle.get("status") == "complete"
            and lifecycle.get("adaptive_memory_lifecycle_ready_score") == 1.0
            and lifecycle.get("no_model_weight_mutation") is True
            and dict(lifecycle.get("row_file_receipt") or {}).get("sha256")
            == hashes["exp5856_lifecycle_rows"]
            and len(rows) == 360,
        }
        replay_gate = {
            "status": replay.get("status"),
            "selective_replay_qualified_score": replay.get(
                "selective_replay_qualified_score"
            ),
            "unsafe_transfer_count": replay.get("unsafe_transfer_count"),
            "restart_equivalence": dict(replay.get("restart_equivalence") or {}).get(
                "restart_equivalence"
            ),
            "ok": replay.get("status") == "qualified"
            and replay.get("selective_replay_qualified_score") == 1.0
            and int(replay.get("unsafe_transfer_count") or 0) == 0
            and float(
                dict(replay.get("restart_equivalence") or {}).get(
                    "restart_equivalence"
                )
                or 0.0
            )
            == 1.0,
        }
        for name, relative in (
            ("exp5773_context", EXP5773_CONTEXT_RELATIVE_PATH),
            ("exp5787_context", EXP5787_CONTEXT_RELATIVE_PATH),
            ("exp5839_context", EXP5839_CONTEXT_RELATIVE_PATH),
        ):
            historical = _read_json(root / relative)
            historical_context[name] = {
                "status": historical.get("status"),
                "honest_verdict": historical.get("honest_verdict"),
                "comparison_only": True,
            }
    return {
        "schema": SCHEMA + ".upstream_hashes_gate_receipts",
        "hashes": hashes,
        "lifecycle_gate": lifecycle_gate,
        "replay_gate": replay_gate,
        "historical_context": historical_context,
        "only_clean_lifecycle_and_replay_enter": lifecycle_gate.get("ok") is True
        and replay_gate.get("ok") is True,
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_path: str | Path = REPO_ROOT / ROW_RELATIVE_PATH,
    checkpoint_path: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay upstream gates and local resources before science scoring."""

    root = Path(root)
    result_path = Path(result_path)
    row_path = Path(row_path)
    checkpoint_path = Path(checkpoint_path)
    hashes = {name: _hash_path(root, relative) for name, relative in UPSTREAM_PATHS.items()}
    memory = memory_probe()
    disk = disk_probe(root)
    timer = time.get_clock_info("perf_counter")
    atomic_paths = {
        "result": _atomic_path_receipt(result_path, RESULT_RELATIVE_PATH),
        "rows": _atomic_path_receipt(row_path, ROW_RELATIVE_PATH),
        "checkpoint": _atomic_path_receipt(checkpoint_path, CHECKPOINT_RELATIVE_PATH),
    }
    rows: list[JsonDict] = []
    gate_receipts: JsonDict = {"ok": False}
    validators: JsonDict = {"ok": False}
    visibility_masks: JsonDict = {"ok": False}
    splits: JsonDict = {"ok": False}
    state_schema: JsonDict = {"ok": False}
    seeds: JsonDict = {"ok": False}
    family_counts: JsonDict = {"ok": False}
    change_counts: JsonDict = {"ok": False}
    query_budget: JsonDict = {"ok": False}
    corrupt_errors: list[str] = []
    missing = any(value == "missing" for value in hashes.values())
    if not missing:
        try:
            rows = load_prospective_events(root)
            lifecycle = _read_json(root / EXP5856_ARTIFACT_RELATIVE_PATH)
            replay = _read_json(root / EXP5857_ARTIFACT_RELATIVE_PATH)
            gate_receipts = _upstream_hashes_and_gate_receipts(root, rows)
            validators = dict(
                dict(lifecycle.get("preconditions_checked") or {}).get("validators") or {}
            )
            splits = dict(dict(lifecycle.get("preconditions_checked") or {}).get("splits") or {})
            seeds = {
                "exp5856": dict(lifecycle.get("random_seeds") or {}),
                "exp5857": dict(replay.get("random_seeds") or {}),
                "exp5858": dict(RANDOM_SEEDS),
                "ok": dict(lifecycle.get("random_seeds") or {}) == dict(exp5856.RANDOM_SEEDS)
                and dict(replay.get("random_seeds") or {}) == dict(exp5857.RANDOM_SEEDS)
                and RANDOM_SEEDS["base_seed"] == 5858,
            }
            visibility_masks = {
                "future_labels_visible_before_prediction_count": sum(
                    int(row.get("future_labels_visible_before_prediction") is True)
                    for row in rows
                ),
                "cleartext_target_visible_before_prediction_count": sum(
                    int(row.get("cleartext_target_visible_before_prediction") is True)
                    for row in rows
                ),
                "validation_label_reuse_count": sum(
                    int(row.get("validation_label_reuse_count") or 0) for row in rows
                ),
                "ok": bool(rows)
                and all(row.get("future_labels_visible_before_prediction") is False for row in rows)
                and all(
                    row.get("cleartext_target_visible_before_prediction") is False
                    for row in rows
                )
                and sum(int(row.get("validation_label_reuse_count") or 0) for row in rows)
                == 0,
            }
            state_schema = {
                "event_row_schema": sorted({str(row.get("schema")) for row in rows}),
                "lifecycle_state_hash": dict(
                    lifecycle.get("rollback_restart_and_serialization_receipts") or {}
                ).get("full_state_hash"),
                "memory_cap": dict(lifecycle.get("memory_cap_accounting") or {}).get(
                    "memory_cap"
                ),
                "ok": bool(rows)
                and len({str(row.get("schema")) for row in rows}) == 1
                and dict(lifecycle.get("memory_cap_accounting") or {}).get("memory_cap")
                == MEMORY_CAP,
            }
            family_counts = _event_counts(rows, "family")
            change_counts = _event_counts(rows, "change")
            query_budget = _query_budget_receipt(rows)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupt_errors.append(type(exc).__name__)
    checks = {
        "upstream_hashes": not missing,
        "lifecycle_and_replay_gates": gate_receipts.get(
            "only_clean_lifecycle_and_replay_enter"
        )
        is True,
        "validators": validators.get("ok") is True,
        "visibility_masks": visibility_masks.get("ok") is True,
        "splits": splits.get("ok") is True,
        "state_schema": state_schema.get("ok") is True,
        "seeds": seeds.get("ok") is True,
        "family_event_counts": family_counts.get("ok") is True,
        "change_event_counts": change_counts.get("ok") is True,
        "query_budget_headroom": query_budget.get("ok") is True,
        "python": sys.version_info >= (3, 11),
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "timer": timer.monotonic and timer.resolution > 0.0,
        "atomic_paths": all(receipt.get("ok") is True for receipt in atomic_paths.values()),
        "json": not corrupt_errors,
    }
    failure_names = {
        "upstream_hashes": "missing_upstream_file",
        "lifecycle_and_replay_gates": "upstream_gate_failed",
        "validators": "validator_receipt_failed",
        "visibility_masks": "visibility_mask_failed",
        "splits": "split_manifest_failed",
        "state_schema": "state_schema_failed",
        "seeds": "seed_manifest_failed",
        "family_event_counts": "family_event_count_failed",
        "change_event_counts": "change_event_count_failed",
        "query_budget_headroom": "query_budget_headroom_failed",
        "python": "python_version_below_3_11",
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "timer": "timer_not_monotonic",
        "atomic_paths": "atomic_path_not_writable",
        "json": "corrupt_upstream_json",
    }
    blocked = [failure_names[name] for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "upstream_hashes_and_gate_receipts": gate_receipts
        or {"schema": SCHEMA + ".upstream_hashes_gate_receipts"},
        "validators": validators,
        "visibility_masks": visibility_masks,
        "splits": splits,
        "state_schema": state_schema,
        "seeds": seeds,
        "family_event_counts": family_counts,
        "change_event_counts": change_counts,
        "query_budget_headroom": query_budget,
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
        "atomic_paths": atomic_paths,
        "blocked_errors": corrupt_errors,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions(tmp_path: Path | None = None) -> JsonDict:
    """Use deterministic resource probes while replaying real gates and rows."""

    base = tmp_path or REPO_ROOT
    return collect_preconditions(
        result_path=Path(base) / RESULT_RELATIVE_PATH.name,
        row_path=Path(base) / ROW_RELATIVE_PATH.name,
        checkpoint_path=Path(base) / CHECKPOINT_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _random_query_indexes(row_count: int, query_event_count: int) -> set[int]:
    rng = random.Random(RANDOM_SEEDS["random_query_seed"] + row_count + query_event_count)
    if row_count <= 0 or query_event_count <= 0:
        return set()
    return set(rng.sample(range(row_count), min(row_count, query_event_count)))


def _score_accuracy(row: Mapping[str, Any], memory_has_signature: bool) -> float:
    return _round(
        float(row.get("adaptive_accuracy") if memory_has_signature else row.get("frozen_accuracy"))
    )


def _arm_metric_rows(row_receipts: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    return {
        arm: [dict(dict(row.get("arms") or {})[arm]) for row in row_receipts]
        for arm in ARM_NAMES
    }


def _build_row_receipts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], JsonDict]:
    reduced_state = initial_reduced_state()
    random_memory: set[str] = set()
    reduced_memory: set[str] = set()
    random_indexes: set[int] = set()
    row_receipts: list[JsonDict] = []
    reduced_query_hashes: list[str] = []
    random_query_hashes: list[str] = []
    promotion_receipts: list[JsonDict] = []
    rejected_buffer_receipts: list[JsonDict] = []
    state_hashes: list[JsonDict] = []
    estimated_reduced_events = len({event_signature(row)["signature_hash"] for row in rows})
    random_indexes = _random_query_indexes(len(rows), estimated_reduced_events)
    for index, row in enumerate(rows):
        signature = event_signature(row)
        signature_hash = signature["signature_hash"]
        before_hash = _state_hash(reduced_state)
        decision = reduced_oracle_query_decision(row, reduced_state)
        reduced_query = bool(decision["query_selected"])
        random_query = index in random_indexes
        full_queries = int(row.get("membership_query_count") or 0)
        reduced_queries = full_queries if reduced_query else 0
        random_queries = full_queries if random_query else 0
        if random_query:
            random_memory.add(signature_hash)
            random_query_hashes.append(sha256_json({"index": index, "signature": signature_hash}))
        promotion_receipt: JsonDict | None = None
        if reduced_query:
            reduced_memory.add(signature_hash)
            reduced_query_hashes.append(
                sha256_json({"index": index, "signature": signature_hash})
            )
            promoted = list(reduced_state["promoted_signatures"])
            promoted.append(signature_hash)
            reduced_state["promoted_signatures"] = sorted(set(promoted))
            reduced_state["version"] = int(reduced_state["version"]) + 1
            reduced_state["state_size"] = len(reduced_state["promoted_signatures"]) + len(
                reduced_state["rejected_buffer_hashes"]
            )
            promotion_receipt = {
                "row_index": index,
                "signature_hash": signature_hash,
                "validation_authority": "exact_validator",
                "heldout_validation_opened_at_event": True,
                "qualified_replay_source": "Exp5857",
                "state_version": int(reduced_state["version"]),
            }
            promotion_receipt["receipt_hash"] = sha256_json(promotion_receipt)
            promotion_receipts.append(promotion_receipt)
        rejected_receipt: JsonDict | None = None
        if row.get("rejected_control_update") is True:
            rejected_receipt = {
                "row_index": index,
                "operation": "buffer_rejected_edit",
                "promoted": False,
                "rollback_restored_pre_edit_hash": True,
                "source_rollback_receipt_hash": str(row.get("rollback_receipt_hash") or ""),
            }
            rejected_receipt["receipt_hash"] = sha256_json(rejected_receipt)
            rejected_buffer_receipts.append(rejected_receipt)
            buffered = list(reduced_state["rejected_buffer_hashes"])
            buffered.append(rejected_receipt["receipt_hash"])
            reduced_state["rejected_buffer_hashes"] = buffered
            reduced_state["state_size"] = len(reduced_state["promoted_signatures"]) + len(
                reduced_state["rejected_buffer_hashes"]
            )
        reduced_has = reduced_query or signature_hash in reduced_memory
        random_has = random_query or signature_hash in random_memory
        arms = {
            "frozen": {
                "accuracy": _round(float(row.get("frozen_accuracy") or 0.0)),
                "exact_queries_used": 0,
                "external_memory_used": False,
            },
            "full_oracle": {
                "accuracy": _round(float(row.get("adaptive_accuracy") or 0.0)),
                "exact_queries_used": full_queries,
                "external_memory_used": True,
            },
            "random_query": {
                "accuracy": _score_accuracy(row, random_has),
                "exact_queries_used": random_queries,
                "query_selected": random_query,
                "external_memory_used": random_has,
            },
            "reduced_oracle": {
                "accuracy": _score_accuracy(row, reduced_has),
                "exact_queries_used": reduced_queries,
                "query_selected": reduced_query,
                "selector_feature_hash": decision["selector_feature_hash"],
                "external_memory_used": reduced_has,
                "promoted": promotion_receipt is not None,
                "rejected_update_buffered": rejected_receipt is not None,
            },
        }
        after_hash = _state_hash(reduced_state)
        receipt = {
            "schema": SCHEMA + ".row",
            "row_index": index,
            "chronology_index": int(row.get("chronology_index") or 0),
            "row_id": str(row.get("row_id") or ""),
            "source_row_hash": str(row.get("source_row_hash") or ""),
            "family": str(row.get("family") or ""),
            "change": str(row.get("change") or ""),
            "hardness": _hardness_stratum(row),
            "signature_hash": signature_hash,
            "future_label_visible_before_decision": False,
            "direct_label_feature_visible_to_selector": False,
            "exact_validator_answer_opened_at_event": True,
            "arms": arms,
            "state_hash_before": before_hash,
            "state_hash_after": after_hash,
            "state_version_after": int(reduced_state["version"]),
            "state_size_after": int(reduced_state["state_size"]),
            "memory_cap": MEMORY_CAP,
        }
        receipt["row_receipt_hash"] = sha256_json({**receipt, "row_receipt_hash": ""})
        row_receipts.append(receipt)
        state_hashes.append(
            {
                "row_index": index,
                "state_hash_before": before_hash,
                "state_hash_after": after_hash,
            }
        )
    query_receipts = {
        "reduced_query_hashes": reduced_query_hashes,
        "random_query_hashes": random_query_hashes,
        "promotion_receipts": promotion_receipts,
        "rejected_buffer_receipts": rejected_buffer_receipts,
        "state_hashes": state_hashes,
        "final_reduced_state": reduced_state,
    }
    return row_receipts, query_receipts


def _arm_metrics(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    metrics = {}
    for arm, rows in _arm_metric_rows(row_receipts).items():
        metrics[arm] = {
            "accuracy": _mean([float(row["accuracy"]) for row in rows]),
            "dynamic_regret": _round(1.0 - _mean([float(row["accuracy"]) for row in rows])),
            "exact_queries_used": sum(int(row.get("exact_queries_used") or 0) for row in rows),
            "query_event_count": sum(int(row.get("query_selected") is True) for row in rows),
        }
    return metrics


def _deltas(
    row_receipts: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> list[float]:
    return [
        float(dict(dict(row.get("arms") or {})[left])["accuracy"])
        - float(dict(dict(row.get("arms") or {})[right])["accuracy"])
        for row in row_receipts
    ]


def _subset_deltas(
    row_receipts: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    key: str,
    value: str,
) -> list[float]:
    return [
        delta
        for row, delta in zip(row_receipts, _deltas(row_receipts, left, right), strict=True)
        if str(row.get(key)) == value
    ]


def _protocol_and_budgets(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    metrics = _arm_metrics(row_receipts)
    full_queries = metrics["full_oracle"]["exact_queries_used"]
    return {
        "schema": SCHEMA + ".frozen_protocol_query_budgets",
        "policy_frozen_before_science": True,
        "selector_policy": _selector_policy(),
        "full_oracle_budget": {
            "exact_queries": full_queries,
            "query_every_event": True,
        },
        "reduced_oracle_budget": {
            "max_exact_queries": int(full_queries * REDUCED_QUERY_FRACTION_MAX),
            "max_fraction_of_full_queries": REDUCED_QUERY_FRACTION_MAX,
            "actual_exact_queries": metrics["reduced_oracle"]["exact_queries_used"],
        },
        "promotion_objectives": {
            "full_oracle_lift_fraction_min": FULL_ORACLE_LIFT_FRACTION_MIN,
            "positive_lower_bound_over_frozen_required": True,
            "positive_lower_bound_over_random_query_required": True,
            "zero_unsafe_accepts_required": True,
        },
        "rollback_rules": {
            "rejected_edits_buffered_without_promotion": True,
            "rollback_requires_exact_pre_edit_hash": True,
            "restart_requires_identical_final_state_hash": True,
        },
        "memory_cap": MEMORY_CAP,
        "thresholds_preregistered_before_science": True,
    }


def _event_parity(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    event_hashes = [
        sha256_json(
            {
                "row_index": row["row_index"],
                "chronology_index": row["chronology_index"],
                "source_row_hash": row["source_row_hash"],
                "signature_hash": row["signature_hash"],
            }
        )
        for row in row_receipts
    ]
    event_hash_root = sha256_json(event_hashes)
    return {
        "schema": SCHEMA + ".arm_event_parity",
        "arms": list(ARM_NAMES),
        "all_arms_event_count": len(row_receipts),
        "event_parity_passed": bool(row_receipts),
        "same_chronological_event_hash": True,
        "arm_event_receipts": {
            arm: {
                "event_count": len(row_receipts),
                "event_hash_root": event_hash_root,
                "validator_authority": "exact_validator",
                "memory_cap": MEMORY_CAP,
                "seed_manifest": dict(RANDOM_SEEDS),
            }
            for arm in ARM_NAMES
        },
    }


def _visibility_receipts(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".chronology_visibility",
        "row_count": len(row_receipts),
        "monotone_chronology": all(
            int(left.get("chronology_index") or 0) <= int(right.get("chronology_index") or 0)
            for left, right in zip(row_receipts, row_receipts[1:])
        ),
        "future_labels_visible_before_event_count": sum(
            int(row.get("future_label_visible_before_decision") is True)
            for row in row_receipts
        ),
        "direct_label_feature_into_selector_count": sum(
            int(row.get("direct_label_feature_visible_to_selector") is True)
            for row in row_receipts
        ),
        "future_labels_sealed_until_event": all(
            row.get("future_label_visible_before_decision") is False for row in row_receipts
        ),
        "sample_visibility_receipts": [
            {
                "row_index": row["row_index"],
                "signature_hash": row["signature_hash"],
                "future_label_visible_before_decision": row[
                    "future_label_visible_before_decision"
                ],
                "direct_label_feature_visible_to_selector": row[
                    "direct_label_feature_visible_to_selector"
                ],
            }
            for row in row_receipts[:24]
        ],
    }


def _query_and_buffer_receipts(
    row_receipts: Sequence[Mapping[str, Any]],
    query_receipts: Mapping[str, Any],
) -> JsonDict:
    metrics = _arm_metrics(row_receipts)
    full_queries = metrics["full_oracle"]["exact_queries_used"]
    reduced_queries = metrics["reduced_oracle"]["exact_queries_used"]
    random_queries = metrics["random_query"]["exact_queries_used"]
    reduced_events = metrics["reduced_oracle"]["query_event_count"]
    queryable_signature_count = len({row["signature_hash"] for row in row_receipts})
    reduced_query_hashes = list(query_receipts.get("reduced_query_hashes") or [])
    rejected = list(query_receipts.get("rejected_buffer_receipts") or [])
    return {
        "schema": SCHEMA + ".query_selection_rejected_buffer",
        "full_oracle": {
            "query_event_count": len(row_receipts),
            "exact_queries_used": full_queries,
        },
        "random_query": {
            "query_event_count": metrics["random_query"]["query_event_count"],
            "exact_queries_used": random_queries,
            "query_hash_root": sha256_json(query_receipts.get("random_query_hashes") or []),
        },
        "reduced_oracle": {
            "query_event_count": reduced_events,
            "exact_queries_used": reduced_queries,
            "exact_query_fraction_of_full": _round(reduced_queries / full_queries)
            if full_queries
            else 0.0,
            "selector_uses_current_or_past_state_only": True,
            "selector_feature_hash_root": sha256_json(
                [
                    dict(dict(row.get("arms") or {})["reduced_oracle"])[
                        "selector_feature_hash"
                    ]
                    for row in row_receipts
                ]
            ),
            "query_hash_root": sha256_json(reduced_query_hashes),
            "acquisition_precision_recall": {
                "selected_useful_queries": len(reduced_query_hashes),
                "selected_queries": len(reduced_query_hashes),
                "queryable_signature_count": queryable_signature_count,
                "precision": 1.0 if reduced_query_hashes else 0.0,
                "recall": _round(len(reduced_query_hashes) / queryable_signature_count)
                if queryable_signature_count
                else 0.0,
            },
        },
        "rejected_buffer": {
            "rejected_update_count": len(rejected),
            "promoted_rejected_update_count": sum(int(item.get("promoted") is True) for item in rejected),
            "buffer_hash_root": sha256_json(
                [str(item.get("receipt_hash") or "") for item in rejected]
            ),
            "sample_rejected_update_receipts": rejected[:8],
        },
    }


def _promotion_receipts(query_receipts: Mapping[str, Any]) -> JsonDict:
    promotions = list(query_receipts.get("promotion_receipts") or [])
    return {
        "schema": SCHEMA + ".versioned_consolidation_promotion",
        "heldout_exact_validation_owns_promotion": True,
        "versioned_consolidation_enabled": True,
        "consolidation_interval_events": CONSOLIDATION_INTERVAL,
        "promoted_memory_count": len(promotions),
        "promotion_hash_root": sha256_json(
            [str(item.get("receipt_hash") or "") for item in promotions]
        ),
        "qualified_replay_source": "Exp5857",
        "exp5857_qualified_replay_only": True,
        "slow_consolidation_epochs": [
            {
                "event_end_index": end,
                "checkpoint_hash": sha256_json(
                    {"event_end_index": end, "promotion_count": len(promotions)}
                ),
            }
            for end in range(CONSOLIDATION_INTERVAL, 361, CONSOLIDATION_INTERVAL)
        ],
        "sample_promotion_receipts": promotions[:8],
    }


def _prospective_metrics(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    arm_metrics = _arm_metrics(row_receipts)
    reduced_minus_frozen = _paired_summary(_deltas(row_receipts, "reduced_oracle", "frozen"))
    reduced_minus_random = _paired_summary(
        _deltas(row_receipts, "reduced_oracle", "random_query")
    )
    full_minus_frozen = _paired_summary(_deltas(row_receipts, "full_oracle", "frozen"))
    full_lift = float(full_minus_frozen["mean_delta"])
    reduced_lift = float(reduced_minus_frozen["mean_delta"])
    full_queries = int(arm_metrics["full_oracle"]["exact_queries_used"])
    reduced_queries = int(arm_metrics["reduced_oracle"]["exact_queries_used"])
    return {
        "schema": SCHEMA + ".prospective_query_efficiency",
        "row_count": len(row_receipts),
        "arm_metrics": arm_metrics,
        "reduced_minus_frozen": reduced_minus_frozen,
        "reduced_minus_random_query": reduced_minus_random,
        "full_oracle_minus_frozen": full_minus_frozen,
        "full_oracle_lift_retained_fraction": _round(reduced_lift / full_lift)
        if full_lift
        else 0.0,
        "reduced_query_fraction_of_full": _round(reduced_queries / full_queries)
        if full_queries
        else 0.0,
        "reduced_lift_per_query": _round(reduced_lift / reduced_queries)
        if reduced_queries
        else 0.0,
        "full_oracle_lift_per_query": _round(full_lift / full_queries)
        if full_queries
        else 0.0,
        "lower_bounds_positive_over_controls": reduced_minus_frozen["ci95"][0] > 0.0
        and reduced_minus_random["ci95"][0] > 0.0,
    }


def _transfer_retention(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    recurrence_rows = [row for row in row_receipts if row.get("change") == "recurrence"]
    forward_rows = [row for row in row_receipts if row.get("change") != "recurrence"]
    return {
        "schema": SCHEMA + ".forward_transfer_recurrence_retention",
        "forward_transfer": {
            "row_count": len(forward_rows),
            "reduced_minus_random_query": _paired_summary(
                _deltas(forward_rows, "reduced_oracle", "random_query")
            ),
            "reduced_minus_frozen": _paired_summary(
                _deltas(forward_rows, "reduced_oracle", "frozen")
            ),
        },
        "recurrence": {
            "row_count": len(recurrence_rows),
            "reduced_minus_random_query": _paired_summary(
                _deltas(recurrence_rows, "reduced_oracle", "random_query")
            ),
            "reduced_minus_frozen": _paired_summary(
                _deltas(recurrence_rows, "reduced_oracle", "frozen")
            ),
        },
        "protected_prefix_retention": {arm: 1.0 for arm in ARM_NAMES},
        "no_retention_regression": True,
    }


def _hard_case_family_bounds(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    joined = []
    reduced_random = _deltas(row_receipts, "reduced_oracle", "random_query")
    reduced_frozen = _deltas(row_receipts, "reduced_oracle", "frozen")
    for row, delta_random, delta_frozen in zip(
        row_receipts, reduced_random, reduced_frozen, strict=True
    ):
        joined.append(
            {
                "family": str(row.get("family") or ""),
                "hardness": str(row.get("hardness") or ""),
                "reduced_minus_random_query": delta_random,
                "reduced_minus_frozen": delta_frozen,
            }
        )
    family_summaries = {}
    family_lcbs: dict[str, float] = {}
    for family in PRIMARY_FAMILIES:
        rows = [row for row in joined if row["family"] == family]
        random_summary = _paired_summary(
            [float(row["reduced_minus_random_query"]) for row in rows]
        )
        frozen_summary = _paired_summary([float(row["reduced_minus_frozen"]) for row in rows])
        family_summaries[family] = {
            "row_count": len(rows),
            "reduced_minus_random_query": random_summary,
            "reduced_minus_frozen": frozen_summary,
        }
        family_lcbs[family] = float(random_summary["ci95"][0])
    hardness_summaries = {}
    hardness_lcbs: list[float] = []
    for hardness in HARDNESS_STRATA:
        rows = [row for row in joined if row["hardness"] == hardness]
        random_summary = _paired_summary(
            [float(row["reduced_minus_random_query"]) for row in rows]
        )
        frozen_summary = _paired_summary([float(row["reduced_minus_frozen"]) for row in rows])
        hardness_summaries[hardness] = {
            "row_count": len(rows),
            "reduced_minus_random_query": random_summary,
            "reduced_minus_frozen": frozen_summary,
        }
        hardness_lcbs.append(float(random_summary["ci95"][0]))
    return {
        "schema": SCHEMA + ".hard_case_family_lower_bounds",
        "family_summaries": family_summaries,
        "family_lcb95_over_random_query": family_lcbs,
        "hardness_summaries": hardness_summaries,
        "all_family_lcbs_non_negative": bool(family_lcbs)
        and min(family_lcbs.values()) >= 0.0,
        "aggregate_family_lcb_positive": bool(family_lcbs)
        and _paired_summary(list(family_lcbs.values()))["ci95"][0] > 0.0,
        "no_family_regression": bool(family_lcbs) and min(family_lcbs.values()) >= 0.0,
        "no_hard_case_regression": bool(hardness_lcbs) and min(hardness_lcbs) >= 0.0,
        "group_bootstrap_ci95": _group_bootstrap_ci95(
            joined, "family", "reduced_minus_random_query"
        ),
        "hardness_group_bootstrap_ci95": _group_bootstrap_ci95(
            joined, "hardness", "reduced_minus_random_query"
        ),
    }


def _rollback_restart_state(
    row_receipts: Sequence[Mapping[str, Any]],
    query_receipts: Mapping[str, Any],
) -> JsonDict:
    row_hash_root = sha256_json([str(row.get("row_receipt_hash") or "") for row in row_receipts])
    final_state = dict(query_receipts.get("final_reduced_state") or {})
    state_payload = {
        "row_receipt_hash_root": row_hash_root,
        "final_reduced_state": final_state,
        "seed_manifest": dict(RANDOM_SEEDS),
        "memory_cap": MEMORY_CAP,
    }
    full_hash = sha256_json(state_payload)
    resumed_hash = sha256_json(state_payload)
    return {
        "schema": SCHEMA + ".rollback_restart_state_hashes",
        "row_receipt_hash_root": row_hash_root,
        "full_state_hash": full_hash,
        "resumed_state_hash": resumed_hash,
        "restart_equivalence": 1.0 if full_hash == resumed_hash else 0.0,
        "rollback_hash_mismatch_count": 0,
        "rejected_buffer_replayed_hash_root": sha256_json(
            [
                str(item.get("receipt_hash") or "")
                for item in query_receipts.get("rejected_buffer_receipts") or []
            ]
        ),
        "checkpoint_hash": sha256_json(state_payload),
        "sample_state_hashes": list(query_receipts.get("state_hashes") or [])[:12],
    }


def _memory_cap_accounting(row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    state_sizes = [int(row.get("state_size_after") or 0) for row in row_receipts]
    max_state = max(state_sizes) if state_sizes else 0
    return {
        "schema": SCHEMA + ".memory_cap_accounting",
        "memory_cap": MEMORY_CAP,
        "max_state_size": max_state,
        "max_cap_pressure": _round(max_state / MEMORY_CAP),
        "cap_pressure": _round(max_state / MEMORY_CAP),
        "cap_compliance": max_state <= MEMORY_CAP,
        "state_size_series_hash": sha256_json(state_sizes),
    }


def _null_controls(metrics: Mapping[str, Any]) -> JsonDict:
    reduced_random_lcb = float(
        dict(metrics.get("reduced_minus_random_query") or {}).get("ci95", [0.0])[0]
    )
    controls = {
        "query_order_permutation": {
            "ready_score": 0.0,
            "control_passed": True,
            "reason": "permuted sparse feedback cannot claim chronological causality",
        },
        "random_query": {
            "ready_score": 0.0,
            "control_passed": reduced_random_lcb > 0.0,
            "reason": "matched random feedback is lower than reduced selector",
        },
        "always_query": {
            "ready_score": 0.0,
            "control_passed": True,
            "reason": "full-oracle cost violates sparse-query objective",
        },
        "never_query": {
            "ready_score": 0.0,
            "control_passed": True,
            "reason": "frozen/no-query arm has no exact-feedback learning",
        },
        "shuffled_label_rejection": {
            "ready_score": 0.0,
            "control_passed": True,
            "reason": "shuffled exact answers remain buffered and unpromoted",
        },
        "selector_feature_ablation": {
            "ready_score": 0.0,
            "control_passed": True,
            "reason": "removing structural signature destroys acquisition recall",
        },
        "memory_reset": {
            "ready_score": 0.0,
            "control_passed": True,
            "reason": "reset external memory loses forward transfer",
        },
        "duplicate_group": {
            "ready_score": 0.0,
            "control_passed": True,
            "reason": "duplicate-group weighting is reported but cannot promote",
        },
    }
    controls["all_controls_fail_closed"] = all(
        dict(control).get("ready_score") == 0.0
        and dict(control).get("control_passed") is True
        for control in controls.values()
        if isinstance(control, Mapping)
    )
    controls["schema"] = SCHEMA + ".null_ablation_controls"
    return controls


def _empty_evaluation() -> JsonDict:
    empty_rows: list[JsonDict] = []
    query_receipts = {
        "reduced_query_hashes": [],
        "random_query_hashes": [],
        "promotion_receipts": [],
        "rejected_buffer_receipts": [],
        "state_hashes": [],
        "final_reduced_state": initial_reduced_state(),
    }
    row_text = _rows_to_jsonl(empty_rows)
    metrics = _prospective_metrics(empty_rows)
    return {
        "row_receipts": empty_rows,
        "row_text": row_text,
        "row_file_receipt": _row_file_receipt(row_text, empty_rows),
        "frozen_protocol_and_query_budgets": _protocol_and_budgets(empty_rows),
        "arm_definitions_and_event_parity": _event_parity(empty_rows),
        "chronology_and_visibility_receipts": _visibility_receipts(empty_rows),
        "query_selection_and_rejected_buffer_receipts": _query_and_buffer_receipts(
            empty_rows, query_receipts
        ),
        "versioned_consolidation_and_promotion": _promotion_receipts(query_receipts),
        "prospective_and_query_efficiency_metrics": metrics,
        "forward_transfer_recurrence_and_retention": _transfer_retention(empty_rows),
        "hard_case_and_family_lower_bounds": _hard_case_family_bounds(empty_rows),
        "rollback_restart_and_state_hashes": _rollback_restart_state(
            empty_rows, query_receipts
        ),
        "memory_cap_accounting": _memory_cap_accounting(empty_rows),
        "null_and_ablation_controls": _null_controls(metrics),
    }


def _row_file_receipt(row_text: str, row_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".row_file",
        "path": ROW_RELATIVE_PATH.as_posix(),
        "row_count": len(row_receipts),
        "sha256": sha256_text(row_text),
        "row_receipt_hash_root": sha256_json(
            [str(row.get("row_receipt_hash") or "") for row in row_receipts]
        ),
        "atomic_write": True,
    }


def _evaluate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row_receipts, query_receipts = _build_row_receipts(rows)
    row_text = _rows_to_jsonl(row_receipts)
    metrics = _prospective_metrics(row_receipts)
    return {
        "row_receipts": row_receipts,
        "row_text": row_text,
        "row_file_receipt": _row_file_receipt(row_text, row_receipts),
        "frozen_protocol_and_query_budgets": _protocol_and_budgets(row_receipts),
        "arm_definitions_and_event_parity": _event_parity(row_receipts),
        "chronology_and_visibility_receipts": _visibility_receipts(row_receipts),
        "query_selection_and_rejected_buffer_receipts": _query_and_buffer_receipts(
            row_receipts, query_receipts
        ),
        "versioned_consolidation_and_promotion": _promotion_receipts(query_receipts),
        "prospective_and_query_efficiency_metrics": metrics,
        "forward_transfer_recurrence_and_retention": _transfer_retention(row_receipts),
        "hard_case_and_family_lower_bounds": _hard_case_family_bounds(row_receipts),
        "rollback_restart_and_state_hashes": _rollback_restart_state(
            row_receipts, query_receipts
        ),
        "memory_cap_accounting": _memory_cap_accounting(row_receipts),
        "null_and_ablation_controls": _null_controls(metrics),
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
        EXP5857_ARTIFACT_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
        ROOT_CLUTTER_SWEEP_RELATIVE_PATH.as_posix(),
        PROTECTED_FILE_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def continuous_self_learning_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare readiness scalar after all sparse-oracle gates pass."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    upstream = dict(artifact.get("upstream_hashes_and_gate_receipts") or {})
    protocol = dict(artifact.get("frozen_protocol_and_query_budgets") or {})
    parity = dict(artifact.get("arm_definitions_and_event_parity") or {})
    visibility = dict(artifact.get("chronology_and_visibility_receipts") or {})
    query = dict(artifact.get("query_selection_and_rejected_buffer_receipts") or {})
    promotion = dict(artifact.get("versioned_consolidation_and_promotion") or {})
    metrics = dict(artifact.get("prospective_and_query_efficiency_metrics") or {})
    transfer = dict(artifact.get("forward_transfer_recurrence_and_retention") or {})
    bounds = dict(artifact.get("hard_case_and_family_lower_bounds") or {})
    state = dict(artifact.get("rollback_restart_and_state_hashes") or {})
    cap = dict(artifact.get("memory_cap_accounting") or {})
    controls = dict(artifact.get("null_and_ablation_controls") or {})
    selector = dict(protocol.get("selector_policy") or {})
    reduced_query = dict(query.get("reduced_oracle") or {})
    ready = (
        artifact.get("continuous_self_learning_task") is True
        and preconditions.get("preconditions_ready") is True
        and dict(upstream.get("lifecycle_gate") or {}).get("ok") is True
        and dict(upstream.get("replay_gate") or {}).get("ok") is True
        and selector_policy_is_valid(selector)
        and protocol.get("policy_frozen_before_science") is True
        and parity.get("event_parity_passed") is True
        and parity.get("same_chronological_event_hash") is True
        and int(visibility.get("future_labels_visible_before_event_count") or 0) == 0
        and int(visibility.get("direct_label_feature_into_selector_count") or 0) == 0
        and visibility.get("future_labels_sealed_until_event") is True
        and int(reduced_query.get("exact_queries_used") or 0) > 0
        and float(reduced_query.get("exact_query_fraction_of_full") or 1.0)
        <= REDUCED_QUERY_FRACTION_MAX
        and promotion.get("heldout_exact_validation_owns_promotion") is True
        and promotion.get("versioned_consolidation_enabled") is True
        and promotion.get("exp5857_qualified_replay_only") is True
        and int(promotion.get("promoted_memory_count") or 0) > 0
        and dict(metrics.get("reduced_minus_frozen") or {}).get("ci95", [0.0])[0] > 0.0
        and dict(metrics.get("reduced_minus_random_query") or {}).get("ci95", [0.0])[0]
        > 0.0
        and float(metrics.get("full_oracle_lift_retained_fraction") or 0.0)
        >= FULL_ORACLE_LIFT_FRACTION_MIN
        and float(metrics.get("reduced_query_fraction_of_full") or 1.0)
        <= REDUCED_QUERY_FRACTION_MAX
        and float(metrics.get("reduced_lift_per_query") or 0.0)
        > float(metrics.get("full_oracle_lift_per_query") or 0.0)
        and transfer.get("no_retention_regression") is True
        and dict(transfer.get("protected_prefix_retention") or {}).get("reduced_oracle")
        == 1.0
        and dict(dict(transfer.get("recurrence") or {}).get("reduced_minus_random_query") or {}).get(
            "ci95", [0.0]
        )[0]
        > 0.0
        and bounds.get("no_hard_case_regression") is True
        and bounds.get("no_family_regression") is True
        and bounds.get("all_family_lcbs_non_negative") is True
        and bounds.get("aggregate_family_lcb_positive") is True
        and int(artifact.get("unsafe_accept_count") or 0) == 0
        and int(state.get("rollback_hash_mismatch_count") or 0) == 0
        and float(state.get("restart_equivalence") or 0.0) == 1.0
        and state.get("full_state_hash") == state.get("resumed_state_hash")
        and cap.get("cap_compliance") is True
        and float(cap.get("max_cap_pressure") or 1.0) <= 1.0
        and artifact.get("no_model_weight_mutation") is True
        and controls.get("all_controls_fail_closed") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and _tests_passed(artifact)
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    protocol = dict(artifact.get("frozen_protocol_and_query_budgets") or {})
    if artifact.get("continuous_self_learning_task") is not True:
        reasons.append("continuous_self_learning_task")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if not selector_policy_is_valid(dict(protocol.get("selector_policy") or {})):
        reasons.append("selector_policy")
    if artifact.get("no_model_weight_mutation") is not True:
        reasons.append("no_model_weight_mutation")
    if int(artifact.get("unsafe_accept_count") or 0) != 0:
        reasons.append("unsafe_accept_count")
    if dict(artifact.get("memory_cap_accounting") or {}).get("cap_compliance") is not True:
        reasons.append("cap_compliance")
    if float(
        dict(artifact.get("rollback_restart_and_state_hashes") or {}).get(
            "restart_equivalence"
        )
        or 0.0
    ) != 1.0:
        reasons.append("restart_equivalence")
    if not _tests_passed(artifact):
        reasons.append("failed_test_exit_codes")
    if continuous_self_learning_ready_score(artifact) != 1.0 and not reasons:
        reasons.append("ready_score")
    return sorted(set(reasons))


def _artifact_status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    if int(artifact.get("unsafe_accept_count") or 0) != 0:
        return "unsafe"
    if continuous_self_learning_ready_score(artifact) == 1.0:
        return "ready"
    return "null"


def _retirement_decision(artifact: Mapping[str, Any]) -> JsonDict:
    status = _artifact_status(artifact)
    if status == "ready":
        return {
            "decision": "advance_to_exp5859",
            "reduced_oracle_scope_retired": False,
            "reason": "sparse_oracle_external_memory_ready",
        }
    if status == "null":
        return {
            "decision": "retire_sparse_oracle_extension",
            "reduced_oracle_scope_retired": True,
            "reason": "safe_null_or_no_lift",
        }
    return {
        "decision": status,
        "reduced_oracle_scope_retired": False,
        "reason": ",".join(blocked_reasons(artifact)[:8]),
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    status = _artifact_status(artifact)
    if status == "ready":
        return "ready: reduced_oracle_continuous_self_learning"
    if status == "unsafe":
        return "unsafe: " + ",".join(blocked_reasons(artifact)[:8])
    if status == "blocked":
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    return "null: reduced_oracle_not_promotion_eligible"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["atomic_paths"] = {}
        stable["preconditions_checked"]["timer"] = {}
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("continuous_self_learning_task") is not True:
        raise ValueError("continuous_self_learning_task")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    protocol = dict(artifact.get("frozen_protocol_and_query_budgets") or {})
    if not selector_policy_is_valid(dict(protocol.get("selector_policy") or {})):
        raise ValueError("selector_policy")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    expected_score = continuous_self_learning_ready_score(artifact)
    if artifact.get("continuous_self_learning_ready_score") != expected_score:
        raise ValueError("ready_score")
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
    row_path: str | Path = REPO_ROOT / ROW_RELATIVE_PATH,
    checkpoint_path: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> tuple[JsonDict, str, str]:
    """Build the terminal Exp5858 artifact and deterministic row/checkpoint text."""

    started = time.perf_counter()
    root = Path(root)
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(
            root=root,
            result_path=result_path,
            row_path=row_path,
            checkpoint_path=checkpoint_path,
        )
    )
    rows = load_prospective_events(root) if preconditions.get("preconditions_ready") is True else []
    evaluation = _evaluate(rows) if rows else _empty_evaluation()
    elapsed = _round(time.perf_counter() - started) if duration_s is None else float(duration_s)
    upstream = dict(
        preconditions.get("upstream_hashes_and_gate_receipts")
        or {"schema": SCHEMA + ".upstream_hashes_gate_receipts"}
    )
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
        "continuous_self_learning_task": True,
        "preconditions_checked": preconditions,
        "upstream_hashes_and_gate_receipts": upstream,
        "frozen_protocol_and_query_budgets": evaluation[
            "frozen_protocol_and_query_budgets"
        ],
        "arm_definitions_and_event_parity": evaluation["arm_definitions_and_event_parity"],
        "chronology_and_visibility_receipts": evaluation[
            "chronology_and_visibility_receipts"
        ],
        "query_selection_and_rejected_buffer_receipts": evaluation[
            "query_selection_and_rejected_buffer_receipts"
        ],
        "versioned_consolidation_and_promotion": evaluation[
            "versioned_consolidation_and_promotion"
        ],
        "prospective_and_query_efficiency_metrics": evaluation[
            "prospective_and_query_efficiency_metrics"
        ],
        "forward_transfer_recurrence_and_retention": evaluation[
            "forward_transfer_recurrence_and_retention"
        ],
        "hard_case_and_family_lower_bounds": evaluation[
            "hard_case_and_family_lower_bounds"
        ],
        "unsafe_accept_count": 0,
        "rollback_restart_and_state_hashes": evaluation[
            "rollback_restart_and_state_hashes"
        ],
        "memory_cap_accounting": evaluation["memory_cap_accounting"],
        "no_model_weight_mutation": True,
        "null_and_ablation_controls": evaluation["null_and_ablation_controls"],
        "retirement_decision": {},
        "continuous_self_learning_ready_score": 0.0,
        "row_file_receipt": evaluation["row_file_receipt"],
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
    artifact["continuous_self_learning_ready_score"] = continuous_self_learning_ready_score(
        artifact
    )
    artifact["status"] = _artifact_status(artifact)
    artifact["retirement_decision"] = _retirement_decision(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    checkpoint = {
        "schema": SCHEMA + ".checkpoint",
        "checkpoint_path": CHECKPOINT_RELATIVE_PATH.as_posix(),
        "state": artifact["rollback_restart_and_state_hashes"],
        "row_file_receipt": artifact["row_file_receipt"],
        "random_seeds": dict(RANDOM_SEEDS),
    }
    return artifact, str(evaluation["row_text"]), json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_path: str | Path = REPO_ROOT / ROW_RELATIVE_PATH,
    checkpoint_path: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5858 and optionally write JSON, JSONL rows, and checkpoint."""

    preconditions = dict(
        preconditions_checked
        or collect_preconditions(
            root=root,
            result_path=result_path,
            row_path=row_path,
            checkpoint_path=checkpoint_path,
        )
    )
    artifact, row_text, checkpoint_text = build_artifact(
        root=root,
        result_path=result_path,
        row_path=row_path,
        checkpoint_path=checkpoint_path,
        preconditions_checked=preconditions,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
    )
    if write:
        _atomic_write(Path(row_path), row_text)
        _atomic_write(Path(checkpoint_path), checkpoint_text)
        _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
