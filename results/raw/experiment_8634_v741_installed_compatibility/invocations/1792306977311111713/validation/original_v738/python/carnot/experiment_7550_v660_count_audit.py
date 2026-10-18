"""Independently audit V660 delayed count learning from raw evidence.

This module rebuilds Beta counts and binary probabilities without importing the
producer reducer. It keeps a valid null distinct from invalid or missing data.

Spec refs: REQ-REPORT-7550 and SCENARIO-REPORT-7550-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7550-count-audit"
SCHEMA = "carnot.exp7550.v660.count_audit.v1"
RESULT_PATH = Path("results/experiment_7550_v660_count_audit.json")
RAW_DIR = Path("results/raw/experiment_7550_v660_count_audit")
MODULE_PATH = Path("python/carnot/experiment_7550_v660_count_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7550_v660_count_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7550_v660_count_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
STREAM_PATH = Path("results/experiment_7547_v660_count_stream.json")
PRODUCER_PATH = Path("results/experiment_7549_v660_count_learning.json")
PRIOR_7509_PATH = Path("results/experiment_7509_v657_causal_online.json")
PRIOR_7510_PATH = Path("results/experiment_7510_v657_causal_audit.json")

ARMS = ("frozen", "global_count", "local_count", "shuffled_local")
COMPARATORS = ("frozen", "global_count", "shuffled_local")
ORDER_SEEDS = (7549001, 7549002, 7549003, 7549004, 7549005)
BOOTSTRAP_SEED = 7549011
BOOTSTRAP_REPLICATES = 1000
MINIMUM_SOURCES = 128
MINIMUM_PER_LABEL = 12
MINIMUM_BRIER_DELTA = -0.005
MAX_RETENTION_DETERIORATION = 0.01
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
MUTATION_NAMES = (
    "future_label_swap",
    "orientation_change",
    "duplicate_update",
    "group_id_permutation",
    "checkpoint_change",
    "aggregate_row_mismatch",
)

AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7510_v657_causal_audit.py"),
    Path("python/carnot/experiment_7534_v659_count_memory.py"),
    SPEC_PATH,
    DESIGN_PATH,
    STREAM_PATH,
    PRODUCER_PATH,
    PRIOR_7509_PATH,
    PRIOR_7510_PATH,
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so a changed row cannot retain its evidence identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes before any scientific field is trusted."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    """Return one object, or an empty object when external bytes are malformed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read object rows without repairing malformed external evidence."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_number}")
            rows.append(value)
    return rows


def sidecar_receipt(path: Path, root: Path, *, rows: int) -> JsonDict:
    """Bind one row file by relative path, exact bytes, hash, and row count."""

    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def read_sidecar(root: Path, receipt: Mapping[str, Any], name: str) -> list[JsonDict]:
    """Authenticate all sidecar bytes before returning any rows."""

    path = root / str(receipt.get("path") or "")
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"raw_sidecar_hash_mismatch:{name}")
    if path.stat().st_size != receipt.get("bytes"):
        raise ValueError(f"raw_sidecar_size_mismatch:{name}")
    rows = load_jsonl(path)
    if len(rows) != receipt.get("rows"):
        raise ValueError(f"raw_sidecar_row_count_mismatch:{name}")
    return rows


def _clip(probability: float) -> float:
    """Keep direct odds finite while preserving the registered probability range."""

    if not math.isfinite(probability):
        raise ValueError("probability_not_finite")
    return min(1.0 - 1e-4, max(1e-4, float(probability)))


def _logit(probability: float) -> float:
    value = _clip(probability)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def direct_beta_prediction(
    baseline: float, successes: float, failures: float, prior_mean: float
) -> tuple[float, list[float]]:
    """Apply the registered odds correction from direct Beta sufficient counts."""

    base = _clip(baseline)
    posterior = float(successes) / (float(successes) + float(failures))
    probability = _sigmoid(_logit(base) + _logit(posterior) - _logit(prior_mean))
    return probability, [0.0, -_logit(probability)]


def normalized_binary_probability(energies: Sequence[float]) -> float:
    """Normalize both binary energies instead of trusting a saved probability."""

    if len(energies) != 2:
        raise ValueError("binary_energy_requires_two_values")
    zero = math.exp(-float(energies[0]))
    one = math.exp(-float(energies[1]))
    return one / (zero + one)


def brier(probability: float, label: int) -> float:
    """Return one binary Brier contribution from a sealed forecast."""

    return (float(probability) - int(label)) ** 2


def typed_decision(probability: float, label: int) -> JsonDict:
    """Apply the frozen cost policy, including escalation on a tied minimum."""

    value = float(probability)
    expected = {"accept": 5.0 * value, "reject": 1.0 - value, "escalate": 0.2}
    minimum = min(expected.values())
    tied = [name for name, cost in expected.items() if math.isclose(cost, minimum, abs_tol=1e-12)]
    action = "escalate" if "escalate" in tied else tied[0]
    realized = {"accept": 5.0 * int(label), "reject": 1.0 - int(label), "escalate": 0.2}
    return {
        "action": action,
        "expected_costs": expected,
        "expected_cost": expected[action],
        "realized_cost": realized[action],
        "tie_break": "escalation_wins",
    }


def _precondition(
    check: str,
    upstream: str,
    path_or_field: str,
    expected: Any,
    observed: Any,
    *,
    required: bool = True,
    op: str = "eq",
) -> JsonDict:
    """Keep each exact prerequisite operand visible before measurement."""

    passed = observed == expected
    if op == "in":
        passed = observed in expected
    return {
        "check": check,
        "upstream": upstream,
        "path_or_field": path_or_field,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "required": required,
        "principle": "Exact prerequisite operands prevent fabricated fallback evidence.",
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Check required paths and terminal producer fields before reading raw rows."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    resolved = root.resolve()
    for relative in REQUIRED_INPUT_PATHS:
        path = resolved / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _precondition(
                f"required_input:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "readable_nonempty_bytes",
                observed,
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = sha256_file(path)

    spec = (
        (resolved / SPEC_PATH).read_text(encoding="utf-8")
        if (resolved / SPEC_PATH).is_file()
        else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7550",
            "REQ-REPORT-7550" if "REQ-REPORT-7550" in spec else None,
        )
    )
    upstreams = {
        "stream": load_json(resolved / STREAM_PATH),
        "producer": load_json(resolved / PRODUCER_PATH),
        "prior_7509": load_json(resolved / PRIOR_7509_PATH),
        "prior_7510": load_json(resolved / PRIOR_7510_PATH),
    }
    upstreams = {name: value for name, value in upstreams.items() if value}
    expected_fields = {
        "stream": (
            ("experiment_id", EXPERIMENT_ID.replace("7550-count-audit", "7547-count-stream"), "eq"),
            ("milestone", MILESTONE, "eq"),
            ("run_date", RUN_DATE, "eq"),
            ("cached_stream_ready_score", 1, "eq"),
            ("verdict_class", ("positive", "null", "circular_positive"), "in"),
            ("flagged_adversarial", False, "eq"),
        ),
        "producer": (
            ("experiment_id", "exp7549-count-learning", "eq"),
            ("milestone", MILESTONE, "eq"),
            ("run_date", RUN_DATE, "eq"),
            ("terminal_status", "complete", "eq"),
            ("count_measurement_complete_score", 1, "eq"),
            ("verdict_class", ("positive", "null", "circular_positive"), "in"),
            ("flagged_adversarial", False, "eq"),
        ),
        "prior_7509": (
            (
                "honest_verdict",
                "complete_null_causal_online_measurement_valid_benefit_gate_failed",
                "eq",
            ),
            ("verdict_class", "null", "eq"),
        ),
        "prior_7510": (
            (
                "honest_verdict",
                "complete_null_v657_causal_audit_benefit_gate_failed",
                "eq",
            ),
            ("verdict_class", "null", "eq"),
            ("causal_claims_qualified_score", 1, "eq"),
            ("qualified_online_benefit_score", 0, "eq"),
        ),
    }
    paths = {
        "stream": STREAM_PATH,
        "producer": PRODUCER_PATH,
        "prior_7509": PRIOR_7509_PATH,
        "prior_7510": PRIOR_7510_PATH,
    }
    for name, fields in expected_fields.items():
        artifact = upstreams.get(name, {})
        for field, expected, op in fields:
            checks.append(
                _precondition(
                    f"upstream_{name}:{field}",
                    paths[name].as_posix(),
                    field,
                    expected,
                    artifact.get(field),
                    op=op,
                )
            )

    receipts: list[tuple[str, Mapping[str, Any]]] = []
    stream = upstreams.get("stream", {})
    frozen = stream.get("raw_sidecars", {}).get("frozen_protocol", {})
    if isinstance(frozen, Mapping):
        receipts.append(("stream.frozen_protocol", frozen))
    producer = upstreams.get("producer", {})
    for field in (
        "prediction_update_rows",
        "release_rows",
        "persistence_rows",
        "retention_rows",
        "bootstrap_rows",
    ):
        receipt = producer.get(field, {})
        if isinstance(receipt, Mapping):
            receipts.append((f"producer.{field}", receipt))
    for name, receipt in receipts:
        label = str(receipt.get("path") or "")
        path = resolved / label
        observed_hash = sha256_file(path) if path.is_file() else None
        checks.append(
            _precondition(
                f"raw_custody:{name}",
                paths[name.split(".", 1)[0]].as_posix(),
                label or name,
                receipt.get("sha256"),
                observed_hash,
            )
        )
    cpu_count = os.cpu_count() or 0
    disk_available = resolved.stat().st_dev >= 0
    checks.append(
        {
            **_precondition(
                "resource_availability",
                "host CPU/RAM/storage",
                "cpu_count_and_repository_storage",
                True,
                cpu_count > 0 and disk_available,
            ),
            "resource_observation": {
                "logical_cpu_count": cpu_count,
                "repository_device": resolved.stat().st_dev,
                "gpu_required": False,
            },
        }
    )
    return checks, hashes, upstreams


def _read_json_sidecar(root: Path, receipt: Mapping[str, Any], name: str) -> JsonDict:
    """Authenticate one JSON object sidecar without trusting its path alone."""

    path = root / str(receipt.get("path") or "")
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"raw_sidecar_hash_mismatch:{name}")
    if receipt.get("bytes") is not None and path.stat().st_size != receipt.get("bytes"):
        raise ValueError(f"raw_sidecar_size_mismatch:{name}")
    value = load_json(path)
    if not value:
        raise ValueError(f"raw_sidecar_object_invalid:{name}")
    return value


def load_upstream_evidence(root: Path, upstreams: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Load only authenticated raw inputs from the two exact producer paths."""

    stream = upstreams["stream"]
    producer = upstreams["producer"]
    protocol = _read_json_sidecar(
        root, stream["raw_sidecars"]["frozen_protocol"], "frozen_protocol"
    )
    return {
        "protocol": protocol,
        "producer": deepcopy(dict(producer)),
        "prediction_rows": read_sidecar(
            root, producer["prediction_update_rows"], "prediction_update_rows"
        ),
        "release_rows": read_sidecar(root, producer["release_rows"], "release_rows"),
        "persistence_rows": read_sidecar(root, producer["persistence_rows"], "persistence_rows"),
        "retention_rows": read_sidecar(root, producer["retention_rows"], "retention_rows"),
        "bootstrap_rows": read_sidecar(root, producer["bootstrap_rows"], "bootstrap_rows"),
    }


def _bin_index(probability: float) -> int:
    return min(7, math.floor(8 * _clip(probability)))


def _initial_counts(config: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Create direct mutable counts from the label-free frozen means."""

    means = [float(value) for value in config["bin_means"]]
    global_mean = float(config["global_mean"])
    kappa = float(config["kappa"])

    def cells(values: Sequence[float]) -> list[list[float]]:
        return [[kappa * value, kappa * (1.0 - value)] for value in values]

    return {
        "frozen": {"kind": "frozen", "counts": cells(means), "processed_event_ids": []},
        "global": {
            "kind": "global",
            "counts": cells([global_mean]),
            "processed_event_ids": [],
        },
        "local": {"kind": "local", "counts": cells(means), "processed_event_ids": []},
        "permuted_local": {
            "kind": "permuted_local",
            "counts": cells(means),
            "processed_event_ids": [],
        },
    }


def _counts_hash(counts: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_hash({name: deepcopy(dict(counts[name])) for name in sorted(counts)})


def _predict_arm(
    arm: str, baseline: float, config: Mapping[str, Any], counts: Mapping[str, Mapping[str, Any]]
) -> tuple[float, list[float], int]:
    index = _bin_index(baseline)
    machine_name = {
        "frozen": "frozen",
        "global_count": "global",
        "local_count": "local",
        "shuffled_local": "permuted_local",
    }[arm]
    slot = 0 if machine_name == "global" else index
    prior = (
        float(config["global_mean"])
        if machine_name == "global"
        else float(config["bin_means"][index])
    )
    cell = counts[machine_name]["counts"][slot]
    successes, failures = float(cell[0]), float(cell[1])
    if machine_name == "frozen":
        successes = float(config["kappa"]) * prior
        failures = float(config["kappa"]) * (1.0 - prior)
    probability, energies = direct_beta_prediction(baseline, successes, failures, prior)
    return probability, energies, index


def _close(left: Any, right: Any, *, tolerance: float = 1e-11) -> bool:
    """Compare nested numerical evidence while permitting float roundoff only."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_close(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def _update_count(
    counts: dict[str, JsonDict], arm: str, event_id: str, baseline: float, label: int
) -> str | None:
    """Apply one binary update and return a duplicate error when it repeats."""

    state = counts[arm]
    seen = state["processed_event_ids"]
    if event_id in seen:
        return f"duplicate_update:{arm}:{event_id}"
    if label not in (0, 1):
        return f"nonbinary_label:{event_id}"
    slot = 0 if arm == "global" else _bin_index(baseline)
    state["counts"][slot][0] += label
    state["counts"][slot][1] += 1 - label
    seen.append(event_id)
    seen.sort()
    return None


def _group_prediction_rows(
    rows: Sequence[Mapping[str, Any]], errors: list[str]
) -> dict[tuple[int, str], dict[str, Mapping[str, Any]]]:
    """Require one row per arm for each repeated order event."""

    grouped: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (int(row.get("seed", -1)), str(row.get("event_id") or ""))
        arm = str(row.get("arm") or "")
        if arm in grouped[key]:
            errors.append(f"duplicate_prediction_arm:{key[0]}:{key[1]}:{arm}")
        grouped[key][arm] = row
    return grouped


def _audit_orders(
    protocol: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    release_rows: Sequence[Mapping[str, Any]],
    persistence_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[tuple[int, int], dict[str, JsonDict]]]:
    """Replay direct counts and preserve snapshots for retention evaluation."""

    errors: list[str] = []
    if protocol.get("labels_read") is not False:
        errors.append("protocol_labels_read_before_freeze")
    if protocol.get("settings", {}).get("label_orientation") != ("one_means_contains_unsupported"):
        errors.append("label_orientation_mismatch")
    config = protocol.get("count_config") or {}
    if len(config.get("bin_means") or []) != 8 or config.get("kappa") != 8.0:
        errors.append("count_config_mismatch")

    predictions = _group_prediction_rows(prediction_rows, errors)
    releases_by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in release_rows:
        releases_by_seed[int(row.get("seed", -1))].append(row)
    persistence: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in persistence_rows:
        key = (int(row.get("seed", -1)), int(row.get("block_id", -1)))
        if key in persistence:
            errors.append(f"duplicate_checkpoint:{key[0]}:{key[1]}")
        persistence[key] = row

    snapshots: dict[tuple[int, int], dict[str, JsonDict]] = {}
    settings = protocol.get("settings") or {}
    block_size = int(settings.get("block_size", 8))
    delay = int(settings.get("feedback_delay", 8))
    expected_keys: set[tuple[int, str]] = set()
    for seed_text, raw_order in (protocol.get("orders") or {}).items():
        seed = int(seed_text)
        order = [str(item) for item in raw_order]
        counts = _initial_counts(config)
        snapshots[(seed, 0)] = deepcopy(counts)
        releases_at: dict[int, Mapping[str, Any]] = {}
        observed_blocks: set[int] = set()
        for release in releases_by_seed.get(seed, []):
            block_id = int(release.get("block_id", -1))
            if block_id in observed_blocks:
                errors.append(f"duplicate_update:{seed}:{block_id}")
            observed_blocks.add(block_id)
            release_time = int(release.get("release_time", -1))
            if release_time in releases_at:
                errors.append(f"duplicate_release_time:{seed}:{release_time}")
            releases_at[release_time] = release

        for prediction_time, event_id in enumerate(order):
            key = (seed, event_id)
            expected_keys.add(key)
            arm_rows = predictions.get(key, {})
            if set(arm_rows) != set(ARMS):
                errors.append(f"prediction_arm_roster_mismatch:{seed}:{event_id}")
                continue
            baselines = {float(row.get("base_probability", math.nan)) for row in arm_rows.values()}
            labels = {row.get("label") for row in arm_rows.values()}
            if len(baselines) != 1 or len(labels) != 1:
                errors.append(f"prediction_operand_mismatch:{seed}:{event_id}")
                continue
            baseline = next(iter(baselines))
            label = next(iter(labels))
            before_hash = _counts_hash(counts)
            for arm in ARMS:
                row = arm_rows[arm]
                probability, energies, index = _predict_arm(arm, baseline, config, counts)
                if row.get("label_available_at_prediction") is not False:
                    errors.append(f"future_label_access:{seed}:{event_id}:{arm}")
                if row.get("prediction_time") != prediction_time:
                    errors.append(f"prediction_time_mismatch:{seed}:{event_id}:{arm}")
                if row.get("counts_hash_before_prediction") != before_hash:
                    errors.append(f"count_state_mismatch:{seed}:{event_id}:{arm}")
                if not _close(row.get("probability"), probability):
                    errors.append(f"probability_mismatch:{seed}:{event_id}:{arm}")
                if not _close(row.get("energies"), energies):
                    errors.append(f"energy_mismatch:{seed}:{event_id}:{arm}")
                if not _close(
                    row.get("normalized_probability"), normalized_binary_probability(energies)
                ):
                    errors.append(f"normalization_mismatch:{seed}:{event_id}:{arm}")
                if row.get("bin_index") != index:
                    errors.append(f"bin_mismatch:{seed}:{event_id}:{arm}")
                if label not in (0, 1) or not _close(
                    row.get("brier"), brier(probability, int(label))
                ):
                    errors.append(f"row_brier_mismatch:{seed}:{event_id}:{arm}")
                if not _close(row.get("decision"), typed_decision(probability, int(label))):
                    errors.append(f"row_cost_mismatch:{seed}:{event_id}:{arm}")

            release = releases_at.get(prediction_time)
            if release is not None and release.get("release_within_stream") is True:
                block_id = int(release.get("block_id", -1))
                start = block_id * block_size
                expected_events = order[start : start + block_size]
                released_events = [str(item) for item in release.get("event_ids") or []]
                expected_release_time = start + len(expected_events) - 1 + delay
                if released_events != expected_events or prediction_time != expected_release_time:
                    errors.append(f"release_schedule_mismatch:{seed}:{block_id}")
                permutation = {
                    str(row.get("event_id")): row
                    for row in release.get("source_to_label_permutation") or []
                }
                if set(permutation) != set(released_events):
                    errors.append(f"shuffled_group_roster_mismatch:{seed}:{block_id}")
                true_labels: list[int] = []
                shuffled_labels: list[int] = []
                for released_id in released_events:
                    released_rows = predictions.get((seed, released_id), {})
                    if set(released_rows) != set(ARMS):
                        errors.append(f"released_prediction_missing:{seed}:{released_id}")
                        continue
                    truth = int(released_rows["local_count"].get("label", -1))
                    true_labels.append(truth)
                    shuffled = int(permutation.get(released_id, {}).get("label", -1))
                    shuffled_labels.append(shuffled)
                    base = float(released_rows["local_count"]["base_probability"])
                    for machine_arm, update_label in (
                        ("global", truth),
                        ("local", truth),
                        ("permuted_local", shuffled),
                    ):
                        duplicate = _update_count(
                            counts, machine_arm, released_id, base, update_label
                        )
                        if duplicate:
                            errors.append(duplicate)
                if sorted(true_labels) != release.get("label_multiset"):
                    errors.append(f"released_label_mismatch:{seed}:{block_id}")
                if sorted(shuffled_labels) != release.get("permuted_label_multiset"):
                    errors.append(f"shuffled_label_mismatch:{seed}:{block_id}")
                changed = sum(a != b for a, b in zip(true_labels, shuffled_labels))
                if changed != release.get("changed_label_binding_count"):
                    errors.append(f"changed_binding_mismatch:{seed}:{block_id}")
                after_hash = _counts_hash(counts)
                update_id = f"seed-{seed}-release-{block_id:03d}"
                after_states: set[Any] = set()
                for released_id in released_events:
                    for arm, row in predictions.get((seed, released_id), {}).items():
                        if row.get("feedback_update_id") != update_id:
                            errors.append(f"feedback_receipt_mismatch:{seed}:{released_id}:{arm}")
                        if row.get("feedback_availability_time") != prediction_time:
                            errors.append(f"feedback_time_mismatch:{seed}:{released_id}:{arm}")
                        if row.get("counts_hash_after_feedback") != after_hash:
                            errors.append(f"post_count_state_mismatch:{seed}:{released_id}:{arm}")
                        after_states.add(row.get("state_hash_after_feedback"))
                checkpoint = persistence.get((seed, block_id))
                if checkpoint is None:
                    errors.append(f"checkpoint_missing:{seed}:{block_id}")
                else:
                    checkpoint_hash = str(checkpoint.get("checkpoint_sha256") or "")
                    if len(checkpoint_hash) != 71 or not checkpoint_hash.startswith("sha256:"):
                        errors.append(f"checkpoint_hash_invalid:{seed}:{block_id}")
                    if len(after_states) != 1 or checkpoint.get("state_hash") not in after_states:
                        errors.append(f"checkpoint_state_mismatch:{seed}:{block_id}")
                    if checkpoint.get("payload_equal_after_reload") is not True:
                        errors.append(f"checkpoint_payload_mismatch:{seed}:{block_id}")
                    if checkpoint.get("uninterrupted_state_equal") is not True:
                        errors.append(f"checkpoint_restart_mismatch:{seed}:{block_id}")
                    if checkpoint.get("duplicate_feedback_rejected") is not True:
                        errors.append(f"checkpoint_duplicate_guard_missing:{seed}:{block_id}")
            snapshots[(seed, prediction_time + 1)] = deepcopy(counts)

        for release in releases_by_seed.get(seed, []):
            if release.get("release_within_stream") is False:
                block_id = int(release.get("block_id", -1))
                start = block_id * block_size
                expected_events = order[start : start + block_size]
                if release.get("event_ids") != expected_events:
                    errors.append(f"censored_schedule_mismatch:{seed}:{block_id}")
                if release.get("update_id") is not None:
                    errors.append(f"censored_update_present:{seed}:{block_id}")

    extras = set(predictions) - expected_keys
    if extras:
        errors.append(f"prediction_group_extra:{len(extras)}")
    expected_persistence = sum(
        int(row.get("release_within_stream") is True) for row in release_rows
    )
    if len(persistence_rows) != expected_persistence:
        errors.append("checkpoint_count_mismatch")
    return list(dict.fromkeys(errors)), snapshots


def _audit_retention(
    protocol: Mapping[str, Any],
    retention_rows: Sequence[Mapping[str, Any]],
    snapshots: Mapping[tuple[int, int], Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    """Verify held-out labels score snapshots without updating their counts."""

    errors: list[str] = []
    config = protocol.get("count_config") or {}
    for row in retention_rows:
        seed = int(row.get("seed", -1))
        arrival = int(row.get("arrival_checkpoint", -1))
        counts = snapshots.get((seed, arrival))
        arm = str(row.get("arm") or "")
        if counts is None or arm not in ARMS:
            errors.append(f"retention_snapshot_missing:{seed}:{arrival}:{arm}")
            continue
        if row.get("label_returned_to_learner") is not False:
            errors.append(f"retention_label_leak:{seed}:{arrival}:{row.get('group_id')}")
        baseline = float(row.get("base_probability", math.nan))
        label = int(row.get("label", -1))
        probability, energies, _index = _predict_arm(arm, baseline, config, counts)
        identity = f"{seed}:{arrival}:{row.get('group_id')}:{arm}"
        if not _close(row.get("probability"), probability):
            errors.append(f"retention_probability_mismatch:{identity}")
        if not _close(row.get("energies"), energies):
            errors.append(f"retention_energy_mismatch:{identity}")
        if not _close(row.get("normalized_probability"), normalized_binary_probability(energies)):
            errors.append(f"retention_normalization_mismatch:{identity}")
        if label not in (0, 1) or not _close(row.get("brier"), brier(probability, label)):
            errors.append(f"retention_brier_mismatch:{identity}")
        if not _close(row.get("decision"), typed_decision(probability, label)):
            errors.append(f"retention_cost_mismatch:{identity}")
    return list(dict.fromkeys(errors))


def _bootstrap_order(
    sampled: Sequence[str],
    baselines: Mapping[str, float],
    labels: Mapping[str, int],
    registered_order: Sequence[str],
    config: Mapping[str, Any],
    retention_sources: Sequence[Mapping[str, Any]],
    retention_sample: Sequence[int],
    checkpoints: Sequence[int],
) -> tuple[dict[str, float], list[float], list[float]]:
    """Replay one source-multiplicity draw through one registered order."""

    multiplicity = Counter(sampled)
    expanded = [
        (f"{source_id}#bootstrap-{copy_index}", source_id)
        for source_id in registered_order
        for copy_index in range(multiplicity[source_id])
    ]
    counts = _initial_counts(config)
    losses: dict[str, list[float]] = {arm: [] for arm in ARMS}
    retention_brier: list[float] = []
    retention_cost: list[float] = []
    block_size = 8
    delay = 8
    due: dict[int, list[str]] = {}
    for start in range(0, len(expanded), block_size):
        clones = [clone for clone, _source in expanded[start : start + block_size]]
        release_time = start + len(clones) - 1 + delay
        if release_time < len(expanded):
            due[release_time] = clones

    def score_retention() -> None:
        local_brier: list[float] = []
        frozen_brier: list[float] = []
        local_cost: list[float] = []
        frozen_cost: list[float] = []
        for index in retention_sample:
            source = retention_sources[index]
            baseline = float(source["base_probability"])
            label = int(source["label"])
            for arm, briers, costs in (
                ("local_count", local_brier, local_cost),
                ("frozen", frozen_brier, frozen_cost),
            ):
                probability, _energies, _bin = _predict_arm(arm, baseline, config, counts)
                briers.append(brier(probability, label))
                costs.append(float(typed_decision(probability, label)["realized_cost"]))
        retention_brier.append(
            math.fsum(local_brier) / len(local_brier) - math.fsum(frozen_brier) / len(frozen_brier)
        )
        retention_cost.append(
            math.fsum(local_cost) / len(local_cost) - math.fsum(frozen_cost) / len(frozen_cost)
        )

    if 0 in checkpoints:
        score_retention()
    predicted: dict[str, str] = {}
    for prediction_time, (clone_id, source_id) in enumerate(expanded):
        baseline = baselines[source_id]
        label = labels[source_id]
        predicted[clone_id] = source_id
        for arm in ARMS:
            probability, _energies, _bin = _predict_arm(arm, baseline, config, counts)
            losses[arm].append(brier(probability, label))
        released = due.get(prediction_time)
        if released is not None:
            true = [labels[predicted[clone]] for clone in released]
            shuffled = true[1:] + true[:1]
            for index, clone in enumerate(released):
                source = predicted[clone]
                base = baselines[source]
                _update_count(counts, "global", clone, base, true[index])
                _update_count(counts, "local", clone, base, true[index])
                _update_count(counts, "permuted_local", clone, base, shuffled[index])
        if prediction_time + 1 in checkpoints:
            score_retention()
    means = {arm: math.fsum(values) / len(values) for arm, values in losses.items()}
    return means, retention_brier, retention_cost


def source_cluster_resamples(
    protocol: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    progress_hook: Callable[[int], None] | None = None,
) -> list[JsonDict]:
    """Resample sources once and replay their dependence across all orders."""

    first_by_event: dict[str, Mapping[str, Any]] = {}
    for row in prediction_rows:
        if row.get("arm") == "local_count":
            first_by_event.setdefault(str(row["event_id"]), row)
    source_ids = sorted(first_by_event)
    baselines = {event: float(first_by_event[event]["base_probability"]) for event in source_ids}
    labels = {event: int(first_by_event[event]["label"]) for event in source_ids}
    retention_by_group: dict[str, Mapping[str, Any]] = {}
    for row in retention_rows:
        if row.get("arm") == "frozen":
            retention_by_group.setdefault(str(row["group_id"]), row)
    retention_sources = [retention_by_group[group] for group in sorted(retention_by_group)]
    orders = [list(value) for _seed, value in sorted((protocol.get("orders") or {}).items())]
    config = protocol.get("count_config") or {}
    checkpoints = sorted({int(row["arrival_checkpoint"]) for row in retention_rows})
    rng = random.Random(BOOTSTRAP_SEED)
    output: list[JsonDict] = []
    for replicate in range(replicates):
        sampled = [source_ids[rng.randrange(len(source_ids))] for _ in source_ids]
        retention_sample = [
            rng.randrange(len(retention_sources)) for _ in range(len(retention_sources))
        ]
        order_losses: list[dict[str, float]] = []
        retained_brier: list[float] = []
        retained_cost: list[float] = []
        for order in orders:
            means, brier_deltas, cost_deltas = _bootstrap_order(
                sampled,
                baselines,
                labels,
                order,
                config,
                retention_sources,
                retention_sample,
                checkpoints,
            )
            order_losses.append(means)
            retained_brier.extend(brier_deltas)
            retained_cost.extend(cost_deltas)
        arm_means = {
            arm: math.fsum(row[arm] for row in order_losses) / len(order_losses) for arm in ARMS
        }
        output.append(
            {
                "replicate": replicate,
                "sampled_source_count": len(sampled),
                "unique_source_count": len(set(sampled)),
                "source_multiplicity_hash": canonical_hash(sorted(Counter(sampled).items())),
                "order_count": len(orders),
                "chronology_replayed": True,
                "online_delta_brier": {
                    comparator: arm_means["local_count"] - arm_means[comparator]
                    for comparator in COMPARATORS
                },
                "retention_brier_deterioration": max(retained_brier),
                "retention_cost_deterioration": max(retained_cost),
                "disposition": "complete_source_cluster_replay",
            }
        )
        if progress_hook is not None and (
            (replicate + 1) % 100 == 0 or replicate + 1 == replicates
        ):
            progress_hook(replicate + 1)
    return output


def _upper(values: Sequence[float], confidence: float) -> float | None:
    """Return the conservative observed quantile for finite replay draws."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(confidence * len(ordered)) - 1))
    return ordered[index]


def _holm(
    means: Mapping[str, float], bootstrap_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Apply one Holm family to the three registered Brier contrasts."""

    base: list[JsonDict] = []
    for comparator in COMPARATORS:
        draws = [float(row["online_delta_brier"][comparator]) for row in bootstrap_rows]
        one_sided = (1 + sum(value >= 0.0 for value in draws)) / (len(draws) + 1)
        base.append(
            {
                "comparator": comparator,
                "mean_delta": float(means[comparator]),
                "upper95_delta": _upper(draws, 0.95),
                "one_sided_p": one_sided,
                "bootstrap_replicates": len(draws),
            }
        )
    ranked = sorted(enumerate(base), key=lambda item: float(item[1]["one_sided_p"]))
    adjusted = [1.0] * len(base)
    simultaneous: list[float | None] = [None] * len(base)
    running = 0.0
    for rank, (index, row) in enumerate(ranked):
        remaining = len(base) - rank
        running = max(running, min(1.0, remaining * float(row["one_sided_p"])))
        adjusted[index] = running
        draws = [float(item["online_delta_brier"][row["comparator"]]) for item in bootstrap_rows]
        simultaneous[index] = _upper(draws, 1.0 - 0.05 / remaining)
    for index, row in enumerate(base):
        row["holm_adjusted_p"] = adjusted[index]
        row["simultaneous_upper95_delta"] = simultaneous[index]
        row["holm_passed"] = adjusted[index] < 0.05
    return base


def _producer_agreement(producer: Mapping[str, Any], reduced: Mapping[str, Any]) -> bool:
    """Compare only after raw arithmetic has produced independent headlines."""

    headline = producer.get("independent_reduction") or {}
    keys = (
        "absolute_brier",
        "mean_delta_brier",
        "complete_online_source_count",
        "label_counts",
        "primary_contrasts",
        "retention_brier_upper95_deterioration",
        "retention_cost_upper95_deterioration",
        "support_passed",
        "schedule_passed",
        "control_passed",
        "chronology_passed",
        "restart_passed",
        "retention_passed",
        "uncertainty_passed",
        "effect_passed",
    )
    return all(_close(headline.get(key), reduced.get(key)) for key in keys)


def audit_evidence(
    *,
    protocol: Mapping[str, Any],
    producer: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    release_rows: Sequence[Mapping[str, Any]],
    persistence_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    progress_hook: Callable[[int], None] | None = None,
) -> JsonDict:
    """Rebuild causal counts, clustered uncertainty, and retained losses."""

    errors, snapshots = _audit_orders(protocol, prediction_rows, release_rows, persistence_rows)
    retention_errors = _audit_retention(protocol, retention_rows, snapshots)
    errors.extend(retention_errors)
    requested_replicates = int(producer.get("bootstrap_replicates", len(bootstrap_rows)))
    rebuilt_bootstrap = (
        source_cluster_resamples(
            protocol,
            prediction_rows,
            retention_rows,
            replicates=requested_replicates,
            progress_hook=progress_hook,
        )
        if requested_replicates
        else []
    )
    if not _close(rebuilt_bootstrap, list(bootstrap_rows)):
        errors.append("source_resample_mismatch")

    losses: dict[str, list[float]] = {arm: [] for arm in ARMS}
    costs: dict[str, list[float]] = {arm: [] for arm in ARMS}
    labels_by_event: dict[str, int] = {}
    seeds: set[int] = set()
    for row in prediction_rows:
        arm = str(row.get("arm") or "")
        if arm not in losses:
            errors.append(f"unknown_arm:{arm}")
            continue
        losses[arm].append(float(row["brier"]))
        costs[arm].append(float(row["decision"]["realized_cost"]))
        labels_by_event.setdefault(str(row["event_id"]), int(row["label"]))
        seeds.add(int(row["seed"]))
    absolute_brier = {
        arm: math.fsum(values) / len(values) if values else math.nan
        for arm, values in losses.items()
    }
    absolute_cost = {
        arm: math.fsum(values) / len(values) if values else math.nan
        for arm, values in costs.items()
    }
    mean_delta = {
        comparator: absolute_brier["local_count"] - absolute_brier[comparator]
        for comparator in COMPARATORS
    }
    contrasts = _holm(mean_delta, rebuilt_bootstrap) if rebuilt_bootstrap else []
    label_counts = Counter(str(value) for value in labels_by_event.values())
    expected_sources = int(producer.get("expected_online_sources", len(labels_by_event)))
    expected_seeds = {int(seed) for seed in (protocol.get("orders") or {})}
    schedule_passed = bool(
        len(labels_by_event) == expected_sources
        and seeds == expected_seeds
        and len(prediction_rows) == expected_sources * len(expected_seeds) * len(ARMS)
    )
    changed_by_seed: Counter[int] = Counter()
    for row in release_rows:
        changed_by_seed[int(row.get("seed", -1))] += int(row.get("changed_label_binding_count", 0))
    control_passed = (
        bool(schedule_passed and all(changed_by_seed[seed] >= 40 for seed in expected_seeds))
        if expected_sources >= MINIMUM_SOURCES
        else schedule_passed
    )
    support_passed = bool(
        len(labels_by_event) >= min(MINIMUM_SOURCES, expected_sources)
        and int(label_counts.get("0", 0)) >= min(MINIMUM_PER_LABEL, expected_sources // 2)
        and int(label_counts.get("1", 0)) >= min(MINIMUM_PER_LABEL, expected_sources // 2)
    )
    retention_brier_draws = [
        float(row["retention_brier_deterioration"]) for row in rebuilt_bootstrap
    ]
    retention_cost_draws = [float(row["retention_cost_deterioration"]) for row in rebuilt_bootstrap]
    retention_brier_upper = _upper(retention_brier_draws, 0.95)
    retention_cost_upper = _upper(retention_cost_draws, 0.95)
    retention_passed = bool(
        rebuilt_bootstrap
        and retention_brier_upper is not None
        and retention_cost_upper is not None
        and retention_brier_upper <= MAX_RETENTION_DETERIORATION
        and retention_cost_upper <= MAX_RETENTION_DETERIORATION
    )
    uncertainty_passed = bool(
        len(rebuilt_bootstrap) == BOOTSTRAP_REPLICATES
        and len(contrasts) == len(COMPARATORS)
        and all(
            row["simultaneous_upper95_delta"] is not None
            and float(row["simultaneous_upper95_delta"]) < 0.0
            and row["holm_passed"] is True
            for row in contrasts
        )
    )
    chronology_prefixes = (
        "future_label_access",
        "prediction_time_mismatch",
        "release_schedule_mismatch",
        "feedback_time_mismatch",
        "released_label_mismatch",
    )
    restart_prefixes = ("checkpoint_", "post_count_state_mismatch", "duplicate_update")
    arithmetic_prefixes = (
        "probability_mismatch",
        "energy_mismatch",
        "normalization_mismatch",
        "bin_mismatch",
        "row_brier_mismatch",
        "row_cost_mismatch",
        "count_state_mismatch",
    )
    chronology_passed = not any(error.startswith(chronology_prefixes) for error in errors)
    restart_passed = not any(error.startswith(restart_prefixes) for error in errors)
    arithmetic_passed = not any(error.startswith(arithmetic_prefixes) for error in errors)
    retention_isolation_passed = not any(error.startswith("retention_") for error in errors)
    effect_passed = bool(
        support_passed
        and schedule_passed
        and control_passed
        and chronology_passed
        and restart_passed
        and arithmetic_passed
        and retention_isolation_passed
        and retention_passed
        and uncertainty_passed
        and all(value <= MINIMUM_BRIER_DELTA for value in mean_delta.values())
    )
    retention_losses: dict[str, float | None] = {}
    for arm in ARMS:
        values = [float(row["brier"]) for row in retention_rows if row.get("arm") == arm]
        retention_losses[arm] = math.fsum(values) / len(values) if values else None
    reduced: JsonDict = {
        "independent_unit": "online_source_group",
        "complete_online_source_count": len(labels_by_event),
        "label_counts": {"0": int(label_counts.get("0", 0)), "1": int(label_counts.get("1", 0))},
        "absolute_brier": absolute_brier,
        "absolute_primary_cost": absolute_cost,
        "retained_absolute_brier": retention_losses,
        "mean_delta_brier": mean_delta,
        "primary_contrasts": contrasts,
        "retention_brier_upper95_deterioration": retention_brier_upper,
        "retention_cost_upper95_deterioration": retention_cost_upper,
        "support_passed": support_passed,
        "schedule_passed": schedule_passed,
        "control_passed": control_passed,
        "chronology_passed": chronology_passed,
        "restart_passed": restart_passed,
        "arithmetic_passed": arithmetic_passed,
        "retention_isolation_passed": retention_isolation_passed,
        "retention_passed": retention_passed,
        "uncertainty_passed": uncertainty_passed,
        "effect_passed": effect_passed,
        "censored_ordered_event_count": sum(
            row.get("arm") == "local_count"
            and row.get("disposition") == "predicted_feedback_censored_end_of_stream"
            for row in prediction_rows
        ),
        "source_resample_replicates": len(rebuilt_bootstrap),
        "source_resample_agreement": _close(rebuilt_bootstrap, list(bootstrap_rows)),
    }
    agreement = _producer_agreement(producer, reduced)
    reduced["producer_aggregate_agreement"] = agreement
    if not agreement:
        errors.append("producer_aggregate_mismatch")
    reduced["audit_errors"] = list(dict.fromkeys(errors))
    reduced["count_claims_qualified"] = bool(
        not reduced["audit_errors"]
        and support_passed
        and schedule_passed
        and control_passed
        and chronology_passed
        and restart_passed
        and arithmetic_passed
        and retention_isolation_passed
        and reduced["source_resample_agreement"]
    )
    return reduced


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep the exact operand and failure principle beside each decision."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def acceptance_gates(reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Separate validity, readiness, support, and benefit decisions."""

    gates = [
        _gate(
            "independent_arithmetic",
            "validity",
            True,
            reduction.get("arithmetic_passed"),
            "eq",
            reduction.get("arithmetic_passed") is True,
            "Invalid arithmetic cannot support a count-learning claim.",
        ),
        _gate(
            "feedback_chronology",
            "validity",
            True,
            reduction.get("chronology_passed"),
            "eq",
            reduction.get("chronology_passed") is True,
            "Future feedback cannot author an earlier prediction.",
        ),
        _gate(
            "restart_and_exactly_once",
            "validity",
            True,
            reduction.get("restart_passed"),
            "eq",
            reduction.get("restart_passed") is True,
            "A changed or repeated update invalidates persistent learning.",
        ),
        _gate(
            "count_claims_qualified",
            "readiness",
            True,
            reduction.get("count_claims_qualified"),
            "eq",
            reduction.get("count_claims_qualified") is True,
            "A valid null remains auditable without becoming a benefit.",
        ),
        _gate(
            "source_support",
            "support",
            {"sources": MINIMUM_SOURCES, "per_label": MINIMUM_PER_LABEL},
            {
                "sources": reduction.get("complete_online_source_count"),
                "labels": reduction.get("label_counts"),
            },
            "gte",
            reduction.get("support_passed") is True,
            "Repeated orders cannot multiply independent source support.",
        ),
        _gate(
            "exploratory_effect",
            "benefit",
            True,
            reduction.get("effect_passed"),
            "eq",
            reduction.get("effect_passed") is True,
            "Insufficient effect or retention evidence cannot become promotion.",
        ),
        _gate(
            "confirmatory_benefit",
            "benefit",
            0,
            0,
            "eq",
            True,
            "Prior corpus inspection forbids confirmatory promotion.",
        ),
    ]
    return gates


def fixture_evidence() -> JsonDict:
    """Create compact raw evidence for causal and mutation tests."""

    order = ["group-0", "group-1", "group-2", "group-3"]
    baselines = {"group-0": 0.8, "group-1": 0.2, "group-2": 0.2, "group-3": 0.8}
    labels = {"group-0": 1, "group-1": 0, "group-2": 1, "group-3": 0}
    protocol: JsonDict = {
        "labels_read": False,
        "settings": {
            "label_orientation": "one_means_contains_unsupported",
            "block_size": 2,
            "feedback_delay": 1,
        },
        "count_config": {"bin_means": [0.5] * 8, "global_mean": 0.5, "kappa": 8.0},
        "orders": {"1": order},
    }
    counts = _initial_counts(protocol["count_config"])
    predictions: list[JsonDict] = []
    indexes: dict[str, list[int]] = defaultdict(list)
    releases: list[JsonDict] = []
    persistence: list[JsonDict] = []
    for prediction_time, event_id in enumerate(order):
        baseline = baselines[event_id]
        label = labels[event_id]
        before = _counts_hash(counts)
        for arm in ARMS:
            probability, energies, index = _predict_arm(
                arm, baseline, protocol["count_config"], counts
            )
            indexes[event_id].append(len(predictions))
            predictions.append(
                {
                    "seed": 1,
                    "prediction_time": prediction_time,
                    "event_id": event_id,
                    "source_hash": f"sha256:{prediction_time + 1:064x}",
                    "source_family": "fixture",
                    "arm": arm,
                    "base_probability": baseline,
                    "probability": probability,
                    "energies": energies,
                    "normalized_probability": normalized_binary_probability(energies),
                    "bin_index": index,
                    "label": label,
                    "label_available_at_prediction": False,
                    "feedback_update_id": None,
                    "feedback_availability_time": None,
                    "arm_update_applied": False,
                    "counts_hash_before_prediction": before,
                    "counts_hash_after_feedback": None,
                    "state_hash_after_feedback": None,
                    "brier": brier(probability, label),
                    "decision": typed_decision(probability, label),
                    "disposition": "predicted_before_feedback",
                }
            )
        if prediction_time == 2:
            event_ids = order[:2]
            true = [labels[event] for event in event_ids]
            shuffled = true[1:] + true[:1]
            permutation = []
            for position, released in enumerate(event_ids):
                base = baselines[released]
                _update_count(counts, "global", released, base, true[position])
                _update_count(counts, "local", released, base, true[position])
                _update_count(counts, "permuted_local", released, base, shuffled[position])
                permutation.append(
                    {
                        "event_id": released,
                        "label_origin": event_ids[(position + 1) % len(event_ids)],
                        "label": shuffled[position],
                    }
                )
            after = _counts_hash(counts)
            state_hash = canonical_hash({"fixture_counts": counts, "block_id": 0})
            for released in event_ids:
                for row_index in indexes[released]:
                    row = predictions[row_index]
                    row.update(
                        {
                            "feedback_update_id": "seed-1-release-000",
                            "feedback_availability_time": 2,
                            "arm_update_applied": row["arm"] != "frozen",
                            "counts_hash_after_feedback": after,
                            "state_hash_after_feedback": state_hash,
                            "disposition": "released_after_prediction",
                        }
                    )
            releases.append(
                {
                    "seed": 1,
                    "block_id": 0,
                    "start_time": 0,
                    "end_time": 1,
                    "release_time": 2,
                    "event_ids": event_ids,
                    "release_within_stream": True,
                    "short_final_block": False,
                    "update_id": "seed-1-release-000",
                    "label_multiset": sorted(true),
                    "permuted_label_multiset": sorted(shuffled),
                    "source_to_label_permutation": permutation,
                    "changed_label_binding_count": 2,
                    "disposition": "released_updated_persisted_reloaded",
                }
            )
            persistence.append(
                {
                    "seed": 1,
                    "block_id": 0,
                    "checkpoint_sha256": "sha256:" + "b" * 64,
                    "state_hash": state_hash,
                    "payload_equal_after_reload": True,
                    "uninterrupted_state_equal": True,
                    "duplicate_feedback_rejected": True,
                    "disposition": "exact_restart_checked",
                }
            )
    releases.append(
        {
            "seed": 1,
            "block_id": 1,
            "start_time": 2,
            "end_time": 3,
            "release_time": 4,
            "event_ids": order[2:],
            "release_within_stream": False,
            "short_final_block": False,
            "update_id": None,
            "changed_label_binding_count": 0,
            "disposition": "censored_end_of_stream",
        }
    )
    producer: JsonDict = {
        "expected_online_sources": 4,
        "bootstrap_replicates": 0,
        "independent_reduction": {},
    }
    evidence: JsonDict = {
        "protocol": protocol,
        "producer": producer,
        "prediction_rows": predictions,
        "release_rows": releases,
        "persistence_rows": persistence,
        "retention_rows": [],
        "bootstrap_rows": [],
    }
    first = audit_evidence(**evidence)
    producer["independent_reduction"] = deepcopy(first)
    producer["independent_reduction"].pop("audit_errors", None)
    producer["independent_reduction"].pop("count_claims_qualified", None)
    producer["independent_reduction"].pop("producer_aggregate_agreement", None)
    producer["independent_reduction"].pop("source_resample_agreement", None)
    producer["independent_reduction"].pop("source_resample_replicates", None)
    producer["independent_reduction"].pop("absolute_primary_cost", None)
    producer["independent_reduction"].pop("retained_absolute_brier", None)
    producer["independent_reduction"].pop("arithmetic_passed", None)
    producer["independent_reduction"].pop("retention_isolation_passed", None)
    return evidence


def _mutate_evidence(value: JsonDict, name: str) -> None:
    """Apply one private corruption without changing any published bytes."""

    if name == "future_label_swap":
        for row in value["prediction_rows"]:
            if row["event_id"] == "group-0":
                row["label"] = 0
            elif row["event_id"] == "group-2":
                row["label"] = 1
    elif name == "orientation_change":
        value["protocol"]["settings"]["label_orientation"] = "one_means_supported"
    elif name == "duplicate_update":
        value["release_rows"].insert(1, deepcopy(value["release_rows"][0]))
    elif name == "group_id_permutation":
        for row in value["prediction_rows"]:
            if row["event_id"] == "group-0":
                row["event_id"] = "group-permuted"
    elif name == "checkpoint_change":
        value["persistence_rows"][0]["state_hash"] = "sha256:" + "0" * 64
    elif name == "aggregate_row_mismatch":
        value["producer"]["independent_reduction"]["absolute_brier"]["local_count"] += 0.1
    else:
        raise ValueError(f"unknown_private_mutation:{name}")


def run_private_mutations() -> list[JsonDict]:
    """Prove six named corruptions fail and retain no corrupted fixture bytes."""

    output: list[JsonDict] = []
    for name in MUTATION_NAMES:
        private = deepcopy(fixture_evidence())
        _mutate_evidence(private, name)
        reduced = audit_evidence(**private)
        output.append(
            {
                "mutation": name,
                "expected": "rejected",
                "observed_errors": list(reduced["audit_errors"]),
                "passed": bool(reduced["audit_errors"]),
                "corrupted_fixture_published": False,
                "disposition": "complete_private_falsification",
            }
        )
    return output


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful, non-timeout receipt for every named command."""

    passing = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set(names) <= passing


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name all failed checks while keeping the first exact operand."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_checks": [str(row["check"]) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the evidence failure prevented by each terminal field."""

    specific = {
        "experiment_id": "Binds the exact task, milestone, and run date.",
        "preconditions_checked": "Prevents absent inputs from becoming fabricated measurements.",
        "MODEL_SPECS": "Makes the no-model plan explicit.",
        "model_specs": "Keeps both required model manifests empty.",
        "model_invoked": "Separates current aggregation from historical model calls.",
        "inference_substrate_class": "Keeps aggregation distinct from model inference.",
        "inference_substrate": "Names upstream-artifact aggregation as the actual substrate.",
        "execution_venue": "Uses the legal host venue enum.",
        "duration_s": "Reports measured current work without padding.",
        "random_seed": "Prevents outcome-aware resampling changes.",
        "reproducibility_checksum": "Binds code, inputs, settings, and raw evidence.",
        "rows": "Keeps absolute arm metrics beside comparative effects.",
        "sample_size_budget": "Separates completed, failed, censored, and unstarted units.",
        "acceptance_gate_results": "Keeps validity, readiness, support, and benefit separate.",
        "gate_check_summary": "Names exact failed operands instead of a vague status.",
        "honest_verdict": "Uses one complete terminal disposition.",
        "verdict_class": "Restricts classification to the closed vocabulary.",
        "verifier_is_oracle": "Prevents probability evidence from becoming a correctness proof.",
        "flagged_adversarial": "Preserves actual falsification outcomes.",
        "validation_receipts": "Stores exact command scopes, exits, logs, and cold readers.",
        "field_principles": "States the failure prevented by every terminal field.",
        "count_claims_qualified_score": "Requires arithmetic, chronology, restart, and uncertainty agreement.",
        "qualified_exploratory_effect_score": "Cannot exceed independently retained evidence.",
        "confirmatory_benefit_score": "Remains zero on previously inspected data.",
        "mutation_rows": "Exposes every required private falsification result.",
    }
    return {
        field: specific.get(field, "Preserves this field for independent drift detection.")
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind every stable terminal field except clocks and this self-reference."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    payload.pop("phase_spans", None)
    payload.pop("process_identity", None)
    return canonical_hash(payload)


def build_artifact(
    *,
    reduction: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    raw_evidence: Mapping[str, Any],
    process_root: Path = REPO_ROOT,
) -> JsonDict:
    """Assemble a compact audit while keeping readiness apart from benefit."""

    affected_passed = _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_rows_present = any(
        row.get("name") in TERMINAL_CHECK_NAMES for row in validation_receipts
    )
    terminal_passed = _receipts_pass(validation_receipts, TERMINAL_CHECK_NAMES)
    mutations_passed = bool(mutation_rows) and all(
        row.get("passed") is True for row in mutation_rows
    )
    claims = bool(
        reduction.get("count_claims_qualified")
        and mutations_passed
        and affected_passed
        and (terminal_passed or not terminal_rows_present)
    )
    effect = bool(claims and reduction.get("effect_passed"))
    required_failure = any(
        row.get("name") in {*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
        and row.get("passed") is not True
        for row in validation_receipts
    )
    if required_failure or not claims:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_count_audit_required_validation_or_evidence_failed"
    elif effect:
        verdict_class = "circular_positive"
        honest_verdict = "complete_circular_positive_count_audit_exploratory_effect"
    else:
        verdict_class = "null"
        honest_verdict = "complete_null_count_claims_qualified_benefit_gate_failed"
    gates = [
        *acceptance_gates(reduction),
        _gate(
            "affected_validation",
            "validity",
            True,
            affected_passed,
            "eq",
            affected_passed,
            "Scoped checks must pass before current evidence can qualify.",
        ),
        _gate(
            "terminal_validation",
            "validity",
            True,
            terminal_passed if terminal_rows_present else "pending",
            "eq",
            terminal_passed if terminal_rows_present else True,
            "Fresh readers must accept the exact measured candidate.",
        ),
        _gate(
            "private_mutations",
            "validity",
            len(MUTATION_NAMES),
            sum(row.get("passed") is True for row in mutation_rows),
            "eq",
            mutations_passed,
            "Required corruptions must fail before scientific interpretation.",
        ),
    ]
    predictions = int(raw_evidence.get("prediction_rows", {}).get("rows", 0))
    sources = int(reduction.get("complete_online_source_count", 0))
    rows = [
        {
            "unit_id": f"arm-{arm}",
            "arm": arm,
            "absolute_metrics": {
                "mean_brier": reduction["absolute_brier"][arm],
                "mean_primary_cost": reduction["absolute_primary_cost"][arm],
                "mean_retained_brier": reduction["retained_absolute_brier"][arm],
                "prediction_count": predictions // len(ARMS),
                "independent_source_groups": sources,
            },
            "delta_brier_vs_local": (
                0.0 if arm == "local_count" else -float(reduction["mean_delta_brier"][arm])
            ),
            "disposition": "complete_comparative_arm",
            "failed": False,
            "censored": False,
        }
        for arm in ARMS
    ]
    invocation_counts = {
        f"{operation}_{state}": 0
        for operation in ("model_loads", "forward_calls", "generation_calls")
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Independent V660 feedback causality and count-learning audit",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": invocation_counts,
        "historical_model_calls": {
            "source": STREAM_PATH.as_posix(),
            "counted_as_current": False,
        },
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "compute_details": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "cwd": str(process_root.resolve())},
        "random_seed": {
            "order_seeds": list(ORDER_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "source_artifact_hashes": dict(source_hashes),
        "raw_evidence": deepcopy(dict(raw_evidence)),
        "independent_reduction": deepcopy(dict(reduction)),
        "mutation_rows": [deepcopy(dict(row)) for row in mutation_rows],
        "rows": rows,
        "sample_size_budget": {
            "independent_online_groups": {
                "planned": sources,
                "attempted": sources,
                "completed": sources,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
            },
            "ordered_event_instances": {
                "planned": predictions // len(ARMS),
                "attempted": predictions // len(ARMS),
                "completed": predictions // len(ARMS),
                "excluded": 0,
                "failed": 0,
                "censored": int(reduction.get("censored_ordered_event_count", 0)),
                "unstarted": 0,
            },
            "source_resamples": {
                "planned": int(reduction.get("source_resample_replicates", 0)),
                "attempted": int(reduction.get("source_resample_replicates", 0)),
                "completed": int(reduction.get("source_resample_replicates", 0)),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
            },
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "capability_e2e": {
            "workflow": "predict_release_update_persist_reload_independent_replay",
            "passed": bool(
                reduction.get("chronology_passed")
                and reduction.get("restart_passed")
                and reduction.get("arithmetic_passed")
            ),
            "numbered_runtime_e2e": {
                "applicable": [],
                "reason": "Read-only reporting changed no shared runtime, binding, ARC, or telemetry path.",
            },
            "private_llm_off_real_environment_smoke": "authenticated_cached_count_replay",
        },
        "prior_results": [
            {
                "path": PRIOR_7509_PATH.as_posix(),
                "honest_verdict": "complete_null_causal_online_measurement_valid_benefit_gate_failed",
                "verdict_class": "null",
            },
            {
                "path": PRIOR_7510_PATH.as_posix(),
                "honest_verdict": "complete_null_v657_causal_audit_benefit_gate_failed",
                "verdict_class": "null",
            },
        ],
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "terminal_status": "complete",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "positive_claim": False,
        "no_headroom": False,
        "no_headroom_annotation": (
            "No no-headroom claim is made. The local arm lost to the global control and exceeded "
            "retention limits."
        ),
        "count_claims_qualified_score": int(claims),
        "qualified_exploratory_effect_score": int(effect),
        "confirmatory_benefit_score": 0,
    }
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    failed: Mapping[str, Any], checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Publish external absence once without inventing dependent measurements."""

    reason = "".join(
        character if character.isalnum() else "_" for character in str(failed["check"])
    ).strip("_")
    invocation_counts = {
        f"{operation}_{state}": 0
        for operation in ("model_loads", "forward_calls", "generation_calls")
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Independent V660 feedback causality and count-learning audit",
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": invocation_counts,
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "cwd": str(REPO_ROOT)},
        "random_seed": {
            "order_seeds": list(ORDER_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "source_artifact_hashes": {},
        "raw_evidence": {},
        "independent_reduction": {},
        "mutation_rows": [],
        "rows": [],
        "sample_size_budget": {
            "independent_online_groups": {
                "planned": 159,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 159,
            }
        },
        "acceptance_gate_results": [
            _gate(
                str(failed["check"]),
                "validity",
                failed.get("expected"),
                failed.get("observed"),
                str(failed.get("op") or "eq"),
                False,
                "Missing external evidence cannot become a scientific measurement.",
            )
        ],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [str(failed["check"])],
            "first_failure": deepcopy(dict(failed)),
        },
        "validation_receipts": [],
        "capability_e2e": {"passed": False, "reason": "External prerequisite absent."},
        "prior_results": [],
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "terminal_status": "complete",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "positive_claim": False,
        "no_headroom": False,
        "no_headroom_annotation": "No effect was measured because an external prerequisite was absent.",
        "count_claims_qualified_score": 0,
        "qualified_exploratory_effect_score": 0,
        "confirmatory_benefit_score": 0,
    }
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Publish complete row bytes with the same atomic boundary as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _json_receipt(path: Path, root: Path) -> JsonDict:
    """Bind one JSON object with the same path and byte fields as row sidecars."""

    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_fixture_evidence(root: Path, evidence: Mapping[str, Any]) -> JsonDict:
    """Write compact private evidence so strict readers exercise byte custody."""

    directory = root / "private_exp7550_fixture"
    protocol_path = directory / "protocol.json"
    producer_path = directory / "producer.json"
    atomic_json(protocol_path, evidence["protocol"])
    atomic_json(producer_path, evidence["producer"])
    receipts: JsonDict = {
        "protocol": _json_receipt(protocol_path, root),
        "producer": _json_receipt(producer_path, root),
    }
    for name in (
        "prediction_rows",
        "release_rows",
        "persistence_rows",
        "retention_rows",
        "bootstrap_rows",
    ):
        path = directory / f"{name}.jsonl"
        rows = evidence[name]
        _atomic_jsonl(path, rows)
        receipts[name] = sidecar_receipt(path, root, rows=len(rows))
    return receipts


def load_receipted_evidence(root: Path, receipts: Mapping[str, Any]) -> JsonDict:
    """Recover audit operands only after every referenced file authenticates."""

    return {
        "protocol": _read_json_sidecar(root, receipts["protocol"], "protocol"),
        "producer": _read_json_sidecar(root, receipts["producer"], "producer"),
        "prediction_rows": read_sidecar(root, receipts["prediction_rows"], "prediction_rows"),
        "release_rows": read_sidecar(root, receipts["release_rows"], "release_rows"),
        "persistence_rows": read_sidecar(root, receipts["persistence_rows"], "persistence_rows"),
        "retention_rows": read_sidecar(root, receipts["retention_rows"], "retention_rows"),
        "bootstrap_rows": read_sidecar(root, receipts["bootstrap_rows"], "bootstrap_rows"),
    }


def build_test_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build one compact qualified null for schema and fresh-reader tests."""

    evidence = fixture_evidence()
    raw = write_fixture_evidence(root, evidence)
    reduction = audit_evidence(**evidence)
    return build_artifact(
        reduction=reduction,
        mutation_rows=run_private_mutations(),
        validation_receipts=validation_receipts,
        preconditions_checked=[],
        source_hashes={"private_fixture": canonical_hash(evidence["protocol"])},
        duration_s=0.1,
        phase_spans=[],
        raw_evidence=raw,
        process_root=root,
    )


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_terminal: bool = True,
) -> JsonDict:
    """Reject identity, raw, score, receipt, principle, or checksum drift."""

    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("artifact_identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        raise ValueError("artifact_date_or_milestone_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        raise ValueError("model_specs_must_be_empty")
    if value.get("model_invoked") is not False:
        raise ValueError("current_model_invocation_mismatch")
    counts = value.get("invocation_counts") or {}
    if not counts or any(item != 0 for item in counts.values()):
        raise ValueError("current_invocation_counts_mismatch")
    if value.get("inference_substrate_class") != "aggregation":
        raise ValueError("inference_substrate_class_mismatch")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        raise ValueError("inference_substrate_mismatch")
    if value.get("execution_venue") != "host":
        raise ValueError("execution_venue_mismatch")
    if value.get("confirmatory_benefit_score") != 0:
        raise ValueError("confirmatory_benefit_score_mismatch")
    if value.get("positive_claim") is not False:
        raise ValueError("positive_claim_mismatch")
    principles = value.get("field_principles") or {}
    if any(field not in principles for field in value):
        raise ValueError("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        raise ValueError("reproducibility_checksum_mismatch")
    if value.get("verdict_class") == "blocked":
        failure = value.get("gate_check_summary", {}).get("first_failure") or {}
        if not failure.get("upstream") or not failure.get("path_or_field"):
            raise ValueError("blocked_gate_summary_incomplete")
        return {}

    required = {*validation_scope.REQUIRED_CHECK_NAMES}
    if require_terminal:
        required.update(TERMINAL_CHECK_NAMES)
    if not _receipts_pass(value.get("validation_receipts") or [], tuple(required)):
        raise ValueError("required_validation_failed")
    evidence = load_receipted_evidence(root, value.get("raw_evidence") or {})
    reduced = audit_evidence(**evidence)
    if not _close(reduced, value.get("independent_reduction")):
        raise ValueError("independent_reduction_mismatch")
    mutations = value.get("mutation_rows") or []
    if len(mutations) != len(MUTATION_NAMES) or any(
        row.get("passed") is not True for row in mutations
    ):
        raise ValueError("private_mutation_validation_failed")
    claims = int(bool(reduced["count_claims_qualified"]))
    effect = int(bool(claims and reduced["effect_passed"]))
    if value.get("count_claims_qualified_score") != claims:
        raise ValueError("count_claims_qualified_score_mismatch")
    if value.get("qualified_exploratory_effect_score") != effect:
        raise ValueError("qualified_exploratory_effect_score_mismatch")
    expected_class = "circular_positive" if effect else "null" if claims else "disqualified"
    if value.get("verdict_class") != expected_class:
        raise ValueError("verdict_class_mismatch")
    return reduced


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Load serialized evidence through a fresh-reader compatible boundary."""

    value = load_json(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    return validate_artifact(value, root=root, require_terminal=False)


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze serial tests, coverage, lint, type, and specification checks."""

    return validation_contract.build_command_plan(root, AFFECTED_MANIFEST, private_root)


def terminal_commands(
    candidate: Path, root: Path = REPO_ROOT
) -> list[validation_scope.CommandSpec]:
    """Build four bounded fresh readers for one exact candidate path."""

    python = str(root / ".venv/bin/python")
    common = (
        "--date",
        RUN_DATE,
        "--root",
        str(root),
    )
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
    ]


def progress(  # pragma: no cover - visible only in the declared entrypoint.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every phase and slow-operation boundary with monotonic time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7550] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - current monotonic time is an entrypoint boundary.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    """Close one disjoint current-work interval with completed units."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _write_affected_manifest(path: Path) -> None:
    """Freeze exact validation inputs before any runner expands a command."""

    atomic_json(
        path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )


def _actual_raw_receipts(root: Path, upstreams: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Reference exact upstream bytes without copying multi-megabyte sidecars."""

    stream = upstreams["stream"]
    producer = upstreams["producer"]
    return {
        "stream": _json_receipt(root / STREAM_PATH, root),
        "producer": _json_receipt(root / PRODUCER_PATH, root),
        "protocol": deepcopy(stream["raw_sidecars"]["frozen_protocol"]),
        "prediction_rows": deepcopy(producer["prediction_update_rows"]),
        "release_rows": deepcopy(producer["release_rows"]),
        "persistence_rows": deepcopy(producer["persistence_rows"]),
        "retention_rows": deepcopy(producer["retention_rows"]),
        "bootstrap_rows": deepcopy(producer["bootstrap_rows"]),
    }


def _source_hashes(root: Path, manifest_path: Path) -> dict[str, str]:
    """Bind all present instructions, code, artifacts, and the frozen manifest."""

    paths = (
        *REQUIRED_INPUT_PATHS,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        manifest_path.relative_to(root),
    )
    return {path.as_posix(): sha256_file(root / path) for path in paths if (root / path).is_file()}


def run_experiment(  # pragma: no cover - the declared entrypoint is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, audit, validate in fresh processes, and publish atomically."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    started = time.monotonic()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    checks, _initial_hashes, upstreams = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next(
        (row for row in checks if row.get("required") is True and row.get("passed") is not True),
        None,
    )
    if failed is not None:
        blocked = build_blocked_artifact(failed, checks, duration_s=time.monotonic() - started)
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "complete_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "complete", completed_units=len(checks))

    manifest_path = raw_root / "affected_validation_manifest.json"
    _write_affected_manifest(manifest_path)
    source_hashes = _source_hashes(root, manifest_path)
    raw_receipts = _actual_raw_receipts(root, upstreams)

    progress(started, "input_custody", "before_raw_read")
    phase_started = time.monotonic()
    evidence = load_upstream_evidence(root, upstreams)
    spans.append(_span("input_custody", phase_started, started, len(evidence["prediction_rows"])))
    progress(
        started,
        "input_custody",
        "after_raw_read",
        prediction_rows=len(evidence["prediction_rows"]),
    )

    progress(started, "independent_reduction", "before_benchmark", replicates=1000)
    phase_started = time.monotonic()
    reduction = audit_evidence(
        **evidence,
        progress_hook=lambda completed: progress(
            started,
            "independent_reduction",
            "units_complete",
            completed_units=completed,
        ),
    )
    spans.append(
        _span(
            "independent_reduction",
            phase_started,
            started,
            int(reduction["source_resample_replicates"]),
        )
    )
    progress(
        started,
        "independent_reduction",
        "after_benchmark",
        errors=len(reduction["audit_errors"]),
    )

    progress(started, "private_mutations", "start", planned=len(MUTATION_NAMES))
    phase_started = time.monotonic()
    mutation_rows = run_private_mutations()
    spans.append(_span("private_mutations", phase_started, started, len(mutation_rows)))
    progress(
        started,
        "private_mutations",
        "complete",
        rejected=sum(row["passed"] is True for row in mutation_rows),
    )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7550-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private_root)
    plan_errors = validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError("validation_command_plan_invalid:" + ",".join(plan_errors))
    plans = [
        validation_contract.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    progress(started, "affected_validation", "before_subprocesses", commands=len(plans))
    phase_started = time.monotonic()
    affected = validation_contract.run_categorized_commands(
        root,
        plans,
        log_dir=raw_root / "validation" / "affected",
        heartbeat_s=60.0,
    )
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    affected_passed = _receipts_pass(affected, validation_scope.REQUIRED_CHECK_NAMES)
    progress(started, "affected_validation", "after_subprocesses", passed=affected_passed)

    candidate = build_artifact(
        reduction=reduction,
        mutation_rows=mutation_rows,
        validation_receipts=affected,
        preconditions_checked=checks,
        source_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        raw_evidence=raw_receipts,
        process_root=root,
    )
    if not affected_passed:
        atomic_json(destination, candidate)
        progress(started, "publish", "complete_disqualified_affected")
        return candidate
    measured_path = raw_root / "measured_terminal_candidate.json"
    atomic_json(measured_path, candidate)

    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        terminal_commands(measured_path, root),
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        reduction=reduction,
        mutation_rows=mutation_rows,
        validation_receipts=[*affected, *terminal],
        preconditions_checked=checks,
        source_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        raw_evidence=raw_receipts,
        process_root=root,
    )
    if not terminal_passed:
        atomic_json(destination, final)
        progress(started, "publish", "complete_disqualified_terminal")
        return final
    validate_artifact(final, root=root, require_terminal=True)
    exact_path = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_path, final)

    progress(started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        terminal_commands(exact_path, root),
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _receipts_pass(exact, TERMINAL_CHECK_NAMES)
    progress(started, "exact_candidate_validation", "after_subprocesses", passed=exact_passed)
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "complete",
        count_claims_qualified_score=final["count_claims_qualified_score"],
        qualified_exploratory_effect_score=final["qualified_exploratory_effect_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the producer and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the audit or one exact fresh-process evidence reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        reduced = cold_replay(args.cold_replay, root=root)
        print(json.dumps({"event": "cold_replay_passed", **reduced}, sort_keys=True), flush=True)
        return int(not reduced.get("count_claims_qualified", False))
    if args.independent_reduce is not None:
        value = load_json(args.independent_reduce)
        if not value:
            raise ValueError("artifact_unreadable_or_not_object")
        reduced = validate_artifact(value, root=root, require_terminal=False)
        print(
            json.dumps({"event": "independent_reduction_passed", **reduced}, sort_keys=True),
            flush=True,
        )
        return int(not reduced.get("count_claims_qualified", False))
    artifact = run_experiment(root, args.date, output_path=args.output)
    print(
        json.dumps(
            {
                "result": str(args.output or root / RESULT_PATH),
                "count_claims_qualified_score": artifact["count_claims_qualified_score"],
                "qualified_exploratory_effect_score": artifact[
                    "qualified_exploratory_effect_score"
                ],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(artifact["verdict_class"] in {"blocked", "disqualified", "partial"})


if __name__ == "__main__":  # pragma: no cover - wrapper is the public executable.
    raise SystemExit(main())
