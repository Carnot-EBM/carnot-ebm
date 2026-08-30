"""Cold-audit temporal exchange and map its logical cost to board receipts.

This evaluator replays serialized Exp6793 inputs in a separate module. It does
not import the source experiment or sampler. The hardware section reads old
receipts and RTL formats only. It never invokes synthesis or a physical board.

Spec: REQ-SAMPLE-100, SCENARIO-SAMPLE-100, SCENARIO-SAMPLE-101,
SCENARIO-SAMPLE-102.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path("results/experiment_6793_temporal_exchange_ising_ab.json")
KV260_RECEIPT_PATH = Path("results/experiment_2477_kv260_bitstream_flash.json")
GATEMATE_RECEIPT_PATH = Path("results/experiment_3866_gatemate_ising_tile_flash_v2.json")
WISHLIST_PATH = Path("research-hardware-wishlist.md")
BRINGUP_PATH = Path("ops/hardware-bringup-prep.md")
RESULT_PATH = Path("results/experiment_6794_temporal_exchange_cold_hardware_audit.json")
MODULE_PATH = Path("python/carnot/experiment_6794_temporal_exchange_cold_hardware_audit.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_6794_temporal_exchange_cold_hardware_audit.py")

EXPERIMENT_ID = "experiment_6794_temporal_exchange_cold_hardware_audit"
SCHEMA_VERSION = "carnot.experiment_6794.temporal_exchange_cold_hardware_audit.v1"
INFERENCE_SUBSTRATE = "fresh-process CPU audit and static board-envelope mapping"
SPEC_REFS = ["REQ-SAMPLE-100", "SCENARIO-SAMPLE-100", "SCENARIO-SAMPLE-101", "SCENARIO-SAMPLE-102"]
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
EVIDENCE_CLASSES = {
    "measured_simulator",
    "derived_estimate",
    "existing_board_receipt",
    "unavailable",
}
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

REQUIRED_FIELDS = {
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hash",
    "existing_hardware_receipt_hashes",
    "rows",
    "exact_target_recomputation",
    "cold_recomputed_metrics",
    "headline_differences",
    "update_accounting_by_arm",
    "denominator_sensitivity",
    "burnin_sensitivity",
    "thinning_sensitivity",
    "initialization_sensitivity",
    "coupling_sensitivity",
    "stationarity_checks",
    "zero_coupling_equivalence",
    "estimated_state_cost",
    "estimated_arithmetic_cost",
    "estimated_memory_traffic",
    "coefficient_precision_range",
    "evidence_class_by_hardware_field",
    "physical_hardware_invoked",
    "source_verdict_supported",
    "temporal_exchange_audit_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}


class TemporalExchangeAuditError(RuntimeError):
    """Stop when cold evidence cannot satisfy the declared artifact schema."""


def canonical_json(value: Any) -> str:
    """Serialize finite evidence in one stable form for every receipt."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TemporalExchangeAuditError("evidence must be finite canonical JSON") from exc


def json_digest(value: Any) -> str:
    """Return a typed SHA-256 digest of canonical JSON evidence."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Hash one required file while keeping a missing file visible."""

    return (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    )


def load_json(path: Path | str) -> JsonDict:
    """Read one JSON object without importing its producing module."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TemporalExchangeAuditError("source artifact must be one JSON object")
    return value


def _rounded(value: float) -> float:
    """Keep stable precision above all audit comparison tolerances."""

    result = float(f"{float(value):.15g}")
    return 0.0 if result == 0.0 else result


def _source_row_digest(row: Mapping[str, Any]) -> str:
    """Rebuild the source row receipt from serialized row content."""

    return json_digest(
        {
            key: deepcopy(value)
            for key, value in row.items()
            if key not in {"row_sha256", "wall_time_s"}
        }
    )


def _hardware_receipt_hashes() -> JsonDict:
    """Bind the two board artifacts and the documents that define their limits."""

    return {
        KV260_RECEIPT_PATH.as_posix(): file_digest(REPO_ROOT / KV260_RECEIPT_PATH),
        GATEMATE_RECEIPT_PATH.as_posix(): file_digest(REPO_ROOT / GATEMATE_RECEIPT_PATH),
        WISHLIST_PATH.as_posix(): file_digest(REPO_ROOT / WISHLIST_PATH),
        BRINGUP_PATH.as_posix(): file_digest(REPO_ROOT / BRINGUP_PATH),
    }


def check_preconditions(source: Mapping[str, Any]) -> list[JsonDict]:
    """Fail closed on missing completion, sample, code, work, or board receipts."""

    rows = source.get("rows") if isinstance(source.get("rows"), list) else []
    implementation = source.get("implementation_receipt", {})
    module_path = REPO_ROOT / str(implementation.get("module_path", "missing"))
    sampler_path = REPO_ROOT / str(implementation.get("sampler_module_path", "missing"))
    declared_module = implementation.get("module_sha256")
    declared_sampler = implementation.get("sampler_module_sha256")
    code_observed = {
        "module_sha256": file_digest(module_path),
        "sampler_module_sha256": file_digest(sampler_path),
    }
    code_ok = bool(
        declared_module == code_observed["module_sha256"]
        and declared_sampler == code_observed["sampler_module_sha256"]
    )

    invalid_stat_rows = []
    for row in rows:
        marginal = row.get("empirical_marginal")
        hashes_present = all(
            isinstance(row.get(key), str) and str(row[key]).startswith("sha256:")
            for key in (
                "row_sha256",
                "trajectory_sha256",
                "empirical_marginal_sha256",
                "exact_target_sha256",
            )
        )
        valid = bool(
            hashes_present
            and isinstance(marginal, list)
            and row.get("empirical_marginal_sha256") == json_digest(marginal)
            and row.get("row_sha256") == _source_row_digest(row)
            and isinstance(row.get("energy_trace"), list)
            and len(row["energy_trace"]) == int(row.get("collected_samples", -1))
        )
        if not valid:
            invalid_stat_rows.append(row.get("row_id"))

    target_hashes = source.get("exact_target_hashes", {})
    row_target_hashes = {row.get("exact_target_sha256") for row in rows}
    target_ok = bool(
        isinstance(target_hashes, dict)
        and len(target_hashes) == 6
        and set(target_hashes.values()) == row_target_hashes
    )

    arm_updates = defaultdict(int)
    invalid_update_rows = []
    for row in rows:
        update_count = row.get("update_count")
        collections = row.get("collection_update_counts")
        valid = bool(
            isinstance(update_count, int)
            and update_count > 0
            and isinstance(collections, list)
            and len(collections) == int(row.get("collected_samples", -1))
            and collections
            and collections[-1] == update_count
        )
        if valid:
            arm_updates[str(row.get("arm"))] += update_count
        else:
            invalid_update_rows.append(row.get("row_id"))
    declared_updates = source.get("update_budget_by_arm", {})
    update_ok = not invalid_update_rows and all(
        declared_updates.get(arm, {}).get("headline_updates") == total
        for arm, total in arm_updates.items()
    )

    hardware_hashes = _hardware_receipt_hashes()
    board_ok = all(
        hardware_hashes[path.as_posix()] != "missing"
        for path in (KV260_RECEIPT_PATH, GATEMATE_RECEIPT_PATH)
    )
    return [
        {
            "check": "source_comparison_completed",
            "passed": source.get("temporal_exchange_comparison_completed") is True,
            "expected": True,
            "observed": source.get("temporal_exchange_comparison_completed"),
        },
        {
            "check": "source_code_identity",
            "passed": code_ok,
            "expected": {
                "module_sha256": declared_module,
                "sampler_module_sha256": declared_sampler,
            },
            "observed": code_observed,
        },
        {
            "check": "raw_or_sufficient_statistic_hashes",
            "passed": bool(rows) and not invalid_stat_rows,
            "expected": {"row_count": 360, "invalid_count": 0},
            "observed": {"row_count": len(rows), "invalid_row_ids": invalid_stat_rows},
        },
        {
            "check": "exact_target_hashes",
            "passed": target_ok,
            "expected": {"target_count": 6, "all_row_hashes_referenced": True},
            "observed": {
                "target_count": len(target_hashes),
                "row_target_hash_count": len(row_target_hashes),
            },
        },
        {
            "check": "update_receipts",
            "passed": update_ok,
            "expected": {"invalid_count": 0, "arm_totals_match": True},
            "observed": {
                "invalid_row_ids": invalid_update_rows,
                "recounted_headline_updates": dict(arm_updates),
            },
        },
        {
            "check": "existing_board_receipts",
            "passed": board_ok,
            "expected": {"kv260_present": True, "gatemate_present": True},
            "observed": hardware_hashes,
        },
    ]


def _graph_arrays(graph: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build one symmetric matrix from a serialized graph receipt."""

    biases = np.asarray(graph["biases"], dtype=np.float64)
    couplings = np.zeros((biases.size, biases.size), dtype=np.float64)
    for left, right, weight in graph["edges"]:
        couplings[int(left), int(right)] = couplings[int(right), int(left)] = float(weight)
    return biases, couplings


def _state_labels(states: np.ndarray) -> list[str]:
    """Use source spin order while deriving labels independently."""

    return ["".join("+" if spin > 0 else "-" for spin in state) for state in states]


def _energy(state: np.ndarray, biases: np.ndarray, couplings: np.ndarray) -> float:
    """Compute the spatial Ising energy without a temporal term."""

    return float(-biases @ state - 0.5 * state @ couplings @ state)


def independent_exact_target(graph: Mapping[str, Any], temperature: float) -> JsonDict:
    """Enumerate one spatial Boltzmann law without calling source helpers."""

    biases, couplings = _graph_arrays(graph)
    state_count = 1 << biases.size
    indices = np.arange(state_count, dtype=np.uint64)[:, None]
    bit_positions = np.arange(biases.size, dtype=np.uint64)[None, :]
    states = np.where(((indices >> bit_positions) & 1) == 1, 1, -1).astype(np.int8)
    energies = -states @ biases - 0.5 * np.einsum("bi,ij,bj->b", states, couplings, states)
    log_weights = -energies / float(temperature)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    probabilities = weights / float(np.sum(weights))
    target: JsonDict = {
        "graph_id": graph["graph_id"],
        "temperature": float(temperature),
        "n_spins": int(biases.size),
        "exact": True,
        "state_order": _state_labels(states),
        "probabilities": [_rounded(value) for value in probabilities],
        "energies": [_rounded(value) for value in energies],
        "expected_energy": _rounded(float(probabilities @ energies)),
        "expected_magnetization": _rounded(float(probabilities @ np.mean(states, axis=1))),
        "optimum_energy": _rounded(float(np.min(energies))),
    }
    target["target_sha256"] = json_digest(target)
    target["_states"] = states
    target["_probabilities"] = probabilities
    target["_energies"] = np.asarray(energies, dtype=np.float64)
    return target


def _logistic(value: float) -> float:
    """Evaluate one conditional probability without overflow."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _trajectory_digest(samples: np.ndarray, energy_trace: np.ndarray) -> str:
    """Rebuild the source trajectory receipt from replayed raw samples."""

    payload = np.asarray(samples, dtype=np.int8).tobytes(order="C")
    payload += np.asarray(energy_trace, dtype="<f8").tobytes(order="C")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _autocorrelation(values: Sequence[float], maximum_lag: int = 100) -> JsonDict:
    """Estimate integrated autocorrelation through the first nonpositive lag."""

    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array))
    variance = float(np.dot(centered, centered) / array.size)
    if variance <= 1.0e-24:
        return {
            "integrated_time": _rounded(float(array.size)),
            "effective_samples": 1.0,
            "positive_lag_count": 0,
        }
    correlation_sum = 0.0
    positive_lags = 0
    for lag in range(1, min(maximum_lag, array.size - 1) + 1):
        correlation = float(np.dot(centered[:-lag], centered[lag:]) / (array.size - lag) / variance)
        if correlation <= 0.0:
            break
        correlation_sum += correlation
        positive_lags += 1
    integrated = min(float(array.size), max(1.0, 1.0 + 2.0 * correlation_sum))
    return {
        "integrated_time": _rounded(integrated),
        "effective_samples": _rounded(float(array.size) / integrated),
        "positive_lag_count": positive_lags,
    }


def _state_indices(samples: np.ndarray) -> np.ndarray:
    """Map bipolar samples to the target's little-endian support order."""

    powers = (1 << np.arange(samples.shape[1], dtype=np.int64)).reshape((-1, 1))
    return ((samples > 0).astype(np.int64) @ powers).ravel()


def _diversity(probabilities: np.ndarray) -> JsonDict:
    """Recompute support and entropy from one empirical marginal."""

    nonzero = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return {
        "unique_state_count": int(nonzero.size),
        "unique_state_rate": _rounded(float(nonzero.size) / probabilities.size),
        "shannon_entropy_nats": _rounded(entropy),
        "effective_support": _rounded(math.exp(entropy)),
    }


def _replay_chain(
    *,
    graph: Mapping[str, Any],
    current: Sequence[int],
    previous: Sequence[int],
    temperature: float,
    temporal_coupling: float,
    arm: str,
    seed: int,
    burn_in_sweeps: int,
    n_samples: int,
    sweeps_per_sample: int,
    optimum_energy: float,
) -> JsonDict:
    """Replay one chain and count logical work at each attempted update."""

    biases, couplings = _graph_arrays(graph)
    state = np.asarray(current, dtype=np.int8).copy()
    prior = np.asarray(previous, dtype=np.int8).copy()
    generator = np.random.default_rng(int(seed))
    n_spins = int(biases.size)
    burn_updates = int(burn_in_sweeps) * n_spins
    interval_updates = int(sweeps_per_sample) * n_spins
    total_updates = burn_updates + int(n_samples) * interval_updates
    samples = np.empty((int(n_samples), n_spins), dtype=np.int8)
    energy_trace = np.empty(int(n_samples), dtype=np.float64)
    magnetization_trace = np.empty(int(n_samples), dtype=np.float64)
    collection_updates = np.empty(int(n_samples), dtype=np.int64)
    degrees = np.count_nonzero(couplings, axis=1)

    spatial_energy = _energy(state, biases, couplings)
    best_energy = spatial_energy
    best_state = state.copy()
    hit = 0 if spatial_energy <= float(optimum_energy) + 1.0e-12 else None
    accepted = 0
    neighbor_reads = 0
    sample_index = 0
    sweep_order = np.arange(n_spins, dtype=np.int64)
    sweep_position = 0
    for attempt in range(total_updates):
        if sweep_position == 0:
            sweep_order = np.asarray(generator.permutation(n_spins), dtype=np.int64)
        site = int(sweep_order[sweep_position])
        old_spin = int(state[site])
        spatial_field = float(biases[site] + couplings[site] @ state)
        total_field = spatial_field + float(temporal_coupling) * float(prior[site])
        new_spin = (
            1 if generator.random() < _logistic(2.0 * total_field / float(temperature)) else -1
        )
        neighbor_reads += int(degrees[site])
        if new_spin != old_spin:
            state[site] = new_spin
            spatial_energy += 2.0 * old_spin * spatial_field
            accepted += 1
        update_count = attempt + 1
        sweep_position += 1
        if sweep_position == n_spins:
            prior = state.copy()
            sweep_position = 0
        if spatial_energy < best_energy - 1.0e-12:
            best_energy = spatial_energy
            best_state = state.copy()
        if hit is None and spatial_energy <= float(optimum_energy) + 1.0e-12:
            hit = update_count
        after_burn = update_count - burn_updates
        if after_burn > 0 and after_burn % interval_updates == 0:
            spatial_energy = _energy(state, biases, couplings)
            samples[sample_index] = state
            energy_trace[sample_index] = spatial_energy
            magnetization_trace[sample_index] = float(np.mean(state))
            collection_updates[sample_index] = update_count
            sample_index += 1

    sweep_count = total_updates // n_spins
    uses_temporal = arm != "ordinary_gibbs"
    previous_reads = total_updates if uses_temporal else 0
    snapshot_reads = sweep_count * n_spins
    snapshot_writes = sweep_count * n_spins
    spatial_adds = neighbor_reads + total_updates
    temporal_operations = total_updates if uses_temporal else 0
    arithmetic_proxy = spatial_adds + 2 * temporal_operations
    return {
        "samples": samples,
        "energy_trace": energy_trace,
        "magnetization_trace": magnetization_trace,
        "collection_update_counts": collection_updates,
        "trajectory_sha256": _trajectory_digest(samples, energy_trace),
        "best_energy": _rounded(best_energy),
        "best_state": best_state.astype(int).tolist(),
        "optimum_hitting_updates": hit,
        "attempted_conditional_updates": total_updates,
        "accepted_state_changes": accepted,
        "temporal_coupling_operations": temporal_operations,
        "random_uniform_draws": total_updates,
        "random_permutation_calls": sweep_count,
        "random_api_calls": total_updates + sweep_count,
        "logical_current_or_neighbor_reads": neighbor_reads + total_updates + snapshot_reads,
        "logical_previous_state_reads": previous_reads,
        "logical_stored_state_reads": neighbor_reads
        + total_updates
        + snapshot_reads
        + previous_reads,
        "logical_current_state_writes": accepted,
        "logical_previous_snapshot_writes": snapshot_writes,
        "logical_stored_state_writes": accepted + snapshot_writes,
        "spatial_add_operations": spatial_adds,
        "arithmetic_operation_proxy": arithmetic_proxy,
    }


def replay_row(
    row: Mapping[str, Any], graph: Mapping[str, Any], target: Mapping[str, Any]
) -> JsonDict:
    """Replay one serialized source row with the independent chain."""

    return _replay_chain(
        graph=graph,
        current=row["initial_state_pair"]["current"],
        previous=row["initial_state_pair"]["previous"],
        temperature=float(row["temperature"]),
        temporal_coupling=float(row["temporal_coupling"]),
        arm=str(row["arm"]),
        seed=int(row["sampler_seed"]),
        burn_in_sweeps=int(row["burn_in_sweeps"]),
        n_samples=int(row["collected_samples"]),
        sweeps_per_sample=int(row["sweeps_per_sample"]),
        optimum_energy=float(target["optimum_energy"]),
    )


def _audit_row_digest(row: Mapping[str, Any]) -> str:
    """Bind one cold row without including its self-digest."""

    return json_digest(
        {key: deepcopy(value) for key, value in row.items() if key != "audit_row_sha256"}
    )


def recompute_source_row(
    source_row: Mapping[str, Any],
    graph: Mapping[str, Any],
    target: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> JsonDict:
    """Reduce replayed raw samples into independently recomputed metrics."""

    samples = np.asarray(replay["samples"], dtype=np.int8)
    energy_trace = np.asarray(replay["energy_trace"], dtype=np.float64)
    magnetization_trace = np.asarray(replay["magnetization_trace"], dtype=np.float64)
    indices = _state_indices(samples)
    empirical = np.bincount(indices, minlength=len(target["probabilities"])).astype(np.float64)
    empirical /= float(samples.shape[0])
    target_probabilities = np.asarray(target["_probabilities"], dtype=np.float64)
    target_energies = np.asarray(target["_energies"], dtype=np.float64)
    states = np.asarray(target["_states"], dtype=np.int8)
    energy_autocorrelation = _autocorrelation(energy_trace)
    magnetization_autocorrelation = _autocorrelation(magnetization_trace)
    attempted = int(replay["attempted_conditional_updates"])
    row: JsonDict = {
        "row_kind": "source_recomputation",
        "source_row_id": source_row["row_id"],
        "graph_id": source_row["graph_id"],
        "graph_family": source_row["graph_family"],
        "temperature": source_row["temperature"],
        "seed": source_row["seed"],
        "arm": source_row["arm"],
        "temporal_coupling": source_row["temporal_coupling"],
        "source_row_sha256": source_row["row_sha256"],
        "source_row_hash_valid": source_row["row_sha256"] == _source_row_digest(source_row),
        "source_trajectory_sha256": source_row["trajectory_sha256"],
        "replayed_trajectory_sha256": replay["trajectory_sha256"],
        "trajectory_hash_match": source_row["trajectory_sha256"] == replay["trajectory_sha256"],
        "source_exact_target_sha256": source_row["exact_target_sha256"],
        "recomputed_exact_target_sha256": target["target_sha256"],
        "exact_target_hash_match": source_row["exact_target_sha256"] == target["target_sha256"],
        "target_total_variation": _rounded(
            0.5 * float(np.sum(np.abs(empirical - target_probabilities)))
        ),
        "energy_error": _rounded(
            abs(float(empirical @ target_energies) - float(target_probabilities @ target_energies))
        ),
        "magnetization_error": _rounded(
            abs(
                float(empirical @ np.mean(states, axis=1))
                - float(target_probabilities @ np.mean(states, axis=1))
            )
        ),
        "autocorrelation": {
            "energy": energy_autocorrelation,
            "magnetization": magnetization_autocorrelation,
        },
        "effective_samples": {
            "energy": energy_autocorrelation["effective_samples"],
            "magnetization": magnetization_autocorrelation["effective_samples"],
        },
        "energy_effective_samples_per_attempted_update": _rounded(
            float(energy_autocorrelation["effective_samples"]) / attempted
        ),
        "optimum_hitting_updates": replay["optimum_hitting_updates"],
        "diversity": _diversity(empirical),
        "empirical_marginal_sha256": json_digest([_rounded(value) for value in empirical]),
        "update_accounting": {
            key: replay[key]
            for key in (
                "attempted_conditional_updates",
                "accepted_state_changes",
                "temporal_coupling_operations",
                "random_uniform_draws",
                "random_permutation_calls",
                "random_api_calls",
                "logical_current_or_neighbor_reads",
                "logical_previous_state_reads",
                "logical_stored_state_reads",
                "logical_current_state_writes",
                "logical_previous_snapshot_writes",
                "logical_stored_state_writes",
                "spatial_add_operations",
                "arithmetic_operation_proxy",
            )
        },
    }
    row["audit_row_sha256"] = _audit_row_digest(row)
    return row


def replay_all_source_rows(source: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay the full source grid and retain one cold row per source row."""

    graphs = {
        item["graph_id"]: item
        for item in source["graph_families"]
        if item.get("exact_target_enumerated")
    }
    targets: dict[tuple[str, float], JsonDict] = {}
    rows = []
    for source_row in source["rows"]:
        key = (str(source_row["graph_id"]), float(source_row["temperature"]))
        if key not in targets:
            targets[key] = independent_exact_target(graphs[key[0]], key[1])
        replay = replay_row(source_row, graphs[key[0]], targets[key])
        rows.append(recompute_source_row(source_row, graphs[key[0]], targets[key], replay))
    target_rows = [
        {
            "graph_id": graph_id,
            "temperature": temperature,
            "target_sha256": target["target_sha256"],
            "source_target_sha256": source["exact_target_hashes"][
                f"{graph_id}:T={temperature:.2f}"
            ],
            "hash_match": target["target_sha256"]
            == source["exact_target_hashes"][f"{graph_id}:T={temperature:.2f}"],
            "normalization_sum": _rounded(float(np.sum(target["_probabilities"]))),
            "state_count": len(target["probabilities"]),
            "expected_energy": target["expected_energy"],
            "expected_magnetization": target["expected_magnetization"],
        }
        for (graph_id, temperature), target in sorted(targets.items())
    ]
    return rows, target_rows


def _mean_ci(values: Sequence[float]) -> JsonDict:
    """Return a paired or unpaired normal 95 percent interval."""

    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": _rounded(mean),
        "ci95": [_rounded(mean - 1.96 * standard_error), _rounded(mean + 1.96 * standard_error)],
    }


def derive_cold_metrics(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate only cold rows within graph, temperature, and arm strata."""

    groups: dict[tuple[str, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["graph_id"]), float(row["temperature"]), str(row["arm"]))].append(row)
    result = []
    for (graph_id, temperature, arm), group in sorted(groups.items()):
        result.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                "arm": arm,
                "target_total_variation": _mean_ci(
                    [float(item["target_total_variation"]) for item in group]
                ),
                "energy_effective_samples_per_attempted_update": _mean_ci(
                    [float(item["energy_effective_samples_per_attempted_update"]) for item in group]
                ),
                "energy_integrated_autocorrelation_time": _mean_ci(
                    [float(item["autocorrelation"]["energy"]["integrated_time"]) for item in group]
                ),
                "magnetization_integrated_autocorrelation_time": _mean_ci(
                    [
                        float(item["autocorrelation"]["magnetization"]["integrated_time"])
                        for item in group
                    ]
                ),
                "optimum_hitting_updates": _mean_ci(
                    [float(item["optimum_hitting_updates"]) for item in group]
                ),
                "unique_state_count": _mean_ci(
                    [float(item["diversity"]["unique_state_count"]) for item in group]
                ),
            }
        )
    return result


def _headline_differences(source: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare cold values with row-level source values, never source headlines."""

    source_index = {row["row_id"]: row for row in source["rows"]}
    fields = {
        "target_total_variation": [],
        "energy_error": [],
        "magnetization_error": [],
        "energy_effective_samples": [],
        "magnetization_effective_samples": [],
    }
    for row in rows:
        original = source_index[row["source_row_id"]]
        fields["target_total_variation"].append(
            abs(float(row["target_total_variation"]) - float(original["target_total_variation"]))
        )
        fields["energy_error"].append(
            abs(float(row["energy_error"]) - float(original["energy_error"]))
        )
        fields["magnetization_error"].append(
            abs(float(row["magnetization_error"]) - float(original["magnetization_error"]))
        )
        fields["energy_effective_samples"].append(
            abs(
                float(row["effective_samples"]["energy"])
                - float(original["effective_samples"]["energy"])
            )
        )
        fields["magnetization_effective_samples"].append(
            abs(
                float(row["effective_samples"]["magnetization"])
                - float(original["effective_samples"]["magnetization"])
            )
        )
    return {
        "comparison_basis": "cold recomputation versus source row fields; source headline aggregates not imported",
        "maximum_absolute_difference": {
            key: _rounded(max(values, default=0.0)) for key, values in fields.items()
        },
        "trajectory_hash_mismatch_count": sum(not row["trajectory_hash_match"] for row in rows),
        "exact_target_hash_mismatch_count": sum(not row["exact_target_hash_match"] for row in rows),
        "source_row_hash_invalid_count": sum(not row["source_row_hash_valid"] for row in rows),
    }


def _update_accounting_by_arm(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Sum every logical work counter by arm."""

    result: JsonDict = {}
    for arm in sorted({str(row["arm"]) for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        keys = arm_rows[0]["update_accounting"]
        result[arm] = {
            "row_count": len(arm_rows),
            **{key: sum(int(row["update_accounting"][key]) for row in arm_rows) for key in keys},
        }
    return result


def _paired_endpoint(rows: Sequence[Mapping[str, Any]], denominator: str) -> list[JsonDict]:
    """Compare temporal and Gibbs ESS under one explicit work denominator."""

    index = {(row["graph_id"], row["temperature"], row["seed"], row["arm"]): row for row in rows}
    result = []
    strata = sorted({(str(row["graph_id"]), float(row["temperature"])) for row in rows})
    for graph_id, temperature in strata:
        deltas = []
        for seed in sorted(
            {
                int(row["seed"])
                for row in rows
                if row["graph_id"] == graph_id and row["temperature"] == temperature
            }
        ):
            ordinary = index[(graph_id, temperature, seed, "ordinary_gibbs")]
            temporal = index[(graph_id, temperature, seed, "temporal_exchange")]
            if denominator in {
                "attempted_conditional_updates",
                "accepted_state_changes",
                "random_api_calls",
            }:
                ordinary_denominator = ordinary["update_accounting"][denominator]
                temporal_denominator = temporal["update_accounting"][denominator]
            elif denominator == "logical_stored_state_accesses":
                ordinary_denominator = (
                    ordinary["update_accounting"]["logical_stored_state_reads"]
                    + ordinary["update_accounting"]["logical_stored_state_writes"]
                )
                temporal_denominator = (
                    temporal["update_accounting"]["logical_stored_state_reads"]
                    + temporal["update_accounting"]["logical_stored_state_writes"]
                )
            else:
                ordinary_denominator = ordinary["update_accounting"]["arithmetic_operation_proxy"]
                temporal_denominator = temporal["update_accounting"]["arithmetic_operation_proxy"]
            deltas.append(
                float(temporal["effective_samples"]["energy"]) / temporal_denominator
                - float(ordinary["effective_samples"]["energy"]) / ordinary_denominator
            )
        interval = _mean_ci(deltas)
        result.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                **interval,
                "lower_bound_above_zero": interval["ci95"][0] > 0.0,
            }
        )
    return result


def derive_denominator_sensitivity(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Show whether a temporal advantage survives less favorable work counts."""

    denominators = [
        "attempted_conditional_updates",
        "accepted_state_changes",
        "random_api_calls",
        "logical_stored_state_accesses",
        "arithmetic_operation_proxy",
    ]
    comparisons = {name: _paired_endpoint(rows, name) for name in denominators}
    baseline = {
        (row["graph_id"], row["temperature"]): row["lower_bound_above_zero"]
        for row in comparisons["attempted_conditional_updates"]
    }
    changes = []
    for name, items in comparisons.items():
        for row in items:
            key = (row["graph_id"], row["temperature"])
            if row["lower_bound_above_zero"] != baseline[key]:
                changes.append({"denominator": name, "graph_id": key[0], "temperature": key[1]})
    return {
        "comparisons": comparisons,
        "favorable_source_denominator": "attempted_conditional_updates",
        "endpoint_depends_on_favorable_denominator": bool(changes),
        "changed_strata": changes,
        "source_all_strata_positive_under_attempted_updates": all(baseline.values()),
    }


def _sensitivity_row(
    *,
    axis: str,
    value: Any,
    graph: Mapping[str, Any],
    target: Mapping[str, Any],
    current: Sequence[int],
    previous: Sequence[int],
    temperature: float,
    coupling: float,
    arm: str,
    seed: int,
    burn: int,
    samples: int,
    thin: int,
) -> tuple[JsonDict, JsonDict]:
    """Run and reduce one bounded sensitivity condition."""

    replay = _replay_chain(
        graph=graph,
        current=current,
        previous=previous,
        temperature=temperature,
        temporal_coupling=coupling,
        arm=arm,
        seed=seed,
        burn_in_sweeps=burn,
        n_samples=samples,
        sweeps_per_sample=thin,
        optimum_energy=float(target["optimum_energy"]),
    )
    indices = _state_indices(np.asarray(replay["samples"]))
    empirical = np.bincount(indices, minlength=len(target["probabilities"])).astype(np.float64)
    empirical /= float(samples)
    energy_ac = _autocorrelation(replay["energy_trace"])
    row: JsonDict = {
        "row_kind": "sensitivity",
        "sensitivity_axis": axis,
        "sensitivity_value": value,
        "graph_id": graph["graph_id"],
        "temperature": temperature,
        "temporal_coupling": coupling,
        "arm": arm,
        "seed": seed,
        "burn_in_sweeps": burn,
        "thinning_sweeps": thin,
        "collected_samples": samples,
        "target_total_variation": _rounded(
            0.5 * float(np.sum(np.abs(empirical - np.asarray(target["_probabilities"]))))
        ),
        "energy_integrated_autocorrelation_time": energy_ac["integrated_time"],
        "energy_effective_samples": energy_ac["effective_samples"],
        "energy_effective_samples_per_attempted_update": _rounded(
            float(energy_ac["effective_samples"]) / replay["attempted_conditional_updates"]
        ),
        "optimum_hitting_updates": replay["optimum_hitting_updates"],
        "diversity": _diversity(empirical),
        "trajectory_sha256": replay["trajectory_sha256"],
    }
    row["audit_row_sha256"] = _audit_row_digest(row)
    return row, replay


def run_sensitivity_panel(source: Mapping[str, Any]) -> JsonDict:
    """Vary one bounded factor at a time and run two stationarity chains."""

    graph = next(
        item for item in source["graph_families"] if item["graph_id"] == "ferromagnetic_ring_n6"
    )
    base_source = next(
        row
        for row in source["rows"]
        if row["graph_id"] == graph["graph_id"]
        and row["temperature"] == 0.75
        and row["seed"] == 679300
        and row["arm"] == "temporal_exchange"
    )
    current = base_source["initial_state_pair"]["current"]
    previous = base_source["initial_state_pair"]["previous"]
    seed = int(base_source["sampler_seed"])
    target_cache: dict[float, JsonDict] = {}

    def target(temperature: float) -> JsonDict:
        if temperature not in target_cache:
            target_cache[temperature] = independent_exact_target(graph, temperature)
        return target_cache[temperature]

    conditions = []
    conditions.extend(
        ("burn_in_sweeps", value, value, 1, 256, previous, 0.75, -0.08)
        for value in (0, 64, 128, 256)
    )
    conditions.extend(
        ("thinning_sweeps", value, 128, value, 256, previous, 0.75, -0.08) for value in (1, 2, 4)
    )
    conditions.extend(
        (
            "initial_previous_state",
            label,
            128,
            1,
            256,
            values,
            0.75,
            -0.08,
        )
        for label, values in (
            ("source_previous", previous),
            ("equal_current", current),
            ("negated_current", [-int(value) for value in current]),
        )
    )
    conditions.extend(
        ("coupling", value, 128, 1, 256, previous, 0.75, value)
        for value in (-0.16, -0.08, 0.0, 0.08, 0.16)
    )
    conditions.extend(
        ("temperature", value, 128, 1, 256, previous, value, -0.08 if value <= 0.75 else 0.08)
        for value in (0.5, 0.75, 2.0, 2.5)
    )
    conditions.extend(
        ("run_length", value, 128, 1, value, previous, 0.75, -0.08) for value in (256, 1024, 2048)
    )
    rows = []
    for axis, value, burn, thin, sample_count, prior, temperature, coupling in conditions:
        row, _replay = _sensitivity_row(
            axis=axis,
            value=value,
            graph=graph,
            target=target(float(temperature)),
            current=current,
            previous=prior,
            temperature=float(temperature),
            coupling=float(coupling),
            arm="temporal_exchange",
            seed=seed,
            burn=int(burn),
            samples=int(sample_count),
            thin=int(thin),
        )
        rows.append(row)

    zero_common = {
        "axis": "coupling",
        "value": 0.0,
        "graph": graph,
        "target": target(0.75),
        "current": current,
        "previous": previous,
        "temperature": 0.75,
        "seed": seed,
        "burn": 128,
        "samples": 512,
        "thin": 1,
    }
    ordinary_row, ordinary = _sensitivity_row(coupling=0.0, arm="ordinary_gibbs", **zero_common)
    disabled_row, disabled = _sensitivity_row(
        coupling=0.0, arm="temporal_exchange_zero_coupling", **zero_common
    )
    zero_equivalence = {
        "bit_identical": bool(
            np.array_equal(ordinary["samples"], disabled["samples"])
            and np.array_equal(ordinary["energy_trace"], disabled["energy_trace"])
        ),
        "trajectory_hash_equal": ordinary["trajectory_sha256"] == disabled["trajectory_sha256"],
        "ordinary_trajectory_sha256": ordinary["trajectory_sha256"],
        "zero_coupling_trajectory_sha256": disabled["trajectory_sha256"],
        "attempted_updates_equal": ordinary["attempted_conditional_updates"]
        == disabled["attempted_conditional_updates"],
    }

    stationarity_checks = []
    for temperature, coupling in ((0.75, -0.08), (2.0, 0.08)):
        _, replay = _sensitivity_row(
            axis="stationarity_full_chain",
            value=f"T={temperature}",
            graph=graph,
            target=target(temperature),
            current=current,
            previous=previous,
            temperature=temperature,
            coupling=coupling,
            arm="temporal_exchange",
            seed=seed + int(temperature * 100),
            burn=512,
            samples=4096,
            thin=1,
        )
        samples_array = np.asarray(replay["samples"])
        windows = []
        for window_index, chunk in enumerate(np.array_split(samples_array, 4)):
            indices = _state_indices(chunk)
            empirical = np.bincount(
                indices, minlength=len(target(temperature)["probabilities"])
            ).astype(float)
            empirical /= float(chunk.shape[0])
            window_row = {
                "row_kind": "sensitivity",
                "sensitivity_axis": "stationarity_window",
                "sensitivity_value": window_index,
                "graph_id": graph["graph_id"],
                "temperature": temperature,
                "temporal_coupling": coupling,
                "arm": "temporal_exchange",
                "seed": seed + int(temperature * 100),
                "sample_start": window_index * chunk.shape[0],
                "sample_stop": (window_index + 1) * chunk.shape[0],
                "target_total_variation": _rounded(
                    0.5 * float(np.sum(np.abs(empirical - target(temperature)["_probabilities"])))
                ),
                "empirical_marginal_sha256": json_digest([_rounded(value) for value in empirical]),
            }
            window_row["audit_row_sha256"] = _audit_row_digest(window_row)
            rows.append(window_row)
            windows.append(window_row)
        stationarity_checks.append(
            {
                "graph_id": graph["graph_id"],
                "temperature": temperature,
                "temporal_coupling": coupling,
                "burn_in_sweeps": 512,
                "collected_samples": 4096,
                "windows": windows,
                "first_to_last_tv_change": _rounded(
                    float(windows[-1]["target_total_variation"])
                    - float(windows[0]["target_total_variation"])
                ),
                "spatial_target_preservation_assumed": False,
            }
        )
    return {
        "rows": rows,
        "zero_coupling_equivalence": zero_equivalence,
        "stationarity_checks": stationarity_checks,
        "zero_rows": [ordinary_row, disabled_row],
    }


def derive_hardware_mapping(source: Mapping[str, Any]) -> JsonDict:
    """Map logical overhead to checked-in envelopes without running a tool."""

    kv260 = load_json(REPO_ROOT / KV260_RECEIPT_PATH)
    gatemate = load_json(REPO_ROOT / GATEMATE_RECEIPT_PATH)
    kv_utilization = kv260["utilization"]
    gm_utilization = gatemate["lut_dff_utilization"]
    state_sizes = [6, 7, 8, 16, 128, 256]
    evidence = {
        "simulator_work_counts": "measured_simulator",
        "extra_previous_state_bits": "derived_estimate",
        "extra_adds_per_update": "derived_estimate",
        "extra_sign_multiply_per_update": "derived_estimate",
        "extra_previous_state_reads_per_update": "derived_estimate",
        "kv260_clb_lut_capacity": "existing_board_receipt",
        "kv260_clb_register_capacity": "existing_board_receipt",
        "gatemate_cpe_ff_capacity": "existing_board_receipt",
        "gatemate_ram_half_capacity": "existing_board_receipt",
        "kv260_q8_8_format": "existing_board_receipt",
        "gatemate_signed_8_bit_format": "existing_board_receipt",
        "gatemate_coefficient_scale": "unavailable",
        "mapped_clock": "unavailable",
        "mapped_power": "unavailable",
        "mapped_throughput": "unavailable",
    }
    assert set(evidence.values()) <= EVIDENCE_CLASSES
    return {
        "physical_hardware_invoked": False,
        "estimated_state_cost": {
            "extra_previous_state_bits_formula": "N",
            "extra_temporal_coefficient_bits": {"kv260": 16, "gatemate": 8},
            "examples": [
                {
                    "n_spins": size,
                    "extra_previous_state_bits": size,
                    "evidence_class": "derived_estimate",
                }
                for size in state_sizes
            ],
            "update_dependency": "all updates in one sweep read one fixed previous configuration; snapshot replacement occurs at the sweep boundary",
        },
        "estimated_arithmetic_cost": {
            "extra_adds_per_update": 1,
            "extra_sign_multiply_per_update": 1,
            "extra_general_multiplier_required": False,
            "reason": "The previous spin is bipolar, so Jv times the spin can use sign selection before the field add.",
        },
        "estimated_memory_traffic": {
            "extra_previous_state_reads_per_update": 1,
            "snapshot_reads_per_sweep_formula": "N",
            "snapshot_writes_per_sweep_formula": "N",
            "off_chip_traffic_required": False,
        },
        "coefficient_precision_range": {
            "source_spatial_coefficient_range": [-0.6, 0.55],
            "audited_temporal_coupling_range": [-0.16, 0.16],
            "audited_temperature_range": [0.5, 2.5],
            "kv260_format": "signed Q8.8, 16-bit",
            "kv260_step": 1.0 / 256.0,
            "kv260_numeric_range": [-128.0, 127.99609375],
            "kv260_temporal_range_representable": True,
            "gatemate_format": "signed 8-bit",
            "gatemate_scale": None,
            "gatemate_temporal_range_representable": None,
        },
        "board_envelope_comparison": {
            "kv260": {
                "receipt_path": KV260_RECEIPT_PATH.as_posix(),
                "receipt_sha256": file_digest(REPO_ROOT / KV260_RECEIPT_PATH),
                "existing_clb_lut_capacity": 117120,
                "existing_clb_register_capacity": 234240,
                "receipt_design_clb_luts_used": kv_utilization["clb_luts"],
                "receipt_design_clb_registers_used": kv_utilization["clb_registers"],
                "extra_state_fit_by_register_count_only": all(
                    size <= 234240 for size in state_sizes
                ),
                "mapping_limit": "No synthesis was run, so routing, timing closure, and exact LUT changes are unavailable.",
            },
            "gatemate": {
                "receipt_path": GATEMATE_RECEIPT_PATH.as_posix(),
                "receipt_sha256": file_digest(REPO_ROOT / GATEMATE_RECEIPT_PATH),
                "existing_cpe_ff_capacity": gm_utilization["nextpnr_resources"]["CPE_FF"][
                    "available"
                ],
                "existing_ram_half_capacity": gm_utilization["nextpnr_resources"]["RAM_HALF"][
                    "available"
                ],
                "receipt_design_dff_count": gm_utilization["dff_count"],
                "receipt_design_lut_count": gm_utilization["lut_count"],
                "extra_state_fit_by_ff_count_only": all(size <= 40960 for size in state_sizes),
                "mapping_limit": "The receipt design differs from the sampler, so no timing or throughput transfer is valid.",
            },
        },
        "evidence_class_by_hardware_field": evidence,
    }


def _source_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute the source's two positive gates from cold rows."""

    attempted = _paired_endpoint(rows, "attempted_conditional_updates")
    efficiency_passed = all(item["lower_bound_above_zero"] for item in attempted)
    index = {(row["graph_id"], row["temperature"], row["seed"], row["arm"]): row for row in rows}
    target_rows = []
    for graph_id, temperature in sorted(
        {(str(row["graph_id"]), float(row["temperature"])) for row in rows}
    ):
        deltas = []
        for seed in sorted(
            {
                int(row["seed"])
                for row in rows
                if row["graph_id"] == graph_id and row["temperature"] == temperature
            }
        ):
            deltas.append(
                float(
                    index[(graph_id, temperature, seed, "temporal_exchange")][
                        "target_total_variation"
                    ]
                )
                - float(
                    index[(graph_id, temperature, seed, "ordinary_gibbs")]["target_total_variation"]
                )
            )
        interval = _mean_ci(deltas)
        target_rows.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                **interval,
                "upper_bound_within_0_03": interval["ci95"][1] <= 0.03,
            }
        )
    target_passed = all(item["upper_bound_within_0_03"] for item in target_rows)
    return {
        "efficiency_by_stratum": attempted,
        "efficiency_gate_passed": efficiency_passed,
        "target_law_by_stratum": target_rows,
        "target_law_gate_passed": target_passed,
        "cold_verdict_class": "positive" if efficiency_passed and target_passed else "null",
    }


def _base_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hash: str,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Create the complete schema before selecting a terminal result."""

    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "run_date": str(run_date),
        "status": "in_progress",
        "spec_refs": list(SPEC_REFS),
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _rounded(duration_s),
        "random_seed": {
            "source_seeds": list(range(679300, 679320)),
            "sensitivity_seed_basis": "source row 679300 plus fixed temperature offsets",
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hash": source_hash,
        "existing_hardware_receipt_hashes": _hardware_receipt_hashes(),
        "preconditions_checked": [dict(row) for row in preconditions],
        "rows": [],
        "exact_target_recomputation": [],
        "cold_recomputed_metrics": [],
        "headline_differences": {},
        "update_accounting_by_arm": {},
        "denominator_sensitivity": {},
        "burnin_sensitivity": [],
        "thinning_sensitivity": [],
        "initialization_sensitivity": [],
        "coupling_sensitivity": [],
        "temperature_sensitivity": [],
        "run_length_sensitivity": [],
        "stationarity_checks": [],
        "zero_coupling_equivalence": {},
        "estimated_state_cost": {},
        "estimated_arithmetic_cost": {},
        "estimated_memory_traffic": {},
        "coefficient_precision_range": {},
        "board_envelope_comparison": {},
        "evidence_class_by_hardware_field": {},
        "physical_hardware_invoked": False,
        "source_verdict_supported": False,
        "source_gate_recomputation": {},
        "temporal_exchange_audit_completed": False,
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "complete_partial: temporal exchange cold audit did not finish",
        "claim_boundary": (
            "Fresh-process CPU replay and static mapping to old receipts only; no synthesis, board command, "
            "clock, power, physical timing, latency, or throughput claim"
        ),
        "implementation_receipt": {
            "module_path": MODULE_PATH.as_posix(),
            "module_sha256": file_digest(REPO_ROOT / MODULE_PATH),
            "source_module_imported": False,
            "source_sampler_imported": False,
        },
    }


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Explain why each top-level field exists in the cold record."""

    specific = {
        "inference_substrate": "This label separates CPU replay and static mapping from hardware execution.",
        "duration_s": "This is measured CPU audit duration and is not a hardware timing result.",
        "source_artifact_hash": "This hash binds the exact Exp6793 input.",
        "existing_hardware_receipt_hashes": "These hashes bind old envelopes without rerunning either board.",
        "rows": "Each source replay and sensitivity condition remains attributable.",
        "exact_target_recomputation": "Independent enumeration checks the target law and its hash.",
        "headline_differences": "Row-level differences expose source and cold-audit disagreement.",
        "denominator_sensitivity": "Alternative work counts expose a favorable denominator.",
        "physical_hardware_invoked": "Bare false forbids a physical hardware interpretation.",
        "source_verdict_supported": "This bare boolean states whether the cold gate matches the source class.",
        "temporal_exchange_audit_completed": "Completion records a full audit even for a null result.",
        "gate_check_summary": "A block names each failed check and observed value.",
        "verifier_is_oracle": "False records that no target outcome controls sampler updates.",
        "verdict_class": "The closed class separates null, blocked, and disqualified outcomes.",
        "honest_verdict": "The terminal prefix keeps the conductor decision visible.",
    }
    return {
        key: specific.get(key, f"The {key} field preserves attributable audit evidence.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic audit content while excluding measured duration."""

    excluded = {"duration_s", "reproducibility_checksum", "field_principles"}
    return json_digest(
        {key: deepcopy(value) for key, value in artifact.items() if key not in excluded}
    )


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    """Add principles and checksum after all terminal fields are stable."""

    artifact["field_principles"] = _field_principles(list(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hash: str,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write the full blocked schema and stop before stochastic replay."""

    artifact = _base_artifact(
        run_date=run_date,
        duration_s=duration_s,
        source_hash=source_hash,
        preconditions=preconditions,
    )
    artifact["status"] = "complete_blocked_temporal_exchange_audit"
    artifact["gate_check_summary"] = [dict(row) for row in preconditions if not row.get("passed")]
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = (
        "complete_blocked_temporal_exchange_audit: one or more source, row, target, update, code, or board-receipt checks failed"
    )
    return _finish_artifact(artifact)


def build_artifact(
    *,
    source: Mapping[str, Any],
    source_hash: str,
    cold_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    sensitivity: Mapping[str, Any],
    hardware: Mapping[str, Any],
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one complete terminal audit from cold rows and static receipts."""

    artifact = _base_artifact(
        run_date=run_date,
        duration_s=duration_s,
        source_hash=source_hash,
        preconditions=preconditions,
    )
    sensitivity_rows = list(sensitivity["rows"])
    artifact["rows"] = [dict(row) for row in cold_rows] + sensitivity_rows
    artifact["exact_target_recomputation"] = [dict(row) for row in target_rows]
    artifact["cold_recomputed_metrics"] = derive_cold_metrics(cold_rows)
    artifact["headline_differences"] = _headline_differences(source, cold_rows)
    artifact["update_accounting_by_arm"] = _update_accounting_by_arm(cold_rows)
    artifact["denominator_sensitivity"] = derive_denominator_sensitivity(cold_rows)
    artifact["burnin_sensitivity"] = [
        row for row in sensitivity_rows if row["sensitivity_axis"] == "burn_in_sweeps"
    ]
    artifact["thinning_sensitivity"] = [
        row for row in sensitivity_rows if row["sensitivity_axis"] == "thinning_sweeps"
    ]
    artifact["initialization_sensitivity"] = [
        row for row in sensitivity_rows if row["sensitivity_axis"] == "initial_previous_state"
    ]
    artifact["coupling_sensitivity"] = [
        row for row in sensitivity_rows if row["sensitivity_axis"] == "coupling"
    ]
    artifact["temperature_sensitivity"] = [
        row for row in sensitivity_rows if row["sensitivity_axis"] == "temperature"
    ]
    artifact["run_length_sensitivity"] = [
        row for row in sensitivity_rows if row["sensitivity_axis"] == "run_length"
    ]
    artifact["stationarity_checks"] = sensitivity["stationarity_checks"]
    artifact["zero_coupling_equivalence"] = sensitivity["zero_coupling_equivalence"]
    for key in (
        "estimated_state_cost",
        "estimated_arithmetic_cost",
        "estimated_memory_traffic",
        "coefficient_precision_range",
        "board_envelope_comparison",
        "evidence_class_by_hardware_field",
    ):
        artifact[key] = deepcopy(hardware[key])
    source_gate = _source_gate(cold_rows)
    artifact["source_gate_recomputation"] = source_gate
    artifact["source_verdict_supported"] = source_gate["cold_verdict_class"] == source.get(
        "verdict_class"
    )
    complete = bool(
        len(cold_rows) == 360
        and all(
            row["trajectory_hash_match"] and row["exact_target_hash_match"] for row in cold_rows
        )
        and all(row["hash_match"] for row in target_rows)
        and sensitivity["zero_coupling_equivalence"]["bit_identical"]
    )
    artifact["temporal_exchange_audit_completed"] = complete
    artifact["gate_check_summary"] = (
        []
        if complete
        else [
            {
                "check": "complete_cold_audit",
                "expected": {
                    "source_rows": 360,
                    "all_hashes_match": True,
                    "zero_coupling_equivalent": True,
                },
                "observed": {
                    "source_rows": len(cold_rows),
                    "trajectory_mismatches": artifact["headline_differences"][
                        "trajectory_hash_mismatch_count"
                    ],
                    "target_mismatches": artifact["headline_differences"][
                        "exact_target_hash_mismatch_count"
                    ],
                    "zero_coupling_equivalent": sensitivity["zero_coupling_equivalence"][
                        "bit_identical"
                    ],
                },
            }
        ]
    )
    if complete:
        artifact["status"] = (
            "complete_null" if source_gate["cold_verdict_class"] == "null" else "complete_positive"
        )
        artifact["verdict_class"] = source_gate["cold_verdict_class"]
        artifact["honest_verdict"] = (
            "complete: cold replay supports the source null; temporal overhead and stationary-law sensitivity remain explicit; no hardware invoked"
            if artifact["verdict_class"] == "null"
            else "complete: cold replay supports the source positive under audited denominators; no hardware invoked"
        )
    else:
        artifact["status"] = "complete_disqualified"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = (
            "complete_disqualified: cold replay or exact-target receipts did not match the source"
        )
    return _finish_artifact(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, digests, terminal state, rows, and claim boundaries."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        return ["required_fields_missing"]
    errors = []
    if set(artifact["field_principles"]) != set(artifact):
        errors.append("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["physical_hardware_invoked"] is not False:
        errors.append("physical_hardware_boundary_mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("oracle_boundary_mismatch")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class_mismatch")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_prefix_mismatch")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    duration = artifact["duration_s"]
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0.0:
        errors.append("duration_invalid")
    blocked = artifact.get("status") == "complete_blocked_temporal_exchange_audit"
    if blocked:
        if (
            artifact["verdict_class"] != "blocked"
            or artifact["temporal_exchange_audit_completed"] is not False
            or artifact["rows"]
            or not artifact["gate_check_summary"]
            or not str(artifact["honest_verdict"]).startswith(
                "complete_blocked_temporal_exchange_audit"
            )
        ):
            errors.append("blocked_terminal_state_mismatch")
        return errors
    source_rows = [row for row in artifact["rows"] if row.get("row_kind") == "source_recomputation"]
    if any(row.get("audit_row_sha256") != _audit_row_digest(row) for row in artifact["rows"]):
        errors.append("row_hash_mismatch")
    if artifact["temporal_exchange_audit_completed"] is True and len(source_rows) != 360:
        errors.append("source_row_grid_mismatch")
    if artifact["temporal_exchange_audit_completed"] is True and artifact["gate_check_summary"]:
        errors.append("completion_gate_mismatch")
    if set(artifact["evidence_class_by_hardware_field"].values()) - EVIDENCE_CLASSES:
        errors.append("hardware_evidence_class_mismatch")
    return errors


def _write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result only after one complete JSON object exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    source_path: Path | str = SOURCE_PATH,
    output_path: Path | str = RESULT_PATH,
    run_date: str,
) -> JsonDict:
    """Check gates, replay, map old receipts, validate, and write once."""

    started = time.perf_counter()
    source_file = Path(source_path)
    source = load_json(source_file)
    source_hash = file_digest(source_file)
    preconditions = check_preconditions(source)
    if any(not row["passed"] for row in preconditions):
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            source_hash=source_hash,
            preconditions=preconditions,
        )
    else:
        cold_rows, target_rows = replay_all_source_rows(source)
        sensitivity = run_sensitivity_panel(source)
        hardware = derive_hardware_mapping(source)
        artifact = build_artifact(
            source=source,
            source_hash=source_hash,
            cold_rows=cold_rows,
            target_rows=target_rows,
            sensitivity=sensitivity,
            hardware=hardware,
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=preconditions,
        )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise TemporalExchangeAuditError(f"artifact validation failed: {validation_errors}")
    _write_atomic(Path(output_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the required dated cold audit command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = run(source_path=args.source, output_path=args.output, run_date=args.date)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "temporal_exchange_audit_completed": artifact["temporal_exchange_audit_completed"],
                "physical_hardware_invoked": artifact["physical_hardware_invoked"],
                "verdict_class": artifact["verdict_class"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    raise SystemExit(main())
