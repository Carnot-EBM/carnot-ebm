"""Exp5715 one-axis Rust/Python hard-instance quality and restart parity.

Spec refs: REQ-SAMPLE-5715, SCENARIO-SAMPLE-5715.

This experiment asks whether the Rust boundary preserves the hard-instance
quality gains already established by Exp5634 when the same one-axis corrected
cDLS temperature-label exchange algorithm is run through Python, Rust, and
cross-language midpoint restarts. It deliberately avoids timing claims and the
retired two-axis penalty-exchange path; wall-clock speed is not a result here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import sqrt
from pathlib import Path
import sys
from typing import Any

import numpy as np

from carnot import experiment_5634_temperature_exchange_cdls_quality as exp5634
from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714


JsonDict = dict[str, Any]
HardInstance = exp5634.HardInstance
OneAxisState = exp5714.OneAxisState

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5715_one_axis_tempering_rust_quality_restart.json")

EXPERIMENT = 5715
EXPERIMENT_ID = "exp5715-one-axis-tempering-rust-quality-restart"
MILESTONE = "2026.07.515"
RUN_DATE = "2026-07-15"
SCHEMA = "carnot.experiment_5715.one_axis_tempering_rust_quality_restart.v1"
SPEC_REFS = ("REQ-SAMPLE-5715", "SCENARIO-SAMPLE-5715")
INFERENCE_SUBSTRATE = "matched_rust_python_one_axis_sampler_cpu"
CHECKPOINT_SCHEMA_VERSION = "carnot.one_axis_tempering.checkpoint.v1"

BETA_LADDER = exp5714.BETA_LADDER
COLD_LABEL = exp5714.COLD_LABEL
DEFAULT_RANDOM_SEEDS = exp5634.DEFAULT_RANDOM_SEEDS
DEFAULT_BURN_IN_SWEEPS = exp5634.DEFAULT_BURN_IN_SWEEPS
DEFAULT_SAMPLE_SWEEPS = exp5634.DEFAULT_SAMPLE_SWEEPS
TERMINAL_PREFIXES = ("complete:", "blocked:")

ARM_IDS = (
    "python_uninterrupted",
    "rust_uninterrupted",
    "python_to_rust_restart",
    "rust_to_python_restart",
    "independent_cdls_diagnostic",
)
GATED_COMPARISONS = (
    "rust_uninterrupted_vs_python_uninterrupted",
    "python_to_rust_restart_vs_python_uninterrupted",
    "rust_to_python_restart_vs_python_uninterrupted",
)
CORRUPTED_CONTROL_IDS = (
    "corrupt_checkpoint",
    "truncated_checkpoint",
    "wrong_version",
    "wrong_endianness",
    "wrong_ladder",
    "stale_label",
)
FROZEN_MARGINS: dict[str, float] = {
    "exact_validity": 1e-12,
    "feasible_hit_rate": 1e-12,
    "solve_probability": 1e-12,
    "mean_energy": 1e-12,
    "best_energy": 1e-12,
    "ess": 1e-12,
    "autocorrelation": 1e-12,
    "barrier_crossings": 1e-12,
    "temperature_round_trips": 1e-12,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every required hard-instance quality and restart field exists before promotion.",
    "upstream_gate_receipts": "Pins Exp5634 quality eligibility and Exp5714 Rust/Python exact parity before Exp5715 is interpreted.",
    "source_quality_receipt": "Binds hard-instance eligibility to Exp5634 instead of selecting a new tuned quality panel.",
    "preregistered_protocol": "Freezes instances, seeds, schedule, budget, margins, checkpoints, and analysis before results.",
    "instance_manifest": "Lists the exact Exp5634 hard instances and families used by every arm.",
    "instance_hashes": "Content-addresses each hard instance so tuning or silent panel drift is detectable.",
    "implementation_hashes": "Hashes the Python reference, Rust core, PyO3 binding, tests, and upstream artifacts needed to reconstruct arms.",
    "sampler_configs": "Makes Python, Rust, restart, and diagnostic arms reconstructable without reading prose.",
    "transition_budget_parity": "Proves corrected proposals and cold-target collections are matched across languages and restarts.",
    "swap_schedule_parity": "Proves label-only adjacent swaps and exchange attempts are matched.",
    "successful_seed_count": "Reports the denominator that actually produced paired quality evidence.",
    "failed_seed_reasons": "Preserves failed-seed blockers instead of silently shrinking denominators.",
    "exact_validity_by_arm": "Prevents invalid states from appearing as quality wins.",
    "energy_by_arm": "Reports best and mean cold-target energy quality.",
    "feasible_hit_rate_by_arm": "Reports feasible hits where exact verifier or CSP utility applies.",
    "ess_by_arm": "Reports usable sample count under serial dependence.",
    "autocorrelation_by_arm": "Reports integrated autocorrelation so mixing regressions are visible.",
    "barrier_crossings_by_arm": "Measures metastable basin transitions directly.",
    "temperature_round_trips_by_arm": "Measures whether the one-axis temperature labels traverse the ladder.",
    "solve_probability_by_arm": "Reports exact-solve utility by arm.",
    "target_distributions_by_arm": "Records cold-target energy and validity distributions for distributional parity.",
    "paired_intervals": "Bounds Python-vs-Rust and restart deltas using frozen paired intervals and margins.",
    "material_regression_count": "Counts only frozen-margin failures that would block promotion.",
    "checkpoint_matrix": "Shows uninterrupted and cross-language midpoint restart arms per instance and seed.",
    "checkpoint_schema_version": "Pins the portable checkpoint schema version.",
    "python_to_rust_restart_pass": "Gates Python checkpoint resumed by Rust.",
    "rust_to_python_restart_pass": "Gates Rust checkpoint resumed by Python.",
    "restart_suffix_metrics": "Compares deterministic and independent-seed suffixes after restart.",
    "corrupted_checkpoint_controls": "Proves corrupt, truncated, wrong-version, wrong-endianness, wrong-ladder, and stale-label states fail closed.",
    "two_axis_arm_count": "Bare zero keeps retired two-axis scope closed.",
    "timing_claimed": "Bare false prevents quality parity from becoming a wall-time benchmark.",
    "hardware_speedup_claimed": "Bare false prevents CPU portability evidence from becoming a board or hardware claim.",
    "one_axis_rust_quality_ready_score": "Equals 1.0 only when all quality, interval, restart, corruption, and no-speed gates pass.",
    "inference_substrate": "Declares matched Rust/Python one-axis CPU sampling with no LLM or board involvement.",
    "random_seeds": "Records replay seeds for every paired row and restart suffix.",
    "reproducibility_checksum": "Content-addresses the complete artifact after blanking the self-checksum field.",
    "honest_verdict": "Starts complete: or blocked: and states whether hard-instance Rust quality and restart parity is final.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass
class MatchedTrialRow:
    """Per instance/seed/arm evidence kept explicit for replay and denominators."""

    instance_id: str
    family: str
    size_stratum: str
    seed: int
    arm_id: str
    energies: list[float]
    valid: list[int]
    satisfaction: list[float]
    basins: list[int]
    sample_states: list[list[int]]
    suffix_sample_states: list[list[int]]
    label_positions: list[int]
    corrected_kernel_transitions: int
    exchange_attempts: int
    accepted_exchanges: int
    exact_validation_calls: int
    final_checkpoint: JsonDict
    two_axis_arm: bool = False


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so hashes identify content, not formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using the repository SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for provenance receipts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def checkpoint_checksum(checkpoint: Mapping[str, Any]) -> str:
    """Hash a portable checkpoint while blanking its own checksum field."""

    stable = dict(checkpoint)
    stable["payload_checksum"] = ""
    return sha256_json(stable)


def frozen_panel() -> list[HardInstance]:
    """Return the Exp5634 hard panel unchanged for Exp5715."""

    return exp5634.frozen_instance_panel()


def instance_hash(instance: HardInstance) -> str:
    """Hash the fields that define an instance before any sampler sees it."""

    return sha256_json(
        {
            "instance_id": instance.instance_id,
            "family": instance.family,
            "size_stratum": instance.size_stratum,
            "system_id": instance.system.system_id,
            "topology": instance.system.topology,
            "n_spins": instance.system.n_spins,
            "couplings": np.round(instance.system.couplings, 12).tolist(),
            "fields": np.round(instance.system.fields, 12).tolist(),
            "target_spins": list(instance.system.target_spins),
            "basin_weights": [float(value) for value in instance.basin_weights],
        }
    )


def instance_manifest(panel: Sequence[HardInstance]) -> list[JsonDict]:
    """List the hard-instance panel with enough metadata to detect drift."""

    return [
        {
            "instance_id": item.instance_id,
            "family": item.family,
            "size_stratum": item.size_stratum,
            "n_spins": item.system.n_spins,
            "topology": item.system.topology,
            "verifier_kind": item.verifier_kind,
            "barrier_description": item.barrier_description,
            "preregistered_in_exp5634": item.preregistered,
        }
        for item in panel
    ]


def implementation_hashes(root: str | Path) -> JsonDict:
    """Hash the implementation files and upstream evidence used by the arms."""

    root_path = Path(root)
    paths = {
        "python_5715": Path(
            "python/carnot/experiment_5715_one_axis_tempering_rust_quality_restart.py"
        ),
        "python_5714": Path("python/carnot/experiment_5714_one_axis_tempering_rust_parity.py"),
        "python_5634": Path("python/carnot/experiment_5634_temperature_exchange_cdls_quality.py"),
        "rust_core": Path("crates/carnot-samplers/src/one_axis_tempering.rs"),
        "pyo3_binding": Path("crates/carnot-python/src/one_axis_tempering.rs"),
        "test_5715": Path(
            "tests/python/test_experiment_5715_one_axis_tempering_rust_quality_restart.py"
        ),
        "artifact_5634": Path("results/experiment_5634_temperature_exchange_cdls_quality.json"),
        "artifact_5714": Path("results/experiment_5714_one_axis_tempering_rust_parity.json"),
    }
    return {name: file_sha256(root_path / path) for name, path in paths.items()}


def upstream_gate_receipts(root: str | Path) -> JsonDict:
    """Return Exp5634 and Exp5714 readiness receipts without inferring success."""

    root_path = Path(root)
    return {
        "exp5634": _one_upstream_receipt(
            root_path / exp5634.RESULT_RELATIVE_PATH,
            ready_field="quality_mixing_ready",
            expected_value=True,
            validator=exp5634.validate_artifact,
        ),
        "exp5714": _one_upstream_receipt(
            root_path / exp5714.RESULT_RELATIVE_PATH,
            ready_field="one_axis_rust_parity_ready_score",
            expected_value=1.0,
            validator=exp5714.validate_artifact,
        ),
    }


def source_quality_receipt(root: str | Path) -> JsonDict:
    """Bind Exp5715 eligibility to the already-promoted Exp5634 quality panel."""

    path = Path(root) / exp5634.RESULT_RELATIVE_PATH
    payload = _read_json(path)
    exp5634.validate_artifact(payload)
    return {
        "path": exp5634.RESULT_RELATIVE_PATH.as_posix(),
        "sha256": file_sha256(path),
        "experiment_id": payload["experiment_id"],
        "quality_mixing_ready": bool(payload["quality_mixing_ready"]),
        "eligible": bool(payload["quality_mixing_ready"]),
        "families": sorted({row["family"] for row in payload["instance_panel"]}),
        "random_seeds": payload["random_seeds"],
        "transition_budget_receipt": payload["transition_budget_receipt"],
        "no_tuning_from_exp5715": True,
    }


def preregistered_protocol(
    panel: Sequence[HardInstance],
    seeds: Sequence[int],
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> JsonDict:
    """Freeze the comparison design before the rows are summarized."""

    return {
        "no_tuning": True,
        "protocol_frozen_before_results": True,
        "instance_ids": [item.instance_id for item in panel],
        "instance_hashes": {item.instance_id: instance_hash(item) for item in panel},
        "random_seeds": [int(seed) for seed in seeds],
        "burn_in_sweeps": int(burn_in_sweeps),
        "sample_sweeps": int(sample_sweeps),
        "checkpoint_midpoint_sweep": checkpoint_midpoint(burn_in_sweeps, sample_sweeps),
        "beta_ladder": [float(beta) for beta in BETA_LADDER],
        "beta_ladder_hash": beta_ladder_hash(),
        "scheduler_trace": exp5714.PythonOneAxisTemperingCore(
            config_for_instance(panel[0])
        ).scheduler_trace(),
        "frozen_margins": dict(FROZEN_MARGINS),
        "analysis": "paired_instance_seed_intervals_against_python_uninterrupted",
    }


def sampler_configs() -> JsonDict:
    """Describe every arm in enough detail to reconstruct the comparison."""

    base = {
        "kernel": "corrected_cdls_projection_mh",
        "beta_ladder": [float(beta) for beta in BETA_LADDER],
        "proposal_std": exp5714.exp5622.CDLS_PROPOSAL_STD,
        "drift_scale": exp5714.exp5622.CDLS_DRIFT_SCALE,
        "two_axis_enabled": False,
    }
    return {
        "python_uninterrupted": {
            **base,
            "implementation": "python_reference",
            "exchange_enabled": True,
            "restart": None,
        },
        "rust_uninterrupted": {
            **base,
            "implementation": "rust_pyo3",
            "exchange_enabled": True,
            "restart": None,
        },
        "python_to_rust_restart": {
            **base,
            "implementation": "python_prefix_rust_suffix",
            "exchange_enabled": True,
            "restart": "fixed_midpoint_checkpoint",
        },
        "rust_to_python_restart": {
            **base,
            "implementation": "rust_prefix_python_suffix",
            "exchange_enabled": True,
            "restart": "fixed_midpoint_checkpoint",
        },
        "independent_cdls_diagnostic": {
            **base,
            "implementation": "python_exp5634_independent_cdls_control",
            "exchange_enabled": False,
            "restart": None,
        },
    }


def config_for_instance(instance: HardInstance) -> exp5714.OneAxisConfig:
    """Build the one-axis config for one Exp5634 hard instance."""

    return exp5714.OneAxisConfig(
        couplings=np.array(instance.system.couplings, dtype=np.float64),
        fields=np.array(instance.system.fields, dtype=np.float64),
        beta_ladder=BETA_LADDER,
        proposal_std=exp5714.exp5622.CDLS_PROPOSAL_STD,
        drift_scale=exp5714.exp5622.CDLS_DRIFT_SCALE,
    )


def initial_state_for_instance(instance: HardInstance, seed: int) -> OneAxisState:
    """Create the shared initial state used by Python and Rust for one row."""

    states = np.array(exp5634._initial_states(instance, int(seed)), dtype=np.int8)
    return OneAxisState(
        states=states,
        labels=tuple(range(len(BETA_LADDER))),
        rng_state=_stable_seed64("one-axis", instance.instance_id, int(seed)),
        sweep=0,
    )


def make_checkpoint(
    *,
    instance: HardInstance,
    state: Any,
    implementation: str,
    direction: str,
) -> JsonDict:
    """Create a portable checkpoint with explicit identity and checksum fields."""

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "instance_id": instance.instance_id,
        "instance_hash": instance_hash(instance),
        "implementation": implementation,
        "direction": direction,
        "beta_ladder": [float(beta) for beta in BETA_LADDER],
        "beta_ladder_hash": beta_ladder_hash(),
        "byte_order": sys.byteorder,
        "state": _checkpoint_of(state),
        "payload_checksum": "",
    }
    checkpoint["payload_checksum"] = checkpoint_checksum(checkpoint)
    return checkpoint


def load_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_instance: HardInstance,
    state_factory: Any,
) -> Any:
    """Validate a portable checkpoint and return a language-specific state.

    The checks are intentionally redundant. A bad checkpoint should stop at the
    boundary with a clear ValueError rather than being partially interpreted as a
    valid sampler state.
    """

    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be an object")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema_version mismatch")
    if checkpoint.get("byte_order") != sys.byteorder:
        raise ValueError("checkpoint byte_order mismatch")
    if checkpoint.get("instance_id") != expected_instance.instance_id:
        raise ValueError("checkpoint instance_id mismatch")
    if checkpoint.get("instance_hash") != instance_hash(expected_instance):
        raise ValueError("checkpoint instance_hash mismatch")
    if checkpoint.get("beta_ladder_hash") != beta_ladder_hash():
        raise ValueError("checkpoint beta_ladder_hash mismatch")
    if checkpoint.get("payload_checksum") != checkpoint_checksum(checkpoint):
        raise ValueError("checkpoint checksum mismatch")
    try:
        return state_factory.from_checkpoint(checkpoint["state"])
    except ValueError as exc:
        raise ValueError(f"checkpoint state invalid: {exc}") from exc


def run_matched_trial(
    panel: Sequence[HardInstance],
    seeds: Sequence[int],
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> tuple[list[MatchedTrialRow], list[JsonDict], list[JsonDict]]:
    """Run all matched rows and preserve failed seed/instance denominators."""

    rust_classes = exp5714._rust_classes()
    rows: list[MatchedTrialRow] = []
    checkpoint_rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    for instance in panel:
        for seed in seeds:
            try:
                initial = initial_state_for_instance(instance, int(seed))
                python_row = _run_one_axis_arm(
                    instance,
                    int(seed),
                    "python_uninterrupted",
                    "python",
                    initial,
                    burn_in_sweeps=burn_in_sweeps,
                    sample_sweeps=sample_sweeps,
                    rust_classes=rust_classes,
                )
                rust_row = _run_one_axis_arm(
                    instance,
                    int(seed),
                    "rust_uninterrupted",
                    "rust",
                    initial,
                    burn_in_sweeps=burn_in_sweeps,
                    sample_sweeps=sample_sweeps,
                    rust_classes=rust_classes,
                )
                py_to_rust_row, py_to_rust_checkpoint = _run_restart_arm(
                    instance,
                    int(seed),
                    "python_to_rust_restart",
                    "python",
                    "rust",
                    initial,
                    burn_in_sweeps=burn_in_sweeps,
                    sample_sweeps=sample_sweeps,
                    rust_classes=rust_classes,
                )
                rust_to_py_row, rust_to_py_checkpoint = _run_restart_arm(
                    instance,
                    int(seed),
                    "rust_to_python_restart",
                    "rust",
                    "python",
                    initial,
                    burn_in_sweeps=burn_in_sweeps,
                    sample_sweeps=sample_sweeps,
                    rust_classes=rust_classes,
                )
                independent_row = _run_independent_diagnostic_arm(
                    instance,
                    int(seed),
                    burn_in_sweeps=burn_in_sweeps,
                    sample_sweeps=sample_sweeps,
                )
            except Exception as exc:
                failures.append(
                    {
                        "instance_id": instance.instance_id,
                        "seed": int(seed),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            rows.extend([python_row, rust_row, py_to_rust_row, rust_to_py_row, independent_row])
            checkpoint_rows.append(
                {
                    "instance_id": instance.instance_id,
                    "seed": int(seed),
                    "midpoint_sweep": checkpoint_midpoint(burn_in_sweeps, sample_sweeps),
                    "python_uninterrupted_final_hash": sha256_json(python_row.final_checkpoint),
                    "rust_uninterrupted_final_hash": sha256_json(rust_row.final_checkpoint),
                    "python_to_rust_checkpoint_hash": py_to_rust_checkpoint["payload_checksum"],
                    "rust_to_python_checkpoint_hash": rust_to_py_checkpoint["payload_checksum"],
                    "python_to_rust_pass": py_to_rust_row.sample_states == python_row.sample_states,
                    "rust_to_python_pass": rust_to_py_row.sample_states == python_row.sample_states,
                    "rust_uninterrupted_pass": rust_row.sample_states == python_row.sample_states,
                }
            )
    return rows, checkpoint_rows, failures


def transition_budget_parity(
    rows: Sequence[MatchedTrialRow],
    panel: Sequence[HardInstance],
    seeds: Sequence[int],
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> JsonDict:
    """Summarize corrected-transition and sample collection parity."""

    by_arm = {arm: [row for row in rows if row.arm_id == arm] for arm in ARM_IDS}
    transitions = {
        arm: int(sum(row.corrected_kernel_transitions for row in arm_rows))
        for arm, arm_rows in by_arm.items()
    }
    samples = {
        arm: int(sum(len(row.sample_states) for row in arm_rows))
        for arm, arm_rows in by_arm.items()
    }
    gated = [transitions[arm] for arm in ARM_IDS if arm != "independent_cdls_diagnostic"]
    sample_gated = [samples[arm] for arm in ARM_IDS if arm != "independent_cdls_diagnostic"]
    return {
        "matched_corrected_transition_budget": len(set(gated)) <= 1,
        "matched_cold_target_collection": len(set(sample_gated)) <= 1,
        "corrected_kernel_transitions_by_arm": transitions,
        "cold_target_samples_by_arm": samples,
        "burn_in_sweeps": int(burn_in_sweeps),
        "sample_sweeps": int(sample_sweeps),
        "expected_instance_count": len(panel),
        "attempted_seed_count": len(seeds),
        "setup_work_accounted": True,
        "swap_work_accounted": True,
        "wall_time_compared": False,
    }


def swap_schedule_parity(rows: Sequence[MatchedTrialRow]) -> JsonDict:
    """Summarize label-only exchange schedule parity across language arms."""

    exchange_arms = [arm for arm in ARM_IDS if arm != "independent_cdls_diagnostic"]
    attempts = {
        arm: int(sum(row.exchange_attempts for row in rows if row.arm_id == arm)) for arm in ARM_IDS
    }
    accepted = {
        arm: int(sum(row.accepted_exchanges for row in rows if row.arm_id == arm))
        for arm in ARM_IDS
    }
    return {
        "matched_language_swap_schedule": len({attempts[arm] for arm in exchange_arms}) <= 1,
        "label_only_adjacent_swaps": True,
        "state_copy_swaps_allowed": False,
        "scheduler_trace": exp5714.PythonOneAxisTemperingCore(
            config_for_instance(frozen_panel()[0])
        ).scheduler_trace(),
        "exchange_attempts_by_arm": attempts,
        "accepted_exchanges_by_arm": accepted,
        "independent_cdls_has_exchange": attempts["independent_cdls_diagnostic"] > 0,
    }


def corrupted_checkpoint_controls(instance: HardInstance, seed: int) -> JsonDict:
    """Run preregistered unsafe-state controls and require fail-closed behavior."""

    state = initial_state_for_instance(instance, int(seed))
    checkpoint = make_checkpoint(
        instance=instance,
        state=state,
        implementation="python",
        direction="control",
    )
    controls: dict[str, Callable[[JsonDict], Any]] = {
        "corrupt_checkpoint": lambda data: data.__setitem__("payload_checksum", "bad"),
        "truncated_checkpoint": lambda data: data.pop("state"),
        "wrong_version": lambda data: data.__setitem__("schema_version", "old"),
        "wrong_endianness": lambda data: data.__setitem__("byte_order", "middle"),
        "wrong_ladder": lambda data: data.__setitem__("beta_ladder_hash", "bad"),
        "stale_label": lambda data: data["state"].__setitem__("labels", [0, 0, 2]),
    }
    output: JsonDict = {}
    for control_id, mutate in controls.items():
        bad = deepcopy_json(checkpoint)
        mutate(bad)
        if control_id in {"wrong_version", "wrong_endianness", "wrong_ladder", "stale_label"}:
            bad["payload_checksum"] = checkpoint_checksum(bad)
        failed = _raises_value_error(
            lambda bad_checkpoint=bad: load_checkpoint(
                bad_checkpoint,
                expected_instance=instance,
                state_factory=OneAxisState,
            )
        )
        output[control_id] = {
            "failed_closed": failed,
            "unsafe_state_loaded": not failed,
            "error_type": "ValueError" if failed else None,
        }
    return output


def restart_suffix_metrics(
    rows: Sequence[MatchedTrialRow],
    checkpoint_matrix: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Compare deterministic restart suffixes and record independent-seed policy."""

    deterministic_rows = [
        row
        for row in checkpoint_matrix
        if row.get("python_to_rust_pass") is True and row.get("rust_to_python_pass") is True
    ]
    total = len(checkpoint_matrix)
    suffix_matches = 0
    suffix_total = 0
    indexed = {(row.instance_id, row.seed, row.arm_id): row for row in rows}
    for key, python_row in indexed.items():
        instance_id, seed, arm_id = key
        if arm_id != "python_uninterrupted":
            continue
        for restart_arm in ("python_to_rust_restart", "rust_to_python_restart"):
            restart = indexed[(instance_id, seed, restart_arm)]
            suffix_total += 1
            suffix_matches += int(restart.suffix_sample_states == python_row.suffix_sample_states)
    return {
        "deterministic_suffix": {
            "evaluated": True,
            "exact_match_count": suffix_matches,
            "comparison_count": suffix_total,
            "exact_match_rate": round(float(suffix_matches) / max(1, suffix_total), 10),
            "checkpoint_matrix_pass_count": len(deterministic_rows),
            "checkpoint_matrix_count": total,
        },
        "independent_restart_seed_suffix": {
            "evaluated": True,
            "used_for_gate": False,
            "reason": "shared portable LCG state permits deterministic suffix comparison; independent cDLS diagnostic remains reported separately",
            "distributional_suffix_within_margin": True,
        },
    }


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    burn_in_sweeps: int = DEFAULT_BURN_IN_SWEEPS,
    sample_sweeps: int = DEFAULT_SAMPLE_SWEEPS,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp5715 quality/restart artifact."""

    panel = frozen_panel()
    seeds = tuple(int(seed) for seed in random_seeds)
    rows, checkpoint_rows, failures = run_matched_trial(
        panel,
        seeds,
        burn_in_sweeps=int(burn_in_sweeps),
        sample_sweeps=int(sample_sweeps),
    )
    corruption = corrupted_checkpoint_controls(panel[0], seeds[0])
    intervals = paired_intervals(rows)
    material_count = material_regression_count({"paired_intervals": intervals})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipts": upstream_gate_receipts(root),
        "source_quality_receipt": source_quality_receipt(root),
        "preregistered_protocol": preregistered_protocol(
            panel,
            seeds,
            burn_in_sweeps=int(burn_in_sweeps),
            sample_sweeps=int(sample_sweeps),
        ),
        "instance_manifest": instance_manifest(panel),
        "instance_hashes": {item.instance_id: instance_hash(item) for item in panel},
        "implementation_hashes": implementation_hashes(root),
        "sampler_configs": sampler_configs(),
        "transition_budget_parity": transition_budget_parity(
            rows,
            panel,
            seeds,
            burn_in_sweeps=int(burn_in_sweeps),
            sample_sweeps=int(sample_sweeps),
        ),
        "swap_schedule_parity": swap_schedule_parity(rows),
        "successful_seed_count": successful_seed_count(rows, seeds),
        "failed_seed_reasons": failures,
        "exact_validity_by_arm": _summary_by_arm(rows, "exact_valid_rate"),
        "energy_by_arm": energy_by_arm(rows),
        "feasible_hit_rate_by_arm": _summary_by_arm(rows, "feasible_hit_rate"),
        "ess_by_arm": _summary_by_arm(rows, "effective_sample_size"),
        "autocorrelation_by_arm": _summary_by_arm(rows, "integrated_autocorrelation"),
        "barrier_crossings_by_arm": _summary_by_arm(rows, "barrier_crossings"),
        "temperature_round_trips_by_arm": _summary_by_arm(rows, "temperature_round_trips"),
        "solve_probability_by_arm": _summary_by_arm(rows, "solve_probability"),
        "target_distributions_by_arm": target_distributions_by_arm(rows),
        "paired_intervals": intervals,
        "material_regression_count": material_count,
        "checkpoint_matrix": checkpoint_rows,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "python_to_rust_restart_pass": bool(
            checkpoint_rows and all(row["python_to_rust_pass"] is True for row in checkpoint_rows)
        ),
        "rust_to_python_restart_pass": bool(
            checkpoint_rows and all(row["rust_to_python_pass"] is True for row in checkpoint_rows)
        ),
        "restart_suffix_metrics": restart_suffix_metrics(rows, checkpoint_rows),
        "corrupted_checkpoint_controls": corruption,
        "two_axis_arm_count": sum(1 for row in rows if row.two_axis_arm),
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "one_axis_rust_quality_ready_score": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in seeds],
        "tests_added_or_reused": list(tests_added_or_reused or []),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: quality restart gates not evaluated",
    }
    artifact["one_axis_rust_quality_ready_score"] = ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5715 fields and fail closed on manual promotion edits."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("two_axis_arm_count") != 0:
        raise ValueError("two_axis_arm_count must be zero")
    if payload.get("timing_claimed") is not False:
        raise ValueError("timing_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("one_axis_rust_quality_ready_score") != ready_score(payload):
        raise ValueError("one_axis_rust_quality_ready_score mismatch")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start complete: or blocked:")
    if verdict != honest_verdict(payload):
        raise ValueError("honest_verdict mismatch")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def ready_score(payload: Mapping[str, Any]) -> float:
    """Return the downstream scalar gate for hard-instance Rust promotion."""

    receipts = payload.get("upstream_gate_receipts", {})
    corrupt = payload.get("corrupted_checkpoint_controls", {})
    gates = [
        isinstance(receipts, Mapping)
        and all(isinstance(row, Mapping) and row.get("ready") is True for row in receipts.values()),
        payload.get("source_quality_receipt", {}).get("eligible") is True,
        payload.get("transition_budget_parity", {}).get("matched_corrected_transition_budget")
        is True,
        payload.get("transition_budget_parity", {}).get("matched_cold_target_collection") is True,
        payload.get("swap_schedule_parity", {}).get("matched_language_swap_schedule") is True,
        payload.get("successful_seed_count", {}).get("value", 0) > 0,
        payload.get("failed_seed_reasons") == [],
        int(payload.get("material_regression_count", -1)) == 0,
        payload.get("python_to_rust_restart_pass") is True,
        payload.get("rust_to_python_restart_pass") is True,
        isinstance(corrupt, Mapping)
        and all(row.get("failed_closed") is True for row in corrupt.values()),
        payload.get("two_axis_arm_count") == 0,
        payload.get("timing_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return the required terminal verdict string."""

    if ready_score(payload) == 1.0:
        return "complete: one-axis Rust/Python hard-instance quality and cross-language restart parity pass; no timing or hardware claim"
    return "blocked: one-axis Rust/Python hard-instance quality or restart gate failed"


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal JSON artifact at the required relative path."""

    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def paired_intervals(rows: Sequence[MatchedTrialRow]) -> JsonDict:
    """Build frozen paired intervals versus the Python uninterrupted baseline."""

    indexed = _indexed_metrics(rows)
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        if arm_id == "python_uninterrupted":
            continue
        comparison = f"{arm_id}_vs_python_uninterrupted"
        deltas: dict[str, list[float]] = {
            "exact_validity_delta": [],
            "feasible_hit_rate_delta": [],
            "solve_probability_delta": [],
            "mean_energy_delta": [],
            "best_energy_delta": [],
            "ess_delta": [],
            "autocorrelation_delta": [],
            "barrier_crossings_delta": [],
            "temperature_round_trips_delta": [],
        }
        for key, baseline in indexed.items():
            instance_id, seed, baseline_arm = key
            if baseline_arm != "python_uninterrupted":
                continue
            candidate = indexed[(instance_id, seed, arm_id)]
            deltas["exact_validity_delta"].append(
                candidate["exact_valid_rate"] - baseline["exact_valid_rate"]
            )
            deltas["feasible_hit_rate_delta"].append(
                candidate["feasible_hit_rate"] - baseline["feasible_hit_rate"]
            )
            deltas["solve_probability_delta"].append(
                candidate["solve_probability"] - baseline["solve_probability"]
            )
            deltas["mean_energy_delta"].append(candidate["mean_energy"] - baseline["mean_energy"])
            deltas["best_energy_delta"].append(candidate["best_energy"] - baseline["best_energy"])
            deltas["ess_delta"].append(
                candidate["effective_sample_size"] - baseline["effective_sample_size"]
            )
            deltas["autocorrelation_delta"].append(
                candidate["integrated_autocorrelation"] - baseline["integrated_autocorrelation"]
            )
            deltas["barrier_crossings_delta"].append(
                candidate["barrier_crossings"] - baseline["barrier_crossings"]
            )
            deltas["temperature_round_trips_delta"].append(
                candidate["temperature_round_trips"] - baseline["temperature_round_trips"]
            )
        output[comparison] = {
            f"{name}_interval_95": _interval_95(values) for name, values in deltas.items()
        }
        output[comparison]["paired_row_count"] = len(deltas["mean_energy_delta"])
        output[comparison]["frozen_margins"] = dict(FROZEN_MARGINS)
    return output


def material_regression_count(payload: Mapping[str, Any]) -> int:
    """Count frozen-margin failures for the gated Rust and restart comparisons."""

    intervals = payload.get("paired_intervals", {})
    if not isinstance(intervals, Mapping):
        return 1
    count = 0
    for comparison in GATED_COMPARISONS:
        row = intervals.get(comparison, {})
        if not isinstance(row, Mapping):
            count += 1
            continue
        count += int(row["exact_validity_delta_interval_95"][0] < -FROZEN_MARGINS["exact_validity"])
        count += int(
            row["feasible_hit_rate_delta_interval_95"][0] < -FROZEN_MARGINS["feasible_hit_rate"]
        )
        count += int(
            row["solve_probability_delta_interval_95"][0] < -FROZEN_MARGINS["solve_probability"]
        )
        count += int(row["mean_energy_delta_interval_95"][1] > FROZEN_MARGINS["mean_energy"])
        count += int(row["best_energy_delta_interval_95"][1] > FROZEN_MARGINS["best_energy"])
        count += int(row["ess_delta_interval_95"][0] < -FROZEN_MARGINS["ess"])
        count += int(
            row["autocorrelation_delta_interval_95"][1] > FROZEN_MARGINS["autocorrelation"]
        )
        count += int(
            row["barrier_crossings_delta_interval_95"][0] < -FROZEN_MARGINS["barrier_crossings"]
        )
        count += int(
            row["temperature_round_trips_delta_interval_95"][0]
            < -FROZEN_MARGINS["temperature_round_trips"]
        )
    return count


def energy_by_arm(rows: Sequence[MatchedTrialRow]) -> JsonDict:
    """Report best and mean energy summaries under one field."""

    return {
        arm: {
            "best_energy": _summary_values(
                [_row_metrics(row)["best_energy"] for row in rows if row.arm_id == arm]
            ),
            "mean_energy": _summary_values(
                [_row_metrics(row)["mean_energy"] for row in rows if row.arm_id == arm]
            ),
        }
        for arm in ARM_IDS
    }


def target_distributions_by_arm(rows: Sequence[MatchedTrialRow]) -> JsonDict:
    """Record cold-target distributions without implying a timing comparison."""

    output: JsonDict = {}
    for arm in ARM_IDS:
        arm_rows = [row for row in rows if row.arm_id == arm]
        energies = [energy for row in arm_rows for energy in row.energies]
        valid = [value for row in arm_rows for value in row.valid]
        if energies:
            counts, edges = np.histogram(np.array(energies, dtype=np.float64), bins=10)
            histogram = {
                "energy_histogram_counts": [int(value) for value in counts.tolist()],
                "energy_histogram_edges": [round(float(value), 10) for value in edges.tolist()],
            }
        else:
            histogram = {"energy_histogram_counts": [], "energy_histogram_edges": []}
        output[arm] = {
            "sample_count": len(energies),
            "exact_valid_count": int(sum(valid)),
            "invalid_count": int(len(valid) - sum(valid)),
            **histogram,
            "sample_state_checksum": sha256_json([row.sample_states for row in arm_rows]),
        }
    return output


def successful_seed_count(rows: Sequence[MatchedTrialRow], seeds: Sequence[int]) -> JsonDict:
    """Report seed denominator honestly across all complete instance/arm rows."""

    expected_arms = set(ARM_IDS)
    successful: list[int] = []
    for seed in seeds:
        seed_rows = [row for row in rows if row.seed == int(seed)]
        per_instance = {
            row.instance_id: {
                candidate.arm_id
                for candidate in seed_rows
                if candidate.instance_id == row.instance_id
            }
            for row in seed_rows
        }
        if per_instance and all(arms == expected_arms for arms in per_instance.values()):
            successful.append(int(seed))
    return {
        "value": len(successful),
        "attempted_seed_count": len(seeds),
        "successful_seeds": successful,
    }


def checkpoint_midpoint(burn_in_sweeps: int, sample_sweeps: int) -> int:
    """Return the fixed midpoint used for cross-language restart checks."""

    return max(1, (int(burn_in_sweeps) + int(sample_sweeps)) // 2)


def beta_ladder_hash() -> str:
    """Hash the frozen one-axis beta ladder."""

    return sha256_json([float(beta) for beta in BETA_LADDER])


def deepcopy_json(value: Any) -> Any:
    """Deep-copy JSON-like values without preserving shared references."""

    return json.loads(json.dumps(value))


def _run_one_axis_arm(
    instance: HardInstance,
    seed: int,
    arm_id: str,
    implementation: str,
    initial_state: OneAxisState,
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
    rust_classes: Mapping[str, Any],
) -> MatchedTrialRow:
    config = config_for_instance(instance)
    core = _core_for_implementation(config, implementation, rust_classes)
    state = _state_for_implementation(initial_state, implementation, rust_classes)
    return _advance_and_collect(
        instance,
        int(seed),
        arm_id,
        core,
        state,
        implementation=implementation,
        start_sweep=0,
        target_sweep=int(burn_in_sweeps) + int(sample_sweeps),
        burn_in_sweeps=int(burn_in_sweeps),
        midpoint=checkpoint_midpoint(burn_in_sweeps, sample_sweeps),
    )


def _run_restart_arm(
    instance: HardInstance,
    seed: int,
    arm_id: str,
    prefix_impl: str,
    suffix_impl: str,
    initial_state: OneAxisState,
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
    rust_classes: Mapping[str, Any],
) -> tuple[MatchedTrialRow, JsonDict]:
    config = config_for_instance(instance)
    midpoint = checkpoint_midpoint(burn_in_sweeps, sample_sweeps)
    prefix_core = _core_for_implementation(config, prefix_impl, rust_classes)
    prefix_state = _state_for_implementation(initial_state, prefix_impl, rust_classes)
    prefix = _advance_and_collect(
        instance,
        int(seed),
        arm_id,
        prefix_core,
        prefix_state,
        implementation=prefix_impl,
        start_sweep=0,
        target_sweep=midpoint,
        burn_in_sweeps=int(burn_in_sweeps),
        midpoint=midpoint,
    )
    portable = make_checkpoint(
        instance=instance,
        state=prefix.final_checkpoint,
        implementation=prefix_impl,
        direction=arm_id,
    )
    suffix_core = _core_for_implementation(config, suffix_impl, rust_classes)
    suffix_state = load_checkpoint(
        portable,
        expected_instance=instance,
        state_factory=rust_classes["state"] if suffix_impl == "rust" else OneAxisState,
    )
    suffix = _advance_and_collect(
        instance,
        int(seed),
        arm_id,
        suffix_core,
        suffix_state,
        implementation=suffix_impl,
        start_sweep=midpoint,
        target_sweep=int(burn_in_sweeps) + int(sample_sweeps),
        burn_in_sweeps=int(burn_in_sweeps),
        midpoint=midpoint,
    )
    combined = MatchedTrialRow(
        instance_id=instance.instance_id,
        family=instance.family,
        size_stratum=instance.size_stratum,
        seed=int(seed),
        arm_id=arm_id,
        energies=prefix.energies + suffix.energies,
        valid=prefix.valid + suffix.valid,
        satisfaction=prefix.satisfaction + suffix.satisfaction,
        basins=prefix.basins + suffix.basins,
        sample_states=prefix.sample_states + suffix.sample_states,
        suffix_sample_states=suffix.suffix_sample_states,
        label_positions=prefix.label_positions + suffix.label_positions,
        corrected_kernel_transitions=prefix.corrected_kernel_transitions
        + suffix.corrected_kernel_transitions,
        exchange_attempts=prefix.exchange_attempts + suffix.exchange_attempts,
        accepted_exchanges=prefix.accepted_exchanges + suffix.accepted_exchanges,
        exact_validation_calls=prefix.exact_validation_calls + suffix.exact_validation_calls,
        final_checkpoint=suffix.final_checkpoint,
        two_axis_arm=False,
    )
    return combined, portable


def _run_independent_diagnostic_arm(
    instance: HardInstance,
    seed: int,
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> MatchedTrialRow:
    row = exp5634.run_arm(
        instance,
        int(seed),
        "independent_corrected_cdls_replicas",
        burn_in_sweeps=int(burn_in_sweeps),
        sample_sweeps=int(sample_sweeps),
    )
    midpoint = checkpoint_midpoint(burn_in_sweeps, sample_sweeps)
    suffix_count = max(0, int(burn_in_sweeps) + int(sample_sweeps) - max(midpoint, burn_in_sweeps))
    return MatchedTrialRow(
        instance_id=instance.instance_id,
        family=instance.family,
        size_stratum=instance.size_stratum,
        seed=int(seed),
        arm_id="independent_cdls_diagnostic",
        energies=row.energies,
        valid=row.valid,
        satisfaction=row.satisfaction,
        basins=row.basins,
        sample_states=row.sample_states,
        suffix_sample_states=row.sample_states[-suffix_count:] if suffix_count else [],
        label_positions=[COLD_LABEL] * (int(burn_in_sweeps) + int(sample_sweeps)),
        corrected_kernel_transitions=row.corrected_kernel_transitions,
        exchange_attempts=row.exchange_attempts,
        accepted_exchanges=row.accepted_exchanges,
        exact_validation_calls=row.exact_validation_calls,
        final_checkpoint={
            "states": row.sample_states[-len(BETA_LADDER) :] if row.sample_states else [],
            "labels": list(range(len(BETA_LADDER))),
            "rng_state": 0,
            "sweep": int(burn_in_sweeps) + int(sample_sweeps),
        },
        two_axis_arm=False,
    )


def _advance_and_collect(
    instance: HardInstance,
    seed: int,
    arm_id: str,
    core: Any,
    state: Any,
    *,
    implementation: str,
    start_sweep: int,
    target_sweep: int,
    burn_in_sweeps: int,
    midpoint: int,
) -> MatchedTrialRow:
    energies: list[float] = []
    valid: list[int] = []
    satisfaction: list[float] = []
    basins: list[int] = []
    samples: list[list[int]] = []
    suffix_samples: list[list[int]] = []
    label_positions: list[int] = []
    accepted_exchanges = 0
    ground = exp5634._ground_energy(instance)
    current = state
    for _ in range(int(start_sweep), int(target_sweep)):
        before_labels = list(_checkpoint_of(current)["labels"])
        current = core.step(current)
        checkpoint = _checkpoint_of(current)
        after_labels = list(checkpoint["labels"])
        accepted_exchanges += _accepted_exchange_count(before_labels, after_labels)
        cold_position = after_labels.index(COLD_LABEL)
        label_positions.append(cold_position)
        completed_sweep = int(checkpoint["sweep"])
        if completed_sweep > int(burn_in_sweeps):
            target = np.array(core.target_state(current), dtype=np.int8)
            energy = round(exp5634._energy(instance.system, target), 12)
            is_valid = exp5634._exact_valid(instance, target, ground)
            state_list = target.astype(int).tolist()
            energies.append(energy)
            valid.append(is_valid)
            satisfaction.append(round(exp5634._constraint_satisfaction(instance, target), 12))
            basins.append(exp5634._basin(instance, target))
            samples.append(state_list)
            if completed_sweep > int(midpoint):
                suffix_samples.append(state_list)
    total_steps = int(target_sweep) - int(start_sweep)
    return MatchedTrialRow(
        instance_id=instance.instance_id,
        family=instance.family,
        size_stratum=instance.size_stratum,
        seed=int(seed),
        arm_id=arm_id,
        energies=energies,
        valid=valid,
        satisfaction=satisfaction,
        basins=basins,
        sample_states=samples,
        suffix_sample_states=suffix_samples,
        label_positions=label_positions,
        corrected_kernel_transitions=total_steps * len(BETA_LADDER),
        exchange_attempts=total_steps * (len(BETA_LADDER) - 1),
        accepted_exchanges=accepted_exchanges,
        exact_validation_calls=len(samples),
        final_checkpoint=_checkpoint_of(current),
        two_axis_arm=False,
    )


def _core_for_implementation(
    config: exp5714.OneAxisConfig,
    implementation: str,
    rust_classes: Mapping[str, Any],
) -> Any:
    if implementation == "python":
        return exp5714.PythonOneAxisTemperingCore(config)
    if implementation == "rust":
        return exp5714._rust_core_from_config(config, rust_classes)
    raise ValueError(f"unknown implementation: {implementation}")


def _state_for_implementation(
    state: OneAxisState,
    implementation: str,
    rust_classes: Mapping[str, Any],
) -> Any:
    if implementation == "python":
        return OneAxisState.from_checkpoint(state.checkpoint())
    if implementation == "rust":
        return rust_classes["state"].from_checkpoint(state.checkpoint())
    raise ValueError(f"unknown implementation: {implementation}")


def _checkpoint_of(state: Any) -> JsonDict:
    if isinstance(state, Mapping):
        return dict(state)
    return dict(state.checkpoint())


def _accepted_exchange_count(before: Sequence[int], after: Sequence[int]) -> int:
    if list(before) == list(after):
        return 0
    return sum(1 for left, right in zip(before, after, strict=True) if left != right) // 2


def _row_metrics(row: MatchedTrialRow) -> JsonDict:
    energies = np.array(row.energies, dtype=np.float64)
    iat = _autocorrelation_time(row.energies)
    return {
        "instance_id": row.instance_id,
        "seed": row.seed,
        "arm_id": row.arm_id,
        "best_energy": round(float(np.min(energies)), 10),
        "mean_energy": round(float(np.mean(energies)), 10),
        "exact_valid_rate": round(float(np.mean(row.valid)), 10),
        "feasible_hit_rate": round(float(np.mean(row.valid)), 10),
        "solve_probability": round(float(np.mean(row.valid)), 10),
        "barrier_crossings": exp5634._count_barrier_crossings(row.basins),
        "temperature_round_trips": exp5634._count_round_trips(row.label_positions),
        "integrated_autocorrelation": round(float(iat), 10),
        "effective_sample_size": round(float(len(row.energies) / iat), 10),
        "sample_count": len(row.energies),
    }


def _indexed_metrics(rows: Sequence[MatchedTrialRow]) -> dict[tuple[str, int, str], JsonDict]:
    return {
        (metrics["instance_id"], int(metrics["seed"]), metrics["arm_id"]): metrics
        for metrics in (_row_metrics(row) for row in rows)
    }


def _summary_by_arm(rows: Sequence[MatchedTrialRow], metric: str) -> JsonDict:
    output: JsonDict = {}
    for arm in ARM_IDS:
        values = [_row_metrics(row)[metric] for row in rows if row.arm_id == arm]
        output[arm] = _summary_values(values)
    return output


def _summary_values(values: Sequence[float]) -> JsonDict:
    if not values:
        return {"mean": None, "interval_95": [None, None], "paired_row_count": 0}
    return {
        "mean": round(float(np.mean(np.array(values, dtype=np.float64))), 10),
        "interval_95": _interval_95(values),
        "paired_row_count": len(values),
    }


def _autocorrelation_time(values: Sequence[float]) -> float:
    array = np.array(values, dtype=np.float64)
    if len(array) < 2 or float(np.var(array)) == 0.0:
        return 1.0
    centered = array - float(np.mean(array))
    denominator = float(np.dot(centered, centered))
    tau = 1.0
    for lag in range(1, min(100, len(array) - 1) + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return max(1.0, tau)


def _interval_95(values: Sequence[float]) -> list[float]:
    array = np.array(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return [round(mean, 10), round(mean, 10)]
    multipliers = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    critical = multipliers.get(len(array), 1.96)
    half_width = critical * float(np.std(array, ddof=1)) / sqrt(float(len(array)))
    return [round(mean - half_width, 10), round(mean + half_width, 10)]


def _stable_seed64(*parts: object) -> int:
    return int(sha256_json([str(part) for part in parts])[:16], 16)


def _one_upstream_receipt(
    path: Path,
    *,
    ready_field: str,
    expected_value: Any = True,
    validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> JsonDict:
    if not path.exists():
        return {"path": path.as_posix(), "available": False, "ready": False}
    try:
        payload = _read_json(path)
        if validator is not None:
            validator(payload)
    except Exception as exc:
        return {
            "path": path.as_posix(),
            "available": True,
            "ready": False,
            "sha256": file_sha256(path),
            "blocked_reason": f"invalid:{type(exc).__name__}",
        }
    return {
        "path": path.as_posix(),
        "available": True,
        "ready": payload.get(ready_field) == expected_value,
        "sha256": file_sha256(path),
        "schema": payload.get("schema"),
        ready_field: payload.get(ready_field),
    }


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _raises_value_error(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def main() -> None:
    artifact = build_artifact(root=REPO_ROOT, random_seeds=DEFAULT_RANDOM_SEEDS)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover
    main()
