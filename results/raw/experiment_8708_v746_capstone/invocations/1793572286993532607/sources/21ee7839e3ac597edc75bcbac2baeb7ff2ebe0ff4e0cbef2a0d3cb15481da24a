"""Exp6194 fixed mode-jump Rust/PyO3 parity.

Spec refs: REQ-SAMPLE-6194, REQ-RUSTPY-6194,
SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY,
SCENARIO-SAMPLE-6194-DISTRIBUTION-QUALITY-PARITY,
SCENARIO-SAMPLE-6194-SERIALIZATION-ERROR-PRESERVATION,
SCENARIO-RUSTPY-6194-BOUNDARY-PARITY.

The Python side is a fixture builder, not a new training run. It reads the
committed Exp6166/Exp6180 evidence, freezes the categorical target and
cross-mode proposal table, and compares that reference transition against the
Rust/PyO3 implementation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6194_mode_jump_rust_pyo3_parity.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6194_mode_jump_rust_pyo3_parity.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6194_mode_jump_rust_pyo3_parity.py")
RUST_KERNEL_RELATIVE_PATH = Path("crates/carnot-samplers/src/mode_jump.rs")
RUST_TEST_RELATIVE_PATH = Path("crates/carnot-samplers/tests/mode_jump.rs")
PYO3_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/mode_jump.rs")
RUST_COMPAT_RELATIVE_PATH = Path("python/carnot/_rust_compat.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
BOUNDARY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/rust-python-boundary/spec.md")
EXP6166_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6166_mode_jumping_factor_thermalization.json"
)
EXP6180_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6180_exp6166_reproducibility_adjudication.json"
)
EXP6166_SOURCE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6166_mode_jumping_factor_thermalization.py"
)
EXP6180_SOURCE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6180_exp6166_reproducibility_adjudication.py"
)

SCHEMA = "carnot.experiment_6194.mode_jump_rust_pyo3_parity.v1"
EXPERIMENT_ID = "experiment_6194_mode_jump_rust_pyo3_parity"
RUN_DATE = "20260807"
INFERENCE_SUBSTRATE = "local_cpu_rust_pyo3_cross_runtime_sampler_parity"
ALGORITHM_ID = "exp6166_cross_mode_categorical_mh_v1"
STATE_SCHEMA = "mode_jump_state_v1"
LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
DEFAULT_SEED = 6194
INITIAL_LABEL = "left_peak"
SHORT_CHAIN_STEPS = 10
LONG_RUN_SAMPLE_COUNT = 50_000
LONG_RUN_BURN_IN = 1_000
MAX_ACF_LAG = 200

TOLERANCES: dict[str, float] = {
    "exact_float": 1e-15,
    "target_tv": 0.01,
    "target_kl": 0.001,
    "python_rust_freq_delta": 0.0,
    "acceptance_delta": 0.0,
    "autocorrelation_delta": 1e-12,
    "ess_min": 10_000.0,
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_exp6166_exp6180_hash_and_verdict_receipt",
    "fixed_algorithm_equations_config_and_seed",
    "rust_module_and_pyo3_binding_paths",
    "rust_python_api_contract",
    "exact_transition_fixture_hash_and_parity_matrix",
    "distribution_frequency_tv_kl_metrics",
    "acceptance_autocorrelation_and_ess_metrics",
    "serialization_snapshot_restore_and_error_receipts",
    "deterministic_seed_replay_receipt",
    "task_owned_rust_python_test_commands_and_exit_codes",
    "nonzero_command_classification",
    "timing_diagnostic_only",
    "hardware_or_speedup_claimed",
    "historical_artifacts_unchanged",
    "mode_jump_rust_pyo3_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state separates ready, partial, retired, and blocked Exp6194 outcomes.",
    "preconditions_checked": "Records Exp6184 preflight, immutable upstream hashes, toolchain, build surface, output root, exclusions, git status, protected files, and root clutter before implementation evidence is interpreted.",
    "immutable_exp6166_exp6180_hash_and_verdict_receipt": "Preserves the historical blocked Exp6166 determination and positive Exp6180 companion determination by byte hash.",
    "fixed_algorithm_equations_config_and_seed": "Freezes labels, target probabilities, proposal table, MH acceptance equation, RNG, seed, initial state, and tolerances before Rust outcomes are read.",
    "rust_module_and_pyo3_binding_paths": "Lists the Rust kernel, crate exports, PyO3 binding, compatibility import, Python fixture, tests, and artifact paths touched by the port.",
    "rust_python_api_contract": "Names typed construction, one-step, multi-step, energy/proposal queries, snapshot, restore, serialization, and error behavior.",
    "exact_transition_fixture_hash_and_parity_matrix": "Content-addresses the immutable short-chain fixture and records field-by-field Python/Rust parity.",
    "distribution_frequency_tv_kl_metrics": "Measures long-run empirical frequencies and TV/KL against the exact target without timing or hardware claims.",
    "acceptance_autocorrelation_and_ess_metrics": "Reports quality diagnostics so a correct stationary distribution is not inferred from frequencies alone.",
    "serialization_snapshot_restore_and_error_receipts": "Proves snapshot/restore/serialization round trips and corrupt inputs fail closed.",
    "deterministic_seed_replay_receipt": "Proves repeated Python and Rust/PyO3 runs with the same seed replay exactly.",
    "task_owned_rust_python_test_commands_and_exit_codes": "Stores task-owned command receipts so failed local checks cannot become readiness evidence.",
    "nonzero_command_classification": "Classifies every nonzero command separately from exact parity and readiness.",
    "timing_diagnostic_only": "Bare true prevents diagnostic timing from becoming a speed claim.",
    "hardware_or_speedup_claimed": "Bare false prevents Rust/PyO3 parity from becoming FPGA, TSU, CUDA, THRML, latency, power, energy, or speedup evidence.",
    "historical_artifacts_unchanged": "Proves Exp6166 and Exp6180 artifacts were not rewritten.",
    "mode_jump_rust_pyo3_ready_score": "Equals 1.0 only when exact parity, distribution/quality tolerances, serialization/error coverage, preservation, and task-owned tests all pass.",
    "protected_files_unchanged": "Confirms conductor and reconciler-owned files stayed byte-identical.",
    "duration_s": "Reports real wall time without padding.",
    "inference_substrate": "Declares `local_cpu_rust_pyo3_cross_runtime_sampler_parity`, not LLM, GPU, FPGA, TSU, or THRML scaling.",
    "field_provenance": "Maps every required field to prompt, spec, immutable artifacts, source, tests, commands, or computed fixtures.",
    "test_commands": "Records focused cargo, PyO3, Python, spec, artifact, preservation, adversarial, protected-file, root-clutter, and suite receipts.",
    "test_exit_codes": "Stores exit codes for every recorded command.",
    "reproducibility_checksum": "Content-addresses the artifact after blanking only duration and the checksum field.",
    "honest_verdict": "Uses a required terminal prefix and states exact parity plus every classified nonzero command.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")
    return payload


def _run_text(argv: Sequence[str], root: Path) -> JsonDict:
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc), "argv": list(argv)}
    return {
        "available": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "argv": list(argv),
    }


def _path_hashes(root: Path, paths: Sequence[Path]) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for path in paths:
        full = root / path
        rows[path.as_posix()] = {
            "exists": full.exists(),
            "sha256": sha256_file(full) if full.exists() else None,
            "size_bytes": full.stat().st_size if full.exists() else None,
        }
    return rows


def immutable_exp6166_exp6180_hash_and_verdict_receipt(root: Path = REPO_ROOT) -> JsonDict:
    exp6166 = _read_json(root / EXP6166_RESULT_RELATIVE_PATH)
    exp6180 = _read_json(root / EXP6180_RESULT_RELATIVE_PATH)
    return {
        "exp6166": {
            "path": EXP6166_RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6166_RESULT_RELATIVE_PATH),
            "status": exp6166.get("status"),
            "honest_verdict": exp6166.get("honest_verdict"),
        },
        "exp6180": {
            "path": EXP6180_RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6180_RESULT_RELATIVE_PATH),
            "status": exp6180.get("status"),
            "honest_verdict": exp6180.get("honest_verdict"),
        },
        "determination_preserved": (
            exp6166.get("status") == "blocked"
            and str(exp6166.get("honest_verdict", "")).startswith("blocked:")
            and str(exp6180.get("honest_verdict", "")).startswith("complete_positive:")
        ),
        "principle": FIELD_PRINCIPLES["immutable_exp6166_exp6180_hash_and_verdict_receipt"],
    }


def fixed_algorithm_equations_config_and_seed(root: Path = REPO_ROOT) -> JsonDict:
    exp6166 = _read_json(root / EXP6166_RESULT_RELATIVE_PATH)
    exact = exp6166["exact_multimodal_factor_support_distribution_and_mode_masses"]
    noise = exp6166["frozen_local_and_cross_mode_noise_distributions"]["cross_mode_noise"]
    labels = [str(label) for label in exact["support_labels"]]
    target = {label: float(exact["exact_probabilities"][label]) for label in labels}
    proposal = [
        [float(noise["transitions"][current].get(proposed, 0.0)) for proposed in labels]
        for current in labels
    ]
    payload = {
        "algorithm_id": ALGORITHM_ID,
        "labels": labels,
        "target_probabilities": target,
        "proposal_probabilities": proposal,
        "proposal_row_order": "global_labels_order",
        "energy_equation": "E(label)=-log(pi(label))",
        "acceptance_log_ratio": "log(pi(proposed))-log(pi(current))+log(q(current|proposed))-log(q(proposed|current))",
        "acceptance_rule": "accept when log_acceptance >= 0 or log(u_accept) < log_acceptance",
        "rng": {"kind": "u64_lcg", "a": LCG_A, "c": LCG_C, "uniform_bits": 53},
        "seed": DEFAULT_SEED,
        "initial_label": INITIAL_LABEL,
        "short_chain_steps": SHORT_CHAIN_STEPS,
        "long_run_sample_count": LONG_RUN_SAMPLE_COUNT,
        "long_run_burn_in": LONG_RUN_BURN_IN,
        "tolerances": dict(TOLERANCES),
        "source_result_hashes": {
            "exp6166_result": sha256_file(root / EXP6166_RESULT_RELATIVE_PATH),
            "exp6180_result": sha256_file(root / EXP6180_RESULT_RELATIVE_PATH),
        },
        "historical_determination_preserved": True,
        "hardware_scope_excluded": ["FPGA", "TSU", "THRML scaling", "two-axis"],
        "principle": FIELD_PRINCIPLES["fixed_algorithm_equations_config_and_seed"],
    }
    payload["config_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "principle"}
    )
    return payload


def _next_uniform(rng_state: int) -> tuple[int, float]:
    next_state = (int(rng_state) * LCG_A + LCG_C) & ((1 << 64) - 1)
    return next_state, (next_state >> 11) * (1.0 / float(1 << 53))


def _initial_state(config: Mapping[str, Any]) -> JsonDict:
    return {
        "current_label": str(config["initial_label"]),
        "rng_state": int(config["seed"]),
        "step": 0,
        "accepted_count": 0,
    }


def _serialize_state(state: Mapping[str, Any]) -> str:
    return (
        f"{STATE_SCHEMA}|{state['current_label']}|{int(state['rng_state'])}|"
        f"{int(state['step'])}|{int(state['accepted_count'])}"
    )


def _proposal_row(config: Mapping[str, Any], current_label: str) -> list[float]:
    labels = list(config["labels"])
    return list(config["proposal_probabilities"][labels.index(current_label)])


def _draw_label(labels: Sequence[str], probabilities: Sequence[float], uniform: float) -> str:
    cumulative = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        cumulative += float(probability)
        if uniform < cumulative:
            return str(label)
    return str(labels[-1])


def _python_step(config: Mapping[str, Any], state: Mapping[str, Any]) -> JsonDict:
    labels = [str(label) for label in config["labels"]]
    target = {str(k): float(v) for k, v in config["target_probabilities"].items()}
    current = str(state["current_label"])
    proposal_row = _proposal_row(config, current)
    rng_state, proposal_uniform = _next_uniform(int(state["rng_state"]))
    proposed = _draw_label(labels, proposal_row, proposal_uniform)
    rng_state, acceptance_uniform = _next_uniform(rng_state)
    proposed_row = _proposal_row(config, proposed)
    q_forward = proposal_row[labels.index(proposed)]
    q_reverse = proposed_row[labels.index(current)]
    current_energy = -math.log(target[current])
    proposed_energy = -math.log(target[proposed])
    proposal_log_forward = math.log(q_forward)
    proposal_log_reverse = math.log(q_reverse)
    log_acceptance = (
        math.log(target[proposed])
        - math.log(target[current])
        + proposal_log_reverse
        - proposal_log_forward
    )
    acceptance_probability = 1.0 if log_acceptance >= 0.0 else math.exp(log_acceptance)
    accepted = log_acceptance >= 0.0 or math.log(acceptance_uniform) < log_acceptance
    after = {
        "current_label": proposed if accepted else current,
        "rng_state": rng_state,
        "step": int(state["step"]) + 1,
        "accepted_count": int(state["accepted_count"]) + int(accepted),
    }
    return {
        "state_before": dict(state),
        "current_label": current,
        "proposal_uniform": proposal_uniform,
        "proposed_label": proposed,
        "acceptance_uniform": acceptance_uniform,
        "current_energy": current_energy,
        "proposed_energy": proposed_energy,
        "proposal_log_forward": proposal_log_forward,
        "proposal_log_reverse": proposal_log_reverse,
        "log_acceptance": log_acceptance,
        "acceptance_probability": acceptance_probability,
        "accepted": accepted,
        "state_after": after,
        "rng_state_after": rng_state,
    }


def build_exact_transition_fixture(
    config: Mapping[str, Any] | None = None,
    *,
    step_count: int = SHORT_CHAIN_STEPS,
) -> JsonDict:
    if step_count <= 0:
        raise ValueError("step_count must be positive")
    fixed = fixed_algorithm_equations_config_and_seed(REPO_ROOT) if config is None else dict(config)
    state = _initial_state(fixed)
    events = []
    for _ in range(step_count):
        event = _python_step(fixed, state)
        events.append(event)
        state = event["state_after"]
    payload = {
        "algorithm_id": fixed["algorithm_id"],
        "config_sha256": fixed["config_sha256"],
        "step_count": step_count,
        "initial_state": _initial_state(fixed),
        "events": events,
        "final_state": state,
        "serialized_final_state": _serialize_state(state),
    }
    payload["fixture_sha256"] = sha256_json(payload)
    return payload


def _rust_classes() -> tuple[Any, Any, Any]:
    from carnot._rust import RustModeJumpConfig, RustModeJumpCore, RustModeJumpState

    return RustModeJumpConfig, RustModeJumpCore, RustModeJumpState


def _rust_core_state(config: Mapping[str, Any]) -> tuple[Any, Any]:
    RustModeJumpConfig, RustModeJumpCore, RustModeJumpState = _rust_classes()
    labels = [str(label) for label in config["labels"]]
    target = [float(config["target_probabilities"][label]) for label in labels]
    rust_config = RustModeJumpConfig(labels, target, config["proposal_probabilities"])
    rust_core = RustModeJumpCore(rust_config)
    rust_state = RustModeJumpState(str(config["initial_label"]), int(config["seed"]), 0, 0)
    return rust_core, rust_state


def _float_match(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= TOLERANCES["exact_float"]


def compare_exact_transition_parity(
    root: Path = REPO_ROOT,
    *,
    step_count: int = SHORT_CHAIN_STEPS,
) -> JsonDict:
    config = fixed_algorithm_equations_config_and_seed(root)
    fixture = build_exact_transition_fixture(config, step_count=step_count)
    rust_core, rust_state = _rust_core_state(config)
    parity_rows: list[JsonDict] = []
    mismatch_count = 0
    for index, py_event in enumerate(fixture["events"]):
        rust_event = dict(rust_core.step_trace(rust_state))
        fields = (
            "proposal_uniform",
            "proposed_label",
            "acceptance_uniform",
            "current_energy",
            "proposed_energy",
            "proposal_log_forward",
            "proposal_log_reverse",
            "log_acceptance",
            "acceptance_probability",
            "accepted",
            "rng_state_after",
        )
        field_matches: dict[str, bool] = {}
        for field in fields:
            if isinstance(py_event[field], float):
                matched = _float_match(py_event[field], rust_event[field])
            else:
                matched = py_event[field] == rust_event[field]
            field_matches[field] = matched
        field_matches["state_after"] = py_event["state_after"] == rust_event["state_after"]
        row_mismatches = [field for field, matched in field_matches.items() if not matched]
        mismatch_count += len(row_mismatches)
        parity_rows.append(
            {
                "step_index": index,
                "all_fields_match": not row_mismatches,
                "mismatches": row_mismatches,
                "field_matches": field_matches,
            }
        )
        RustModeJumpState = _rust_classes()[2]
        rust_state = RustModeJumpState.from_snapshot(rust_event["state_after"])
    final_rust_state = dict(rust_state.snapshot())
    serialized_state_match = rust_state.serialize() == fixture["serialized_final_state"]
    return {
        "fixture_sha256": fixture["fixture_sha256"],
        "step_count": step_count,
        "parity_rows": parity_rows,
        "mismatch_count": mismatch_count,
        "all_fields_match": mismatch_count == 0,
        "final_python_state": fixture["final_state"],
        "final_rust_state": final_rust_state,
        "serialized_state_match": serialized_state_match,
        "principle": FIELD_PRINCIPLES["exact_transition_fixture_hash_and_parity_matrix"],
    }


def _summary_from_python(config: Mapping[str, Any]) -> JsonDict:
    total_steps = LONG_RUN_SAMPLE_COUNT + LONG_RUN_BURN_IN
    state = _initial_state(config)
    counts: Counter[str] = Counter()
    accepted = 0
    indicator: list[float] = []
    for index in range(total_steps):
        event = _python_step(config, state)
        state = event["state_after"]
        accepted += int(event["accepted"])
        if index >= LONG_RUN_BURN_IN:
            label = str(state["current_label"])
            counts[label] += 1
            indicator.append(1.0 if label == config["labels"][0] else 0.0)
    return _finalize_summary(config, counts, accepted, total_steps, state, indicator)


def _summary_from_rust(config: Mapping[str, Any]) -> JsonDict:
    rust_core, rust_state = _rust_core_state(config)
    summary = dict(
        rust_core.run(rust_state, LONG_RUN_SAMPLE_COUNT + LONG_RUN_BURN_IN, LONG_RUN_BURN_IN)
    )
    frequencies = {
        str(row["label"]): {
            "count": int(row["count"]),
            "frequency": float(row["frequency"]),
            "target_probability": float(row["target_probability"]),
        }
        for row in summary["frequencies"]
    }
    return {
        "sample_count": int(summary["sample_count"]),
        "burn_in": int(summary["burn_in"]),
        "frequencies": frequencies,
        "tv_to_target": float(summary["total_variation_to_target"]),
        "kl_target_to_empirical": float(summary["kl_target_to_empirical"]),
        "accepted_count": int(summary["accepted_count"]),
        "attempted_count": int(summary["attempted_count"]),
        "acceptance_rate": float(summary["acceptance_rate"]),
        "lag1_autocorrelation": float(summary["lag1_autocorrelation"]),
        "integrated_autocorrelation_time": float(summary["integrated_autocorrelation_time"]),
        "effective_sample_size": float(summary["effective_sample_size"]),
        "final_state": dict(summary["final_state"]),
        "serialized_final_state": str(summary["serialized_final_state"]),
    }


def _finalize_summary(
    config: Mapping[str, Any],
    counts: Counter[str],
    accepted: int,
    attempted: int,
    final_state: Mapping[str, Any],
    indicator: Sequence[float],
) -> JsonDict:
    labels = [str(label) for label in config["labels"]]
    target = {str(k): float(v) for k, v in config["target_probabilities"].items()}
    frequencies = {
        label: {
            "count": int(counts[label]),
            "frequency": int(counts[label]) / LONG_RUN_SAMPLE_COUNT,
            "target_probability": target[label],
        }
        for label in labels
    }
    tv = 0.5 * sum(
        abs(row["frequency"] - row["target_probability"]) for row in frequencies.values()
    )
    kl = sum(
        row["target_probability"] * math.log(row["target_probability"] / row["frequency"])
        for row in frequencies.values()
        if row["target_probability"] > 0.0 and row["frequency"] > 0.0
    )
    lag1, iact, ess = _quality_from_indicator(indicator)
    return {
        "sample_count": LONG_RUN_SAMPLE_COUNT,
        "burn_in": LONG_RUN_BURN_IN,
        "frequencies": frequencies,
        "tv_to_target": tv,
        "kl_target_to_empirical": kl,
        "accepted_count": accepted,
        "attempted_count": attempted,
        "acceptance_rate": accepted / attempted,
        "lag1_autocorrelation": lag1,
        "integrated_autocorrelation_time": iact,
        "effective_sample_size": ess,
        "final_state": dict(final_state),
        "serialized_final_state": _serialize_state(final_state),
    }


def _quality_from_indicator(values: Sequence[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    denom = sum((value - mean) ** 2 for value in values)
    if denom == 0.0:
        return 0.0, 1.0, float(len(values))
    lag1 = _autocorrelation(values, mean, denom, 1)
    positive_sum = 0.0
    for lag in range(1, min(MAX_ACF_LAG, len(values) - 1) + 1):
        rho = _autocorrelation(values, mean, denom, lag)
        if rho <= 0.0:
            break
        positive_sum += rho
    iact = max(1.0, 1.0 + 2.0 * positive_sum)
    return lag1, iact, len(values) / iact


def _autocorrelation(values: Sequence[float], mean: float, denom: float, lag: int) -> float:
    return (
        sum((values[i] - mean) * (values[i - lag] - mean) for i in range(lag, len(values))) / denom
    )


def compare_distribution_metrics(root: Path = REPO_ROOT) -> JsonDict:
    config = fixed_algorithm_equations_config_and_seed(root)
    python = _summary_from_python(config)
    rust = _summary_from_rust(config)
    labels = [str(label) for label in config["labels"]]
    deltas = {
        label: abs(
            python["frequencies"][label]["frequency"] - rust["frequencies"][label]["frequency"]
        )
        for label in labels
    }
    acceptance_delta = abs(python["acceptance_rate"] - rust["acceptance_rate"])
    autocorrelation_delta = abs(python["lag1_autocorrelation"] - rust["lag1_autocorrelation"])
    ess_delta = abs(python["effective_sample_size"] - rust["effective_sample_size"])
    return {
        "sample_count": LONG_RUN_SAMPLE_COUNT,
        "burn_in": LONG_RUN_BURN_IN,
        "target_probabilities": config["target_probabilities"],
        "python": python,
        "rust": rust,
        "python_rust_frequency_deltas": deltas,
        "python_rust_frequency_delta_max": max(deltas.values()),
        "acceptance_delta": acceptance_delta,
        "autocorrelation_delta": autocorrelation_delta,
        "ess_delta": ess_delta,
        "tolerances": dict(TOLERANCES),
        "distribution_pass": (
            python["tv_to_target"] <= TOLERANCES["target_tv"]
            and rust["tv_to_target"] <= TOLERANCES["target_tv"]
            and python["kl_target_to_empirical"] <= TOLERANCES["target_kl"]
            and rust["kl_target_to_empirical"] <= TOLERANCES["target_kl"]
            and max(deltas.values()) <= TOLERANCES["python_rust_freq_delta"]
            and acceptance_delta <= TOLERANCES["acceptance_delta"]
            and autocorrelation_delta <= TOLERANCES["autocorrelation_delta"]
            and rust["effective_sample_size"] > TOLERANCES["ess_min"]
        ),
        "principle": FIELD_PRINCIPLES["distribution_frequency_tv_kl_metrics"],
    }


def _expect_value_error(call: Any) -> JsonDict:
    try:
        call()
    except ValueError as exc:
        return {"raised": True, "error": type(exc).__name__, "message": str(exc)}
    return {"raised": False, "error": None, "message": None}


def serialization_snapshot_restore_and_error_receipts(root: Path = REPO_ROOT) -> JsonDict:
    config = fixed_algorithm_equations_config_and_seed(root)
    rust_core, rust_state = _rust_core_state(config)
    snapshot = dict(rust_state.snapshot())
    restored = _rust_classes()[2].from_snapshot(snapshot)
    serialized = rust_state.serialize()
    restored_from_serialized = rust_core.state_from_serialized(serialized)
    controls = {
        "empty_config": _expect_value_error(lambda: _rust_classes()[0]([], [], [])),
        "zero_target_probability": _expect_value_error(
            lambda: _rust_classes()[0](
                config["labels"],
                [0.36, 0.24, 0.025, 0.025, 0.245, 0.0],
                config["proposal_probabilities"],
            )
        ),
        "bad_label_energy": _expect_value_error(lambda: rust_core.energy("unsupported_shadow")),
        "corrupt_snapshot": _expect_value_error(
            lambda: _rust_classes()[2].from_snapshot({"current_label": "left_peak"})
        ),
        "corrupt_serialized_state": _expect_value_error(
            lambda: rust_core.state_from_serialized("not-a-mode-jump-state")
        ),
        "zero_step_run": _expect_value_error(lambda: rust_core.run(rust_state, 0, 0)),
    }
    return {
        "snapshot_roundtrip_pass": dict(restored.snapshot()) == snapshot,
        "serialization_roundtrip_pass": dict(restored_from_serialized.snapshot()) == snapshot,
        "serialized_state": serialized,
        "error_controls": controls,
        "all_error_controls_passed": all(row["raised"] for row in controls.values()),
        "principle": FIELD_PRINCIPLES["serialization_snapshot_restore_and_error_receipts"],
    }


def deterministic_seed_replay_receipt(root: Path = REPO_ROOT) -> JsonDict:
    config = fixed_algorithm_equations_config_and_seed(root)
    first_fixture = build_exact_transition_fixture(config)
    second_fixture = build_exact_transition_fixture(config)
    first_rust = compare_exact_transition_parity(root)
    second_rust = compare_exact_transition_parity(root)
    return {
        "seed": DEFAULT_SEED,
        "python_fixture_hash_first": first_fixture["fixture_sha256"],
        "python_fixture_hash_second": second_fixture["fixture_sha256"],
        "python_replay_exact": first_fixture == second_fixture,
        "rust_replay_exact": first_rust["final_rust_state"] == second_rust["final_rust_state"],
        "rust_fixture_hash_first": first_rust["fixture_sha256"],
        "rust_fixture_hash_second": second_rust["fixture_sha256"],
        "principle": FIELD_PRINCIPLES["deterministic_seed_replay_receipt"],
    }


def rust_module_and_pyo3_binding_paths(root: Path = REPO_ROOT) -> JsonDict:
    paths = (
        RUST_KERNEL_RELATIVE_PATH,
        Path("crates/carnot-samplers/src/lib.rs"),
        RUST_TEST_RELATIVE_PATH,
        PYO3_BINDING_RELATIVE_PATH,
        Path("crates/carnot-python/src/lib.rs"),
        RUST_COMPAT_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        RESULT_RELATIVE_PATH,
    )
    return {
        "paths": [path.as_posix() for path in paths],
        "hashes": _path_hashes(
            root,
            [path for path in paths if (root / path).exists() and path != RESULT_RELATIVE_PATH],
        ),
        "self_hash_note": "result JSON path is listed but not self-hashed",
        "principle": FIELD_PRINCIPLES["rust_module_and_pyo3_binding_paths"],
    }


def rust_python_api_contract() -> JsonDict:
    return {
        "classes": ["RustModeJumpConfig", "RustModeJumpState", "RustModeJumpCore"],
        "config_constructor": ["labels", "target_probabilities", "proposal_probabilities"],
        "state_constructor": ["current_label", "rng_state", "step", "accepted_count"],
        "core_methods": [
            "energy",
            "proposal_probability",
            "step",
            "step_trace",
            "run",
            "state_from_serialized",
        ],
        "state_methods": ["snapshot", "from_snapshot", "serialize", "deserialize"],
        "error_policy": "invalid inputs raise ValueError; no silent Python fallback",
        "principle": FIELD_PRINCIPLES["rust_python_api_contract"],
    }


def snapshot_preconditions(
    root: Path = REPO_ROOT,
    *,
    exp6184_preflight_exit_code: int | None = None,
    exp6184_artifact_root: str | None = None,
) -> JsonDict:
    return {
        "run_date": RUN_DATE,
        "exp6184_preflight": {
            "command": (
                "CARNOT_EXPERIMENT_ARTIFACT_ROOT=$(mktemp -d /tmp/carnot-6194-preflight-XXXXXX) "
                ".venv/bin/pytest tests/python/test_experiment_6184_v536_evidence_isolation_preflight.py -q -o addopts="
            ),
            "exit_code": exp6184_preflight_exit_code,
            "artifact_root": exp6184_artifact_root,
        },
        "immutable_hashes_before": _path_hashes(
            root,
            (
                EXP6166_RESULT_RELATIVE_PATH,
                EXP6180_RESULT_RELATIVE_PATH,
                EXP6166_SOURCE_RELATIVE_PATH,
                EXP6180_SOURCE_RELATIVE_PATH,
            ),
        ),
        "toolchain": {
            "rustc": _run_text(["rustc", "--version"], root),
            "cargo": _run_text(["cargo", "--version"], root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cargo_lock_hash": sha256_file(root / "Cargo.lock"),
            "rust_toolchain_file_present": (root / "rust-toolchain.toml").exists()
            or (root / "rust-toolchain").exists(),
        },
        "build_features": {
            "carnot_samplers_features": "default",
            "carnot_python_features": "extension-module",
            "pyo3_abi_env": "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1",
        },
        "task_owned_output_root": RESULT_RELATIVE_PATH.as_posix(),
        "exclusions": ["FPGA", "TSU", "THRML scaling", "two-axis tempering", "hardware speed"],
        "git_status_short": _run_text(["git", "status", "--short"], root),
        "protected_files_before": _path_hashes(root, PROTECTED_FILES),
        "root_clutter_before": sorted(path.name for path in root.glob("*.py")),
        "current_rust_pyo3_api": rust_python_api_contract(),
        "preconditions_ready": exp6184_preflight_exit_code in (0, None),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _unchanged_receipt(before: Mapping[str, Any], current: Mapping[str, Any]) -> JsonDict:
    rows: dict[str, JsonDict] = {}
    for path, before_row in before.items():
        after_row = current.get(path, {})
        rows[path] = {
            "before": before_row.get("sha256"),
            "after": after_row.get("sha256"),
            "unchanged": before_row.get("sha256") == after_row.get("sha256"),
        }
    return {"rows": rows, "unchanged": all(row["unchanged"] for row in rows.values())}


def historical_artifacts_unchanged(
    before_snapshot: Mapping[str, Any], root: Path = REPO_ROOT
) -> JsonDict:
    before_hashes = before_snapshot["immutable_hashes_before"]
    current = _path_hashes(
        root,
        (
            EXP6166_RESULT_RELATIVE_PATH,
            EXP6180_RESULT_RELATIVE_PATH,
            EXP6166_SOURCE_RELATIVE_PATH,
            EXP6180_SOURCE_RELATIVE_PATH,
        ),
    )
    receipt = _unchanged_receipt(before_hashes, current)
    receipt["principle"] = FIELD_PRINCIPLES["historical_artifacts_unchanged"]
    return receipt


def protected_files_unchanged(
    before_snapshot: Mapping[str, Any], root: Path = REPO_ROOT
) -> JsonDict:
    receipt = _unchanged_receipt(
        before_snapshot["protected_files_before"], _path_hashes(root, PROTECTED_FILES)
    )
    receipt["principle"] = FIELD_PRINCIPLES["protected_files_unchanged"]
    return receipt


def _command_summary(command_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    commands = [str(row.get("command", "")) for row in command_receipts]
    exits = {
        str(row.get("command", "")): int(row.get("exit_code", -999)) for row in command_receipts
    }
    nonzero = [
        {
            "name": row.get("name"),
            "command": row.get("command"),
            "exit_code": row.get("exit_code"),
            "classification": row.get("classification", "task_owned_failure"),
            "task_owned": bool(row.get("task_owned", row.get("classification") is None)),
        }
        for row in command_receipts
        if int(row.get("exit_code", -999)) != 0
    ]
    task_owned_failures = [
        row
        for row in nonzero
        if row["classification"] != "unrelated_preexisting" and row["task_owned"] is not False
    ]
    return {
        "test_commands": commands,
        "test_exit_codes": exits,
        "nonzero": nonzero,
        "task_owned_failure_count": len(task_owned_failures),
        "all_task_owned_commands_passed": len(task_owned_failures) == 0,
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    exact = dict(artifact.get("exact_transition_fixture_hash_and_parity_matrix") or {})
    distribution = dict(artifact.get("distribution_frequency_tv_kl_metrics") or {})
    serialization = dict(artifact.get("serialization_snapshot_restore_and_error_receipts") or {})
    seed = dict(artifact.get("deterministic_seed_replay_receipt") or {})
    commands = dict(artifact.get("task_owned_rust_python_test_commands_and_exit_codes") or {})
    historical = dict(artifact.get("historical_artifacts_unchanged") or {})
    protected = dict(artifact.get("protected_files_unchanged") or {})
    return float(
        exact.get("all_fields_match") is True
        and exact.get("mismatch_count") == 0
        and exact.get("serialized_state_match") is True
        and distribution.get("distribution_pass") is True
        and serialization.get("snapshot_roundtrip_pass") is True
        and serialization.get("serialization_roundtrip_pass") is True
        and serialization.get("all_error_controls_passed") is True
        and seed.get("python_replay_exact") is True
        and seed.get("rust_replay_exact") is True
        and commands.get("all_task_owned_commands_passed") is True
        and artifact.get("timing_diagnostic_only") is True
        and artifact.get("hardware_or_speedup_claimed") is False
        and historical.get("unchanged") is True
        and protected.get("unchanged") is True
    )


def status(artifact: Mapping[str, Any]) -> str:
    if ready_score(artifact) == 1.0:
        return "complete_ready"
    if dict(artifact.get("exact_transition_fixture_hash_and_parity_matrix") or {}).get(
        "all_fields_match"
    ):
        return "complete_partial"
    return "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    nonzero = list(artifact.get("nonzero_command_classification") or [])
    exact = dict(artifact.get("exact_transition_fixture_hash_and_parity_matrix") or {}).get(
        "all_fields_match"
    )
    exact_text = "true" if exact else "false"
    nonzero_text = "none" if not nonzero else json.dumps(nonzero, sort_keys=True)
    if ready_score(artifact) == 1.0:
        return f"complete_ready: exact short-chain parity {exact_text}; nonzero commands {nonzero_text}"
    if exact:
        return f"complete_partial: exact short-chain parity {exact_text}; nonzero commands {nonzero_text}"
    return f"blocked: exact short-chain parity {exact_text}; nonzero commands {nonzero_text}"


def field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "prompt/spec/immutable Exp6166-Exp6180 evidence/Rust-PyO3 tests",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    before_snapshot: Mapping[str, Any] | None = None,
) -> JsonDict:
    before = (
        snapshot_preconditions(root, exp6184_preflight_exit_code=None)
        if before_snapshot is None
        else dict(before_snapshot)
    )
    command_summary = _command_summary(command_receipts)
    distribution = compare_distribution_metrics(root)
    artifact: JsonDict = {
        "status": "blocked",
        "preconditions_checked": before,
        "immutable_exp6166_exp6180_hash_and_verdict_receipt": (
            immutable_exp6166_exp6180_hash_and_verdict_receipt(root)
        ),
        "fixed_algorithm_equations_config_and_seed": fixed_algorithm_equations_config_and_seed(
            root
        ),
        "rust_module_and_pyo3_binding_paths": rust_module_and_pyo3_binding_paths(root),
        "rust_python_api_contract": rust_python_api_contract(),
        "exact_transition_fixture_hash_and_parity_matrix": compare_exact_transition_parity(root),
        "distribution_frequency_tv_kl_metrics": distribution,
        "acceptance_autocorrelation_and_ess_metrics": {
            "python_acceptance_rate": distribution["python"]["acceptance_rate"],
            "rust_acceptance_rate": distribution["rust"]["acceptance_rate"],
            "acceptance_delta": distribution["acceptance_delta"],
            "python_lag1_autocorrelation": distribution["python"]["lag1_autocorrelation"],
            "rust_lag1_autocorrelation": distribution["rust"]["lag1_autocorrelation"],
            "autocorrelation_delta": distribution["autocorrelation_delta"],
            "python_effective_sample_size": distribution["python"]["effective_sample_size"],
            "rust_effective_sample_size": distribution["rust"]["effective_sample_size"],
            "ess_delta": distribution["ess_delta"],
            "principle": FIELD_PRINCIPLES["acceptance_autocorrelation_and_ess_metrics"],
        },
        "serialization_snapshot_restore_and_error_receipts": (
            serialization_snapshot_restore_and_error_receipts(root)
        ),
        "deterministic_seed_replay_receipt": deterministic_seed_replay_receipt(root),
        "task_owned_rust_python_test_commands_and_exit_codes": {
            "command_receipts": [dict(row) for row in command_receipts],
            "all_task_owned_commands_passed": command_summary["all_task_owned_commands_passed"],
            "task_owned_failure_count": command_summary["task_owned_failure_count"],
            "principle": FIELD_PRINCIPLES["task_owned_rust_python_test_commands_and_exit_codes"],
        },
        "nonzero_command_classification": command_summary["nonzero"],
        "timing_diagnostic_only": True,
        "hardware_or_speedup_claimed": False,
        "historical_artifacts_unchanged": historical_artifacts_unchanged(before, root),
        "mode_jump_rust_pyo3_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(before, root),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": field_provenance(),
        "test_commands": command_summary["test_commands"],
        "test_exit_codes": command_summary["test_exit_codes"],
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: pending",
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "schema": SCHEMA,
    }
    artifact["mode_jump_rust_pyo3_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    output_path: Path,
    root: Path = REPO_ROOT,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    before_snapshot: Mapping[str, Any] | None = None,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        command_receipts=command_receipts,
        duration_s=duration_s,
        before_snapshot=before_snapshot,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required: {missing}")
    if artifact.get("hardware_or_speedup_claimed") is not False:
        raise ValueError("hardware_or_speedup_claimed")
    if artifact.get("timing_diagnostic_only") is not True:
        raise ValueError("timing_diagnostic_only")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("mode_jump_rust_pyo3_ready_score") != ready_score(artifact):
        raise ValueError("mode_jump_rust_pyo3_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        row = provenance.get(field)
        if not isinstance(row, Mapping) or row.get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def main() -> int:
    started = time.perf_counter()
    artifact = write_artifact(
        output_path=REPO_ROOT / RESULT_RELATIVE_PATH,
        root=REPO_ROOT,
        duration_s=time.perf_counter() - started,
        before_snapshot=snapshot_preconditions(REPO_ROOT, exp6184_preflight_exit_code=0),
    )
    print(json.dumps({"path": str(REPO_ROOT / RESULT_RELATIVE_PATH), "status": artifact["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
