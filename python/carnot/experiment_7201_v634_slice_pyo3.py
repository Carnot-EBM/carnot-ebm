"""Measure the persistent PyO3 boundary for fixed-cardinality pair swaps.

The experiment reuses the shipped Rust kernel and the Exp7189 Python control.
It keeps parity evidence separate from cost attribution. The result does not
create a speed gate or claim a sampler-quality improvement.

Spec: REQ-RUSTPY-7201 and SCENARIO-RUSTPY-7201-PERSISTENT-PARITY.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7189_v633_rust_slice_parity as exp7189


JsonDict = dict[str, Any]

RUN_DATE = "20260911"
TASK_ID = "exp7201-slice-pyo3"
MILESTONE = "2026.09.634"
RESULT_PATH = Path("results/experiment_7201_v634_slice_pyo3.json")
CHECKPOINT_DIR = Path("results/checkpoints")
SPEC_PATH = Path("openspec/capabilities/rust-python-boundary/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
UPSTREAM_RESULT_PATH = Path("results/experiment_7189_v633_rust_slice_parity.json")
BRIDGE_RELATIVE_PATH = Path("target/release/fixed-cardinality-bridge")
RUST_LIBRARY_PATH = Path("target/release/libcarnot_python.so")
TOLERANCE = 1.0e-12
REPLAY_STEPS = 32
REPLAY_SEED = 720100
DISTRIBUTION_SEEDS = tuple(range(720100, 720110))
DISTRIBUTION_BURN_IN = exp7189.DISTRIBUTION_BURN_IN
DISTRIBUTION_RETAINED = exp7189.DISTRIBUTION_RETAINED
TV_LIMIT = exp7189.TV_LIMIT
ENERGY_MEAN_LIMIT = exp7189.ENERGY_MEAN_LIMIT
PROFILE_SIZES = exp7189.THROUGHPUT_SIZES
PROFILE_CARDINALITIES = exp7189.THROUGHPUT_CARDINALITIES
PROFILE_SEEDS = slices.BENCHMARK_SEEDS
PROFILE_STEPS = slices.ENERGY_BUDGET
MODEL_SPECS: list[JsonDict] = []

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": None,
    "prior_failures": [
        {
            "experiment_id": "exp7189-rust-slice-parity",
            "verdict": (
                "null: compiled Rust matched every explicit Python transition, all independent "
                "streams matched the Exp7187 exact law, and the bounded serialized E2E passed, "
                "but at least one measured deployment latency speedup was below NFR-01's 10x "
                "target. The parity path is ready; the speed result is retained as a performance "
                "null."
            ),
            "addressed_by": (
                "Replace per-call subprocess serialization with a persistent compiled PyO3 "
                "boundary; preserve the same kernel and report speed separately after parity."
            ),
            "retire_if_same_verdict": True,
        }
    ],
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    UPSTREAM_RESULT_PATH,
    Path("crates/carnot-samplers/src/fixed_cardinality.rs"),
    Path("crates/carnot-samplers/src/bin/fixed-cardinality-bridge.rs"),
    Path("crates/carnot-python/Cargo.toml"),
    Path("crates/carnot-python/src/fixed_cardinality.rs"),
    Path("crates/carnot-python/src/lib.rs"),
    Path("python/carnot/experiment_7189_v633_rust_slice_parity.py"),
    Path("python/carnot/experiment_7201_v634_slice_pyo3.py"),
    Path("python/carnot/samplers/backend.py"),
    SPEC_PATH,
    Path("scripts/experiments/experiment_7201_v634_slice_pyo3.py"),
    Path("tests/python/test_experiment_7201_v634_slice_pyo3.py"),
)

FIELD_PRINCIPLES = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260911, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": (
        "Record planned and completed counts, independent units and exclusions."
    ),
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": (
        "Every blocked verdict names the failed check, upstream, field, expected and observed "
        "value."
    ),
    "verifier_is_oracle": (
        "True when verification uses the same correctness authority; separate implementations "
        "alone do not remove circularity."
    ),
    "verdict_class": (
        "Use positive | circular_positive | null | blocked | disqualified | partial. Only "
        "incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings, including nulls; blocked_* for "
        "external blocks. Never promote infrastructure readiness as scientific benefit."
    ),
    "pyo3_slice_ready_score": ("Readiness requires real compiled binding execution and parity."),
    "phase_cost_rows": "Measured attribution tests the bridge-overhead explanation.",
    "compiled_binding_receipt": "Binary hash and loaded module path prove Rust execution.",
    "transition_rows": "Explicit proposals isolate language parity from RNG differences.",
    "e2e_receipts": "Binding and serialization round trips must execute.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES) | {
    "task_id",
    "milestone",
    "host_identity",
    "distribution_rows",
    "bridge_overhead_hypothesis",
    "bridge_overhead_hypothesis_supported",
    "buffer_reuse_receipt",
    "upstream_performance_null",
    "spec_refs",
}

canonical_json = exp7189.canonical_json
sha256_bytes = exp7189.sha256_bytes
sha256_json = exp7189.sha256_json
sha256_file = exp7189.sha256_file


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush each numbered boundary so the conductor can see real work."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash stable artifact content without hashing the checksum itself."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _task_contract(root: Path) -> JsonDict | None:
    """Read the exact active task contract instead of inferring its gates."""

    path = root / ROADMAP_PATH
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return None
    task = next(
        (row for row in document["tasks"] if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if task is None:
        return None
    return {
        "id": task.get("id"),
        "milestone": task.get("milestone"),
        "deliverable": task.get("deliverable"),
        "gated_on": task.get("gated_on"),
        "prior_failures": task.get("prior_failures"),
    }


def _manifest_mentions_experiment(value: Any, experiment_number: str) -> bool:
    """Inspect identifier fields without matching unrelated prose or dates."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"experiment_id", "experiment_ids"}:
                identifiers = child if isinstance(child, list) else [child]
                if any(experiment_number in str(identifier) for identifier in identifiers):
                    return True
            if _manifest_mentions_experiment(child, experiment_number):
                return True
    elif isinstance(value, list):
        return any(_manifest_mentions_experiment(child, experiment_number) for child in value)
    return False


def upstream_quarantine_observation(
    upstream: Mapping[str, Any], *, manifest_match: bool
) -> JsonDict:
    """Combine artifact flags with the exclusion manifest before data use."""

    flag_fields = (
        "flagged_adversarial",
        "quarantined",
        "quarantine",
        "quarantine_flags",
        "disqualified",
    )
    observed = {field: upstream.get(field) for field in flag_fields}
    active = [field for field, value in observed.items() if value not in (None, False, "", [], {})]
    if manifest_match:
        active.append("exclusion_manifest")
    return {
        **observed,
        "exclusion_manifest_match": manifest_match,
        "active_flags": active,
        "quarantined": bool(active),
    }


def collect_preconditions(
    root: Path,
    *,
    result_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Print before each check and retain the observed prerequisite values."""

    result = result_path or root / RESULT_PATH
    checkpoints = checkpoint_dir or root / CHECKPOINT_DIR
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    def announce(name: str) -> None:
        print(f"[phase 0 check start] {name}", flush=True)

    announce("required source bytes and hashes")
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    hashes = {
        str(path): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[str(path)] is not None and sizes[str(path)] > 0
    }
    announce("driving capability specification")
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""
    announce("active roadmap task fields")
    task_contract = _task_contract(root)
    announce("upstream artifact bytes and fields")
    upstream_file = root / UPSTREAM_RESULT_PATH
    try:
        upstream = json.loads(upstream_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, AttributeError):
        upstream = {}
    announce("exclusion manifest quarantine flags")
    exclusion_file = root / EXCLUSION_PATH
    try:
        exclusion = yaml.safe_load(exclusion_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        exclusion = None
    manifest_match = _manifest_mentions_experiment(exclusion, "7189")
    quarantine = upstream_quarantine_observation(upstream, manifest_match=manifest_match)
    announce("host tools")
    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    tools = {
        "python": bool(sys.executable),
        "numpy": bool(np.__version__),
        "pyyaml": bool(yaml.__version__),
        "cargo": cargo is not None,
        "rustc": rustc is not None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyyaml_version": yaml.__version__,
        "cargo_path": cargo,
        "rustc_path": rustc,
    }
    announce("output directories")
    output_observed = {
        "result_parent": result.parent.is_dir(),
        "checkpoint_dir": checkpoints.is_dir(),
    }
    checks = [
        {
            "check": "required_source_bytes",
            "upstream": "repository",
            "field": "byte_count",
            "expected_value": "every required source is nonempty",
            "observed_value": sizes,
            "passed": all(size is not None and size > 0 for size in sizes.values()),
        },
        {
            "check": "driving_capability_spec",
            "upstream": str(SPEC_PATH),
            "field": "REQ-RUSTPY-7201",
            "expected_value": {"exists": True, "req_present": True},
            "observed_value": {
                "exists": spec_file.is_file(),
                "req_present": "REQ-RUSTPY-7201" in spec_text,
            },
            "passed": spec_file.is_file() and "REQ-RUSTPY-7201" in spec_text,
        },
        {
            "check": "same_milestone_gate_fields",
            "upstream": str(ROADMAP_PATH),
            "field": "task_contract",
            "expected_value": EXPECTED_TASK_CONTRACT,
            "observed_value": task_contract,
            "passed": task_contract == EXPECTED_TASK_CONTRACT,
        },
        {
            "check": "upstream_artifact_bytes",
            "upstream": str(UPSTREAM_RESULT_PATH),
            "field": "byte_count",
            "expected_value": "nonempty valid JSON object",
            "observed_value": upstream_file.stat().st_size if upstream_file.is_file() else None,
            "passed": bool(upstream),
        },
        {
            "check": "upstream_quarantine_flags",
            "upstream": f"{UPSTREAM_RESULT_PATH} and {EXCLUSION_PATH}",
            "field": "quarantined",
            "expected_value": False,
            "observed_value": quarantine,
            "passed": quarantine["quarantined"] is False,
        },
        {
            "check": "upstream_compiled_parity_gate",
            "upstream": str(UPSTREAM_RESULT_PATH),
            "field": "rust_slice_parity_score",
            "expected_value": 1,
            "observed_value": upstream.get("rust_slice_parity_score"),
            "passed": upstream.get("rust_slice_parity_score") == 1,
        },
        {
            "check": "upstream_known_failed_value",
            "upstream": str(UPSTREAM_RESULT_PATH),
            "field": "nfr_01_10x_met",
            "expected_value": False,
            "observed_value": upstream.get("nfr_01_10x_met"),
            "passed": upstream.get("nfr_01_10x_met") is False,
        },
        {
            "check": "required_tools",
            "upstream": "host_toolchain",
            "field": "python_numpy_pyyaml_cargo_rustc",
            "expected_value": {
                "python": True,
                "numpy": True,
                "pyyaml": True,
                "cargo": True,
                "rustc": True,
            },
            "observed_value": tools,
            "passed": all(tools[name] for name in ("python", "numpy", "pyyaml", "cargo", "rustc")),
        },
        {
            "check": "output_directories",
            "upstream": "filesystem",
            "field": "result_and_checkpoint_parent",
            "expected_value": {"result_parent": True, "checkpoint_dir": True},
            "observed_value": output_observed,
            "passed": all(output_observed.values()),
        },
        {
            "check": "source_artifact_hashes",
            "upstream": "required_source_bytes",
            "field": "sha256",
            "expected_value": len(REQUIRED_SOURCE_PATHS),
            "observed_value": len(hashes),
            "passed": len(hashes) == len(REQUIRED_SOURCE_PATHS),
        },
    ]
    return checks, hashes


def _extension_path(root: Path) -> Path:
    """Use this interpreter's exact extension suffix for the load receipt."""

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension suffix is unavailable")
    return root / "python" / "carnot" / f"_rust{suffix}"


def _stream_process(command: Sequence[str], *, root: Path, operation: str) -> None:
    """Stream compiler output while an external heartbeat bounds silent waits."""

    with exp7189._Heartbeat(operation) as heartbeat:
        process = subprocess.Popen(
            list(command),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
        returncode = process.wait(timeout=600)
        heartbeat.completed = 1
    if returncode != 0:
        raise RuntimeError(f"{operation} failed with exit {returncode}")


def build_pyo3_extension(root: Path) -> Path:
    """Build and atomically install the local compiled extension for this Python."""

    _stream_process(
        ["cargo", "build", "--release", "-p", "carnot-python"],
        root=root,
        operation="cargo build carnot-python",
    )
    library = root / RUST_LIBRARY_PATH
    if not library.is_file():
        raise RuntimeError(f"compiled PyO3 library is missing: {library}")
    destination = _extension_path(root)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(library, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():  # pragma: no cover - interrupted replacement only.
            temporary.unlink()
    return destination


def load_compiled_binding(root: Path, extension: Path | None = None) -> tuple[ModuleType, JsonDict]:
    """Load the exact extension path and reject absent compiled symbols."""

    expected = (extension or _extension_path(root)).resolve()
    class_name = "RustFixedCardinalitySampler"
    cached = sys.modules.get("carnot._rust")
    cached_path = Path(getattr(cached, "__file__", "") or "")
    if (
        isinstance(cached, ModuleType)
        and hasattr(cached, class_name)
        and cached_path.is_file()
        and sha256_file(cached_path) == sha256_file(expected)
    ):
        module = cached
        load_path = cached_path
    else:
        load_dir = root / "target" / "exp7201-pyo3-load"
        load_dir.mkdir(parents=True, exist_ok=True)
        load_path = load_dir / expected.name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{load_path.name}.", suffix=".tmp", dir=load_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(expected, temporary)
            os.replace(temporary, load_path)
        finally:
            if temporary.exists():  # pragma: no cover - interrupted replacement only.
                temporary.unlink()
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location("carnot._rust", load_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot create a loader for compiled PyO3 path {load_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    loaded = Path(module.__file__ or "").resolve()
    if loaded != load_path.resolve():
        raise RuntimeError(f"loaded PyO3 path {loaded} does not match copied path {load_path}")
    if not hasattr(module, class_name):
        raise RuntimeError(f"compiled binding is missing {class_name}")
    receipt = {
        "compiled": True,
        "binary_sha256": sha256_file(loaded),
        "loaded_module_path": str(loaded),
        "class_name": class_name,
        "module_name": module.__name__,
        "module_version": getattr(module, "__version__", None),
        "python_fallback_used": False,
    }
    return module, receipt


def _new_sampler(
    binding: ModuleType, instance: slices.SliceInstance, cardinality: int, beta: float = 2.0
) -> Any:
    """Construct the persistent class from the shipped edge-once model bytes."""

    return binding.RustFixedCardinalitySampler(
        list(instance.edges), list(instance.fields), cardinality, beta
    )


def _tape_arrays(tapes: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[np.ndarray, ...]:
    """Build caller-owned contiguous arrays that repeated PyO3 calls can reuse."""

    positive = np.asarray(
        [[draw["positive_index"] for draw in tape] for tape in tapes], dtype=np.uintp
    )
    negative = np.asarray(
        [[draw["negative_index"] for draw in tape] for tape in tapes], dtype=np.uintp
    )
    uniforms = np.asarray([[draw["uniform"] for draw in tape] for tape in tapes], dtype=np.float64)
    return positive, negative, uniforms


def _finish_row(row: JsonDict) -> JsonDict:
    """Add the common comparison contract and a hash after all fields exist."""

    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = sha256_json(row)
    return row


def run_transition_checks(binding: ModuleType) -> tuple[list[JsonDict], JsonDict]:
    """Replay the ordinary and singleton Exp7189 tapes through compiled PyO3."""

    instance = slices.make_frustrated_instance(8, 718701)
    rows: list[JsonDict] = []
    buffer_receipts: list[JsonDict] = []
    for offset, cardinality in enumerate((2, 0, 8)):
        seed = REPLAY_SEED + offset
        initial = slices.enumerate_slice(8, cardinality)[3 if cardinality == 2 else 0]
        tape = exp7189.make_replay_tape(8, cardinality, seed=seed, steps=REPLAY_STEPS)
        states = np.asarray([initial], dtype=np.int8)
        positive, negative, uniforms = _tape_arrays([tape])
        sampler = _new_sampler(binding, instance, cardinality)
        first = sampler.replay_batch(states, positive, negative, uniforms)[0]
        before = dict(sampler.buffer_receipt())
        second = sampler.replay_batch(states, positive, negative, uniforms)[0]
        after = dict(sampler.buffer_receipt())
        python = exp7189.python_replay(instance, cardinality, 2.0, initial, tape)
        compared = exp7189.compare_replays(f"n8:k{cardinality}:seed{seed}", tape, python, first)
        for row in compared:
            row.pop("row_sha256", None)
            row.update(
                {
                    "arm": "persistent_pyo3_vs_python_control",
                    "seed": seed,
                    "metric": "exact_transition_parity",
                    "error": None,
                    "abstention": False,
                    "energy_tolerance": TOLERANCE,
                    "magnetization_preserved": row["rust_state"].count(1) == cardinality,
                }
            )
            row["passed"] = (
                row["passed"]
                and row["delta_energy_error"] <= TOLERANCE
                and row["magnetization_preserved"]
            )
            rows.append(_finish_row(row))
        buffer_receipts.append(
            {
                "cardinality": cardinality,
                "same_result_on_reused_inputs": first == second,
                "capacity_reused": before["tape_capacity"] == after["tape_capacity"],
                "before": before,
                "after": after,
            }
        )
    receipt = {
        "calls": buffer_receipts,
        "passed": all(
            item["same_result_on_reused_inputs"] and item["capacity_reused"]
            for item in buffer_receipts
        ),
    }
    return rows, receipt


def run_distribution_checks(binding: ModuleType) -> list[JsonDict]:
    """Compare independent Python and Rust streams with the complete finite law."""

    instance = slices.make_frustrated_instance(8, 718701)
    law = slices.independent_exact_law(instance, 2, 2.0)
    initial = law.states[0]
    sampler = _new_sampler(binding, instance, 2)
    rust_seeds = [exp7189._derived_seed("rust", seed) for seed in DISTRIBUTION_SEEDS]
    states = np.asarray([initial] * len(rust_seeds), dtype=np.int8)
    with exp7189._Heartbeat("persistent PyO3 independent distribution batch") as heartbeat:
        rust_chains = sampler.run_seeded_batch(
            states, rust_seeds, DISTRIBUTION_BURN_IN, DISTRIBUTION_RETAINED
        )
        heartbeat.completed = len(rust_chains)
    rows: list[JsonDict] = []
    started = time.monotonic()
    last_report = started
    for index, seed in enumerate(DISTRIBUTION_SEEDS):
        python_chain = exp7189._python_seeded_chain(
            instance,
            cardinality=2,
            beta=2.0,
            initial_state=initial,
            seed=exp7189._derived_seed("python", seed),
            burn_in=DISTRIBUTION_BURN_IN,
            retained=DISTRIBUTION_RETAINED,
        )
        for language, arm, chain in (
            ("python", "python_control", python_chain),
            ("rust", "persistent_pyo3", rust_chains[index]),
        ):
            row = exp7189._distribution_row(
                language=language,
                seed=seed,
                chain=chain,
                law=law,
                cardinality=2,
                tv_limit=TV_LIMIT,
                energy_mean_limit=ENERGY_MEAN_LIMIT,
                receipt=None,
            )
            row.pop("row_sha256", None)
            row.update(
                {
                    "arm": arm,
                    "metric": "finite_law_distribution_check",
                    "error": None,
                    "abstention": False,
                }
            )
            rows.append(_finish_row(row))
        now = time.monotonic()
        if now - last_report >= 50.0:  # pragma: no cover - slow host only.
            print(
                f"[heartbeat] elapsed_s={now - started:.3f} "
                f"completed={index + 1}/{len(DISTRIBUTION_SEEDS)} "
                "operation=distribution controls",
                flush=True,
            )
            last_report = now
    return rows


def _phase_row(
    *, unit: str, arm: str, seed: int, phase: str, duration_s: float, **extra: Any
) -> JsonDict:
    """Retain one raw monotonic phase measurement with common row fields."""

    return _finish_row(
        {
            "row_type": "phase_cost",
            "unit_id": f"{unit}:{arm}:{phase}",
            "arm": arm,
            "seed": seed,
            "metric": "duration_s",
            "phase": phase,
            "value": duration_s,
            "error": None,
            "abstention": False,
            **extra,
        }
    )


def profile_phase_costs(binding: ModuleType, bridge: Path) -> tuple[list[JsonDict], JsonDict]:
    """Measure setup, JSON, launch, kernel, and parsing on matched workloads."""

    rows: list[JsonDict] = []
    total = len(PROFILE_SIZES) * len(PROFILE_CARDINALITIES) * len(PROFILE_SEEDS)
    started_all = time.monotonic()
    last_report = started_all
    completed_units = 0
    for n in PROFILE_SIZES:
        for cardinality in PROFILE_CARDINALITIES:
            for seed in PROFILE_SEEDS:
                instance = slices.make_frustrated_instance(n, seed)
                initial = slices.enumerate_slice(n, cardinality)[0]
                tape = exp7189.make_replay_tape(n, cardinality, seed=seed, steps=PROFILE_STEPS)
                states = np.asarray([initial], dtype=np.int8)
                positive, negative, uniforms = _tape_arrays([tape])
                unit = f"n{n}:k{cardinality}:seed{seed}"

                started = time.monotonic()
                sampler = _new_sampler(binding, instance, cardinality)
                setup_s = time.monotonic() - started
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="persistent_pyo3",
                        seed=seed,
                        phase="setup",
                        duration_s=setup_s,
                        batch_size=1,
                    )
                )
                sampler.replay_batch(states, positive, negative, uniforms)
                started = time.monotonic()
                pyo3_result = sampler.replay_batch(states, positive, negative, uniforms)[0]
                kernel_s = time.monotonic() - started
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="persistent_pyo3",
                        seed=seed,
                        phase="kernel",
                        duration_s=kernel_s,
                        batch_size=1,
                        proposal_count=PROFILE_STEPS,
                    )
                )

                started = time.monotonic()
                python_result = exp7189.python_replay(instance, cardinality, 2.0, initial, tape)
                python_s = time.monotonic() - started
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="python_control",
                        seed=seed,
                        phase="kernel",
                        duration_s=python_s,
                        batch_size=1,
                        proposal_count=PROFILE_STEPS,
                    )
                )

                request = exp7189.replay_request(instance, cardinality, 2.0, initial, tape)
                subprocess_started = time.monotonic()
                started = time.monotonic()
                request_bytes = canonical_json(request).encode("utf-8")
                serialization_s = time.monotonic() - started
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="subprocess_bridge",
                        seed=seed,
                        phase="serialization",
                        duration_s=serialization_s,
                        byte_count=len(request_bytes),
                    )
                )

                launch_started = time.monotonic()
                process = subprocess.Popen(
                    [str(bridge)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                process_launch_s = time.monotonic() - launch_started
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="subprocess_bridge",
                        seed=seed,
                        phase="process_launch",
                        duration_s=process_launch_s,
                    )
                )
                io_started = time.monotonic()
                with exp7189._Heartbeat(f"profile bridge {unit}") as heartbeat:
                    stdout, stderr = process.communicate(input=request_bytes, timeout=120)
                    heartbeat.completed = 1
                bridge_wait_s = time.monotonic() - io_started
                if process.returncode != 0:
                    raise RuntimeError(
                        f"profile bridge failed with exit {process.returncode}: "
                        f"{stderr.decode('utf-8', errors='replace')}"
                    )
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="subprocess_bridge",
                        seed=seed,
                        phase="bridge_io_kernel_wait",
                        duration_s=bridge_wait_s,
                        stdout_bytes=len(stdout),
                    )
                )
                parse_started = time.monotonic()
                parsed = json.loads(stdout)
                parsing_s = time.monotonic() - parse_started
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="subprocess_bridge",
                        seed=seed,
                        phase="parsing",
                        duration_s=parsing_s,
                        stdout_bytes=len(stdout),
                    )
                )
                subprocess_total_s = time.monotonic() - subprocess_started
                rows.append(
                    _phase_row(
                        unit=unit,
                        arm="subprocess_bridge",
                        seed=seed,
                        phase="total",
                        duration_s=subprocess_total_s,
                    )
                )
                if parsed["final_state"] != pyo3_result["final_state"]:
                    raise ValueError(f"profile arms diverged for {unit}")
                if python_result["final_state"] != pyo3_result["final_state"]:
                    raise ValueError(f"Python control diverged for {unit}")

                completed_units += 1
                now = time.monotonic()
                if now - last_report >= 50.0:  # pragma: no cover - slow host only.
                    print(
                        f"[heartbeat] elapsed_s={now - started_all:.3f} "
                        f"completed={completed_units}/{total} operation=phase profiling",
                        flush=True,
                    )
                    last_report = now

    medians: JsonDict = {}
    for arm, phase in (
        ("persistent_pyo3", "setup"),
        ("persistent_pyo3", "kernel"),
        ("python_control", "kernel"),
        ("subprocess_bridge", "serialization"),
        ("subprocess_bridge", "process_launch"),
        ("subprocess_bridge", "bridge_io_kernel_wait"),
        ("subprocess_bridge", "parsing"),
        ("subprocess_bridge", "total"),
    ):
        values = [row["value"] for row in rows if row["arm"] == arm and row["phase"] == phase]
        medians[f"{arm}:{phase}"] = statistics.median(values)
    matched = []
    for n in PROFILE_SIZES:
        for cardinality in PROFILE_CARDINALITIES:
            for seed in PROFILE_SEEDS:
                unit = f"n{n}:k{cardinality}:seed{seed}"
                by_arm_phase = {
                    (row["arm"], row["phase"]): row["value"]
                    for row in rows
                    if row["unit_id"].startswith(f"{unit}:")
                }
                matched.append(
                    by_arm_phase[("subprocess_bridge", "total")]
                    > by_arm_phase[("persistent_pyo3", "kernel")]
                )
    subprocess_total = medians["subprocess_bridge:total"]
    measured_non_kernel = (
        medians["subprocess_bridge:serialization"]
        + medians["subprocess_bridge:process_launch"]
        + medians["subprocess_bridge:parsing"]
    )
    analysis = {
        "supported": all(matched),
        "matched_units": len(matched),
        "pyo3_lower_latency_units": sum(matched),
        "median_phase_duration_s": medians,
        "median_subprocess_to_pyo3_kernel_ratio": (
            subprocess_total / medians["persistent_pyo3:kernel"]
        ),
        "measured_serialization_launch_parsing_share_of_subprocess_total": (
            measured_non_kernel / subprocess_total
        ),
        "launch_share_of_subprocess_total": (
            medians["subprocess_bridge:process_launch"] / subprocess_total
        ),
        "interpretation": (
            "The persistent boundary removes the measured process path when every matched unit "
            "is lower latency. The launch share is reported separately; unisolated child I/O "
            "and kernel wait remain in their own residual, so launch is not assumed to explain "
            "all subprocess cost."
        ),
        "speed_claim_authorized": False,
        "new_10x_gate_created": False,
    }
    return rows, analysis


def run_e2e_receipts(binding: ModuleType) -> list[JsonDict]:
    """Execute compiled binding parity and bidirectional state JSON round trips."""

    instance = slices.SliceInstance(
        n=4,
        seed=7201,
        edges=((0, 1, 1.0), (0, 2, 1.0), (1, 2, -1.0)),
        fields=(0.1, -0.2, 0.3, -0.2),
    )
    initial = slices.enumerate_slice(4, 2)[0]
    tape = exp7189.make_replay_tape(4, 2, seed=7201, steps=4)
    states = np.asarray([initial], dtype=np.int8)
    positive, negative, uniforms = _tape_arrays([tape])
    sampler = _new_sampler(binding, instance, 2)
    rust = sampler.replay_batch(states, positive, negative, uniforms)[0]
    python = exp7189.python_replay(instance, 2, 2.0, initial, tape)
    parity_pass = rust == python
    e2e003 = {
        "scenario": "E2E-003",
        "spec_ref": "REQ-RUSTPY-7201-REPLAY",
        "compiled_class": type(sampler).__name__,
        "python_result_sha256": sha256_json(python),
        "rust_result_sha256": sha256_json(rust),
        "energy_error_max": max(
            abs(left["delta_energy"] - right["delta_energy"])
            for left, right in zip(python["steps"], rust["steps"], strict=True)
        ),
        "passed": parity_pass,
    }

    seeded = sampler.run_seeded_batch(states, [7201], 3, 8)[0]
    rust_serialized = sampler.serialize_state(seeded["final_state"])
    python_loaded = json.loads(rust_serialized)
    python_serialized = canonical_json(python_loaded)
    rust_loaded = dict(sampler.deserialize_state(python_serialized))
    e2e004 = {
        "scenario": "E2E-004",
        "spec_ref": "REQ-RUSTPY-7201-SERIALIZATION",
        "rust_serialized_sha256": sha256_bytes(rust_serialized.encode("utf-8")),
        "python_serialized_sha256": sha256_bytes(python_serialized.encode("utf-8")),
        "state": python_loaded,
        "passed": rust_loaded == python_loaded == seeded["final_state"],
    }

    owned = sampler.replay_batch(states, positive, negative, uniforms)
    saved = json.loads(canonical_json(owned))
    states[:] = states[:, ::-1]
    positive[:] = 1
    lifetime = {
        "scenario": "SCENARIO-RUSTPY-7201-PERSISTENT-PARITY",
        "spec_ref": "REQ-RUSTPY-7201-BUFFER",
        "result_sha256": sha256_json(owned),
        "passed": owned == saved,
    }
    return [e2e003, e2e004, lifetime]


def _base_artifact(
    *, root: Path, run_date: str, checks: list[JsonDict], hashes: dict[str, str]
) -> JsonDict:
    """Create every terminal field before the run chooses complete or blocked."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": checks,
        "inference_substrate": "not_started",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "host_identity": platform.node() or "unknown-host",
        "duration_s": 0.0,
        "source_artifact_hashes": hashes,
        "rows": [],
        "sample_size_budget": {
            "planned_transition_rows": 3 * REPLAY_STEPS,
            "completed_transition_rows": 0,
            "planned_distribution_rows": 2 * len(DISTRIBUTION_SEEDS),
            "completed_distribution_rows": 0,
            "distribution_retained_per_row": DISTRIBUTION_RETAINED,
            "planned_profile_units": (
                len(PROFILE_SIZES) * len(PROFILE_CARDINALITIES) * len(PROFILE_SEEDS)
            ),
            "completed_profile_units": 0,
            "independent_distribution_seeds": len(DISTRIBUTION_SEEDS),
            "exclusions": [],
        },
        "random_seed": {
            "replay_seed": REPLAY_SEED,
            "distribution_seeds": list(DISTRIBUTION_SEEDS),
            "profile_seeds": list(PROFILE_SEEDS),
            "python_rng": "CPython random.Random",
            "rust_rng": "independent 64-bit LCG in carnot-samplers",
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "pyo3_slice_ready_score": 0,
        "phase_cost_rows": [],
        "compiled_binding_receipt": {},
        "transition_rows": [],
        "distribution_rows": [],
        "e2e_receipts": [],
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_invoked": False,
        "bridge_overhead_hypothesis": {},
        "bridge_overhead_hypothesis_supported": False,
        "buffer_reuse_receipt": {},
        "upstream_performance_null": {
            "artifact": str(UPSTREAM_RESULT_PATH),
            "field": "nfr_01_10x_met",
            "observed_value": False,
            "promoted": False,
        },
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "root": str(root.resolve()),
        "spec_refs": ["REQ-RUSTPY-7201", "SCENARIO-RUSTPY-7201-PERSISTENT-PARITY"],
    }


def _block_artifact(artifact: JsonDict, *, failed: Mapping[str, Any], started: float) -> JsonDict:
    """Preserve an exact external error without fabricating measurement rows."""

    artifact.update(
        {
            "status": "blocked_external_precondition",
            "inference_substrate": "blocked_no_run",
            "inference_substrate_class": "blocked_no_run",
            "duration_s": time.monotonic() - started,
            "gate_check_summary": {
                "passed": False,
                "failed_check": failed.get("check"),
                "upstream": failed.get("upstream"),
                "field": failed.get("field"),
                "expected_value": failed.get("expected_value"),
                "observed_value": failed.get("observed_value"),
            },
            "verdict_class": "blocked",
            "honest_verdict": (
                f"blocked_external_precondition: {failed.get('check')} failed before measurement"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path,
    run_date: str = RUN_DATE,
    binding_module: ModuleType | None = None,
    binding_receipt: Mapping[str, Any] | None = None,
    bridge_path: Path | None = None,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> JsonDict:
    """Build measured evidence or a terminal external-prerequisite artifact."""

    started = time.monotonic()
    _progress(0, "start", "precondition checks")
    if preconditions is None or source_hashes is None:
        measured_checks, measured_hashes = collect_preconditions(root)
        checks = measured_checks if preconditions is None else preconditions
        hashes = measured_hashes if source_hashes is None else source_hashes
    else:
        checks, hashes = preconditions, source_hashes
    artifact = _base_artifact(root=root, run_date=run_date, checks=checks, hashes=hashes)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    _progress(0, "end", "precondition checks")
    if failed is not None:
        return _block_artifact(artifact, failed=failed, started=started)
    artifact["gate_check_summary"] = {
        "passed": True,
        "failed_check": None,
        "upstream": "repository_preflight",
        "field": "all_required_preconditions",
        "expected_value": "all pass",
        "observed_value": "all pass",
    }

    _progress(1, "start", "compiled PyO3 binding and subprocess baseline")
    try:
        if binding_module is None or binding_receipt is None:
            extension = build_pyo3_extension(root)
            binding, measured_receipt = load_compiled_binding(root, extension)
            receipt = measured_receipt
        else:
            binding, receipt = binding_module, dict(binding_receipt)
        bridge = bridge_path or exp7189.build_rust_bridge(root)
        if not bridge.is_file() or not os.access(bridge, os.X_OK):
            raise RuntimeError(f"compiled subprocess baseline is unavailable: {bridge}")
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        _progress(1, "end", f"compiled prerequisite failed error={exc}")
        return _block_artifact(
            artifact,
            failed={
                "check": "compiled_binding_or_bridge",
                "upstream": "host_toolchain",
                "field": "carnot._rust.RustFixedCardinalitySampler",
                "expected_value": "compiled binding and bridge load successfully",
                "observed_value": str(exc),
            },
            started=started,
        )
    artifact["compiled_binding_receipt"] = receipt
    _progress(1, "end", f"compiled binding path={receipt['loaded_module_path']}")

    _progress(2, "start", "explicit transition parity")
    transition_rows, buffer_receipt = run_transition_checks(binding)
    artifact["transition_rows"] = transition_rows
    artifact["buffer_reuse_receipt"] = buffer_receipt
    _progress(2, "end", f"explicit transition parity rows={len(transition_rows)}")

    _progress(3, "start", "independent distribution checks")
    distribution_rows = run_distribution_checks(binding)
    artifact["distribution_rows"] = distribution_rows
    _progress(3, "end", f"independent distribution checks rows={len(distribution_rows)}")

    _progress(4, "start", "phase cost profiling")
    phase_rows, hypothesis = profile_phase_costs(binding, bridge)
    artifact["phase_cost_rows"] = phase_rows
    artifact["bridge_overhead_hypothesis"] = hypothesis
    artifact["bridge_overhead_hypothesis_supported"] = hypothesis["supported"]
    _progress(4, "end", f"phase cost profiling rows={len(phase_rows)}")

    _progress(5, "start", "E2E-003 and E2E-004 compiled receipts")
    e2e_receipts = run_e2e_receipts(binding)
    artifact["e2e_receipts"] = e2e_receipts
    _progress(5, "end", f"compiled E2E receipts={len(e2e_receipts)}")

    parity_ready = (
        receipt.get("compiled") is True
        and receipt.get("python_fallback_used") is False
        and len(transition_rows) == 3 * REPLAY_STEPS
        and all(row["passed"] for row in transition_rows)
        and len(distribution_rows) == 2 * len(DISTRIBUTION_SEEDS)
        and all(row["passed"] for row in distribution_rows)
        and buffer_receipt["passed"] is True
        and all(row["passed"] for row in e2e_receipts)
    )
    artifact["sample_size_budget"].update(
        {
            "completed_transition_rows": len(transition_rows),
            "completed_distribution_rows": len(distribution_rows),
            "completed_profile_units": len(PROFILE_SIZES)
            * len(PROFILE_CARDINALITIES)
            * len(PROFILE_SEEDS),
        }
    )
    artifact["rows"] = transition_rows + distribution_rows + phase_rows
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": (
                "cpu_exact_solver_or_simulator: compiled persistent PyO3 pair-swap kernel, "
                "independent Python control, exact finite-slice law, and retained subprocess "
                "phase timings"
            ),
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": time.monotonic() - started,
            "pyo3_slice_ready_score": int(parity_ready),
            "verdict_class": "null",
            "honest_verdict": (
                "complete: genuine compiled PyO3 execution matched every explicit transition, "
                "preserved magnetization and restart state, and passed independent finite-law "
                f"checks. The measured bridge-overhead hypothesis was "
                f"{'supported' if hypothesis['supported'] else 'not supported'}. This prototype "
                "creates no 10x gate and makes no sampler-speed or scientific-benefit claim."
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Recompute completeness, compiled provenance, parity, and terminal state."""

    if not REQUIRED_ARTIFACT_FIELDS.issubset(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if payload.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    if payload.get("MODEL_SPECS") != [] or payload.get("model_invoked") is not False:
        errors.append("model_declaration_invalid")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_authority_invalid")
    if payload.get("verdict_class") == "blocked":
        gate = payload.get("gate_check_summary", {})
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("pyo3_slice_ready_score") != 0
            or any(
                payload.get(field)
                for field in (
                    "rows",
                    "transition_rows",
                    "distribution_rows",
                    "phase_cost_rows",
                    "e2e_receipts",
                )
            )
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not gate.get("failed_check")
            or not gate.get("upstream")
            or not gate.get("field")
        ):
            errors.append("blocked_state_invalid")
        return list(dict.fromkeys(errors))

    transitions = payload.get("transition_rows", [])
    distributions = payload.get("distribution_rows", [])
    phases = payload.get("phase_cost_rows", [])
    receipts = payload.get("e2e_receipts", [])
    expected_units = {
        f"n8:k{cardinality}:seed{REPLAY_SEED + offset}:step{step}"
        for offset, cardinality in enumerate((2, 0, 8))
        for step in range(REPLAY_STEPS)
    }
    transitions_valid = (
        len(transitions) == 3 * REPLAY_STEPS
        and {row.get("unit_id") for row in transitions} == expected_units
        and all(
            row.get("passed") is True
            and row.get("abstention") is False
            and row.get("error") is None
            and row.get("delta_energy_error", math.inf) <= TOLERANCE
            and row.get("magnetization_preserved") is True
            and row.get("row_sha256")
            == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
            for row in transitions
        )
    )
    if not transitions_valid:
        errors.append("transition_rows_invalid")
    distribution_valid = len(distributions) == 2 * len(DISTRIBUTION_SEEDS) and all(
        row.get("passed") is True
        and row.get("abstention") is False
        and row.get("error") is None
        and row.get("cardinality_valid") is True
        and row.get("retained") == DISTRIBUTION_RETAINED
        and row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        for row in distributions
    )
    if not distribution_valid:
        errors.append("distribution_rows_invalid")
    required_phases = {"setup", "serialization", "process_launch", "kernel", "parsing"}
    phases_valid = (
        bool(phases)
        and required_phases.issubset({row.get("phase") for row in phases})
        and all(
            row.get("value", -1.0) >= 0.0
            and row.get("abstention") is False
            and row.get("error") is None
            and row.get("row_sha256")
            == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
            for row in phases
        )
    )
    if not phases_valid:
        errors.append("phase_cost_rows_invalid")
    if payload.get("rows") != transitions + distributions + phases:
        errors.append("rows_invalid")
    if len(receipts) != 3 or any(row.get("passed") is not True for row in receipts):
        errors.append("e2e_receipts_invalid")
    binding = payload.get("compiled_binding_receipt", {})
    binding_valid = (
        isinstance(binding, Mapping)
        and binding.get("compiled") is True
        and binding.get("python_fallback_used") is False
        and binding.get("class_name") == "RustFixedCardinalitySampler"
        and isinstance(binding.get("binary_sha256"), str)
        and binding["binary_sha256"].startswith("sha256:")
        and isinstance(binding.get("loaded_module_path"), str)
    )
    if binding_valid and root is not None:
        loaded = Path(binding["loaded_module_path"])
        binding_valid = loaded.is_file() and binding["binary_sha256"] == sha256_file(loaded)
    if not binding_valid:
        errors.append("compiled_binding_receipt_invalid")
    buffer_valid = payload.get("buffer_reuse_receipt", {}).get("passed") is True
    if not buffer_valid:
        errors.append("buffer_reuse_receipt_invalid")
    readiness = (
        transitions_valid
        and distribution_valid
        and phases_valid
        and len(receipts) == 3
        and all(row.get("passed") is True for row in receipts)
        and binding_valid
        and buffer_valid
    )
    if payload.get("pyo3_slice_ready_score") != int(readiness):
        errors.append("readiness_invalid")
    if payload.get("status") != "complete":
        errors.append("terminal_status_invalid")
    if payload.get("verdict_class") != "null":
        errors.append("terminal_verdict_invalid")
    if not str(payload.get("honest_verdict", "")).startswith("complete:"):
        errors.append("honest_verdict_invalid")
    if payload.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    if payload.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("gate_summary_invalid")
    if payload.get("upstream_performance_null", {}).get("promoted") is not False:
        errors.append("upstream_failed_value_promoted")
    if root is not None:
        recorded = payload.get("source_artifact_hashes", {})
        expected = {str(path) for path in REQUIRED_SOURCE_PATHS}
        if (
            not isinstance(recorded, Mapping)
            or set(recorded) != expected
            or any(
                not (root / path).is_file() or recorded[path] != sha256_file(root / path)
                for path in expected
            )
        ):
            errors.append("source_artifact_hashes_invalid")
    return list(dict.fromkeys(errors))


def atomic_write(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish complete JSON with one same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - interrupted replacement only.
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(*, root: Path, output: Path, run_date: str) -> JsonDict:
    """Build, validate, and atomically write the terminal Exp7201 artifact."""

    artifact = build_artifact(root=root, run_date=run_date)
    _progress(6, "start", "artifact validation")
    errors = validate_artifact(artifact, root=root)
    _progress(6, "end", f"artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7201 artifact: {errors}")
    _progress(7, "start", "final atomic write")
    receipt = atomic_write(output, artifact)
    _progress(7, "end", f"final atomic write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional read-only validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study or validate existing bytes without changing them."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        _progress(6, "start", f"validation path={args.validate}")
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            _progress(6, "end", "validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        _progress(6, "end", f"validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    try:
        output = args.output if args.output.is_absolute() else root / args.output
        run_experiment(root=root, output=output, run_date=args.date)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - the entrypoint executes this branch.
    raise SystemExit(main())
