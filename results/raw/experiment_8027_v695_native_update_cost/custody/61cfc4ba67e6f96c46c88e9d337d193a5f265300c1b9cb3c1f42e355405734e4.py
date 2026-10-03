"""REQ-REPORT-8027: measure an opt-in numerical port without learning promotion.

The loaded extension consumes the current calibrated trajectory. Numerical
parity and CPU timing cannot turn its exposed-development null into benefit.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
import resource
import re
from pathlib import Path
import shutil
import sqlite3
import sys
import sysconfig
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # type: ignore[import-untyped]

from carnot import experiment_7230_v636_native_belief as build
from carnot import experiment_8008_v694_conditioned_energy_fit as conditioned
from carnot import experiment_8019_v695_eligible_targets as eligible
from carnot import experiment_8025_v695_causal_online_updates as upstream
from carnot.experiment_8007_v694_conditioning_diagnosis import copy_evidence
from carnot.experiment_8021_v695_typed_decision_test import action
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import causal_online_8025 as causal
from carnot.verify import evidence_features_7980 as features

Json = dict[str, Any]
Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8027_v695_native_update_cost"
TASK = "exp8027-native-update-cost"
OWNED = [f"python/carnot/{NAME}.py", f"scripts/experiments/{NAME}.py"]
RUST = [
    "crates/carnot-core/src/numerical_update_8027.rs",
    "crates/carnot-python/src/numerical_update_8027.rs",
    "crates/carnot-core/src/lib.rs",
    "crates/carnot-python/src/lib.rs",
]
TEST = "tests/python/test_native_update_8027.py"
CONFIG: Json = dict(
    seed=8027,
    stress_inputs=256,
    repetitions=30,
    batches=[1, 16, 64],
    tolerance=1e-10,
    learning_rate=0.01,
    l2=0.001,
    tier1_target_ns=1000,
    nfr01_target=10,
    eligibility="exact original public masks and released labels",
    scope="cached exposed development; stress has no independent natural credit",
    defaults=False,
    budget_s=900,
    thread_limit=1,
)
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report completed work without adding synthetic waiting time."""
    print(
        f"[exp8027] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def extension() -> tuple[Any, Json]:
    """Reuse interpreter build and loader while retaining task-owned binary bytes."""
    receipt: Json
    supplied = os.environ.get("CARNOT_8027_EXTENSION")
    if supplied:
        path, receipt = Path(supplied), dict(path=supplied)
    else:
        progress("build_before")
        path, receipt = build.build_native_extension(ROOT)
        destination = ROOT / "target/experiment-8027-load" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        path = destination
        progress("build_after")
    progress("extension_load_before")
    loaded_at = time.perf_counter_ns()
    module = build.load_native_extension(path)
    if not hasattr(module, "RustNumericalUpdate8027"):
        raise ValueError("missing_numerical_entrypoint")
    receipt.update(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        actual_loaded=True,
        extension_load_ns=time.perf_counter_ns() - loaded_at,
        build_helper="Exp7230 interpreter-bound build; task-owned copied binary",
    )
    progress("extension_load_after")
    return module, receipt


def coefficients(head: Json) -> Array:
    """The durable multiplier defines logical coefficients after restart."""
    return np.asarray(head["parameters"], dtype=float) * float(head["decay_scale"])


def python_design(head: Json, raw: Array) -> Array:
    """Use the qualified Python geometry, rather than fit another scaler."""
    return np.asarray(conditioned.design("conditioned_energy", raw, head["geometry"]), dtype=float)


def python_batch(
    head: Json, matrix: Array, labels: list[int | None]
) -> tuple[list[float], int, int]:
    """Time equivalent sparse arithmetic without the upstream audit overhead."""
    began, hot, probabilities = time.perf_counter_ns(), 0, []
    for x, y in zip(matrix, labels, strict=True):
        p = float(
            expit(head["calibration"][0] + head["calibration"][1] * float(x @ coefficients(head)))
        )
        probabilities.append(p)
        if y is not None:
            ids = np.flatnonzero(x)
            gradients = (p - y) * head["calibration"][1] * x[ids]
            start = time.perf_counter_ns()
            scale = head["decay_scale"] * (1 - 2 * CONFIG["l2"] * CONFIG["learning_rate"])
            for i, g in zip(ids, gradients, strict=True):
                head["parameters"][int(i)] -= CONFIG["learning_rate"] * float(g) / scale
            head["decay_scale"] = scale
            hot += time.perf_counter_ns() - start
    return probabilities, time.perf_counter_ns() - began, hot


def percentiles(values: list[float]) -> Json:
    """Tied durations retain NumPy's fixed linear quantile convention."""
    return dict(p50=float(np.quantile(values, 0.5)), p95=float(np.quantile(values, 0.95)))


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Bind current readiness to exact producer bytes before replay opens labels."""
    refs, gates, producers = [], [], {}
    for stem, field in [
        ("experiment_8020_v695_qualified_energy_fit", "energy_fit_ready_score"),
        (upstream.NAME, "learning_measurement_ready_score"),
        ("experiment_8002_v693_service_cost", "service_measurement_ready_score"),
    ]:
        path = root / "results" / (stem + ".json")
        value = json.loads(path.read_text()) if path.exists() else {}
        gate = dict(
            upstream_id=value.get("task_id", stem),
            path=str(path),
            hash=sha256_file(path) if path.exists() else None,
            artifact_field=field,
            expected=1,
            observed=value.get(field, "MISSING_CONTRACT_FIELD"),
            passed=type(value.get(field)) is int and value.get(field) == 1,
        )
        validity = [
            gate,
            dict(
                gate,
                artifact_field="flagged_adversarial",
                expected=False,
                observed=value.get("flagged_adversarial", "MISSING_CONTRACT_FIELD"),
                passed=value.get("flagged_adversarial") is False,
            ),
            dict(
                gate,
                artifact_field="verdict_class",
                expected=["null", "positive"],
                observed=value.get("verdict_class", "MISSING_CONTRACT_FIELD"),
                passed=value.get("verdict_class") in {"null", "positive"},
            ),
        ]
        gates.extend(validity)
        producers[stem] = value
        if all(g["passed"] for g in validity):
            refs.append(copy_evidence(reference(path), raw))
    failed = [g for g in gates if not g["passed"]]
    if failed:
        return dict(references=refs), failed
    trajectory = root / "results/raw" / upstream.NAME
    missing = [
        trajectory / p for p in ("inputs.json", "ledger.sqlite") if not (trajectory / p).is_file()
    ]
    if missing:
        return dict(references=refs), [
            dict(
                gates[3],
                artifact_field="trajectory_files",
                expected="inputs and durable ledger",
                observed=[str(p) for p in missing],
                passed=False,
            )
        ]
    for r in producers[upstream.NAME].get("raw_shard_hashes", []):
        if Path(r["path"]).name in {"inputs.json", "ledger.sqlite"}:
            checked(r)
    progress("small_head_load_before")
    data = json.loads((trajectory / "inputs.json").read_text())
    refs.append(copy_evidence(reference(trajectory / "inputs.json"), raw))
    db = sqlite3.connect(f"file:{trajectory / 'ledger.sqlite'}?mode=ro", uri=True)
    updates = [
        json.loads(r[0])
        for r in db.execute("SELECT payload FROM events WHERE kind='update' ORDER BY seq")
    ]
    db.close()
    # Keep only the immutable natural update rows; the upstream owns full ledger validation.
    atomic_json(raw / "natural_updates.json", dict(rows=updates))
    for ref in producers[upstream.NAME]["checkpoint_references"]:
        refs.append(copy_evidence(ref, raw))
    service = producers["experiment_8002_v693_service_cost"]
    imported = dict(
        acquisition=service["acquisition_receipt_joins"],
        load=service["load_amortization"],
        original_reference=refs[2],
        scope="historical_model_receipts",
    )
    capture_refs = [
        r for r in service.get("cited_upstream_artifacts", []) if r.get("producer_id") == 7995
    ]
    if capture_refs:
        capture_ref = copy_evidence(capture_refs[0]["original_reference"], raw)
        refs.append(capture_ref)
        capture = json.loads(checked(capture_ref).read_text())
        imported["original_capture_reference"] = capture_ref
        imported["acquisition"] = [
            dict(
                duration_s=r["duration_s"],
                source_hash=r["source_cluster_id"],
                public_hash=r["public_hash"],
                response_hash=canonical_hash(r["raw_response"]),
                producer_id=7995,
                producer_date=capture["execution_date"],
                scope="historical_model_receipts",
            )
            for r in capture["rows"]
            if r["role"] == "stream" and r["status"] == "generated"
        ]
    atomic_json(raw / "imported_history.json", imported)
    data.update(updates=updates, references=refs, imported=imported)
    progress("small_head_load_after", len(updates), 0)
    return data, []


def parity(data: Json, native: Any, raw: Path) -> Json:
    """Replay all natural updates and isolated stress while retaining original masks."""
    progress("parity_before", 0, len(data["updates"]) + 256)
    h = data["head"]
    random = np.random.default_rng(CONFIG["seed"]).uniform(-0.2, 1.2, (256, 9))
    random[:10] = np.arange(10)[:, None] / 9
    atomic_json(raw / "stress.json", dict(inputs=random.tolist(), scope="numerical_fixture"))
    rows, states = [], {}
    vectors = python_design(h, random)
    n = native.RustNumericalUpdate8027(json.dumps(h))
    native_vectors = np.asarray(n.design(random.tolist()))
    for i, x in enumerate(vectors):
        stress_head = copy.deepcopy(h)
        stress_native = native.RustNumericalUpdate8027(json.dumps(h))
        p = python_batch(stress_head, x[None, :], [i % 2])[0][0]
        q = stress_native.update_batch([native_vectors[i].tolist()], [i % 2])[0][0]
        rows.append(
            dict(
                kind="stress",
                index=i,
                probability_error=abs(p - q),
                basis_error=float(np.max(np.abs(x - native_vectors[i]))),
                action_equal=action(p) == action(q),
                coefficient_error=float(
                    np.max(np.abs(coefficients(stress_head) - stress_native.effective()))
                ),
                eligible=False,
                numerator=abs(p - q),
                denominator=1,
                exclusion_reason="stress_not_natural",
            )
        )
    checkpoints = []
    restarts = []
    for i, update in enumerate(data["updates"]):
        identity = f"{update['arm']}/{update['seed']}"
        if identity not in states:
            states[identity] = native.RustNumericalUpdate8027(json.dumps(h))
        current = states[identity]
        source = data["sources"][update["origin_slot"]]
        x = causal.design(h, source)
        nx = np.asarray(current.design([[source["q"], *source["features"]]])[0])
        before = copy.deepcopy(h)
        before.update(parameters=update["before_coefficients"], decay_scale=1.0)
        expected = causal.probability(before, x)
        observed = current.update_batch([nx.tolist()], [update["y"]])[0][0]
        error = float(
            np.max(np.abs(np.asarray(current.effective()) - update["after_coefficients"]))
        )
        rows.append(
            dict(
                kind="natural",
                arm=update["arm"],
                seed=update["seed"],
                index=update["update_index"],
                family_id=update["family_id"],
                probability_error=abs(expected - observed),
                basis_error=float(np.max(np.abs(x - nx))),
                action_equal=action(expected) == action(observed),
                coefficient_error=error,
                eligible=source["public_eligible"],
                numerator=abs(expected - observed),
                denominator=1,
                exclusion_reason=None,
            )
        )
        if i % 512 == 0:
            progress("parity_natural", i + 1, len(data["updates"]) - i - 1)
    for identity, current in states.items():
        path = raw / "checkpoints" / (identity.replace("/", "-") + ".json")
        atomic_json(path, json.loads(current.state_json()))
        restored = native.RustNumericalUpdate8027(path.read_text())
        restart_error = float(
            np.max(np.abs(np.asarray(restored.effective()) - current.effective()))
        )
        if restart_error > CONFIG["tolerance"]:
            raise ValueError("restart_coefficients")
        restarts.append(
            dict(
                identity=identity,
                checkpoint=reference(path),
                coefficient_error=restart_error,
                bitwise_equal=restored.effective() == current.effective(),
                numerator=restart_error,
                denominator=110,
                eligibility=True,
                passed=True,
            )
        )
        checkpoints.append(reference(path))
    atomic_json(raw / "parity_rows.json", dict(rows=rows))
    masks = [r["public_eligible"] for r in data["sources"]]
    passed = all(
        r["probability_error"] <= 1e-10
        and r["coefficient_error"] <= 1e-10
        and r["basis_error"] <= 1e-10
        and r["action_equal"]
        and (r["eligible"] or r["kind"] == "stress")
        for r in rows
    )
    progress("parity_after", len(rows), 0)
    return dict(
        parity_rows=rows,
        checkpoint_references=checkpoints,
        serialized_restart_rows=restarts,
        parity_passed=passed,
        exact_eligibility_masks=masks,
        non_bitwise_math=any(r["probability_error"] > 0 for r in rows)
        or any(not r["bitwise_equal"] for r in restarts),
        natural_independent_n=len({r["family_id"] for r in rows if r["kind"] == "natural"}),
    )


def benchmark(data: Json, native: Any, raw: Path) -> Json:
    """Separate kernel, binding, durable transaction and imported acquisition costs.

    Each arm starts from the same state and submits the same labels and rows.
    Transaction includes design, update, serialization, fsync and state reopen.
    """
    progress("benchmark_before", 0, 180)
    natural = data["updates"]
    joins = {r["source_hash"]: r for r in data["imported"]["acquisition"]}
    timing: list[Json] = []
    h = data["head"]
    for batch in CONFIG["batches"]:
        for repetition in range(CONFIG["repetitions"]):
            progress("benchmark_block_before", len(timing), 180 - len(timing))
            chosen = [natural[(repetition * batch + i) % len(natural)] for i in range(batch)]
            selected = [data["sources"][r["origin_slot"]] for r in chosen]
            raw_x = np.asarray([[r["q"], *r["features"]] for r in selected])
            labels = [r["y"] for r in chosen]
            x = python_design(h, raw_x)
            ffi_start = time.perf_counter_ns()
            echo = native.numerical_echo_8027(x.tolist())
            ffi_ns = time.perf_counter_ns() - ffi_start
            if not np.array_equal(echo, x):
                raise ValueError("ffi_conversion")
            pair = {}
            for arm in ["python", "rust"] if repetition % 2 == 0 else ["rust", "python"]:
                initialized = time.perf_counter_ns()
                state = (
                    copy.deepcopy(h)
                    if arm == "python"
                    else native.RustNumericalUpdate8027(json.dumps(h))
                )
                initialization_ns = time.perf_counter_ns() - initialized
                start = time.perf_counter_ns()
                public = [
                    features.extract(
                        {k: r[k] for k in ("family_id", "source_bytes", "answer_bytes")}
                    )["values"]
                    for r in selected
                ]
                fresh = np.asarray([[r["q"], *v] for r, v in zip(selected, public, strict=True)])
                if not np.array_equal(fresh, raw_x):
                    raise ValueError("public_feature_drift")
                matrix = (
                    python_design(h, fresh)
                    if arm == "python"
                    else np.asarray(state.design(fresh.tolist()))
                )
                feature_ns = time.perf_counter_ns() - start
                start = time.perf_counter_ns()
                predictions, kernel_ns, arithmetic_ns = (
                    python_batch(state, matrix, labels)
                    if arm == "python"
                    else state.update_batch(matrix.tolist(), labels)
                )
                binding_ns = time.perf_counter_ns() - start
                start = time.perf_counter_ns()
                encoded = json.dumps(state) if arm == "python" else state.state_json()
                serialization_ns = time.perf_counter_ns() - start
                file = raw / "transactions" / f"{batch}-{repetition}-{arm}.json"
                file.parent.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter_ns()
                with file.open("w") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory = os.open(file.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                fsync_ns = time.perf_counter_ns() - start
                start = time.perf_counter_ns()
                restored = (
                    json.loads(file.read_text())
                    if arm == "python"
                    else native.RustNumericalUpdate8027(file.read_text())
                )
                effective = (
                    coefficients(restored).tolist() if arm == "python" else restored.effective()
                )
                restart_ns = time.perf_counter_ns() - start
                transaction_ns = (
                    initialization_ns
                    + feature_ns
                    + binding_ns
                    + serialization_ns
                    + fsync_ns
                    + restart_ns
                )
                acquisition = [joins.get(r["source_cluster_id"]) for r in selected]
                matched = all(r is not None for r in acquisition)
                acquisition_s = sum(r["duration_s"] for r in acquisition if r is not None)
                row = dict(
                    arm=arm,
                    batch=batch,
                    repetition=repetition,
                    kernel_ns=kernel_ns,
                    arithmetic_ns=arithmetic_ns / batch,
                    binding_ns=binding_ns,
                    ffi_conversion_baseline_ns=ffi_ns,
                    feature_construction_ns=feature_ns,
                    serialization_ns=serialization_ns,
                    persistence_fsync_ns=fsync_ns,
                    restart_ns=restart_ns,
                    state_initialization_ns=initialization_ns,
                    transaction_ns=transaction_ns,
                    effective=effective,
                    predictions=predictions,
                    checkpoint=reference(file),
                    input_hash=canonical_hash(dict(raw=raw_x.tolist(), labels=labels)),
                    cpu_input_bytes=int(x.nbytes),
                    cpu_state_bytes=110 * 8 + 8 + 2 * 8,
                    serialized_bytes=len(encoded.encode()),
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    feature_input_bytes=int(raw_x.nbytes),
                    imported_acquisition_s=acquisition_s,
                    service_estimate_s=acquisition_s + transaction_ns / 1e9 if matched else None,
                    acquisition_matched=matched,
                    numerator=transaction_ns,
                    denominator=batch,
                    eligibility=True,
                    failure_reason=None,
                    censor_reason=None,
                    exclusion_reason=None if matched else "no_matching_historical_acquisition",
                )
                timing.append(row)
                pair[arm] = row
            if (
                max(
                    abs(a - b)
                    for a, b in zip(
                        pair["python"]["effective"], pair["rust"]["effective"], strict=True
                    )
                )
                > 1e-10
            ):
                raise ValueError("benchmark_parity")
            progress("benchmark_block_after", len(timing), 180 - len(timing))
    atomic_json(raw / "timing_rows.json", dict(rows=timing))
    progress("benchmark_after", len(timing), 0)
    return timing_summary(timing)


def timing_summary(timing: list[Json]) -> Json:
    """Reduce p50/p95 and gate complete costs rather than only the fast kernel."""
    distributions: list[Json] = []
    service: list[Json] = []
    speedups: dict[str, Json] = {k: {} for k in ("kernel", "binding", "transaction")}
    for batch in CONFIG["batches"]:
        for arm in ("python", "rust"):
            rows = [r for r in timing if r["batch"] == batch and r["arm"] == arm]
            matched = [r["service_estimate_s"] for r in rows if r["acquisition_matched"]]
            service.append(
                dict(
                    batch=batch,
                    arm=arm,
                    matched=len(matched),
                    excluded=len(rows) - len(matched),
                    service_seconds=percentiles(matched) if matched else None,
                    imported_acquisition_seconds=percentiles(
                        [r["imported_acquisition_s"] for r in rows]
                    ),
                    scope="hashed historical acquisition plus current complete transaction; historical load amortization separate",
                )
            )
            distributions.append(
                dict(
                    batch=batch,
                    arm=arm,
                    observations=len(rows),
                    **{
                        key: percentiles([r[key] for r in rows])
                        for key in rows[0]
                        if key.endswith("_ns") or key.endswith("_bytes")
                    },
                )
            )
        for component in speedups:
            medians = [
                next(r for r in distributions if r["batch"] == batch and r["arm"] == arm)[
                    component + "_ns"
                ]["p50"]
                for arm in ("python", "rust")
            ]
            speedups[component][str(batch)] = medians[0] / medians[1]
    rust_hot = [r["arithmetic_ns"] for r in timing if r["arm"] == "rust"]
    return dict(
        timing_rows=timing,
        timing_distributions=distributions,
        complete_service_estimates=service,
        kernel_speedup=speedups["kernel"],
        binding_speedup=speedups["binding"],
        transaction_speedup=speedups["transaction"],
        nfr01_met=all(v >= 10 for v in speedups["transaction"].values()),
        tier1_arithmetic_met=percentiles(rust_hot)["p95"] <= 1000,
    )


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse existing bounded checks, adding the changed Rust crates only."""
    commands = eligible.validation_plan(scratch)
    config = scratch / "coverage.ini"
    config.write_text(config.read_text().replace(eligible.NAME, NAME))
    strict = scratch / "mypy.ini"
    strict.write_text("[mypy]\npython_version = 3.12\nstrict = True\nfollow_imports = silent\n")
    changed = [
        replace(
            c,
            argv=tuple(a.replace(eligible.NAME, NAME).replace(eligible.TEST, TEST) for a in c.argv),
        )
        for c in commands
    ]
    changed = [
        replace(c, argv=(str(ROOT / ".venv/bin/mypy"), "--config-file=" + str(strict), *OWNED))
        if c.name == "strict_mypy"
        else c
        for c in changed
    ]
    return changed + [
        CommandSpec(
            "cargo_test",
            (
                "env",
                "PYO3_PYTHON=" + sys.executable,
                "RUSTFLAGS=-C link-arg=-L"
                + str(sysconfig.get_config_var("LIBDIR"))
                + " -C link-arg=-lpython3.12",
                "LD_LIBRARY_PATH=" + str(sysconfig.get_config_var("LIBDIR")),
                "cargo",
                "test",
                "-p",
                "carnot-core",
                "-p",
                "carnot-python",
                "numerical_update_8027",
            ),
            "owned",
            120,
        ),
        CommandSpec(
            "cargo_fmt",
            (
                "cargo",
                "fmt",
                "-p",
                "carnot-core",
                "-p",
                "carnot-python",
                "--",
                "--check",
                "--config",
                "skip_children=true",
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "cargo_clippy",
            ("cargo", "clippy", "-p", "carnot-core", "-p", "carnot-python", "--no-deps"),
            "owned",
            180,
        ),
        CommandSpec("changed_native_fmt", ("rustfmt", "--check", *RUST[:2]), "owned", 60),
    ]


def native_coverage(scratch: Path, raw: Path) -> tuple[list[Json], Json]:
    """Measure executable Rust lines in the two added modules with LLVM counters.

    Test binaries link the interpreter explicitly because extension builds leave
    those symbols for Python. Profiles and build scratch stay outside results.
    """
    target = scratch / "native-target"
    library = str(sysconfig.get_config_var("LIBDIR"))
    command = CommandSpec(
        "rust_coverage_tests",
        (
            "env",
            "PYO3_PYTHON=" + sys.executable,
            "CARGO_TARGET_DIR=" + str(target),
            "LLVM_PROFILE_FILE=" + str(scratch / "native-%p-%m.profraw"),
            "LD_LIBRARY_PATH=" + library,
            "RUSTFLAGS=-C instrument-coverage -C link-arg=-L"
            + library
            + " -C link-arg=-lpython3.12",
            "cargo",
            "test",
            "-p",
            "carnot-core",
            "-p",
            "carnot-python",
            "numerical_update_8027",
        ),
        "owned",
        180,
    )
    atomic_json(raw / "native_validation_manifest.json", dict(commands=[asdict(command)]))
    receipts = run_commands(ROOT, [command], log_dir=raw / "native_validation_logs", heartbeat_s=30)
    if not all(r["passed"] for r in receipts):
        return receipts, {}
    objects = [
        next(p for p in (target / "debug/deps").glob(stem + "-*") if p.is_file() and p.suffix == "")
        for stem in ("carnot_core", "carnot_python")
    ]
    profile = scratch / "native.profdata"
    commands = [
        CommandSpec(
            "rust_profile_merge",
            (
                "llvm-profdata",
                "merge",
                "-sparse",
                *(str(p) for p in scratch.glob("native-*.profraw")),
                "-o",
                str(profile),
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "rust_coverage_export",
            (
                "llvm-cov",
                "export",
                str(objects[0]),
                "-object=" + str(objects[1]),
                "-instr-profile=" + str(profile),
                "-summary-only",
                *RUST[:2],
            ),
            "owned",
            60,
        ),
    ]
    commands += [
        CommandSpec(
            "rust_source_lines_" + str(i),
            (
                "llvm-cov",
                "show",
                str(objects[0]),
                "-object=" + str(objects[1]),
                "-instr-profile=" + str(profile),
                p,
            ),
            "owned",
            60,
        )
        for i, p in enumerate(RUST[:2])
    ]
    atomic_json(
        raw / "native_reduction_manifest.json",
        dict(
            commands=[asdict(c) for c in commands],
            executable_hashes=[reference(p) for p in objects],
        ),
    )
    reduced = run_commands(ROOT, commands, log_dir=raw / "native_reduction_logs", heartbeat_s=30)
    if not all(r["passed"] for r in reduced):
        return receipts + reduced, {}
    log = Path(reduced[1]["log_path"])
    exported = json.loads((ROOT / log).read_text())
    counts: Json = {}
    for i, p in enumerate(RUST[:2]):
        f = next(f for f in exported["data"][0]["files"] if Path(f["filename"]) == ROOT / p)
        llvm = f["summary"]["lines"]
        show = (ROOT / reduced[i + 2]["log_path"]).read_text()
        zeros = [
            (int(m[1]), m[2].strip())
            for line in show.splitlines()
            if (m := re.match(r"^\s*(\d+)\|\s*0\|(.*)$", line))
        ]
        missing = [line for line, text in zeros if not text.startswith("#[")]
        counts[p] = dict(
            count=llvm["covered"] + len(missing),
            covered=llvm["covered"],
            missing_source_statement_lines=missing,
            raw_llvm_lines=llvm,
            generated_attribute_lines=[line for line, text in zeros if text.startswith("#[")],
            generated_expansion_entries_excluded=llvm["count"] - llvm["covered"] - len(missing),
            units="LLVM executable source lines; PyO3 attribute expansions are generated, not source statements",
        )
    return receipts + reduced, counts


def base(failures: list[Json]) -> Json:
    """Terminal metadata separates numerical readiness from scientific benefit."""
    value = upstream.base(failures)
    value.update(
        experiment_id=8027,
        task_id=TASK,
        schema="carnot.v695.native_update_cost.v1",
        honest_verdict="complete_blocked_native_update_cost"
        if failures
        else "complete_null_native_update_cost",
        claim_scope="Opt-in float64 numerical parity and matched CPU costs on exposed cached development; no deployment promotion or hardware speed claim.",
        native_update_ready_score=0,
        default_enabled=False,
        learning_benefit_import={},
        loaded_extension_receipt={},
        parity_rows=[],
        checkpoint_hashes=[],
        timing_rows=[],
        kernel_speedup={},
        binding_speedup={},
        transaction_speedup={},
        nfr01_met=False,
        tier1_arithmetic_met=False,
        checkpoint_references=[],
        acceptance_gate_results=dict(parity=False, measurement=False, owned_checks=False),
        sample_size_budget={},
        rows=[],
        gate_check_summary=failures,
        genuine_headroom=dict(natural_learning_benefit=False),
        positive_control_results=dict(scope="isolated numerical stress; no natural benefit"),
    )
    value.pop("learning_measurement_ready_score", None)
    return value


def replay(path: Path) -> Json:
    """Cold reduction reads durable rows and reopens the actual binary state."""
    value = json.loads(path.read_text())
    for ref in (
        value["raw_shard_hashes"] + value["code_config_hashes"] + value["checkpoint_references"]
    ):
        checked(ref)
    if value["acceptance_gate_results"]["measurement"]:
        raw = Path(value["raw_directory"])
        timing = json.loads((raw / "timing_rows.json").read_text())["rows"]
        for k, v in timing_summary(timing).items():
            if value[k] != v:
                raise ValueError("reduction_drift:" + k)
        rows = json.loads((raw / "parity_rows.json").read_text())["rows"]
        if rows != value["parity_rows"] or not all(
            r["action_equal"]
            and r["probability_error"] <= 1e-10
            and r["coefficient_error"] <= 1e-10
            for r in rows
        ):
            raise ValueError("parity_drift")
        os.environ["CARNOT_8027_EXTENSION"] = value["loaded_extension_receipt"]["path"]
        native, receipt = extension()
        if receipt["sha256"] != value["loaded_extension_receipt"]["sha256"]:
            raise ValueError("binary_drift")
        with tempfile.TemporaryDirectory(prefix="carnot-8027-cold-") as temporary:
            cold = parity(
                json.loads((raw / "replay_inputs.json").read_text()), native, Path(temporary)
            )
            if cold["parity_rows"] != rows:
                raise ValueError("cold_equation_drift")
        for ref in value["checkpoint_references"]:
            state = json.loads(checked(ref).read_text())
            restored = native.RustNumericalUpdate8027(json.dumps(state))
            if np.max(np.abs(coefficients(state) - restored.effective())) > 1e-10:
                raise ValueError("restart_drift")
    if value["native_update_ready_score"] and (
        value["verdict_class"] in {"blocked", "disqualified"}
        or not all(value["acceptance_gate_results"].values())
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True)


def terminal(path: Path) -> Json:
    """Use the established cold/adversarial/strict reader orchestration unchanged."""
    return terminal_commands(path)


def terminal_commands(path: Path) -> Json:
    """Only the current producer's cold CLI differs from the existing readers."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / OWNED[-1]), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            120,
        ),
    ]
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=path.parent / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Run a current measurement or private fixtures with the same terminal readers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        print(json.dumps(replay(args.cold_replay)), flush=True)
        return 0
    started = time.monotonic()
    output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / output.stem
    raw.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    progress("freeze_before")
    with tempfile.TemporaryDirectory(prefix="carnot-8027-") as temporary:
        scratch = Path(temporary)
        commands = [] if args.validation_worker else validation_plan(scratch)
        health_once = raw / "repository_health_once.json"
        prior_health = json.loads(health_once.read_text()) if health_once.exists() else None
        if prior_health is not None:
            checked(dict(path=prior_health["log_path"], sha256=prior_health["log_sha256"]))
            commands = [c for c in commands if c.scope != "repository_health"]
        atomic_json(
            raw / "methods.json",
            dict(
                config=CONFIG,
                commands=[asdict(c) for c in commands],
                artifact_guard_enabled=True,
                source_roles="qualified head and released natural stream updates",
                exclusions="original public masks; no unknown target imputation; unmatched acquisition excluded",
                affinity=sorted(os.sched_getaffinity(0)),
                threads={
                    k: os.environ.get(k)
                    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
            ),
        )
        data, failures = (
            (json.loads(args.fixture_input.read_text()), [])
            if args.fixture_input
            else load_inputs(args.root, raw)
        )
        value = base(failures)
        value.update(
            config=CONFIG,
            random_seed=CONFIG["seed"],
            raw_directory=str(raw),
            cited_upstream_artifacts=data.get("references", []),
            continuous_self_learning_task=False,
            methodology_note="Matched float64 calibrated sparse updates; complete transactions include original public byte features, basis, binding, serialization, fsync and restart. Imported acquisition is separate and never a current model call.",
        )
        progress("freeze_after")
        if not failures:
            measured = time.monotonic()
            native, receipt = extension()
            binary = raw / "extension" / Path(receipt["path"]).name
            binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(receipt["path"], binary)
            os.environ["CARNOT_8027_EXTENSION"] = str(binary)
            native, loaded = extension()
            receipt.update(loaded)
            value["loaded_extension_receipt"] = receipt
            atomic_json(raw / "replay_inputs.json", data)
            os.environ["CARNOT_8027_EXTENSION"] = receipt["path"]
            value.update(parity(data, native, raw))
            value.update(benchmark(data, native, raw))
            value["rows"] = value["timing_rows"]
            value["acceptance_gate_results"].update(parity=value["parity_passed"], measurement=True)
            value["checkpoint_references"] += [r["checkpoint"] for r in value["timing_rows"]]
            value["checkpoint_hashes"] = [r["sha256"] for r in value["checkpoint_references"]]
            value["learning_benefit_import"] = dict(
                generalized_learning_benefit_score=0,
                scope="original complete null, no deployment benefit",
                producer_reference=(data.get("references", []) + [None, None])[1],
            )
            value["input_source_roster"] = [
                dict(
                    slot=r["slot"],
                    family_id=r["family_id"],
                    source_cluster_id=r["source_cluster_id"],
                    numerator=r["q"],
                    denominator=int(r["public_eligible"]),
                    eligibility=r["public_eligible"],
                    exclusion_reason=r.get("exclusion_reason"),
                    failure_reason=None,
                    censor_reason=None,
                )
                for r in data["sources"]
            ]
            value["trained_head_specs"] = [
                dict(
                    pretrained=False,
                    parameters=110,
                    operation="numerical replay of original CPU small-head updates",
                    updates=len(data["updates"]),
                    scientific_benefit_credit=False,
                )
            ]
            intended = len(data["updates"])
            value["sample_size_budget"] = dict(
                intended=intended,
                eligible=intended,
                started=intended,
                completed=intended,
                excluded=0,
                failed=0,
                censored=0,
                independent=value["natural_independent_n"],
                seeds_are_independent=False,
                stress=256,
                matched_repetitions=30,
                timing_rows=180,
                input_sources=dict(
                    intended=len(data["sources"]),
                    eligible=sum(r["public_eligible"] for r in data["sources"]),
                    excluded=sum(not r["public_eligible"] for r in data["sources"]),
                ),
            )
            value["phase_spans"].append(
                dict(phase="parity_and_cost", duration_s=time.monotonic() - measured)
            )
        progress("validation_before")
        validation_began = time.monotonic()
        receipts = (
            run_commands(
                ROOT,
                commands,
                log_dir=raw / "validation_logs",
                heartbeat_s=30,
                extra_env={"PYO3_PYTHON": sys.executable},
            )
            if commands
            else []
        )
        if prior_health is not None:
            receipts.append(dict(prior_health, current_task_diagnostic_reused=True))
        else:
            for r in receipts:
                if r["scope"] == "repository_health":
                    atomic_json(health_once, dict(r, log_path=str(ROOT / r["log_path"])))
        counts = (
            json.loads((scratch / "coverage.json").read_text())["files"]
            if (scratch / "coverage.json").exists()
            else {}
        )
        counts = {k: v["summary"] for k, v in counts.items()}
        owned = [r for r in receipts if r["scope"] == "owned"]
        native_receipts, native_counts = native_coverage(scratch, raw) if commands else ([], {})
        owned += native_receipts
        good = bool(owned) and all(r["passed"] for r in owned) and len(counts) == len(OWNED)
        good = good and all(r["missing_lines"] == 0 for r in counts.values())
        good = (
            good
            and len(native_counts) == 2
            and all(r["covered"] == r["count"] and r["count"] > 0 for r in native_counts.values())
        )
        value.update(
            validation_receipts=owned,
            coverage_statement_counts=counts,
            rust_executable_line_coverage=native_counts,
            repository_health=[r for r in receipts if r["scope"] == "repository_health"],
        )
        value["acceptance_gate_results"]["owned_checks"] = good
        if not failures and (not good or not value["parity_passed"]):
            value.update(
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_native_update_cost",
            )
        value["native_update_ready_score"] = int(not failures and good and value["parity_passed"])
        value["raw_shard_hashes"] = [
            reference(p)
            for p in raw.rglob("*")
            if p.is_file() and p.name not in {"terminal_candidate.json", "publication.lock"}
        ]
        value["code_config_hashes"] = [
            copy_evidence(reference(ROOT / p), raw)
            for p in [
                *OWNED,
                *RUST,
                TEST,
                "python/carnot/verify/causal_online_8025.py",
                "python/carnot/experiment_8008_v694_conditioned_energy_fit.py",
                "crates/carnot-python/Cargo.toml",
                "Cargo.lock",
            ]
        ]
        value["reproducibility_checksum"] = canonical_hash(
            dict(
                config=CONFIG,
                code=value["code_config_hashes"],
                inputs=value["cited_upstream_artifacts"],
            )
        )
        value["duration_s"] = time.monotonic() - started
        value["model_invocation_counts"] = dict(ZERO_INVOCATION_COUNTS)
        value["phase_spans"].append(
            dict(phase="validation", duration_s=time.monotonic() - validation_began)
        )
        value["terminal_validation_sidecar_path"] = str(raw / "validators")
        value["field_principles"] = {
            k: "Retain " + k + " to expose current work and its precise numerical-only limits."
            for k in value
        }
        value["field_principles"].update(
            {
                "honest_verdict": "Terminal completion does not promote the upstream null learner.",
                "verdict_class": "Numerical parity and speed cannot establish natural learning benefit.",
                "gate_check_summary": "Exact failed prerequisite operands prevent missing-field success.",
                "native_update_ready_score": "One requires complete parity, measurement and all owned checks, independent of benefit.",
                "loaded_extension_receipt": "The actually imported binary has a durable path and exact byte hash.",
                "parity_rows": "Every natural update and isolated stress case retains its numerical errors and typed actions.",
                "checkpoint_references": "Durable serialized lazy decay must survive a cold cross-language restart.",
                "timing_rows": "Matched inputs separate arithmetic, FFI, public features, storage and reopen costs.",
                "nfr01_met": "Tenfold throughput must hold for complete durable transactions at all registered batch sizes.",
                "tier1_arithmetic_met": "The measured p95 arithmetic cost must meet 1000 ns; it is not a promise.",
                "complete_service_estimates": "Hashed historical acquisition remains separate from current CPU work and model counters.",
                "coverage_statement_counts": "Only added Python and direct CLI statements require 100 percent execution.",
                "rust_executable_line_coverage": "Preserve raw LLVM counts; generated PyO3 attribute expansions are not source statements.",
                "sample_size_budget": "Seeds, repeats and stress cases do not enlarge independent natural source counts.",
                "model_invocation_counts": "Current pretrained loads and generations are zero; imported spans cannot change these counters.",
                "learning_benefit_import": "Import the valid null without claiming useful deployment or generalization.",
                "default_enabled": "The prototype requires explicit construction and changes no production default.",
                "validation_receipts": "Frozen argv, exits and byte-bound logs prove owned checks; broad health stays separate.",
                "code_config_hashes": "Contemporaneous source and build configuration bytes bind measurements to their implementation.",
                "raw_shard_hashes": "Cold readers reject drift in original durable evidence bytes.",
            }
        )
        progress("validation_after")
        publication = publish_primary(output, value, terminal)
        rechecked = terminal(output)
        atomic_json(
            raw / "published_readers.json",
            dict(
                publication=publication,
                terminal=rechecked,
                reader=reader_receipt(
                    TASK,
                    output.parent,
                    field="native_update_ready_score",
                    expected=value["native_update_ready_score"],
                ),
            ),
        )
        progress("published", 1, 0)
    return 0
