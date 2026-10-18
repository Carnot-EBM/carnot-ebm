"""REQ-REPORT-8105: loaded radial parity is fixture conformance, not learning.

The worker retains all arithmetic operands. Current validation and cold native
restoration qualify only this bounded Python/Rust component.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.primary_publication import (
    publish_primary,
    read_bound_sidecar,
    reader_receipt as reader_receipt,
)
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import native_radial_8105 as k
from carnot.verify import radial_memory_8085 as original

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8105_v701_native_radial_kernel"
TASK = "exp8105-native-radial-kernel"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_native_radial_8105.py"
RUST = "crates/carnot-python/src/radial_8105.rs"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/verify/native_radial_8105.py", CLI]
INPUTS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "python/carnot/verify/radial_memory_8085.py",
    "scripts/experiments/experiment_8027_v695_native_update_cost.py",
    "crates/carnot-python/Cargo.toml",
    "results/experiment_8085_v700_radial_memory_kernel.json",
    "results/experiment_8027_v695_native_update_cost.json",
]
CONFIG: Json = dict(
    seed=7018105,
    systems=64,
    dimensions=[1, 9],
    centers=[16, 20, 24, 28],
    probability_atol=1e-10,
    fallback_atol=1e-8,
    thresholds=[0.1, 0.5],
    ridge=0.01,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so parent processes can detect a stalled child."""
    print(f"[exp8105] {phase} completed={completed} pending={pending}", flush=True)


def prerequisites(root: Path, raw: Path) -> Json:
    """Observe exact input bytes; historical failures are provenance, not science gates."""
    observations, refs = [], []
    for label in INPUTS:
        path = root / label
        exists = path.is_file()
        row = dict(
            check="input_exists",
            upstream=path.stem,
            path=str(path),
            hash=sha256_file(path) if exists else None,
            field="exists",
            op="==",
            expected=True,
            observed=exists,
        )
        observations.append(row)
        if exists:
            snapshot = raw / "inputs" / label
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            refs.append(dict(path=str(snapshot), sha256=sha256_file(snapshot)))
            if label.startswith("results/"):
                digest = sha256_file(path).split(":")[-1]
                sidecar = path.parent / "raw" / path.stem / "validators" / (digest + ".json")
                try:
                    bound = read_bound_sidecar(path, sidecar)
                    observed: Any = bound["primary_sha256"] == sha256_file(path)
                    snapshot = raw / "inputs" / (path.stem + "-terminal.json")
                    atomic_json(snapshot, bound)
                    refs.append(dict(path=str(snapshot), sha256=sha256_file(snapshot)))
                except (OSError, ValueError, KeyError) as exc:
                    observed = str(exc)
                observations.append(
                    dict(
                        row,
                        check="terminal_binding",
                        path=str(sidecar),
                        field="primary_hash_bound",
                        observed=observed,
                    )
                )
    for tool in [
        "cargo",
        "rustfmt",
        "llvm-cov",
        "llvm-profdata",
        "python",
        "pytest",
        "coverage",
        "ruff",
        "mypy",
    ]:
        operand = (
            shutil.which(tool)
            if tool in {"cargo", "rustfmt", "llvm-cov", "llvm-profdata"}
            else str(ROOT / ".venv/bin" / tool)
        )
        observations.append(
            dict(
                check="tool_exists",
                upstream="environment",
                path=operand,
                hash=None,
                field="exists",
                op="==",
                expected=True,
                observed=bool(operand and Path(operand).is_file()),
            )
        )
    return dict(
        preconditions_checked=observations,
        gate_check_summary=[r for r in observations if r["observed"] != r["expected"]],
        source_artifact_hashes=refs,
    )


def fixture(dim: int, count: int, seed: int) -> Json:
    """Fixed public fixture geometry includes a constant column and duplicate centers."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(count, dim))
    centers[-1] = centers[0]
    return dict(
        geometry=dict(mean=[0.0] * dim, std=[1.0] * dim, sigma=2.0),
        centers=[
            dict(x=c.tolist(), source_id=f"fixture-center-{i}", feedback_origin="public_fixture")
            for i, c in enumerate(centers)
        ],
        coefficients=rng.normal(size=count + 1).tolist(),
        version=3,
        commit_hash="fixture",
    )


def measure(raw: Path, native: Any) -> Json:
    """Every comparison retains operands, native results and independent float64 results."""
    progress("benchmark_before", 0, 512)
    began = time.monotonic()
    systems, parity, serialization, fallback = [], [], [], []
    for dim in CONFIG["dimensions"]:
        for count in CONFIG["centers"]:
            for seed in range(CONFIG["systems"]):
                s = fixture(dim, count, CONFIG["seed"] + seed)
                x = np.random.default_rng(seed).normal(size=(8, dim))
                x[:, -1] = 2.0
                y = np.arange(8) % 2
                n = native.RustRadial8105(json.dumps(s))
                _, expected, gradient = k.reference(s, x, y)
                start = time.perf_counter_ns()
                probabilities = np.asarray(n.predict(x.tolist()))
                native_ns = time.perf_counter_ns() - start
                actual_gradient = np.asarray(n.gradient(x.tolist(), y.tolist(), CONFIG["ridge"]))
                np_p, np_actions, np_fallback = k.service(s, x)
                native_p, native_actions, native_fallback = k.service(s, x, n)
                unit = f"d{dim}-c{count}-seed{seed}"
                checkpoint = n.checkpoint()
                restored = native.RustRadial8105.restore(checkpoint)
                serialization.append(
                    dict(
                        unit_id=unit,
                        nonzero_coefficients=True,
                        state_equal=json.loads(restored.state_json()) == s,
                        probability_error=float(
                            np.max(abs(np.asarray(restored.predict(x.tolist())) - probabilities))
                        ),
                    )
                )
                if count < 28:
                    grown = deepcopy(s)
                    grown["centers"] += [
                        dict(x=[0.0] * dim, source_id=f"zero-{i}") for i in range(4)
                    ]
                    grown["coefficients"] += [0.0] * 4
                    serialization[-1]["zero_extension_unchanged"] = (
                        native.RustRadial8105(json.dumps(grown)).predict(x.tolist())
                        == probabilities.tolist()
                    )
                parity.append(
                    dict(
                        source_id="supplied_fixture",
                        unit_id=unit,
                        arm="native",
                        condition="numpy_float64",
                        issued_state=canonical_hash(s),
                        metric="numerical_conformance",
                        numerator=int(
                            float(np.max(abs(probabilities - expected))) <= 1e-10
                            and bool(np.isfinite(actual_gradient).all())
                            and native_actions == np_actions
                        ),
                        denominator=1,
                        status="completed",
                        exclusion_reason=None,
                        probability_error=float(np.max(abs(probabilities - expected))),
                        gradient_error=float(np.max(abs(actual_gradient - gradient))),
                        action_equal=native_actions == np_actions,
                        native_duration_ns=native_ns,
                    )
                )
                systems.append(
                    dict(
                        unit_id=unit,
                        state=s,
                        x=x.tolist(),
                        y=y.tolist(),
                        native_probabilities=probabilities.tolist(),
                        numpy_probabilities=expected.tolist(),
                        native_gradient=actual_gradient.tolist(),
                        numpy_gradient=gradient.tolist(),
                        checkpoint=checkpoint,
                        native_actions=native_actions,
                        numpy_actions=np_actions,
                    )
                )
                for arm, flags in [("native", native_fallback), ("numpy", np_fallback)]:
                    fallback.append(
                        dict(unit_id=unit, arm=arm, numerator=sum(flags), denominator=len(flags))
                    )
            progress("configuration_complete", len(systems), 512 - len(systems))
    s = fixture(9, 16, CONFIG["seed"])
    x = np.random.default_rng(91).normal(size=(32, 9))
    phi, _, _ = k.reference(s, x, np.arange(32) % 2)
    fitted = original.solve(phi, np.arange(32) % 2, np.zeros(17))
    s["coefficients"] = fitted["coefficients"]
    baseline = native.RustRadial8105(json.dumps(s))
    restored = native.RustRadial8105.restore(baseline.checkpoint())
    serialization.append(
        dict(
            unit_id="nonzero_fitted_baseline",
            fit=fitted,
            nonzero_coefficients=bool(np.any(s["coefficients"])),
            state_equal=json.loads(restored.state_json()) == s,
            probability_error=float(
                np.max(abs(np.asarray(restored.predict(x.tolist())) - baseline.predict(x.tolist())))
            ),
        )
    )
    fitted_baseline = dict(
        state=deepcopy(s),
        checkpoint=baseline.checkpoint(),
        x=x.tolist(),
        numpy_probabilities=baseline.predict(x.tolist()),
    )
    boundaries = []
    for p in [0.1 - 1e-9, 0.1, 0.1 + 1e-9, 0.5 - 1e-9, 0.5, 0.5 + 1e-9]:
        s["coefficients"] = [float(np.log(p / (1 - p)))] + [0.0] * 16
        n = native.RustRadial8105(json.dumps(s))
        a, b = k.service(s, x, n), k.service(s, x)
        boundaries.append(
            dict(
                condition="threshold",
                probability=p,
                passed=a[1] == b[1] and all(a[2]) and all(b[2]),
            )
        )
        for arm, result in [("native", a), ("numpy", b)]:
            fallback.append(
                dict(
                    unit_id=f"threshold-{p}", arm=arm, numerator=sum(result[2]), denominator=len(x)
                )
            )
    invalid = [
        ("nonfinite", lambda: n.predict([[float("nan")] * 9])),
        (
            "overflow_centers",
            lambda: native.RustRadial8105(json.dumps(dict(s, centers=s["centers"] * 2))),
        ),
        ("corrupt_checkpoint", lambda: native.RustRadial8105.restore("{}")),
    ]
    for condition, operation in invalid:
        try:
            operation()
            passed = False
        except ValueError:
            passed = True
        boundaries.append(dict(condition=condition, passed=passed))
    saturated = deepcopy(s)
    saturated["coefficients"] = [1000.0] + [0.0] * 16
    boundaries.append(
        dict(
            condition="logit_overflow",
            passed=native.RustRadial8105(json.dumps(saturated)).predict(x.tolist())
            == [1.0] * len(x),
        )
    )
    zero = fixture(1, 16, 0)
    for center in zero["centers"]:
        center["x"] = [2.0]
    boundaries.append(
        dict(
            condition="constant_duplicate_centers",
            passed=native.RustRadial8105(json.dumps(zero)).design([[2.0]]) == [[1.0] * 17],
        )
    )
    evidence = dict(
        fitted_baseline=fitted_baseline,
        systems=systems,
        parity_rows=parity,
        serialization_rows=serialization,
        fallback_rows=fallback,
        boundary_rows=boundaries,
        duration_s=time.monotonic() - began,
    )
    atomic_json(raw / "evidence.json", evidence)
    progress("benchmark_after", len(systems), 0)
    return evidence


def reduction(evidence: Json) -> Json:
    """Recompute denominators and acceptance from individual observations."""
    rows = evidence.get("parity_rows", [])
    primitive_valid = True
    for system, row in zip(evidence.get("systems", []), rows, strict=True):
        _, expected, gradient = k.reference(system["state"], system["x"], system["y"])
        probability_error = float(
            np.max(abs(np.asarray(system["native_probabilities"]) - expected))
        )
        gradient_error = float(np.max(abs(np.asarray(system["native_gradient"]) - gradient)))
        primitive_valid &= (
            probability_error == row["probability_error"]
            and gradient_error == row["gradient_error"]
        )
    passed = (
        primitive_valid
        and len(rows) == 512
        and all(
            r["numerator"] == 1
            and r["probability_error"] <= 1e-10
            and r["gradient_error"] <= 1e-10
            and r["action_equal"]
            for r in rows
        )
    )
    passed = passed and all(
        r["state_equal"]
        and r["probability_error"] <= 1e-10
        and r.get("zero_extension_unchanged", True)
        for r in evidence.get("serialization_rows", [])
    )
    passed = passed and all(r["passed"] for r in evidence.get("boundary_rows", []))
    fallback = evidence.get("fallback_rows", [])
    return dict(
        passed=bool(passed),
        completed_count=len(rows),
        failed_count=sum(r["numerator"] != 1 for r in rows),
        fallback_numerator=sum(r["numerator"] for r in fallback),
        fallback_denominator=sum(r["denominator"] for r in fallback),
    )


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze required commands before measuring; global health remains diagnostic."""
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ninclude =\n" + "".join(f"    {ROOT / p}\n" for p in OWNED)
    )
    commands = build_scoped_commands(
        ROOT,
        [TEST, "tests/python/test_primary_publication_7928.py"],
        OWNED[:2],
        static_paths=[CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    commands = [
        c
        for c in commands
        if c.name not in {"changed_module_coverage", "changed_module_coverage_report"}
    ]
    py = str(ROOT / ".venv/bin/python")
    commands[1:1] = [
        CommandSpec(
            "changed_module_coverage",
            (
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "--data-file=" + str(private / ".coverage"),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                TEST,
                "-q",
                "--basetemp=" + str(private / "coverage-tests"),
            ),
            "owned",
            300,
        ),
        CommandSpec(
            "coverage_combine",
            (
                py,
                "-m",
                "coverage",
                "combine",
                "--data-file=" + str(private / ".coverage"),
                str(private),
            ),
            "owned",
        ),
        CommandSpec(
            "changed_module_coverage_report",
            (
                py,
                "-m",
                "coverage",
                "json",
                "--data-file=" + str(private / ".coverage"),
                "-o",
                str(private / "coverage.json"),
                "--fail-under=100",
            ),
            "owned",
        ),
    ]
    commands = [
        CommandSpec(
            c.name,
            (*c.argv, "--strict", "--follow-imports=silent")
            if c.name == "changed_module_mypy"
            else c.argv,
            "owned",
            c.timeout_s,
        )
        for c in commands
    ]
    libdir = str(__import__("sysconfig").get_config_var("LIBDIR"))
    commands += [
        CommandSpec(
            "cargo_test",
            (
                "env",
                "PYO3_PYTHON=" + py,
                "LD_LIBRARY_PATH=" + libdir,
                "LLVM_PROFILE_FILE=" + str(private / "native-%p-%m.profraw"),
                "CARGO_TARGET_DIR=" + str(ROOT / "target/experiment-8105-coverage"),
                "RUSTFLAGS=-C instrument-coverage -C link-arg=-L"
                + libdir
                + " -C link-arg=-lpython3.12",
                "cargo",
                "test",
                "-p",
                "carnot-python",
                "radial_8105",
            ),
            "owned",
            300,
        ),
        CommandSpec(
            "cargo_fmt",
            (
                "cargo",
                "fmt",
                "-p",
                "carnot-python",
                "--",
                "--check",
                "--config",
                "skip_children=true",
            ),
            "owned",
        ),
        CommandSpec("radial_fmt", ("rustfmt", "--check", RUST), "owned"),
        CommandSpec(
            "cargo_clippy",
            ("cargo", "clippy", "-p", "carnot-python", "--no-deps"),
            "owned",
            300,
        ),
    ]
    return commands


def validate(commands: list[CommandSpec], raw: Path, private: Path) -> list[Json]:
    """Reuse bounded subprocess receipts with current hashes and normal exits."""
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=raw / "validation_logs",
        heartbeat_s=30,
        extra_env=dict(
            CARNOT_8105_COVERAGE_START=str(private / "coverage.ini"),
            COVERAGE_FILE=str(private / ".coverage"),
            JAX_PLATFORMS="cpu",
        ),
    )

    receipts += native_coverage(private, raw)
    if (private / "coverage.json").is_file():
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
    for index, saved in enumerate(sorted(private.rglob("cli_receipts.json"))):
        for item in json.loads(saved.read_text())["receipts"]:
            log = raw / "cli_validation" / f"{index}-{item['name']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["log_path"], log)
            receipts.append(dict(item, log_path=str(log)))
    return receipts


def worker(root: Path, raw: Path) -> None:
    """Measure in a child and retain its exact loaded library for cold replay."""
    began = time.monotonic()
    work = prerequisites(root, raw)
    evidence, receipt = {}, {}
    if not work["gate_check_summary"]:
        native, receipt = k.extension()
        destination = raw / Path(receipt["path"]).name
        k.copy_extension(receipt["path"], destination)
        os.environ["CARNOT_8105_EXTENSION"] = str(destination)
        native, loaded = k.extension()
        build_duration = receipt["build_duration_s"]
        receipt.update(loaded)
        receipt["build_duration_s"] = build_duration
        evidence = measure(raw, native)
    work.update(
        evidence=evidence,
        loaded_binding_receipt=receipt,
        duration_s=time.monotonic() - began,
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                RUST,
                "crates/carnot-python/src/lib.rs",
                "crates/carnot-python/Cargo.toml",
            ]
        },
        raw_shard_hashes=[
            dict(path=str(raw / "evidence.json"), sha256=sha256_file(raw / "evidence.json"))
        ]
        if evidence
        else [],
        config_hash=canonical_hash(CONFIG),
        phase_spans=[dict(phase="native_qualification", duration_s=time.monotonic() - began)],
    )
    atomic_json(raw / "work.json", work)


def build(work: Json, receipts: list[Json], raw: Path, fixture_only: bool, mutate: bool) -> Json:
    """Readiness requires current owned checks, while supplied fixtures grant no science credit."""
    evidence = work["evidence"]
    reduced = reduction(evidence)
    blocked = bool(work["gate_check_summary"])
    owned = bool(receipts) and all(
        r["passed"] for r in receipts if r.get("scope") != "repository_health"
    )
    valid = reduced["passed"] and owned and not mutate
    verdict = "blocked" if blocked else "circular_positive" if valid else "disqualified"
    ready = int(valid and not blocked and not fixture_only)
    lib = work["loaded_binding_receipt"]
    value: Json = dict(
        experiment_id=8105,
        task_id=TASK,
        schema="carnot.v701.native_radial.v1",
        run_date="20261004",
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            str(work["gate_check_summary"][0]["upstream"]).lower().replace("-", "_")
            if blocked
            else "native_radial_kernel"
        ),
        verdict_class=verdict,
        native_kernel_ready_score=ready,
        verifier_is_oracle=True,
        claim_scope="Supplied numerical fixture conformance through a loaded Rust/PyO3 binding; no natural learning, FPGA execution or 10x service speedup.",
        exposure_scope="Public numerical fixtures only; no natural-data inference",
        generalized_learning_benefit_score=0,
        methodology_note="Independent float64 broadcast Gaussian equations and ridge gradients; bounded dimensions and centers; canonical threshold fallback and cold native checkpoints. No current model calls.",
        flagged_adversarial=False,
        required_checks_passed=bool(valid and not blocked and not fixture_only),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        substrate_declaration="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[
            dict(
                kind="supplied_fixture_radial_ridge_fit",
                coefficients=17,
                generator_weights_changed=False,
            )
        ],
        model_invoked=False,
        rows=evidence.get("parity_rows", []),
        intended_count=512,
        eligible_count=reduced["completed_count"],
        independent_count=0,
        completed_count=reduced["completed_count"],
        excluded_count=512 - reduced["completed_count"],
        censored_count=0,
        failed_count=reduced["failed_count"],
        sample_size_budget=dict(
            seeded_systems_per_configuration=64,
            configurations=8,
            natural_sources=0,
            seeds_are_not_independent=True,
        ),
        duration_s=work["duration_s"],
        random_seed=CONFIG["seed"],
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        native_library_path=lib.get("path"),
        native_library_sha256=lib.get("sha256"),
        loaded_binding_receipt=lib,
        build_duration_s=lib.get("build_duration_s", 0),
        parity_rows=evidence.get("parity_rows", []),
        fallback_rows=evidence.get("fallback_rows", []),
        serialization_rows=evidence.get("serialization_rows", []),
        boundary_rows=evidence.get("boundary_rows", []),
        reduction=reduced,
        fallback_frequency=reduced["fallback_numerator"] / max(1, reduced["fallback_denominator"]),
        acceptance_gates=dict(
            numerical_parity=reduced["passed"],
            owned_validation=owned,
            actual_loaded=bool(lib.get("actual_loaded")),
            full_validation=not fixture_only,
            mutation_absent=not mutate,
        ),
        config=CONFIG,
        mutation_fixture=mutate,
        fixture_only=fixture_only,
    )
    value["field_principles"] = {
        key: f"Recorded {key} binds the current fixture observations and prevents unsupported model or learning credit."
        for key in value
    }
    if (raw / "repository_health_once.json").is_file():
        value["repository_health"] = json.loads((raw / "repository_health_once.json").read_text())[
            "receipts"
        ]
        value["field_principles"]["repository_health"] = (
            "A bounded global diagnostic cannot be substituted for owned numerical validation."
        )
    value["field_principles"].update(
        native_kernel_ready_score="Only current owned checks plus loaded native conformance qualify a usable binding.",
        independent_count="Seed repetitions are not independently acquired natural sources.",
        fallback_frequency="Threshold fixtures are included explicitly; this is not a natural service distribution.",
        verdict_class="External missing operands are terminal blocked; owned validation failure is terminal disqualified.",
        acceptance_gates="Each gate prevents source-only compilation or numerical fixture success from becoming deployment or learning evidence.",
    )
    return value


def replay(path: Path) -> bool:
    """Independent reads reject changed logs, state, native bytes and aggregate reductions."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "work.json").read_text())
        if value["mutation_fixture"]:
            return False
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for p, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / p) != digest:
                return False
        for receipt_row in [*value["validation_receipts"], *value.get("repository_health", [])]:
            if (
                receipt_row.get("log_path")
                and sha256_file(ROOT / receipt_row["log_path"]) != receipt_row["log_sha256"]
            ):
                return False
        if work["evidence"]:
            native, receipt = load_saved(value)
            if receipt["sha256"] != value["native_library_sha256"]:
                return False
            evidence = json.loads((raw / "evidence.json").read_text())
            if evidence != work["evidence"]:
                return False
            for system in [*evidence["systems"], evidence["fitted_baseline"]]:
                n = native.RustRadial8105.restore(system["checkpoint"])
                expected = np.asarray(system["numpy_probabilities"])
                if np.max(abs(np.asarray(n.predict(system["x"])) - expected)) > 1e-10:
                    return False
        return build(work, value["validation_receipts"], raw, value["fixture_only"], False) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_saved(value: Json) -> tuple[Any, Json]:
    """Select the durable task-owned inode explicitly for a cold native process."""
    os.environ["CARNOT_8105_EXTENSION"] = value["native_library_path"]
    return k.extension()


def terminal(path: Path) -> Json:
    """Authenticate cold replay and both established artifact consumers before publication."""
    raw = path.parent
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_replay", (py, "-u", str(ROOT / CLI), "--cold-replay", str(path)), "terminal"
        ),
        CommandSpec(
            "adversarial",
            (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
        ),
        CommandSpec(
            "strict_rows",
            (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
        ),
    ]
    receipts = run_commands(ROOT, commands, log_dir=raw / "terminal_logs", heartbeat_s=30)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze validation first, require normal worker exit, and preserve historical primaries."""
    began = time.monotonic()
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    if args.worker_output:
        worker(args.root, args.worker_output.parent)
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if output.exists() or (raw / "work.json").exists():
        progress("existing_evidence_preserved")
        return 1
    with tempfile.TemporaryDirectory(prefix="carnot-8105-") as tmp:
        private = Path(tmp)
        commands = validation_plan(private)
        child = dict(
            name="measurement_normal_exit",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--root",
                str(args.root),
                "--worker-output",
                str(raw / "work.json"),
            ],
            expected_exit=0,
            deadline_s=300,
        )
        atomic_json(
            raw / "validation_commands.json",
            dict(commands=[asdict(c) for c in commands], measurement=child, config=CONFIG),
        )
        progress("before_measurement_subprocess")
        receipt = run_check(private, child, private, raw / "validation_logs")
        progress("after_measurement_subprocess")
        if not receipt["passed"]:
            atomic_json(
                raw / "work.json",
                dict(
                    evidence={},
                    loaded_binding_receipt={},
                    gate_check_summary=[],
                    preconditions_checked=[],
                    source_artifact_hashes=[],
                    raw_shard_hashes=[],
                    code_config_hashes={p: sha256_file(ROOT / p) for p in OWNED},
                    phase_spans=[],
                    duration_s=time.monotonic() - began,
                    owned_failure=True,
                ),
            )
        work = json.loads((raw / "work.json").read_text())
        receipts = [dict(receipt, scope="owned")]
        if not args.fixture_output:
            os.environ["CARNOT_8105_EXTENSION"] = work["loaded_binding_receipt"].get("path", "")
            receipts += validate(commands, raw, private)
        work["duration_s"] = time.monotonic() - began
        work["phase_spans"].append(
            dict(
                phase="parent_validation",
                duration_s=sum(r.get("duration_s", 0) for r in receipts[1:]),
            )
        )
        for label in [
            "coverage.json",
            "native_coverage.json",
            "validation_commands.json",
            "repository_health_once.json",
        ]:
            if (raw / label).is_file():
                work["raw_shard_hashes"].append(
                    dict(path=str(raw / label), sha256=sha256_file(raw / label))
                )
        atomic_json(raw / "work.json", work)
        value = build(work, receipts, raw, bool(args.fixture_output), args.mutate)
        if (raw / "repository_health_once.json").is_file():
            value["repository_health"] = json.loads(
                (raw / "repository_health_once.json").read_text()
            )["receipts"]
            value["field_principles"]["repository_health"] = (
                "A bounded global diagnostic cannot be substituted for owned numerical validation."
            )
        if args.mutate:
            atomic_json(raw / "failed_mutation_candidate.json", value)
            progress("mutation_rejected")
            return 1
        else:
            publication = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, measurement_exit_receipt=receipt),
            )
        progress("terminal_published", value["completed_count"], 0)
    return 0


def native_coverage(private: Path, raw: Path) -> list[Json]:
    """LLVM counters cover owned Rust source; generated PyO3 attributes are separate."""
    profiles = list(private.glob("native-*.profraw"))
    objects = [
        p
        for p in (ROOT / "target/experiment-8105-coverage/debug/deps").glob("carnot_python-*")
        if p.is_file() and p.suffix == ""
    ]
    if not profiles or not objects:
        return [
            dict(
                name="native_statement_coverage",
                scope="owned",
                passed=False,
                observed="missing_profile_or_object",
            )
        ]
    merged = private / "native.profdata"
    specs = [
        CommandSpec(
            "native_profile_merge",
            ("llvm-profdata", "merge", "-sparse", *map(str, profiles), "-o", str(merged)),
            "owned",
        ),
        CommandSpec(
            "native_source_coverage",
            (
                "llvm-cov",
                "show",
                str(objects[0]),
                "-instr-profile=" + str(merged),
                str(ROOT / RUST),
            ),
            "owned",
        ),
    ]
    receipts = run_commands(ROOT, specs, log_dir=raw / "native_coverage_logs", heartbeat_s=30)
    if not all(r["passed"] for r in receipts):
        return receipts
    show = (ROOT / receipts[-1]["log_path"]).read_text()
    lines = [
        (int(m[1]), int(m[2]), m[3].strip())
        for line in show.splitlines()
        if (m := re.match(r"^\s*(\d+)\|\s*(\d+)\|(.*)$", line))
    ]
    excluded = [number for number, count, source in lines if count == 0 and source.startswith("#[")]
    missing = [
        number for number, count, source in lines if count == 0 and not source.startswith("#[")
    ]
    result = dict(
        path=RUST,
        executable_source_lines=len(lines) - len(excluded),
        covered_source_lines=sum(count > 0 for _, count, _ in lines),
        missing_source_lines=missing,
        generated_attribute_lines=excluded,
        units="LLVM executable source lines excluding generated PyO3 attribute expansions",
    )
    atomic_json(raw / "native_coverage.json", result)
    receipts.append(
        dict(
            name="native_statement_coverage",
            scope="owned",
            passed=bool(lines and not missing),
            coverage=result,
        )
    )
    return receipts
