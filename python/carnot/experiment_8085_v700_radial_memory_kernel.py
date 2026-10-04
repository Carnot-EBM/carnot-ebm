"""REQ-REPORT-8085: qualify Gaussian arithmetic and durable public-center memory.

All labels are supplied numerical fixtures. Passing these checks provides no
independent decision or learning benefit and requires no current model load.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy import __version__ as scipy_version
from scipy.special import expit
from scipy.spatial.distance import cdist

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import constraint_projection_8075 as projected
from carnot import experiment_8008_v694_conditioned_energy_fit as old_basis
from carnot.verify import radial_memory_8085 as kernel

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8085_v700_radial_memory_kernel"
TASK = "exp8085-radial-memory-kernel"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/verify/radial_memory_8085.py", CLI]
TEST = "tests/python/test_radial_memory_8085.py"
SEED = 7008085
START = time.monotonic()
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
    "python/carnot/verify/sparse_energy_7996.py",
    "python/carnot/verify/constraint_projection_8075.py",
    "python/carnot/verify/projected_online_8076.py",
    "python/carnot/verify/fresh_feedback_8064.py",
    "python/carnot/verify/causal_online_8025.py",
    "python/carnot/experiment_8008_v694_conditioned_energy_fit.py",
    "results/experiment_8075_v699_constraint_projection_kernel.json",
    "results/experiment_8076_v699_projected_online_learning.json",
]
CONFIG = dict(
    seed=SEED,
    fixture_count=64,
    dimensions=9,
    ridge=0.01,
    maxiter=256,
    optimizer_deadline_s=600,
    fixture_deadline_s=300,
    tolerance=1e-10,
    dictionary=kernel.LIMITS,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts and flushed text keep a waiting parent observable."""
    print(
        f"[exp8085] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def prerequisites(root: Path, raw: Path) -> Json:
    """Snapshot actual operands and authenticate terminal hashes without science gates."""
    progress("preconditions_before")
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
        if not exists:
            continue
        target = raw / "inputs" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        refs.append(dict(path=str(path), snapshot_path=str(target), sha256=sha256_file(target)))
        if label.startswith("results/"):
            try:
                value = json.loads(path.read_text())
                terminal_path = Path(value["terminal_validation_sidecar_path"])
                terminal = json.loads(terminal_path.read_text())
                bound = Path(terminal["publication"]["sidecar_path"])
                observed: Any = read_bound_sidecar(path, bound)["report"]["passed"]
                for sidecar in [terminal_path, bound]:
                    snapshot = raw / "inputs" / (path.stem + "-" + sidecar.name)
                    shutil.copyfile(sidecar, snapshot)
                    refs.append(
                        dict(
                            path=str(sidecar),
                            snapshot_path=str(snapshot),
                            sha256=sha256_file(snapshot),
                        )
                    )
            except (KeyError, ValueError, OSError, TypeError) as exc:
                observed = str(exc)
            observations.append(
                dict(
                    row,
                    check="terminal_binding",
                    field="terminal.report.passed",
                    expected=True,
                    observed=observed,
                )
            )
    for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
        path = ROOT / ".venv/bin" / tool
        observations.append(
            dict(
                check="tool_exists",
                upstream="environment",
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                field="exists",
                op="==",
                expected=True,
                observed=path.is_file(),
            )
        )
    observations.append(
        dict(
            check="cpu_environment",
            upstream="environment",
            path="JAX_PLATFORMS",
            hash=None,
            field="environment",
            op="==",
            expected="cpu",
            observed=os.environ.get("JAX_PLATFORMS"),
        )
    )
    failures = [r for r in observations if r["expected"] != r["observed"]]
    progress("preconditions_after", len(observations), len(failures))
    return dict(
        preconditions_checked=observations,
        failures=failures,
        source_artifact_hashes=refs,
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            numpy=np.__version__,
            scipy=scipy_version,
            PYTHONUNBUFFERED=os.environ.get("PYTHONUNBUFFERED"),
            JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
        ),
    )


def fixture_feedback(slot: int, role: str = "update", count: int = 64) -> list[Json]:
    """Supplied releases include the original issued action for error ranking."""
    return [
        dict(
            source_id=f"{role}-{slot}-{i:03}",
            x=[i / 10] * 9,
            y=i % 2,
            role=role,
            eligible=True,
            release_slot=slot - count + i,
            observed_slot=slot,
            issued_action="accept" if i % 2 else "reject",
        )
        for i in range(count)
    ]


def controls(state: Json, raw: Path) -> list[Json]:
    """Retain invalid operands and cold reads at an abandoned atomic boundary."""
    path = raw / "controls" / "memory.json"
    kernel.save(path, state)
    zero = kernel.initialize(np.ones((32, 9)), [f"zero-{i}" for i in range(32)])
    rows = [
        dict(
            condition="zero_distances",
            passed=bool(np.all(kernel.design(zero, np.ones((1, 9))) == 1)),
            observed=zero["geometry"]["sigma"],
            expected=1,
        ),
        dict(
            condition="constant_columns",
            passed=zero["geometry"]["std"] == [1.0] * 9,
            observed=zero["geometry"]["std"],
            expected=[1.0] * 9,
        ),
    ]
    overflow = deepcopy(state)
    overflow["centers"] *= 2
    stale = deepcopy(state)
    stale["coefficients"].pop()
    tampered = json.loads(path.read_text())
    tampered["dictionary_records"][0]["coefficient_version"] += 1
    altered = path.parent / "tampered.json"
    atomic_json(altered, tampered)
    operations: list[tuple[str, Callable[[], Any], str]] = [
        ("nonfinite_input", lambda: kernel.matrix([[float("nan")] * 9]), "nonfinite_or_shape"),
        (
            "duplicate_source_ids",
            lambda: kernel.initialize(np.ones((32, 9)), ["duplicate"] * 32),
            "fit_source_identity",
        ),
        (
            "dictionary_overflow",
            lambda: kernel.candidate(overflow, fixture_feedback(64), 64, "feedback_grown"),
            "dictionary_overflow_or_coefficients",
        ),
        (
            "missing_feedback",
            lambda: kernel.candidate(state, [], 64, "feedback_grown"),
            "feedback_contract",
        ),
        (
            "stale_coefficients",
            lambda: kernel.predict(stale, np.zeros((1, 9))),
            "stale_coefficient_shape",
        ),
        (
            "dictionary_tamper",
            lambda: kernel.load(altered),
            "dictionary_hash_or_stale_coefficients",
        ),
    ]
    for condition, call, expected in operations:
        try:
            call()
            observed = "accepted_invalid"
        except ValueError as exc:
            observed = str(exc)
        rows.append(
            dict(
                condition=condition,
                passed=observed == expected,
                expected=expected,
                observed=observed,
            )
        )
    interrupted = path.with_name(".memory.json.tmp-interrupted")
    interrupted.write_bytes(b'{"incomplete":')
    rows.append(
        dict(
            condition="interrupted_commit",
            passed=kernel.load(path) == state,
            observed=canonical_hash(kernel.load(path)),
            expected=canonical_hash(state),
            temporary_path=str(interrupted),
            interruption_scope="abandoned atomic temporary before replacement",
        )
    )
    decisions = [kernel.action(p) for p in [0.1, 0.5]]
    rows.append(
        dict(
            condition="action_boundary",
            passed=decisions == ["escalate", "escalate"],
            observed=decisions,
            expected=["escalate", "escalate"],
        )
    )
    return rows


def historical(values: Any, labels: Any) -> Json:
    """Keep the qualified historical additive basis and projection as a separate fixture."""
    began = time.monotonic()
    x = np.asarray(values, dtype=float).copy()
    x[:, 0] = expit(x[:, 0])
    y = np.asarray(labels, dtype=float)
    geometry = old_basis.geometry(x)
    basis = old_basis.design("conditioned_energy", x, geometry)
    theta = np.zeros(basis.shape[1])
    initial = theta.copy()
    for _ in range(4):
        theta -= 0.01 * (basis.T @ (expit(basis @ theta) - y) / len(y) + 0.002 * theta)
    phi, w0 = projected.calibrated(basis, initial, [0.0, 1.0])
    constraints = [
        projected.constraint(vector, w0, int(label), f"historical-fixture-{i}")
        for i, (vector, label) in enumerate(zip(phi, y, strict=True))
    ]
    projection = projected.project(
        np.append(theta, 1.0), constraints, w0, w0, seed=SEED, frozen_last=True
    )
    return dict(
        basis_name="historical_conditioned_energy",
        scope="supplied numerical fixture using historical code, no historical learning credit",
        x=x.tolist(),
        y=y.tolist(),
        geometry=geometry,
        basis=basis.tolist(),
        initial=w0.tolist(),
        constraints=constraints,
        gradient_steps=4,
        raw_proposal=np.append(theta, 1.0).tolist(),
        projection=projection,
        duration_s=time.monotonic() - began,
    )


def measure(raw: Path) -> Json:
    """Retain every fixture operand, independent equation and real solve cost."""
    progress("benchmark_before", 0, 64)
    began = time.monotonic()
    systems, rows, lifecycle, edge_controls = [], [], [], []
    for i in range(64):
        rng = np.random.default_rng(SEED + i)
        x = rng.normal(size=(32, 9))
        x[:, -1] = 2
        state = kernel.initialize(x, [f"fit-{j:03}" for j in range(32)])
        theta = rng.normal(size=17)
        state["coefficients"] = theta.tolist()
        y = rng.integers(0, 2, 32).astype(float)
        phi = kernel.design(state, x)
        p = kernel.predict(state, x)
        g = state["geometry"]
        z = (x - np.asarray(g["mean"])) / np.asarray(g["std"])
        reference = np.column_stack(
            (
                np.ones(32),
                np.exp(
                    -cdist(z, np.asarray([r["x"] for r in state["centers"]]), "sqeuclidean")
                    / (2 * g["sigma"] ** 2)
                ),
            )
        )
        expected = expit(reference @ theta)
        _, gradient = kernel.objective(theta, phi, y, 0.01)
        numeric = []
        for j in range(17):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numeric.append(
                (kernel.objective(plus, phi, y, 0.01)[0] - kernel.objective(minus, phi, y, 0.01)[0])
                / 2e-5
            )
        solved = kernel.solve(phi, y, theta)
        rb = time.monotonic()
        refsolve = minimize(
            lambda w: (
                np.mean(np.logaddexp(0, reference @ w) - y * (reference @ w)) + 0.005 * (w @ w)
            ),
            np.zeros(17),
            jac=lambda w: reference.T @ (expit(reference @ w) - y) / 32 + 0.01 * w,
            method="BFGS",
            options=dict(gtol=1e-9, maxiter=256),
        )
        system = dict(
            seed=SEED + i,
            x=x.tolist(),
            y=y.tolist(),
            state=state,
            phi=phi.tolist(),
            probabilities=p.tolist(),
            reference_probabilities=expected.tolist(),
            gradient=gradient.tolist(),
            finite_difference=numeric,
            solve=solved,
            reference_solve=dict(
                coefficients=refsolve.x.tolist(),
                objective=float(refsolve.fun),
                success=bool(refsolve.success),
                iterations=int(refsolve.nit),
                duration_s=time.monotonic() - rb,
            ),
        )
        systems.append(system)
        error = float(np.max(np.abs(p - expected)))
        grad_error = float(np.max(np.abs(gradient - numeric)))
        parity = [kernel.action(float(v)) for v in p] == [kernel.action(float(v)) for v in expected]
        rows.append(
            dict(
                source=f"supplied-{i}",
                unit=f"fixture-{i}",
                arm="gaussian_energy_vs_logistic",
                condition="private_numerical",
                primitive_metric=error,
                probability_error=error,
                gradient_error=grad_error,
                action_parity=parity,
                numerator=int(error <= 1e-10 and grad_error <= 1e-8 and parity),
                denominator=1,
                status="completed",
                exclusion_reason=None,
            )
        )
        if i == 0:
            initial = kernel.initialize(x, [f"fit-{j:03}" for j in range(32)])
            edge_controls = controls(initial, raw)
            kernel.save(raw / "dictionary.json", initial)
            for slot in [64, 128, 192]:
                grown = kernel.candidate(initial, fixture_feedback(slot), slot, "feedback_grown")
                fixed = kernel.candidate(initial, fixture_feedback(slot), slot, "fixed_center")
                committed = kernel.commit(
                    raw / "dictionary.json",
                    initial,
                    grown,
                    fixture_feedback(slot + 80, "admission", 12),
                )
                lifecycle.append(
                    dict(
                        slot=slot,
                        initial=initial,
                        feedback_grown=grown,
                        fixed_center=fixed,
                        committed=committed,
                        replay_equal=kernel.load(raw / "dictionary.json") == committed,
                    )
                )
                initial = committed
        if (i + 1) % 8 == 0:
            progress("benchmark_systems", i + 1, 63 - i)
    progress("benchmark_after", 64, 0)
    evidence = dict(
        systems=systems,
        rows=rows,
        lifecycle_rows=lifecycle,
        edge_controls=edge_controls,
        historical_projected_control=historical(systems[0]["x"], systems[0]["y"]),
        duration_s=time.monotonic() - began,
    )
    atomic_json(raw / "evidence.json", evidence)
    return evidence


def reduce(evidence: Json) -> bool:
    """Rebuild primitive equations independently of the Gaussian implementation."""
    checks = [len(evidence["systems"]) == 64, len(evidence["rows"]) == 64]
    checks += [r["passed"] and r["observed"] == r["expected"] for r in evidence["edge_controls"]]
    for row, system in zip(evidence["rows"], evidence["systems"], strict=True):
        x, y = np.asarray(system["x"]), np.asarray(system["y"])
        state, g = system["state"], system["state"]["geometry"]
        std = x.std(0)
        std[std == 0] = 1
        z = (x - x.mean(0)) / std
        phi = np.column_stack(
            (
                np.ones(len(x)),
                np.exp(
                    -np.sum(
                        (z[:, None, :] - np.asarray([r["x"] for r in state["centers"]])[None, :, :])
                        ** 2,
                        axis=2,
                    )
                    / (2 * g["sigma"] ** 2)
                ),
            )
        )
        theta = np.asarray(state["coefficients"])
        p = 1 / (1 + np.exp(-(phi @ theta)))
        gradient = phi.T @ (p - y) / len(y) + 0.01 * theta
        fitted = np.asarray(system["solve"]["coefficients"])
        fit_objective = float(
            np.mean(np.logaddexp(0, phi @ fitted) - y * (phi @ fitted)) + 0.005 * (fitted @ fitted)
        )
        actions = ["accept" if 5 * v < 0.5 else "reject" if 1 - v < 0.5 else "escalate" for v in p]
        checks += [
            np.allclose(g["mean"], x.mean(0), atol=1e-14, rtol=0),
            np.allclose(g["std"], std, atol=1e-14, rtol=0),
            np.allclose(system["phi"], phi, atol=1e-14, rtol=0),
            np.max(np.abs(p - system["probabilities"])) <= 1e-10,
            np.allclose(gradient, system["gradient"], atol=1e-12, rtol=0),
            np.max(np.abs(gradient - system["finite_difference"])) <= 1e-8,
            actions == [kernel.action(v) for v in system["probabilities"]],
            abs(fit_objective - system["solve"]["objective"]) <= 1e-12,
            abs(fit_objective - system["reference_solve"]["objective"]) <= 1e-9,
            row["numerator"] == 1,
            row["probability_error"]
            == float(
                np.max(
                    np.abs(np.asarray(system["probabilities"]) - system["reference_probabilities"])
                )
            ),
        ]
    for lifecycle in evidence["lifecycle_rows"]:
        initial, grown, fixed, committed = [
            lifecycle[key] for key in ["initial", "feedback_grown", "fixed_center", "committed"]
        ]
        checks += [
            grown["centers"][:16] == initial["centers"][:16],
            len(grown["coefficients"]) == len(fixed["coefficients"]),
            len(grown["centers"]) <= 28,
            grown["proposal_hash"]
            == canonical_hash({k: v for k, v in grown.items() if k != "proposal_hash"}),
            committed["version"] == initial["version"] + 1,
            committed["commit_hash"]
            == canonical_hash({k: v for k, v in committed.items() if k != "commit_hash"}),
            lifecycle["replay_equal"],
            len(committed["last_commit"]["admissions"]) == 12,
        ]
    return bool(all(checks))


def manifest(private: Path) -> list[Json]:
    """Freeze exact owned validation and diagnostic argv before measurement."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    paths = [str(ROOT / p) for p in OWNED + [TEST]]
    commands: list[tuple[str, list[str], int, int]] = [
        (
            "focused_unit",
            [
                cov,
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *common,
                str(ROOT / TEST),
                "--basetemp=" + str(private / "unit"),
            ],
            0,
            180,
        ),
        (
            "consumer_e2e015",
            [
                pytest,
                *common,
                str(ROOT / "tests/python/test_primary_publication_7928.py"),
                str(ROOT / "tests/python/test_source_boundary_7852.py"),
                "--basetemp=" + str(private / "consumer"),
            ],
            0,
            180,
        ),
    ]
    success = private / "success" / (NAME + ".json")
    for condition in ["success", "blocked", "mutation"]:
        output = private / condition / (NAME + ".json")
        command = [
            cov,
            "run",
            "--rcfile=" + str(config),
            str(ROOT / CLI),
            "--fixture-output",
            str(output),
        ]
        if condition == "blocked":
            command += ["--root", str(private / "missing")]
        if condition == "mutation":
            command += ["--mutate"]
        commands.append(("private_cli_" + condition, command, 0, 120))
    commands += [
        (
            "private_cold_replay",
            [cov, "run", "--rcfile=" + str(config), str(ROOT / CLI), "--cold-replay", str(success)],
            0,
            60,
        ),
        ("coverage_combine", [cov, "combine", "--rcfile=" + str(config)], 0, 60),
        (
            "coverage_report",
            [cov, "report", "--rcfile=" + str(config), "--fail-under=100", "--show-missing"],
            0,
            60,
        ),
        (
            "coverage_json",
            [cov, "json", "--rcfile=" + str(config), "-o", str(private / "coverage.json")],
            0,
            60,
        ),
        ("ruff_check", [ruff, "check", *paths], 0, 60),
        ("ruff_format", [ruff, "format", "--check", *paths], 0, 60),
        (
            "mypy_strict",
            [
                mypy,
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                "--ignore-missing-imports",
                *paths[:-1],
            ],
            0,
            180,
        ),
        (
            "scoped_spec",
            [py, str(ROOT / "scripts/check_spec_coverage.py"), str(ROOT / TEST)],
            0,
            60,
        ),
        (
            "e2e016",
            [
                py,
                str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                "--date",
                "20260929",
                "--fixture-e2e",
                str(private / "e2e016.json"),
            ],
            0,
            120,
        ),
        (
            "e2e016_replay",
            [
                py,
                str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                "--date",
                "20260929",
                "--cold-replay",
                str(private / "e2e016.json"),
            ],
            0,
            60,
        ),
        (
            "repository_full_suite",
            [pytest, "tests/python", "-q", "--basetemp=" + str(private / "full")],
            0,
            180,
        ),
    ]
    for _name, argv, _expected, _deadline in commands:
        if argv[0] == cov:
            argv.insert(2, "--data-file=" + str(private / ".coverage"))
    return [
        dict(
            name=n,
            argv=a,
            expected_exit=e,
            deadline_s=d,
            classification="diagnostic" if n == "repository_full_suite" else "required",
        )
        for n, a, e, d in commands
    ]


def build(work: Json, raw: Path, receipts: list[Json], coverage: Json, fixture: bool) -> Json:
    """A terminal fixture result separates missing inputs from failed owned work."""
    evidence = work["evidence"]
    blocked = bool(work["failures"])
    valid = bool(evidence and reduce(evidence))
    passed = (
        all(r["passed"] for r in receipts if r.get("classification") != "diagnostic")
        and valid
        and not work.get("owned_failure", False)
    )
    covered = all(
        p in coverage and coverage[p]["num_statements"] == coverage[p]["covered_lines"]
        for p in OWNED
    )
    ready = int(passed and covered and not fixture and not blocked)
    disposition = (
        "blocked"
        if blocked
        else "circular_positive"
        if passed and (fixture or covered)
        else "disqualified"
    )
    rows = evidence.get("rows", [])
    result: Json = dict(
        experiment_id=8085,
        task_id=TASK,
        run_date="20261004",
        honest_verdict="complete_"
        + disposition
        + "_"
        + ("input" if blocked else "radial_memory_kernel"),
        verdict_class=disposition,
        kernel_ready_score=ready,
        verifier_is_oracle=True,
        claim_scope="64 supplied numerical systems and private memory lifecycle only; no independent verification or learning benefit",
        methodology_note="Fit-only nine-feature scaling; exact Gaussian energies; independent NumPy/SciPy equations, finite differences, bounded ridge fits, fresh one-use admission and durable cold restart. No model loaded.",
        flagged_adversarial=False,
        required_checks_passed=bool(passed and (fixture or covered) and not blocked),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        substrate_declaration="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[
            dict(
                kind="private Gaussian ridge energy",
                features=9,
                initial_centers=16,
                maximum_centers=28,
                maximum_coefficients=29,
                generator_weights_changed=False,
            )
        ],
        rows=rows,
        intended_count=64,
        eligible_count=len(rows),
        independent_count=0,
        completed_count=len(rows),
        excluded_count=64 - len(rows),
        censored_count=0,
        failed_count=sum(r["numerator"] == 0 for r in rows),
        sample_size_budget=dict(
            seeded_systems=64, natural_sources=0, seeds_are_not_independent=True
        ),
        random_seed=SEED,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        duration_s=work["duration_s"],
        exposure_scope="supplied-label numerical fixtures; unknown historical and model pretraining exposure",
        generalized_learning_benefit_score=0,
        basis_definition="E0=0; E1=-f; f=intercept+sum(w_j*exp(-||scaled_x-center_j||^2/(2*sigma^2))); fit-only mean/std, constant std=1; sigma=median positive fit distance or 1",
        dictionary_limits=kernel.LIMITS,
        gradient_checks=[
            dict(seed=s["seed"], analytic=s["gradient"], finite_difference=s["finite_difference"])
            for s in evidence.get("systems", [])
        ],
        reference_parity_rows=rows,
        lifecycle_rows=evidence.get("lifecycle_rows", []),
        memory_bytes=(raw / "dictionary.json").stat().st_size
        if (raw / "dictionary.json").is_file()
        else 0,
        edge_controls=evidence.get("edge_controls", []),
        environment=work.get("environment", {}),
        coverage_statement_counts=coverage,
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
        config=CONFIG,
        historical_projected_control=evidence.get("historical_projected_control", {}),
        solve_cost_rows=[
            dict(seed=s["seed"], owned=s["solve"], reference=s["reference_solve"])
            for s in evidence.get("systems", [])
        ],
    )
    result["field_principles"] = {
        key: f"Exact {key} binds this fixture observation and prevents unsupported current model or learning credit."
        for key in result
    }
    result["field_principles"].update(
        honest_verdict="Completed negative science is terminal and must not trigger retries.",
        verdict_class="Missing external inputs are blocked; owned validation failure is disqualified; partial means unfinished owned work.",
        kernel_ready_score="Exact fixture conformance and current owned checks qualify only a numerical kernel.",
        independent_count="Seed repetitions create no independent source groups.",
        memory_bytes="Serialized dictionary includes public provenance and admission audit, rather than generator weights.",
    )
    return result


def replay(path: Path) -> bool:
    """Cold reconstruction rejects altered evidence, dictionaries, code or logs."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        checks = [
            sha256_file(Path(r.get("snapshot_path", r["path"]))) == r["sha256"]
            for r in value["source_artifact_hashes"] + value["raw_shard_hashes"]
        ]
        checks += [sha256_file(ROOT / p) == h for p, h in value["code_config_hashes"].items()]
        checks += [
            sha256_file(Path(r["log_path"])) == r["log_sha256"]
            for r in value["validation_receipts"]
        ]
        checks.append(
            build(work, raw, validation["receipts"], validation["coverage"], validation["fixture"])
            == value
        )
        return all(checks)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path) -> Json:
    """Both real auditors and a cold process must accept the exact candidate."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    py = str(ROOT / ".venv/bin/python")
    specs = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    receipts: list[Json] = []
    with tempfile.TemporaryDirectory(prefix="carnot-8085-terminal-") as tmp:
        for name, argv in specs:
            progress("subprocess_before_" + name, len(receipts), len(specs) - len(receipts))
            receipts.append(
                run_check(
                    Path(tmp),
                    dict(name=name, argv=argv, expected_exit=0, deadline_s=60),
                    Path(tmp),
                    raw / "terminal_logs",
                )
            )
            progress("subprocess_after_" + name, len(receipts), len(specs) - len(receipts))
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def worker(root: Path, raw: Path, mutate: bool) -> None:
    """Save immutable observations in a child before the parent publishes."""
    work = prerequisites(root, raw)
    evidence = measure(raw) if not work["failures"] else {}
    if mutate and evidence:
        evidence["systems"][0]["state"]["centers"][0]["x"][0] += 1
        atomic_json(raw / "evidence.json", evidence)
    work.update(
        evidence=evidence,
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                *[p for p in INPUTS if p.endswith(".py")],
                "python/carnot/reporting/v686_contract_validation.py",
                "python/carnot/reporting/experiment_7303_validation_scope.py",
            ]
        },
        phase_spans=[dict(phase="qualification", duration_s=evidence.get("duration_s", 0))],
        duration_s=time.monotonic() - START,
        raw_shard_hashes=[
            dict(path=str(p), sha256=sha256_file(p))
            for p in sorted(raw.rglob("*"))
            if p.is_file() and "inputs" not in p.relative_to(raw).parts
        ],
    )
    atomic_json(raw / "work.json", work)


def main(argv: list[str] | None = None) -> int:
    """Run bounded private checks and preserve all existing terminal evidence."""
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
        passed = replay(args.cold_replay)
        progress("cold_replay_passed" if passed else "cold_replay_rejected")
        return 0 if passed else 1
    if args.worker_output:
        worker(args.root, args.worker_output.parent, args.mutate)
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if output.exists() or (raw / "work.json").exists():
        progress("existing_evidence_preserved")
        return 1
    with tempfile.TemporaryDirectory(prefix="carnot-8085-") as tmp:
        private = Path(tmp)
        specs = manifest(private)
        py = str(ROOT / ".venv/bin/python")
        cmd = [
            py,
            "-u",
            str(ROOT / CLI),
            "--root",
            str(args.root),
            "--worker-output",
            str(raw / "work.json"),
        ]
        if args.mutate:
            cmd += ["--mutate"]
        config = os.environ.get("CARNOT_8085_COVERAGE_CONFIG")
        if config:
            cmd = [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + config,
                "--data-file=" + str(Path(config).parent / ".coverage"),
                *cmd[2:],
            ]
        child = dict(name="measurement_normal_exit", argv=cmd, expected_exit=0, deadline_s=300)
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[child, *specs],
                config=CONFIG,
                code_hashes={p: sha256_file(ROOT / p) for p in OWNED},
                terminal_checks=["cold_replay", "adversarial", "strict_rows"],
            ),
        )
        progress("subprocess_before_measurement", 0, 1)
        receipt = run_check(private, child, private, raw / "validation_logs")
        progress("subprocess_after_measurement", 1, 0)
        if not receipt["passed"]:
            atomic_json(
                raw / "work.json",
                dict(
                    evidence={},
                    failures=[],
                    owned_failure=True,
                    source_artifact_hashes=[],
                    preconditions_checked=[],
                    code_config_hashes={p: sha256_file(ROOT / p) for p in OWNED},
                    raw_shard_hashes=[],
                    phase_spans=[],
                    duration_s=time.monotonic() - START,
                ),
            )
        work = json.loads((raw / "work.json").read_text())
        receipts, coverage = [receipt], {}
        if not args.fixture_output:
            os.environ["CARNOT_8085_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            for spec in specs:
                progress(
                    "subprocess_before_" + spec["name"],
                    len(receipts) - 1,
                    len(specs) - len(receipts) + 1,
                )
                cwd = (
                    ROOT
                    if spec["name"] in {"repository_full_suite", "consumer_e2e015"}
                    else private
                )
                receipts.append(run_check(cwd, spec, private, raw / "validation_logs"))
                progress(
                    "subprocess_after_" + spec["name"],
                    len(receipts) - 1,
                    len(specs) - len(receipts) + 1,
                )
            if (private / "coverage.json").is_file():
                data = json.loads((private / "coverage.json").read_text())
                coverage = {
                    str(Path(p).resolve().relative_to(ROOT)): v["summary"]
                    for p, v in data["files"].items()
                }
                atomic_json(raw / "coverage.json", data)
        atomic_json(
            raw / "validation.json",
            dict(receipts=receipts, coverage=coverage, fixture=bool(args.fixture_output)),
        )
        value = build(work, raw, receipts, coverage, bool(args.fixture_output))
        publication = publish_primary(output, value, terminal)
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, measurement_exit_receipt=receipt),
        )
        progress("terminal_published", len(value["rows"]), 0)
    return 0
