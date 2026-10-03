"""REQ-REPORT-8075: qualify numerical geometry without learning-benefit credit.

Private supplied-label fixtures compare a bounded correction with a convex QP.
The existing learner is not changed. Publication requires normally exited work
and current checks; historical model calls do not become current model calls.
"""

from __future__ import annotations

import argparse
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
from scipy import __version__ as scipy_version
from scipy.optimize import Bounds, LinearConstraint, minimize, nnls
from scipy.special import expit

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import constraint_projection_8075 as kernel

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8075_v699_constraint_projection_kernel"
TASK = "exp8075-constraint-projection-kernel"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
KERNEL = "python/carnot/verify/constraint_projection_8075.py"
TEST = "tests/python/test_constraint_projection_8075.py"
OWNED = [MODULE, KERNEL, CLI]
SEED = 6998075
START = time.monotonic()
INPUTS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "python/carnot/verify/sparse_energy_7996.py",
    "python/carnot/verify/fresh_feedback_8064.py",
    "python/carnot/verify/feedback_constrained_8051.py",
    "python/carnot/verify/guarded_transaction_8053.py",
    "results/experiment_8063_v698_admission_opportunity_audit.json",
    "research-references.md",
    "results/experiment_8072_v699_sealed_methods.json",
]
CONFIG: Json = dict(
    memory_capacity=64,
    batch=8,
    budget=256,
    tolerance=1e-8,
    coefficient_box=0.5,
    fixture_count=64,
    fixture_budget_s=300,
    calibration_order="intercept,slope",
    frozen_augmented_coefficient=1,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts and elapsed time so quiet subprocesses remain visible."""
    print(
        f"[exp8075] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def prerequisites(root: Path, raw: Path) -> Json:
    """Preserve missing operands and original bytes instead of inventing inputs."""
    progress("preconditions_before")
    refs, failures = [], []
    for label in INPUTS:
        path = root / label
        if not path.is_file():
            failures.append(
                dict(
                    check="input_exists",
                    upstream=path.stem,
                    path=str(path),
                    hash=None,
                    field="exists",
                    op="==",
                    expected=True,
                    observed="missing",
                )
            )
            continue
        target = raw / "inputs" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        refs.append(dict(path=str(path), snapshot_path=str(target), sha256=sha256_file(target)))
        if label.startswith("results/experiment_"):
            try:
                value = json.loads(path.read_text())
                terminal = Path(value["terminal_validation_sidecar_path"])
            except (ValueError, KeyError, TypeError):
                failures.append(
                    dict(
                        check="input_json",
                        upstream=path.stem,
                        path=str(path),
                        hash=sha256_file(path),
                        field="terminal_validation_sidecar_path",
                        op="present",
                        expected="readable JSON and path",
                        observed="unreadable_or_absent",
                    )
                )
                continue
            observed: Any = "missing"
            if terminal.is_file():
                try:
                    binding = json.loads(terminal.read_text())["publication"]["sidecar_path"]
                    observed = read_bound_sidecar(path, Path(binding))["report"]["passed"]
                    for sidecar in [terminal, Path(binding)]:
                        frozen = raw / "inputs" / (sidecar.name + path.stem)
                        shutil.copyfile(sidecar, frozen)
                        refs.append(
                            dict(
                                path=str(sidecar),
                                snapshot_path=str(frozen),
                                sha256=sha256_file(frozen),
                            )
                        )
                except (ValueError, KeyError, OSError):
                    observed = "invalid_terminal_binding"
            checks: list[tuple[str, Any, Any]] = [("terminal.passed", True, observed)]
            if "8072" in label:
                checks += [
                    (
                        "learning_protocol_ready_score",
                        1,
                        value.get("learning_protocol_ready_score"),
                    ),
                    ("flagged_adversarial", False, value.get("flagged_adversarial")),
                    (
                        "verdict_class_allowed",
                        True,
                        value.get("verdict_class") in {"null", "positive"},
                    ),
                    (
                        "candidate_equations",
                        CONFIG["budget"],
                        value.get("candidate_equations", {}).get("max_steps"),
                    ),
                ]
            for field, expected, actual in checks:
                if actual != expected:
                    failures.append(
                        dict(
                            check=field,
                            upstream=path.stem,
                            path=str(path),
                            hash=sha256_file(path),
                            field=field,
                            op="==",
                            expected=expected,
                            observed=actual,
                        )
                    )
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / tool
        if not path.is_file():
            failures.append(
                dict(
                    check="required_tool",
                    upstream="python_environment",
                    path=str(path),
                    hash=None,
                    field="exists",
                    op="==",
                    expected=True,
                    observed="missing",
                )
            )
    if sys.version_info < (3, 11):  # noqa: UP036 - an explicit environment gate is required.
        failures.append(
            dict(
                check="python_version",
                upstream="python_environment",
                path=sys.executable,
                hash=None,
                field="version",
                op=">=",
                expected="3.11",
                observed=str(sys.version_info),
            )
        )
    progress("preconditions_after", len(refs), len(failures))
    return dict(
        source_artifact_hashes=refs,
        failures=failures,
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            numpy=np.__version__,
            scipy=scipy_version,
            tools=[
                str(ROOT / ".venv/bin" / p)
                for p in ("python", "pytest", "coverage", "ruff", "mypy")
            ],
        ),
    )


def qp(proposal: Any, rows: list[Json], initial: Any, *, frozen_last: bool = False) -> Json:
    """Use an independent convex optimizer; nearest distance is a separate claim."""
    raw, w0 = np.asarray(proposal, dtype=float), np.asarray(initial, dtype=float)
    matrix = np.asarray([r["normal"] for r in rows], dtype=float)
    rhs = np.asarray([r["rhs"] for r in rows])
    low, high = w0 - 0.5, w0 + 0.5
    if frozen_last:
        low[-1] = high[-1] = 1
    began = time.perf_counter_ns()
    result = minimize(
        lambda w: 0.5 * np.sum((w - raw) ** 2),
        w0,
        jac=lambda w: w - raw,
        method="SLSQP",
        bounds=Bounds(low, high),
        constraints=[LinearConstraint(matrix, rhs, np.inf)],
        options=dict(ftol=1e-12, maxiter=1000),
    )
    maximum = float(np.max(kernel.residuals(result.x, matrix, rhs, low, high)))
    faces = np.vstack((matrix, np.eye(len(w0)), -np.eye(len(w0))))
    bounds = np.concatenate((rhs, low, -high))
    active = (faces @ result.x - bounds) <= 1e-7
    multipliers, stationarity = nnls(faces[active].T, result.x - raw, maxiter=10000)
    return dict(
        point=result.x.tolist(),
        feasible=bool(maximum <= kernel.TOL),
        reference_certified=bool(maximum <= kernel.TOL and stationarity <= 1e-6),
        stationarity_residual=float(stationarity),
        active_face_indices=np.flatnonzero(active).tolist(),
        dual_multipliers=multipliers.tolist(),
        success=bool(result.success),
        message=str(result.message),
        max_residual=maximum,
        distance=float(np.linalg.norm(result.x - raw)),
        iterations=int(result.nit),
        duration_ns=time.perf_counter_ns() - began,
    )


def measure(raw: Path, *, fixture: bool = False) -> Json:
    """Persist supplied systems, released-label receipts and all bounded work."""
    progress("benchmark_before", 0, 64)
    raw.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    rows, systems = [], []
    for seed in range(64):
        if time.monotonic() - began > 300:
            raise TimeoutError("fixture_budget_300s")
        rng = np.random.default_rng(SEED + seed)
        basis = rng.normal(size=(16, 4))
        phi, w0 = kernel.calibrated(basis, rng.normal(0, 0.15, 4), [0.2, 1.1])
        labels = rng.integers(0, 2, 16)
        storage_began = time.perf_counter_ns()
        memory = kernel.Memory(raw / "memories" / f"seed-{seed}.json", w0)
        for slot, (vector, label) in enumerate(zip(phi, labels, strict=True)):
            memory.add(
                dict(
                    source_id=f"seed-{seed}/source-{slot:02}",
                    release_slot=slot,
                    observed_slot=slot,
                    role="update",
                    eligible=True,
                    y=int(label),
                    phi=vector.tolist(),
                )
            )
        storage_ns = time.perf_counter_ns() - storage_began
        gradient_began = time.perf_counter_ns()
        gradient = phi.T @ (expit(phi @ w0) - labels) / len(labels)
        gradient[-1] = 0
        proposal = w0 - 3 * gradient
        gradient_ns = time.perf_counter_ns() - gradient_began
        projected = kernel.project(
            proposal, memory.rows, w0, w0, seed=SEED + seed, frozen_last=True
        )
        reference = qp(proposal, memory.rows, w0, frozen_last=True)
        systems.append(
            dict(
                seed=SEED + seed,
                initial=w0.tolist(),
                incumbent=w0.tolist(),
                proposal=proposal.tolist(),
                gradient=gradient.tolist(),
                constraint_construction_storage_ns=storage_ns,
                gradient_ns=gradient_ns,
                constraints=memory.rows,
                projection=projected,
                qp=reference,
            )
        )
        rows.append(
            dict(
                unit=f"fixture-{seed}",
                source=f"supplied-system-{seed}",
                arm="skm_vs_qp",
                seed=SEED + seed,
                condition="private_feasible",
                status="completed",
                exclusion_reason=None,
                feasible=projected["feasible"],
                candidate_feasible=projected["candidate_feasible"],
                max_residual=projected["max_residual"],
                distance=projected["distance"],
                qp_distance=reference["distance"],
                distance_gap=projected["distance"] - reference["distance"],
                projection_steps=projected["projection_steps"],
                numerator=int(projected["feasible"]),
                denominator=1,
            )
        )
        if (seed + 1) % 8 == 0:
            progress("benchmark_systems", seed + 1, 64 - seed - 1)
    controls = []
    simple = [dict(source_id="control", normal=[1.0], rhs=0.0)]
    for label, incumbent, expected in [
        ("incumbent", [0.2], "incumbent"),
        ("reset", [-0.2], "initial"),
    ]:
        result = kernel.project([-0.3], simple, [0.0], incumbent, seed=SEED, budget=0)
        controls.append(
            dict(
                condition=label,
                passed=result["fallback"] == expected and not result["candidate_feasible"],
                result=result,
            )
        )
    for label, bad, expected in [
        ("zero_norm", [dict(source_id="zero", normal=[0.0], rhs=1.0)], "zero_norm_violated"),
        (
            "contradictory",
            [
                dict(source_id="a", normal=[1.0], rhs=0.1),
                dict(source_id="b", normal=[-1.0], rhs=0.1),
            ],
            "initial_infeasible",
        ),
        (
            "nonfinite",
            [dict(source_id="bad", normal=[float("nan")], rhs=0.0)],
            "nonfinite_constraint",
        ),
    ]:
        try:
            kernel.project([0.0], bad, [0.0], [0.0], seed=SEED)
            reason = "accepted_invalid"
        except ValueError as exc:
            reason = str(exc)
        controls.append(
            dict(condition=label, passed=reason == expected, observed=reason, expected=expected)
        )
    ref = qp([0.0], [dict(normal=[1.0], rhs=0.1), dict(normal=[-1.0], rhs=0.1)], [0.0])
    controls.append(dict(condition="qp_contradictory", passed=not ref["feasible"], result=ref))
    memory = kernel.Memory(raw / "release_controls.json", [0.0])
    for slot in range(66):
        memory.add(
            dict(
                source_id=f"control-{slot:02}",
                release_slot=slot,
                observed_slot=slot,
                role="update",
                eligible=True,
                y=slot % 2,
                phi=[1.0],
            )
        )
    late = dict(
        source_id="late",
        release_slot=0,
        observed_slot=66,
        role="update",
        eligible=True,
        y=1,
        phi=[1.0],
    )
    memory.add(late)
    duplicate = memory.add(late)
    replayed = kernel.Memory(memory.path, [0.0])
    controls.append(
        dict(
            condition="memory_roundtrip_eviction_late_duplicate",
            passed=len(memory.rows) == 64
            and memory.rows[0]["source_id"] == "control-02"
            and duplicate == "duplicate"
            and replayed.state == memory.state,
            events=memory.state["events"],
            active_sources=[r["source_id"] for r in memory.rows],
        )
    )
    progress("benchmark_after", 64, 0)
    evidence = dict(
        rows=rows,
        systems=systems,
        controls=controls,
        fixture=fixture,
        phase_spans=[dict(name="fixtures", duration_s=time.monotonic() - began)],
    )
    atomic_json(raw / "evidence.json", evidence)
    return evidence


def reduce(evidence: Json) -> bool:
    """Rebuild the correction equations without invoking the producer kernel."""
    if len(evidence["systems"]) != 64 or not all(r["passed"] for r in evidence["controls"]):
        return False
    for row, system in zip(evidence["rows"], evidence["systems"], strict=True):
        initial, proposal = np.asarray(system["initial"]), np.asarray(system["proposal"])
        constraints = system["constraints"]
        for constraint in constraints:
            receipt = constraint["release_receipt"]
            if (
                receipt["role"] != "update"
                or not receipt["eligible"]
                or receipt["observed_slot"] < receipt["release_slot"]
                or receipt["y"] not in (0, 1)
            ):
                return False
            sign = 2 * receipt["y"] - 1
            vector = np.asarray(receipt["phi"])
            threshold = min(
                float(sign * (vector @ initial)), 0.0 if receipt["y"] else float(np.log(9))
            )
            if (
                constraint["normal"] != (sign * vector).tolist()
                or abs(constraint["rhs"] - threshold) > 1e-14
            ):
                return False
        design = np.asarray([r["release_receipt"]["phi"] for r in constraints])
        labels = np.asarray([r["release_receipt"]["y"] for r in constraints])
        gradient = design.T @ (1 / (1 + np.exp(-(design @ initial))) - labels) / len(labels)
        gradient[-1] = 0
        if not np.allclose(gradient, system["gradient"], atol=1e-14, rtol=0) or not np.allclose(
            initial - 3 * gradient, proposal, atol=1e-14, rtol=0
        ):
            return False
        matrix = np.asarray([r["normal"] for r in constraints])
        rhs = np.asarray([r["rhs"] for r in constraints])
        low, high = initial - 0.5, initial + 0.5
        low[-1] = high[-1] = 1
        point = np.clip(proposal, low, high)
        rng = np.random.default_rng(system["seed"])
        projected, reference = system["projection"], system["qp"]
        for step in projected["projection_rows"]:
            if not np.allclose(step["before"], point, atol=1e-13, rtol=0):
                return False
            point = np.asarray(step["before"])
            sample = rng.choice(len(constraints), min(8, len(constraints)), replace=False).tolist()
            chosen = min(
                sample,
                key=lambda i: (-float(rhs[i] - matrix[i] @ point), constraints[i]["source_id"]),
            )
            violation = max(0.0, float(rhs[chosen] - matrix[chosen] @ point))
            if (
                step["sample"] != sample
                or step["chosen"] != chosen
                or not np.allclose(step["before"], point, atol=1e-13, rtol=0)
                or abs(step["violation"] - violation) > 1e-13
            ):
                return False
            if violation > 0:
                point = np.clip(
                    point + violation * matrix[chosen] / np.dot(matrix[chosen], matrix[chosen]),
                    low,
                    high,
                )
            if not np.allclose(point, step["after"], atol=1e-14, rtol=0):
                return False
            point = np.asarray(step["after"])
        candidate_ok = bool(np.max(rhs - matrix @ point) <= 1e-8)
        if projected["candidate_feasible"] != candidate_ok or projected["projection_steps"] > 256:
            return False
        if not candidate_ok:
            point = np.asarray(
                system["incumbent"] if projected["fallback"] == "incumbent" else initial
            )
        maximum = float(
            max(0, np.max(rhs - matrix @ point), np.max(low - point), np.max(point - high))
        )
        qpoint = np.asarray(reference["point"])
        faces = np.vstack((matrix, np.eye(len(initial)), -np.eye(len(initial))))
        bounds = np.concatenate((rhs, low, -high))
        active = np.flatnonzero((faces @ qpoint - bounds) <= 1e-7)
        multipliers = np.asarray(reference["dual_multipliers"])
        if (
            active.tolist() != reference["active_face_indices"]
            or np.any(multipliers < 0)
            or np.linalg.norm(faces[active].T @ multipliers - (qpoint - proposal)) > 1e-6
        ):
            return False
        if (
            not reference["reference_certified"]
            or np.max(rhs - matrix @ qpoint) > 1e-8
            or np.max(low - qpoint) > 1e-8
            or np.max(qpoint - high) > 1e-8
        ):
            return False
        distance, qdistance = (
            float(np.linalg.norm(point - proposal)),
            float(np.linalg.norm(qpoint - proposal)),
        )
        if (
            not np.allclose(point, projected["point"], atol=1e-14, rtol=0)
            or abs(row["max_residual"] - maximum) > 1e-14
            or abs(row["distance"] - distance) > 1e-14
            or abs(row["qp_distance"] - qdistance) > 1e-14
            or abs(row["distance_gap"] - (distance - qdistance)) > 1e-14
            or distance - qdistance < -1e-6
            or maximum > 1e-8
        ):
            return False
    return True


def build(work: Json, raw: Path, receipts: list[Json], coverage: Json, *, fixture: bool) -> Json:
    """Readiness qualifies owned finite controls and never independent learning."""
    evidence = work["evidence"]
    complete_coverage = all(
        p in coverage and coverage[p]["num_statements"] > 0 and coverage[p]["missing_lines"] == 0
        for p in OWNED
    )
    checks = (
        bool(receipts)
        and all(r["passed"] for r in receipts if r.get("classification") != "diagnostic")
        and complete_coverage
    )
    geometry = bool(evidence) and reduce(evidence)
    gates = deepcopy(work["failures"])
    for receipt in receipts:
        if not receipt["passed"] and receipt.get("classification") != "diagnostic":
            gates.append(
                dict(
                    check=receipt["name"],
                    upstream=TASK,
                    path=receipt["log_path"],
                    hash=receipt["log_sha256"],
                    field="passed",
                    op="==",
                    expected=True,
                    observed=False,
                )
            )
    if not fixture and not complete_coverage:
        gates.append(
            dict(
                check="owned_statement_coverage",
                upstream=TASK,
                path=str(raw / "coverage.json"),
                hash=None,
                field="missing_lines",
                op="==",
                expected=0,
                observed=coverage,
            )
        )
    if evidence and not geometry:
        gates.append(
            dict(
                check="independent_reduction",
                upstream=TASK,
                path=str(raw / "evidence.json"),
                hash=None,
                field="valid",
                op="==",
                expected=True,
                observed=False,
            )
        )
    kind = (
        "disqualified"
        if work.get("owned_failure")
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if geometry and (checks or fixture)
        else "disqualified"
    )
    systems = evidence.get("systems", [])
    rows = evidence.get("rows", [])
    result: Json = dict(
        schema="carnot.v699.constraint_projection.v1",
        experiment_id=8075,
        task_id=TASK,
        milestone="2026.10.699",
        run_date="20261003",
        honest_verdict=(
            f"complete_blocked_{work['failures'][0]['upstream']}"
            if kind == "blocked"
            else f"complete_{kind}_constraint_projection_kernel"
        ),
        verdict_class=kind,
        projection_kernel_ready_score=int(kind == "circular_positive" and checks),
        generalized_learning_benefit_score=0,
        verifier_is_oracle=True,
        claim_scope="finite supplied-label numerical fixtures only; no unseen safety, learning benefit or nearest-point SKM claim",
        methodology_note="64 seeded private feasible systems; independent SLSQP convex QP; finite full-set residual and fallback controls; no model operations",
        flagged_adversarial=False,
        required_checks_passed=checks,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="aggregation_from_upstream_artifacts",
        substrate_declaration="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[
            dict(
                kind="private frozen linear coefficient heads",
                dimension=4,
                calibration_frozen=True,
                deployed=False,
            )
        ],
        rows=rows,
        intended_count=64,
        eligible_count=len(rows),
        independent_count=0,
        completed_count=len(rows),
        censored_count=0,
        excluded_count=64 - len(rows),
        failed_count=sum(not r["feasible"] for r in rows),
        sample_size_budget=dict(
            seeded_systems=64, independent_natural_sources=0, controls_are_circular=True
        ),
        gate_check_summary=gates,
        random_seed=SEED,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        duration_s=work["duration_s"],
        coverage_statement_counts=coverage,
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
        memory_capacity=64,
        constraint_bytes=sum(len(json.dumps(s["constraints"]).encode()) for s in systems),
        projection_steps=sum(s["projection"]["projection_steps"] for s in systems),
        constraint_rows=[dict(seed=s["seed"], **r) for s in systems for r in s["constraints"]],
        projection_rows=[
            dict(seed=s["seed"], step=i, **r)
            for s in systems
            for i, r in enumerate(s["projection"]["projection_rows"])
        ],
        residual_rows=[
            dict(
                seed=s["seed"],
                candidate_feasible=s["projection"]["candidate_feasible"],
                candidate=s["projection"]["candidate_residuals"],
                final=s["projection"]["residuals"],
            )
            for s in systems
        ],
        qp_reference_rows=[dict(seed=s["seed"], **s["qp"]) for s in systems],
        fallback_rows=[
            dict(
                seed=s["seed"],
                fallback=s["projection"]["fallback"],
                termination=s["projection"]["termination"],
                cost_ns=s["projection"]["cost"]["fallback_ns"],
            )
            for s in systems
        ],
        release_order_controls=evidence.get("controls", []),
        projection_cost_rows=[
            dict(
                seed=s["seed"],
                constraint_construction_storage_ns=s["constraint_construction_storage_ns"],
                gradient_ns=s["gradient_ns"],
                **s["projection"]["cost"],
            )
            for s in systems
        ],
        hardware_operation_counts=dict(
            cpu_projection_fixture_calls=len(systems),
            cpu_projection_control_attempts=5 * int(bool(evidence)),
            cpu_qp_reference_calls=len(systems) + int(bool(evidence)),
            row_dot_products=sum(s["projection"]["cost"]["row_dot_products"] for s in systems),
            rust_calls=0,
            gpu_calls=0,
        ),
        config=CONFIG,
        environment=work.get("environment", {}),
    )
    result["field_principles"] = {
        k: f"Preserve {k} as an exact owned operand so fixture evidence cannot silently become scientific benefit."
        for k in result
    }
    result["field_principles"].update(
        honest_verdict="Completed negative findings are terminal and must not trigger accidental retries.",
        verdict_class="Blocked external prerequisites differ from unfinished owned work and failed owned checks.",
        projection_kernel_ready_score="All owned checks qualify the numerical component only.",
        generalized_learning_benefit_score="Reused development sources and private fixtures grant no generalized learning credit.",
        independent_count="Seed repetitions create no independent natural source groups.",
        constraint_rows="Only original released update labels may define inequalities; known-label memory cannot imply unseen safety.",
        projection_rows="Persist sampled rows and arithmetic so a finite iteration budget cannot masquerade as convergence.",
        residual_rows="Full inequalities and box faces must pass after correction and after fallback.",
        qp_reference_rows="Feasibility, optimizer status and convex optimality certificate are distinct from SKM distance.",
        fallback_rows="An unfinished candidate remains unfinished even when a validated incumbent is restored; charge reset work.",
        release_order_controls="Arrival order cannot displace newer original release slots or apply duplicate labels twice.",
        constraint_bytes="Serialized active fixture constraints include label receipts; this total spans separate fixtures, not one memory occupancy.",
        projection_cost_rows="Charge gradient, durable constraint construction, projection and fallback rather than claiming free correction.",
        hardware_operation_counts="Count executed CPU work; no Rust or GPU execution follows from a future implementation path.",
        memory_capacity="The active set is bounded at 64; the retained audit journal is not an additional active constraint set.",
        validation_receipts="Current argv, exit, elapsed time and immutable logs authenticate owned checks.",
        repository_health="Pre-existing global health failures are diagnostics, never evidence that owned checks passed.",
        code_config_hashes="Code changes invalidate replay rather than silently borrowing old qualification.",
        environment="Record actual Python and numerical libraries so CPU execution remains attributable.",
        trained_head_specs="Private numerical coefficients are not a loaded LLM or a changed shipped learner.",
    )
    return result


def replay(path: Path) -> bool:
    """Reconstruct reductions and authenticate snapshots without repeating work."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        if any(sha256_file(ROOT / p) != h for p, h in value["code_config_hashes"].items()):
            return False
        for receipt in value["validation_receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        return bool(
            build(
                work,
                raw,
                validation["receipts"],
                validation["coverage"],
                fixture=validation["fixture"],
            )
            == value
        )
    except (ValueError, OSError, KeyError, TypeError):
        return False


def manifest(private: Path) -> list[Json]:
    """Freeze bounded owned checks before numerical observations are measured."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / p) for p in ("python", "coverage", "pytest", "ruff", "mypy")
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
    commands = [
        (
            "focused_unit_and_cli",
            [
                cov,
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *common,
                str(ROOT / TEST),
                "--basetemp=" + str(private / "focused"),
            ],
            300,
        ),
        (
            "consumer_and_e2e_015",
            [
                pytest,
                *common,
                str(ROOT / "tests/python/test_source_boundary_7852.py"),
                str(ROOT / "tests/python/test_primary_publication_7928.py"),
                "--basetemp=" + str(private / "consumers"),
            ],
            180,
        ),
        (
            "coverage_combine",
            [
                cov,
                "combine",
                "--rcfile=" + str(config),
                "--data-file=" + str(private / ".coverage"),
            ],
            60,
        ),
        (
            "coverage_report",
            [
                cov,
                "report",
                "--rcfile=" + str(config),
                "--data-file=" + str(private / ".coverage"),
                "--fail-under=100",
                "--show-missing",
            ],
            60,
        ),
        (
            "coverage_json",
            [
                cov,
                "json",
                "--rcfile=" + str(config),
                "--data-file=" + str(private / ".coverage"),
                "-o",
                str(private / "coverage.json"),
            ],
            60,
        ),
        ("ruff_check", [ruff, "check", *paths], 60),
        ("ruff_format", [ruff, "format", "--check", *paths], 60),
        (
            "mypy_strict",
            [
                mypy,
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                "--ignore-missing-imports",
                *[str(ROOT / p) for p in OWNED],
            ],
            180,
        ),
        (
            "scoped_spec_coverage",
            [py, str(ROOT / "scripts/check_spec_coverage.py"), str(ROOT / TEST)],
            60,
        ),
        (
            "e2e_016_fixture",
            [
                py,
                str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                "--date",
                "20260929",
                "--fixture-e2e",
                str(private / "e2e016.json"),
            ],
            120,
        ),
        (
            "e2e_016_replay",
            [
                py,
                str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                "--date",
                "20260929",
                "--cold-replay",
                str(private / "e2e016.json"),
            ],
            60,
        ),
        (
            "repository_full_suite",
            [pytest, "tests/python", "-q", "--basetemp=" + str(private / "full")],
            900,
        ),
    ]
    return [
        dict(
            name=n,
            argv=a,
            deadline_s=d,
            expected_exit=0,
            classification="diagnostic" if n == "repository_full_suite" else "required",
        )
        for n, a, d in commands
    ]


def terminal(path: Path) -> Json:
    """Cold replay and both real auditors must pass on exact candidate bytes."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    py = str(ROOT / ".venv/bin/python")
    checks = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="carnot-8075-terminal-") as tmp:
        receipts: list[Json] = []
        for name, argv in checks:
            progress("subprocess_before_" + name, len(receipts), len(checks) - len(receipts))
            receipts.append(
                run_check(
                    Path(tmp),
                    dict(name=name, argv=argv, deadline_s=60, expected_exit=0),
                    Path(tmp),
                    raw / "terminal_logs",
                )
            )
            progress("subprocess_after_" + name, len(receipts), len(checks) - len(receipts))
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def worker(root: Path, raw: Path, *, fixture: bool, mutate: bool) -> Json:
    """Measure in a child whose normal exit precedes terminal publication."""
    plan = prerequisites(root, raw)
    evidence = measure(raw, fixture=fixture) if not plan["failures"] else {}
    if mutate and evidence:
        evidence["rows"][0]["max_residual"] = 1
        atomic_json(raw / "evidence.json", evidence)
    plan.update(
        evidence=evidence,
        code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
        phase_spans=evidence.get("phase_spans", []),
        duration_s=time.monotonic() - START,
        raw_shard_hashes=[
            dict(path=str(p), sha256=sha256_file(p))
            for p in sorted(raw.rglob("*.json"))
            if "inputs" not in p.parts and p.name != "validation_commands.json"
        ],
    )
    atomic_json(raw / "work.json", plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    """Preserve existing evidence and publish only current authenticated results."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
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
        worker(
            args.root,
            args.worker_output.parent,
            fixture=bool(args.fixture_output),
            mutate=args.mutate,
        )
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if output.exists() or (raw / "work.json").exists():
        progress("existing_terminal_evidence_preserved")
        return 1
    with tempfile.TemporaryDirectory(prefix="carnot-8075-") as temp:
        private = Path(temp)
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
        if args.fixture_output:
            cmd += ["--fixture-output", str(output)]
        if args.mutate:
            cmd += ["--mutate"]
        coverage_config = os.environ.get("CARNOT_8075_COVERAGE_CONFIG")
        if coverage_config:
            cmd = [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + coverage_config,
                "--data-file=" + str(Path(coverage_config).parent / ".coverage"),
                "--parallel-mode",
                *cmd[2:],
            ]
        child = dict(name="measurement_normal_exit", argv=cmd, deadline_s=300, expected_exit=0)
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[child, *specs],
                config=CONFIG,
                terminal_checks=["cold_replay", "adversarial", "strict_rows"],
                code_hashes={p: sha256_file(ROOT / p) for p in OWNED},
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
                    raw_shard_hashes=[],
                    code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
                    phase_spans=[],
                    duration_s=time.monotonic() - START,
                ),
            )
        work = json.loads((raw / "work.json").read_text())
        receipts, coverage = [receipt], {}
        if not args.fixture_output:
            os.environ["CARNOT_8075_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            for spec in specs:
                progress(
                    "subprocess_before_" + spec["name"],
                    len(receipts) - 1,
                    len(specs) - len(receipts) + 1,
                )
                receipts.append(run_check(ROOT, spec, private, raw / "validation_logs"))
                progress(
                    "subprocess_after_" + spec["name"],
                    len(receipts) - 1,
                    len(specs) - len(receipts) + 1,
                )
            if (private / "coverage.json").is_file():
                data = json.loads((private / "coverage.json").read_text())["files"]
                coverage = {
                    str(Path(p).resolve().relative_to(ROOT)): v["summary"] for p, v in data.items()
                }
                atomic_json(raw / "coverage.json", dict(files=data))
        atomic_json(
            raw / "validation.json",
            dict(receipts=receipts, coverage=coverage, fixture=bool(args.fixture_output)),
        )
        value = build(work, raw, receipts, coverage, fixture=bool(args.fixture_output))
        publication = publish_primary(output, value, terminal)
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, measurement_exit_receipt=receipt),
        )
        progress("terminal_published", len(value["rows"]), 0)
    return 0
