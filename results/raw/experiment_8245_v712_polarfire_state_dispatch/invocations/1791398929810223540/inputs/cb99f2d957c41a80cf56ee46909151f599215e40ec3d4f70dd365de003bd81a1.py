"""REQ-VERIFY-8180: qualify small convex heads without changing the generator.

The existing release clock controls label access. Calibration can reduce a
saturated offset, but fixture success supplies no independent evidence of benefit.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit
import yaml

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import admission_horizon_methods_8152 as schedule

Json = dict[str, Any]
Array = NDArray[np.float64]
engine = schedule.engine
ROOT = schedule.ROOT
NAME = "experiment_8180_v707_calibrated_memory_methods"
TASK = "exp8180-calibrated-memory-methods"
MODULE = "python/carnot/verify/calibrated_memory_methods_8180.py"
RUNNER = "python/carnot/reporting/calibrated_memory_execution_8180.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_calibrated_memory_methods_8180.py"
PROTOCOL = "openspec/change-proposals/v707-calibrated-memory-protocol.json"
PROTOCOL_HASH = "sha256:b6b1f522d511b19e2106a31c14d3acd2b992988b8042dd161800884ad1896202"
UPSTREAM = "results/experiment_8172_v706_learning_benefit_audit.json"
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
OLD_ARMS = list(engine.ARMS)
ARMS = OLD_ARMS + ["calibration_only"]
NAMES = dict(
    error_center="calibrated_error_center",
    fixed_public_center="calibrated_fixed_center",
    random_past_center="calibrated_random_center",
    calibration_only="calibration_only",
    frozen_qwen_offset="frozen_qwen_offset",
)
BASE_GENESIS, BASE_INTERPOLATE, BASE_PROPOSE = engine.genesis, engine.interpolate, schedule.propose


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so supervisors can observe the finite CPU workload."""
    print(f"[exp8180] phase={phase} completed={completed} pending={pending}", flush=True)


def operands(head: Json, geometry: Json, pool: list[Json]) -> tuple[Array, Array]:
    """Keep Gaussian centers fixed, making the parameter fit a convex problem."""
    phi = engine.historical.radial.design(
        dict(centers=head["centers"], geometry=geometry), [r["values"] for r in pool]
    )
    offsets = logit(np.clip(expit([r["values"][0] for r in pool]), 1e-6, 1 - 1e-6))
    return np.column_stack((offsets, phi)), np.asarray([r["y"] for r in pool], dtype=float)


def objective(theta: Array, x: Array, y: Array, *, reference: bool = False) -> tuple[float, Array]:
    """Independent scalar sums check logistic loss and its selective penalties."""
    penalty = np.array([theta[0] - 1, 0, *theta[2:]])
    if reference:
        scores = [sum(float(v) * float(t) for v, t in zip(row, theta, strict=True)) for row in x]
        errors = [float(expit(s)) - float(t) for s, t in zip(scores, y, strict=True)]
        loss = sum(
            max(s, 0) + math.log1p(math.exp(-abs(s))) - float(t) * s
            for s, t in zip(scores, y, strict=True)
        ) / len(y)
        gradient = np.asarray(
            [
                sum(float(x[i, j]) * errors[i] for i in range(len(y))) / len(y)
                for j in range(len(theta))
            ]
        )
    else:
        scores_array = x @ theta
        loss = float(np.mean(np.logaddexp(0, scores_array) - y * scores_array))
        gradient = x.T @ (expit(scores_array) - y) / len(y)
    return float(loss + 0.005 * (penalty @ penalty)), np.asarray(gradient + 0.01 * penalty)


def train(head: Json, geometry: Json, pool: list[Json], *, maxiter: int = 200) -> Json:
    """Reject an unfinished solve even when its last iterate looks useful."""
    pool = pool[-64:]
    if not pool:
        raise ValueError("optimizer_operands")
    x, y = operands(head, geometry, pool)
    theta = np.array([head["scale"], head["intercept"], *head["weights"]])
    low, high = (
        np.array([0, -8, *([-4] * len(head["weights"]))]),
        np.array([2, 8, *([4] * len(head["weights"]))]),
    )
    result = minimize(
        objective,
        theta,
        args=(x, y),
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(low, high, strict=True)),
        options=dict(maxiter=maxiter, gtol=1e-6, ftol=0, maxls=50),
    )
    loss, gradient = objective(result.x, x, y)
    scalar_loss, scalar_gradient = objective(result.x, x, y, reference=True)
    projected = float(np.max(np.abs(result.x - np.clip(result.x - gradient, low, high))))
    agreement = max(abs(loss - scalar_loss), float(np.max(np.abs(gradient - scalar_gradient))))
    h = deepcopy(head)
    h.update(
        scale=float(result.x[0]),
        intercept=float(result.x[1]),
        weights=result.x[2:].tolist(),
        optimizer_step=head["optimizer_step"] + int(result.nit),
    )
    h["fit"] = dict(
        objective=loss,
        initial_objective=objective(theta, x, y)[0],
        objective_agreement=agreement,
        projected_gradient=projected,
        converged=bool(result.success and projected <= 1e-6 and agreement <= 1e-8),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        sample_count=len(pool),
        sample_ids=[r["source_cluster_id"] for r in pool],
    )
    return h


def probability(head: Json, geometry: Json, values: list[float]) -> float:
    """Recover the clipped generator logit before applying learned calibration."""
    phi = engine.historical.radial.design(
        dict(centers=head["centers"], geometry=geometry), [values]
    )[0]
    offset = float(logit(np.clip(expit(values[0]), 1e-6, 1 - 1e-6)))
    return float(
        expit(head["scale"] * offset + phi @ np.array([head["intercept"], *head["weights"]]))
    )


def scalar_probability(head: Json, geometry: Json, values: list[float]) -> float:
    """Scalar distances catch a wrong feature column or calibration parameter."""
    z = [(x - m) / s for x, m, s in zip(values, geometry["mean"], geometry["std"], strict=True)]
    phi = [
        math.exp(
            -sum((x - v) ** 2 for x, v in zip(z, c["x"], strict=True))
            / (2 * geometry["sigma"] ** 2)
        )
        for c in head["centers"]
    ]
    p = min(1 - 1e-6, max(1e-6, float(expit(values[0]))))
    score = (
        head["scale"] * math.log(p / (1 - p))
        + head["intercept"]
        + sum(w * f for w, f in zip(head["weights"], phi, strict=True))
    )
    return float(expit(score))


def genesis(rows: list[Json], seed: int) -> Json:
    """Every arm starts at the historical generator with zero residual."""
    s = BASE_GENESIS(rows, seed)
    for h in s["arms"].values():
        h["scale"] = 1.0
    return s


def interpolate(candidate: Json, step: float) -> Json:
    """Shrink scale, intercept and weights together without another label fit."""
    h = BASE_INTERPOLATE(candidate, step)
    h["scale"] = candidate["base"]["scale"] + step * (h["scale"] - candidate["base"]["scale"])
    return h


def propose(state: Json, slot: int, opportunity: int) -> None:
    """Reuse qualified center choices, then exclude every unfinished proposal."""
    with patch.object(engine, "ARMS", OLD_ARMS):
        BASE_PROPOSE(state, slot, opportunity)
    if not state["candidates"]:
        return
    h = state["arms"]["calibration_only"]
    state["candidates"]["calibration_only"] = dict(
        head=train(h, state["geometry"], state["pool"][-64:]),
        base=deepcopy(h),
        labels=[],
        commit=slot,
    )
    rejected = [a for a, c in state["candidates"].items() if not c["head"]["fit"]["converged"]]
    state["candidates"] = {a: c for a, c in state["candidates"].items() if a not in rejected}
    engine.event(
        state,
        "qualified_candidates",
        slot,
        rejected=rejected,
        candidates=deepcopy(state["candidates"]),
    )


def run(rows: list[Json], labels: list[int | None], seed: int, **kwargs: Any) -> Json:
    """Scoped adapters preserve the historical event clock and admission safety."""
    with ExitStack() as stack:
        for owner, name, value in [
            (engine, "ARMS", ARMS),
            (engine, "genesis", genesis),
            (engine, "train", train),
            (engine, "probability", probability),
            (engine, "interpolate", interpolate),
            (schedule, "propose", propose),
        ]:
            stack.enter_context(patch.object(owner, name, value))
        return schedule.run(rows, labels, seed, **kwargs)


def fixture(case: str) -> tuple[list[Json], list[int | None]]:
    """Known targets test decision movement with a deliberately saturated offset."""
    rows, labels = schedule.fixture("positive")
    for r, y in zip(rows, labels, strict=True):
        r["values"] = [
            float(logit(1 - 1e-6)),
            float(y) if case == "learnable" else 0.0,
            *([0.0] * 7),
        ]
    return rows, labels


def stable(value: Any) -> Any:
    """Compare restart arithmetic without pretending wall-clock timings repeat."""
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k != "duration_s"}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def fixture_summary(state: Json, rows: list[Json]) -> Json:
    """Only predictions strictly after an installed state count as future use."""
    installs = [
        v["slot"]
        for v in state["events"]
        if v["kind"] == "admit_once" and v["steps"].get("error_center", 0)
    ]
    first = min(installs, default=257)
    changed = sum(
        p["slot"] > first
        and engine.historical.radial.action(p["predictions"]["error_center"])
        != engine.historical.radial.action(p["predictions"]["frozen_qwen_offset"])
        for p in state["issued"]
    )
    return dict(
        seed=state["seed"],
        install_slot=first,
        changed_later_decisions=changed,
        passed=first <= 208 and changed >= 32,
        benefit_claim=False,
        original_sources=len(rows),
        lost_feedback=len(state["lost"]),
    )


def seed_child(inputs: Path, output: Path, resume: Path | None, crash: int) -> None:
    """A real hard exit after durable issue tests recovery before target release."""
    data = json.loads(inputs.read_text())
    state = json.loads(resume.read_text()) if resume else None

    def seal(kind: str, current: Json) -> None:
        if current["cursor"] == crash:
            atomic_json(output / "crash.json", current)
            progress("intentional_hard_exit", current["cursor"], len(current["pending"]))
            # Flush coverage before os._exit because Python cleanup does not run.
            import coverage

            active = coverage.Coverage.current()
            if active:
                active.save()
            sys.stdout.flush()
            os._exit(73)

    final = run(data["rows"], data["labels"], data["seed"], state=state, seal=seal)
    atomic_json(output / "final.json", final)


def restart_specs(raw: Path, private: Path | None = None) -> list[Json]:
    """Freeze direct child argv before any private fixture fit is measured."""
    cli = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / CLI),
        "--seed-input",
        str((private or raw) / "restart-input.json"),
        "--seed-output",
        str(raw / "restart"),
    ]
    if os.environ.get("COVERAGE_RCFILE"):
        config = os.environ["COVERAGE_RCFILE"]
        cli[3:3] = [
            "COVERAGE_PROCESS_START=" + config,
            "COVERAGE_FILE=" + str(Path(config).parent / ".coverage"),
        ]
    return [
        dict(
            name="hard_restart_crash",
            argv=cli + ["--crash-slot", "90"],
            expected_exit=73,
            deadline_s=120,
            classification="required",
        ),
        dict(
            name="hard_restart_resume",
            argv=cli + ["--resume-state", str(raw / "restart/crash.json")],
            expected_exit=0,
            deadline_s=120,
            classification="required",
        ),
    ]


def qualify(raw: Path, private: Path) -> Json:
    """Twenty repeated seeds qualify mechanics while adding no independent units."""
    records, summaries, fits, rows = [], [], [], []
    for case in ["learnable", "no_signal"]:
        public, labels = fixture(case)
        for seed in range(101, 121):
            progress("before_fixture_benchmark_" + case, seed - 101, 121 - seed)
            s = run(public, labels, seed)
            records.append(dict(case=case, seed=seed, state=engine.shard(raw, s)))
            summaries.append(dict(case=case, **fixture_summary(s, public)))
            for event in s["events"]:
                if event["kind"] == "qualified_candidates":
                    fits.extend(
                        dict(
                            case=case,
                            seed=seed,
                            slot=event["slot"],
                            arm=NAMES[a],
                            **c["head"]["fit"],
                        )
                        for a, c in event["candidates"].items()
                    )
            for r, prediction, y in zip(public[64:], s["issued"][64:], labels[64:], strict=True):
                scored = engine.historical.scored(
                    r,
                    prediction,
                    dict(y=y, exclusion_reason=None),
                    "private_" + case + "_fixture",
                    seed,
                )
                rows.extend(dict(t, arm=NAMES[t["arm"]]) for t in scored)
            progress("after_fixture_benchmark_" + case, seed - 100, 120 - seed)
    public, labels = fixture("learnable")
    overflow = run(public, labels, 101, capacity=8)
    atomic_json(private / "restart-input.json", dict(rows=public, labels=labels, seed=101))
    receipts = []
    for spec in restart_specs(raw, private):
        progress("before_subprocess_" + spec["name"], len(receipts), 2 - len(receipts))
        receipts.append(run_check(ROOT, spec, private, raw / "restart_logs", heartbeat_s=30))
        progress("after_subprocess_" + spec["name"], len(receipts), 2 - len(receipts))
    restarted = json.loads((raw / "restart/final.json").read_text())
    baseline = json.loads(Path(records[0]["state"]["path"]).read_text())
    return dict(
        rows=rows,
        fixture_states=records,
        fixture_summaries=summaries,
        optimizer_fixture_rows=fits,
        future_decision_fixture_score=int(
            all(r["passed"] for r in summaries if r["case"] == "learnable")
        ),
        overflow_fixture=dict(
            lost_count=len(overflow["lost"]), passed=bool(overflow["lost"]), lossless_claim=False
        ),
        restart_fixture=dict(
            passed=all(r["passed"] for r in receipts) and stable(restarted) == stable(baseline),
            receipts=receipts,
            resumed_state=engine.shard(raw, restarted),
        ),
    )


def diagnose(upstream: Json, public: list[Json]) -> Json:
    """Separate actual clipping and small residuals from an unfinished old fit.

    Positive gradients show that four SGD steps did not solve the old objective.
    They do not prove that a converged fit or different features improve decisions.
    """
    thresholds = [math.log(0.1 / 0.9), 0.0]
    saturation, optimizer = [], []
    states = {
        r["seed"]: json.loads(Path(r["state"]["path"]).read_text())
        for r in upstream["state_manifest"]
    }
    for row in upstream["rows"]:
        if (
            row["condition"] != "later_stream"
            or row["metric"] != "brier"
            or row["status"] != "completed"
        ):
            continue
        original = public[row["slot"] - 1]
        base = original["values"][0]
        predicted = float(logit(row["prediction"]))
        margin = min(abs(base - t) for t in thresholds)
        saturation.append(
            dict(
                unit_id=row["unit_id"],
                source_cluster_id=row["source_cluster_id"],
                slot=row["slot"],
                seed=row["seed"],
                arm=row["arm"],
                base_logit=base,
                calibrated_logit=predicted,
                residual=predicted - base,
                residual_magnitude=abs(predicted - base),
                decision_threshold_logits=thresholds,
                decision_margin=margin,
                probability_clipped=original.get("probability_clipped", False),
                action=engine.historical.radial.action(row["prediction"]),
            )
        )
    for seed, s in states.items():
        for event in s["events"]:
            if event["kind"] != "commit_candidate" or not event.get("candidates"):
                continue
            pool = [r for r in s["pool"] if r["slot"] + 20 <= event["slot"]][-64:]
            for arm, candidate in event["candidates"].items():
                head = candidate["head"]
                phi = engine.historical.radial.design(
                    dict(centers=head["centers"], geometry=s["geometry"]),
                    [r["values"] for r in pool],
                )
                theta = np.array([head["intercept"], *head["weights"]])
                scores = np.array([r["values"][0] for r in pool]) + phi @ theta
                targets = np.array([r["y"] for r in pool])
                grad = phi.T @ (expit(scores) - targets) / len(pool)
                grad[1:] += 0.01 * theta[1:]
                low = np.array([-np.inf, *([-4] * len(head["weights"]))])
                high = -low
                pg = float(np.max(np.abs(theta - np.clip(theta - grad, low, high))))
                optimizer.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        slot=event["slot"],
                        sample_count=len(pool),
                        objective=float(
                            np.mean(np.logaddexp(0, scores) - targets * scores)
                            + 0.005 * np.sum(theta[1:] ** 2)
                        ),
                        projected_gradient=pg,
                        converged=pg <= 1e-6,
                        old_steps=4,
                    )
                )
    selected = [r for r in saturation if r["seed"] == 101 and r["arm"] == "error_center"]
    return dict(
        saturation_rows=saturation,
        historical_optimizer_rows=optimizer,
        diagnosis=dict(
            completed_sources=len(selected),
            clipped_sources=sum(r["probability_clipped"] for r in selected),
            minimum_decision_margin=min((r["decision_margin"] for r in selected), default=None),
            maximum_residual=max((r["residual_magnitude"] for r in selected), default=None),
            residuals_crossing_nearest_threshold=sum(
                r["residual_magnitude"] >= r["decision_margin"] for r in selected
            ),
            unconverged_old_candidates=sum(not r["converged"] for r in optimizer),
            representational_signal="unresolved; probability changes and short SGD do not distinguish absent signal from optimization limits",
            preserved_null_verdict=upstream["honest_verdict"],
        ),
    )


def authenticate(root: Path, b: Any, fixture_mode: bool) -> Json:
    """A specific checked upstream receipt and exact primitive bytes gate reuse."""
    b.upstream = "exp8172-learning-benefit-audit"
    for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
        path = ROOT / ".venv/bin" / name
        b.require(path, "runtime_" + name, True, path.is_file())
    path = root / UPSTREAM
    value = b.read(
        path,
        None
        if fixture_mode
        else "sha256:5cb110c748960181176793a2d46b18702a53f2e5e49bfed113b3c46c28c29028",
    )
    for field, expected in [
        ("experiment_id", 8172),
        ("learning_audit_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        b.require(path, field, expected, value.get(field))
    if not fixture_mode:
        terminal = engine.methods.historical.terminal(path, value, b)
        b.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    exclusion = ROOT / "ops/exclusion_manifest.yaml"
    b.bind(exclusion)
    retired = yaml.safe_load(exclusion.read_text()).get("retired_experiments", [])
    b.require(
        exclusion,
        "experiment8180_not_retired",
        True,
        not any(r.get("experiment_id") == 8180 for r in retired),
    )
    b.bind(ROOT / PROTOCOL, PROTOCOL_HASH)
    b.bind(ROOT / schedule.PROTOCOL, schedule.PROTOCOL_HASH)
    for name, digest in value["code_config_hashes"].items():
        b.bind(ROOT / name, digest)
    for ref in value["raw_shard_hashes"] + [r["state"] for r in value["state_manifest"]]:
        b.bind(Path(ref["path"]), ref["sha256"])
    for role in ["stream", "retention"]:
        for ref in [
            value["input_manifests"][role + "_feature_manifest"],
            value["input_manifests"]["evaluator_label_manifests"][role],
        ]:
            b.bind(Path(ref["path"]), ref["sha256"])
    return dict(value)


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate historical inputs before fitting only private known targets."""
    began = time.monotonic()
    progress("preconditions_start")
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(mkdtemp(prefix="carnot-8180-private-fixtures-"))
    atomic_json(raw / "restart_commands.json", dict(commands=restart_specs(raw, private)))
    b = engine.methods.Custody(raw)
    work: Json = dict(
        input_ready=0,
        fixture_mode=fixture,
        rows=[],
        saturation_rows=[],
        optimizer_fixture_rows=[],
        historical_optimizer_rows=[],
        fixture_states=[],
        fixture_summaries=[],
        future_decision_fixture_score=0,
        overflow_fixture=dict(passed=False),
        restart_fixture=dict(passed=False),
        diagnosis={},
        input_manifests={},
        historical_model_provenance={},
        cited_upstream_artifacts=[],
    )
    try:
        upstream = authenticate(root, b, fixture)
        manifests = upstream["input_manifests"]
        public = json.loads(Path(manifests["stream_feature_manifest"]["path"]).read_text())["rows"]
        retained = json.loads(Path(manifests["retention_feature_manifest"]["path"]).read_text())[
            "rows"
        ]
        engine.historical.public_rows(public, 256)
        engine.historical.public_rows(retained, 64)
        work.update(
            input_ready=1,
            input_manifests=manifests,
            historical_model_provenance=upstream.get("historical_model_provenance", {}),
            cited_upstream_artifacts=[
                dict(
                    experiment_id=8172,
                    fields_imported=[
                        "rows",
                        "state_manifest",
                        "input_manifests",
                        "historical_model_provenance",
                    ],
                    sha256=sha256_file(root / UPSTREAM),
                )
            ],
        )
        progress("before_diagnosis_benchmark", 0, len(upstream["rows"]))
        work.update(diagnose(upstream, public))
        progress("after_diagnosis_benchmark", len(work["saturation_rows"]))
        work.update(qualify(raw, private))
    except engine.methods.historical.InputFailure:
        progress("external_operand_blocked", len(b.checks), 0)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    (raw / "protocol.json").write_bytes((ROOT / PROTOCOL).read_bytes())
    atomic_json(raw / "primitive_evidence.json", work)
    work.update(
        gate_check_summary=b.checks,
        preconditions_checked=dict(
            runtime_executable=str(ROOT / ".venv/bin/python"),
            model_loads=0,
            natural_fitting=False,
            checks=len(b.checks),
        ),
        protocol_path=str(raw / "protocol.json"),
        protocol_sha256=sha256_file(raw / "protocol.json"),
        method_map=protocol["method_map"],
        acceptance_gates=protocol["statistical_plan"],
        H2=dict(
            protocol["H2_analysis"],
            status="registered_not_measured",
            support=protocol["statistical_plan"]["H2_support"],
            retention=protocol["statistical_plan"]["retention"],
        ),
        source_artifact_hashes=b.refs,
        raw_shard_hashes=[
            dict(path=str(p), sha256=sha256_file(p))
            for p in sorted(raw.rglob("*.json"))
            if "custody" not in p.parts
        ],
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                MODULE,
                RUNNER,
                CLI,
                TEST,
                PROTOCOL,
                schedule.MODULE,
                "python/carnot/verify/learning_protocol_8138.py",
                "python/carnot/verify/radial_memory_8085.py",
            ]
        },
        duration_s=time.monotonic() - began,
        phase_spans=[dict(name="qualification", duration_s=time.monotonic() - began)],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["fixture_states"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Normal owned checks qualify mechanics; they never grant natural benefit."""
    owned = bool(receipts) and all(
        r["passed"] and r.get("normal_exit", r.get("actual_exit", 0) >= 0) for r in receipts
    )
    mechanics = bool(
        len(work["fixture_states"]) == 40
        and work["future_decision_fixture_score"]
        and work["overflow_fixture"]["passed"]
        and work["restart_fixture"]["passed"]
    )
    owned = owned and (not work["input_ready"] or mechanics)
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if not work["input_ready"]
        else "circular_positive"
    )
    operand = next(
        (r["check"] for r in work["gate_check_summary"] if not r["passed"]),
        "calibrated_memory_mechanics_qualified",
    )
    units = [
        r
        for r in work["rows"]
        if r["seed"] == 101 and r["arm"] == "frozen_qwen_offset" and r["metric"] == "brier"
    ]
    count = len(units)
    value = dict(
        work,
        experiment_id=8180,
        task_id=TASK,
        milestone="2026.10.707",
        honest_verdict="complete_" + verdict + "_" + ("owned_validation" if not owned else operand),
        verdict_class=verdict,
        calibrated_memory_ready_score=int(owned and work["input_ready"] and mechanics),
        stream_input_ready_score=int(owned and work["input_ready"]),
        verifier_is_oracle=True,
        claim_scope="Private mechanics qualification and historical diagnosis; natural benefit reserved for Exp8186/8187",
        exposure_scope="private_circular_fixtures_and_exposed_historical_diagnosis",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[
            dict(
                kind="bounded_calibrated_Gaussian_residual",
                arms=[NAMES[a] for a in ARMS[1:]],
                seeds=list(range(101, 121)),
                optimizer="L-BFGS-B",
                maxiter=200,
                projected_gradient_tolerance=1e-6,
                bounds=dict(a=[0, 2], b=[-8, 8], w=[-4, 4]),
                scope="private_known_target_fixtures",
            )
        ],
        intended_count=count,
        eligible_count=count,
        independent_count=0,
        completed_count=count,
        excluded_count=0,
        censored_count=0,
        failed_count=0,
        sample_size_budget=dict(
            original_stream=256,
            retention=64,
            fixture_cases=2,
            robustness_seeds=20,
            arms=5,
            repeats_add_independent_sources=0,
            H2_support=128,
            H1_alpha=0.025,
            H2_alpha=0.025,
        ),
        run_date=RUN_DATE,
        random_seed=101,
        reductions=engine.historical.reductions(work["rows"]),
        methodology_note="No LLM loaded. Released update labels fit bounded convex calibration; future one-use labels select joint interpolation. Fixtures establish mechanics only. V706 rows and four-step gradients diagnose prior limits without natural refitting.",
    )
    value["field_principles"] = {
        k: "Bind actual primitives; fixtures and repeated seeds add zero independent benefit credit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reexecute private trajectories and reduce headlines from primitive rows.

    A fresh outer checksum cannot legitimize changed predictions, thresholds,
    convergence receipts or a fabricated readiness headline.
    """
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        for name, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                return False
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            if (
                "log_path" in receipt
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        if value["experiment_id"] != 8180 or value["protocol_sha256"] != PROTOCOL_HASH:
            return False
        primitive = next(
            r for r in value["raw_shard_hashes"] if r["path"].endswith("primitive_evidence.json")
        )
        saved = json.loads(Path(primitive["path"]).read_text())
        if any(value[k] != v for k, v in saved.items()):
            return False
        rebuilt_rows, summaries, fits = [], [], []
        for index, entry in enumerate(value["fixture_states"]):
            public, labels = fixture(entry["case"])
            s = run(public, labels, entry["seed"])
            original = json.loads(Path(entry["state"]["path"]).read_text())
            if stable(s) != stable(original):
                return False
            summaries.append(dict(case=entry["case"], **fixture_summary(s, public)))
            for event in s["events"]:
                if event["kind"] == "qualified_candidates":
                    fits.extend(
                        dict(
                            case=entry["case"],
                            seed=entry["seed"],
                            slot=event["slot"],
                            arm=NAMES[a],
                            **c["head"]["fit"],
                        )
                        for a, c in event["candidates"].items()
                    )
            for row, p, y in zip(public[64:], s["issued"][64:], labels[64:], strict=True):
                rebuilt_rows.extend(
                    dict(r, arm=NAMES[r["arm"]])
                    for r in engine.historical.scored(
                        row,
                        p,
                        dict(y=y, exclusion_reason=None),
                        "private_" + entry["case"] + "_fixture",
                        entry["seed"],
                    )
                )
            progress("cold_replay_states", index + 1, len(value["fixture_states"]) - index - 1)
        if (
            rebuilt_rows != value["rows"]
            or summaries != value["fixture_summaries"]
            or fits != value["optimizer_fixture_rows"]
        ):
            return False
        if value["input_ready"]:
            original_ref = next(
                r for r in value["source_artifact_hashes"] if r["path"].endswith(UPSTREAM)
            )
            original_upstream = json.loads(Path(original_ref["snapshot_path"]).read_text())
            public = json.loads(
                Path(value["input_manifests"]["stream_feature_manifest"]["path"]).read_text()
            )["rows"]
            diagnosis = diagnose(original_upstream, public)
            if any(value[k] != v for k, v in diagnosis.items()):
                return False
        rebuilt = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        for key in [
            "reductions",
            "completed_count",
            "intended_count",
            "calibrated_memory_ready_score",
            "stream_input_ready_score",
            "honest_verdict",
            "verdict_class",
        ]:
            if rebuilt[key] != value[key]:
                return False
        return True
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return False
