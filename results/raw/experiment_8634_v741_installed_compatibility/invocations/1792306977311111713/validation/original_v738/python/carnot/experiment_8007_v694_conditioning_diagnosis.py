"""REQ-REPORT-8007: diagnose frozen development conditioning without a model load.

The offset diagnostic tests optimizer scale on historical fitting data. It does
not select a new deployed head or turn exposed development into validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify import qwen_energy_calibration_7972 as scalar
from carnot.verify import qwen_development_capture_7995 as capture_helper

Json = dict[str, Any]
Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8007_v694_conditioning_diagnosis"
TASK = "exp8007-conditioning-diagnosis"
MODEL_SPECS: list[str] = []
OWNED = [f"python/carnot/{NAME}.py", f"scripts/experiments/{NAME}.py"]
TEST = "tests/python/test_conditioning_diagnosis_8007.py"
CONFIG = dict(
    seed=69407,
    arms=["raw_q", "historical_fixed_step", "intercept_only"],
    optimizer=dict(
        bracket=[-40, 40],
        xtol=1e-12,
        maxiter=128,
        gradient_tolerance=1e-8,
        objective="fit_BCE_plus_0.001_intercept_squared",
        tune_selection=False,
    ),
    costs=dict(unsupported_accept=5, supported_reject=1, escalate=0.5, correct=0),
    primary_outcomes=["brier", "decision_loss"],
    draws=10000,
    block_size=16,
    minimum_groups=32,
    minimum_per_class=4,
    correction="Holm",
    alpha=0.05,
    saturation_derivative_cutoff=0.001,
    primary_seed=17,
    benefit_gate="both upper_95_gain_bounds_below_zero_and_Holm_p_below_0.05",
    exposure="all roles are previously exposed development",
    groups=["raw_accept", "raw_escalate", "raw_reject"],
)
INPUTS = {
    7994: ("development_cohort", "cohort_ready_score"),
    7995: ("qwen_development_capture", "capture_ready_score"),
    7996: ("sparse_energy_training", "sparse_fit_ready_score"),
    7999: ("learning_causal_audit", "learning_audit_ready_score"),
}


def progress(phase: str, units: int = 0, pending: int = 0) -> None:
    """Actual elapsed time makes phase boundaries useful without padding work."""
    print(
        f"[exp8007] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"units={units} pending={pending}",
        flush=True,
    )


BEGAN = time.monotonic()


def policy(p: float | None) -> str:
    """Escalating exact ties avoids choosing an action by floating point ordering."""
    return (
        "accept"
        if p is not None and p < 0.1
        else ("reject" if p is not None and p > 0.5 else "escalate")
    )


def loss(action: str, y: int) -> float:
    """The cost is fixed before targets so wrong accepts stay five times as costly."""
    return float(5 * y if action == "accept" else 1 - y if action == "reject" else 0.5)


def distribution(x: Array) -> Json:
    """Quantiles show scale and tails rather than hiding saturation in a mean."""
    return dict(
        count=len(x),
        minimum=float(x.min()),
        maximum=float(x.max()),
        mean=float(x.mean()),
        quantiles=np.quantile(x, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    )


def copy_evidence(ref: Json, raw: Path) -> Json:
    """Copy exact checked bytes to a durable content address, preserving history."""
    source = checked(ref)
    target = raw / "custody" / (ref["sha256"].split(":")[-1] + source.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied = reference(target)
    if copied["sha256"] != ref["sha256"]:
        raise ValueError("copy_hash")
    return dict(copied, original_path=ref["path"])


def materialize(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Read existing primaries and freeze hashes before opening fit-only targets."""
    upstream, refs, checks, failures = {}, [], [], []
    for eid, (suffix, ready) in INPUTS.items():
        path = root / "results" / f"experiment_{eid}_v693_{suffix}.json"
        value = json.loads(path.read_text()) if path.is_file() else {}
        if value and ready not in value:
            raise ValueError("upstream_contract:" + ready)
        receipt = reader_receipt(
            value.get("task_id", f"exp{eid}-absent"), root / "results", field=ready
        )
        check = dict(
            upstream_id=f"exp{eid}",
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=ready,
            expected=1,
            observed=value.get(ready),
            passed=value.get(ready) == 1 and receipt["passed"],
        )
        checks.append(check)
        if not check["passed"]:
            failures.append(check)
        else:
            upstream[eid] = value
            refs.append(
                dict(
                    copy_evidence(reference(path), raw),
                    imported_fields=[ready, "rows", "checkpoints", "public_role_manifests"],
                )
            )
    if failures:
        return dict(upstream_checks=checks, references=refs), failures
    custody = {
        f"{eid}-{key}": copy_evidence(ref, raw)
        for eid in (7996, 7999)
        for key, ref in upstream[eid]["checkpoints"].items()
    }
    public = {
        role: copy_evidence(ref, raw)
        for role, ref in upstream[7994]["public_role_manifests"].items()
    }
    evaluator = {
        role: copy_evidence(ref, raw)
        for role, ref in upstream[7994]["evaluator_role_manifests"].items()
    }
    capture = upstream[7995]
    shards = []
    for i, ref in enumerate(capture["raw_response_shards"]):
        shards.append(copy_evidence(ref, raw))
        if i % 32 == 0:
            progress("copy_capture", i + 1, len(capture["raw_response_shards"]) - i - 1)
    saved = [json.loads(checked(ref).read_text()) for ref in shards]
    if (
        saved != capture["rows"]
        or capture_helper.reduce(saved)["role_completion_counts"]
        != capture["role_completion_counts"]
    ):
        raise ValueError("capture_custody")
    atomic_json(
        raw / "role_freeze.json",
        dict(
            config=CONFIG,
            public=public,
            evaluator=evaluator,
            custody=custody,
            captures=shards,
            upstream=refs,
        ),
    )
    progress("roles_hashes_budgets_frozen")
    data = json.loads(checked(custody["7996-inputs"]).read_text())["data"]
    heads = json.loads(checked(custody["7996-heads"]).read_text())["heads"]
    feature_rows = [
        r for ref in public.values() for r in json.loads(checked(ref).read_text())["features"]
    ]
    clusters = {r["family_id"]: r["source_normalized_hash"] for r in feature_rows}
    slots = [
        dict(
            family_id=r["family_id"],
            role=r["role"],
            status=r["status"],
            q=r.get("parsed", {}).get("probability"),
            source_cluster_id=clusters[r["family_id"]],
            exclusion_reason=r.get("exclusion_reason"),
            eligibility=r.get("public_eligible", False)
            and r.get("parsed", {}).get("completed", False),
        )
        for r in capture["rows"]
    ]
    refs += list(custody.values()) + list(public.values()) + list(evaluator.values()) + shards
    fit_roles = {
        role: dict(
            slot_ids=[r["family_id"] for r in rows],
            slot_hash=canonical_hash([r["family_id"] for r in rows]),
            inputs=custody["7996-inputs"],
        )
        for role, rows in data.items()
    }
    return dict(
        data=data,
        heads=heads,
        slots=slots,
        references=refs,
        upstream_checks=checks,
        role_manifests=dict(public=public, evaluator=evaluator, historical_fit_tune=fit_roles),
        original_exposure_rows=upstream[7994]["known_exposure_rows"],
        historical_headroom=upstream[7999]["headroom_diagnostics"],
    ), []


def bootstrap(rows: list[Json]) -> Json:
    """Resample sources and contiguous source blocks without counting seeds as data."""
    comparisons, pvalues = {}, {}
    rng = np.random.default_rng(CONFIG["seed"])
    for role in ("fit", "tune"):
        subset = [r for r in rows if r["role"] == role and r["denominator"]]
        indexed = {(r["family_id"], r["arm"]): r for r in subset}
        ids = [r["family_id"] for r in subset if r["arm"] == "intercept_only"]
        for metric in CONFIG["primary_outcomes"]:
            groups: Json = {}
            for fid in ids:
                a, b = indexed[fid, "intercept_only"], indexed[fid, "spline"]
                groups.setdefault(a["source_cluster_id"], []).append(a[metric] - b[metric])
            delta = np.asarray([np.mean(v) for v in groups.values()])
            n = len(delta)
            draws = {"source": [], "block": []}
            for unit in draws:
                for offset in range(0, CONFIG["draws"], 1000):
                    if unit == "source":
                        indices = rng.integers(n, size=(1000, n))
                    else:
                        starts = rng.integers(n, size=(1000, (n + 15) // 16))
                        indices = ((starts[:, :, None] + np.arange(16)) % n).reshape(1000, -1)[
                            :, :n
                        ]
                    draws[unit].extend(delta[indices].mean(axis=1).tolist())
                    progress(
                        "bootstrap_" + role + "_" + metric + "_" + unit,
                        offset + 1000,
                        CONFIG["draws"] - offset - 1000,
                    )
            summaries = {
                unit: dict(
                    interval=np.quantile(v, [0.025, 0.975]).tolist(),
                    p=float((1 + np.sum(np.asarray(v) >= 0)) / (1 + len(v))),
                )
                for unit, v in draws.items()
            }
            name = role + "_" + metric
            comparisons[name] = dict(
                mean_difference=float(delta.mean()),
                independent=n,
                uncertainty=summaries,
                source_differences=delta.tolist(),
            )
            pvalues[name] = max(v["p"] for v in summaries.values())
    adjusted = scalar.holm(pvalues)
    for name, value in comparisons.items():
        value["adjusted_p"] = adjusted[name]
    return comparisons


def diagnose(bundle: Json) -> Json:
    """Use only fit/tune targets to distinguish scale problems from benefit claims."""
    data, heads = bundle["data"], bundle["heads"]
    sparse.validate_data(data)
    f = sparse.usable(data["fit"])
    x = sparse.inputs(f)
    y = np.asarray([r["y"] for r in f], dtype=float)
    offset = np.log(np.clip(x[:, 0], 1e-4, 1 - 1e-4) / (1 - np.clip(x[:, 0], 1e-4, 1 - 1e-4)))
    progress("benchmark_intercept_before", len(f))
    intercept, solved = brentq(
        lambda b: float(np.mean(expit(offset + b) - y) + 0.002 * b),
        -40,
        40,
        xtol=1e-12,
        maxiter=128,
        full_output=True,
    )
    gradient = float(np.mean(expit(offset + intercept) - y) + 0.002 * intercept)
    checkpoint = dict(
        intercept=intercept,
        gradient_norm=abs(gradient),
        iterations=solved.iterations,
        function_calls=solved.function_calls,
        converged=solved.converged and abs(gradient) < 1e-8,
        fit_ids=[r["family_id"] for r in f],
    )
    progress("benchmark_intercept_after", solved.iterations)
    rows, conditioning, roles = [], [], {}
    for role, original in data.items():
        valid = sparse.usable(original)
        rx = sparse.inputs(valid)
        ry = np.asarray([r["y"] for r in valid], dtype=float)
        q = rx[:, 0]
        raw = np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
        roles[role] = dict(
            slot_ids=[r["family_id"] for r in original],
            slot_hash=canonical_hash([r["family_id"] for r in original]),
            support=scalar.support(valid, CONFIG["minimum_groups"], CONFIG["minimum_per_class"]),
            raw_logits=distribution(raw),
            raw_q=distribution(q),
            target_counts={str(c): int(np.sum(ry == c)) for c in (0, 1)},
            saturated_count=int(np.sum(expit(raw) * (1 - expit(raw)) < 0.001)),
            saturated_wrong_target_count=int(
                np.sum(((raw < 0) & (ry == 1)) | ((raw > 0) & (ry == 0)))
            ),
            group_support={
                "raw_" + a: scalar.support([r for r in valid if policy(r["q"]) == a], 32, 4)
                for a in ("accept", "escalate", "reject")
            },
        )
        predictions = dict(raw_q=q, intercept_only=expit(raw + intercept))
        for arm, roster in heads.items():
            if arm not in sparse.COUNTS:
                continue
            for head in roster:
                progress("benchmark_" + role + "_" + arm + "_before", len(valid))
                h = dict(head, temperature=1.0)
                z, jac = sparse.logits_jacobian(h, rx)
                theta = sparse.parameters(h)
                g = jac.T @ (expit(z) - ry) / len(ry) + 0.002 * theta
                initial = dict(h, parameters=np.zeros(len(theta)).tolist(), decay_scale=1.0)
                if arm == "mlp":
                    weights = np.random.default_rng(head["seed"]).normal(0, 0.1, len(theta))
                    weights[-1] = 0
                    initial["parameters"] = weights.tolist()
                conditioning.append(
                    dict(
                        role=role,
                        arm=arm,
                        seed=head["seed"],
                        gradient_norm=float(np.linalg.norm(g)),
                        coefficient_scale=distribution(theta),
                        residual_scale=distribution(z - raw),
                        logits=distribution(z),
                        initial_loss=sparse.objective(initial, rx, ry),
                        final_loss=sparse.objective(h, rx, ry),
                        loss_decrease=sparse.objective(initial, rx, ry)
                        - sparse.objective(h, rx, ry),
                        probability_threshold_distance=distribution(
                            np.minimum(
                                abs(expit(z / head["temperature"]) - 0.1),
                                abs(expit(z / head["temperature"]) - 0.5),
                            )
                        ),
                        logit_threshold_distance=distribution(
                            np.minimum(
                                abs(z / head["temperature"] - np.log(1 / 9)),
                                abs(z / head["temperature"]),
                            )
                        ),
                        optimizer_converged=bool(np.linalg.norm(g) < 1e-8),
                        original_steps=200,
                    )
                )
                if head["seed"] == 17:
                    predictions[arm] = sparse.predict(head, rx)
                progress("benchmark_" + role + "_" + arm + "_after", len(valid))
        for arm, probabilities in predictions.items():
            indexed = {r["family_id"]: float(p) for r, p in zip(valid, probabilities, strict=True)}
            for source in original:
                p = indexed.get(source["family_id"])
                eligible = p is not None
                brier = (p - source["y"]) ** 2 if eligible else None
                rows.append(
                    dict(
                        family_id=source["family_id"],
                        source_cluster_id=source["source_cluster_id"],
                        role=role,
                        arm=arm,
                        seed=17,
                        probability=p,
                        action=policy(p),
                        metric="brier",
                        numerator=brier,
                        denominator=int(eligible),
                        brier=brier,
                        decision_loss=loss(policy(p), source["y"]) if eligible else None,
                        status="completed" if eligible else "excluded",
                        eligibility=eligible,
                        exclusion_reason=None if eligible else "missing_q_features_or_target",
                        censor_reason=None,
                    )
                )
    progress("benchmark_synthetic_before", 160)
    sx = np.asarray([[0.5] + [float(i % 2)] * 8 for i in range(128)])
    sy = np.asarray([i % 2 for i in range(128)], dtype=float)
    control = sparse.fit_one("spline", 17, sparse.fit_scaler(sx), sx, sy, sx[:32], sy[:32])
    cp = sparse.predict(control, sx[:32])
    changed = sum(policy(float(p)) != policy(0.5) for p in cp)
    positive = dict(
        passed=changed > 0,
        changed_actions=changed,
        intended=32,
        independent=32,
        scope="circular_separable_synthetic_fixture_only",
        head=control,
        probabilities=cp.tolist(),
        fixture_ids=[f"synthetic-holdout-{i}" for i in range(32)],
    )
    positive["head"].pop("duration_s")
    progress("benchmark_synthetic_after", 32)
    return dict(
        rows=rows,
        conditioning_rows=conditioning,
        role_diagnostics=roles,
        gradient_norms=[
            dict(role=r["role"], arm=r["arm"], seed=r["seed"], value=r["gradient_norm"])
            for r in conditioning
        ],
        intercept_checkpoint=checkpoint,
        positive_control_results=positive,
        policy_cost_matrix=CONFIG["costs"],
        statistical_plan=CONFIG,
        paired_comparisons=bootstrap(rows),
        diagnosis=dict(
            under_convergence="hypothesis; fixed-head gradients measured, not a causal replay result",
            saturation="raw offset and opposing targets measured on fit/tune only",
            no_signal="unresolved; intercept response cannot identify semantic signal",
            inadequate_support="deployment and group support remain unresolved",
            observed_fit_nonconverged_heads=sum(
                not r["optimizer_converged"] for r in conditioning if r["role"] == "fit"
            ),
            raw_saturated_fraction={
                role: r["saturated_count"] / r["raw_q"]["count"] for role, r in roles.items()
            },
            insufficient_groups={
                role: [
                    group for group, support in r["group_support"].items() if not support["passed"]
                ]
                for role, r in roles.items()
            },
            deployment_benefit_established=False,
        ),
    )


def base(failures: list[Json]) -> Json:
    """Every terminal outcome names its limits and retains the full consumer schema."""
    return dict(
        experiment_id=8007,
        task_id=TASK,
        milestone="2026.10.694",
        run_date="20261002",
        schema="carnot.v694.conditioning_diagnosis.v1",
        execution_date="20261002",
        honest_verdict="complete_blocked_conditioning_diagnosis"
        if failures
        else "complete_null_conditioning_diagnosis",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=0.0,
        phase_spans=[],
        rows=[],
        conditioning_rows=[],
        role_manifests={},
        original_exposure_rows=[],
        gradient_norms=[],
        diagnosis={},
        checkpoints={},
        policy_cost_matrix=CONFIG["costs"],
        statistical_plan=CONFIG,
        claim_scope="Fit/tune conditioning diagnostic on exposed development. No new cohort, evaluation-selected fit, live model call, deployment benefit or FPGA claim.",
        sample_size_budget=dict(
            intended=0,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=0,
            independent=0,
        ),
        random_seed=CONFIG["seed"],
        reproducibility_checksum=None,
        verifier_is_oracle=False,
        acceptance_gate_results=dict(measurement=False, natural_benefit=False),
        genuine_headroom={},
        positive_control_results={},
        field_principles={},
        cited_upstream_artifacts=[],
        raw_shard_hashes=[],
        code_config_hashes=[],
        validation_receipts=[],
        coverage_statement_counts={},
        methods_ready_score=0,
        terminal_validation_sidecar_path=None,
        flagged_adversarial=False,
        method_map=[],
        citations=[],
        headroom_scope="Natural fit/tune action response and circular synthetic sensitivity remain separate.",
        methodology_note="Exact zeros in historical action changes are observed nulls. Synthetic labels are constructed; synthetic responsiveness gives no natural benefit credit.",
    )


def replay(path: Path) -> Json:
    """Cold-reduce checked raw rows and heads; reject any altered terminal aggregate."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    if value["checkpoints"]:
        bundle = json.loads(checked(value["checkpoints"]["bundle"]).read_text())
        expected = diagnose(bundle)
        for key, observed in expected.items():
            if value[key] != observed:
                raise ValueError("reduction:" + key)
        if value["role_manifests"] != bundle["role_manifests"]:
            raise ValueError("role_manifests")
    if value["methods_ready_score"] and (
        value["verdict_class"] in ("blocked", "disqualified")
        or not value["validation_receipts"]
        or not all(r["passed"] for r in value["validation_receipts"])
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True, sha256=sha256_file(path))


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Freeze explicit owned checks; global repository health has its own scope."""
    py, cov = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/coverage")
    include = ",".join(str(ROOT / p) for p in OWNED)
    prefix = [
        cov,
        "run",
        "--parallel-mode",
        "--data-file=" + str(scratch / ".coverage"),
        "--include=" + include,
    ]
    commands = [
        CommandSpec(
            "owned_units",
            tuple(
                prefix
                + [
                    "-m",
                    "pytest",
                    "-n0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "--basetemp=" + str(scratch / "pytest"),
                    TEST,
                ]
            ),
            "owned",
            180,
        )
    ]
    commands += [
        CommandSpec(
            "consumer_and_E2E-015",
            (
                py,
                "-m",
                "pytest",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "--basetemp=" + str(scratch / "consumers"),
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_sparse_energy_7996.py",
                "tests/python/test_qwen_energy_calibration_7972.py",
                "tests/python/test_source_boundary_7852.py",
            ),
            "owned",
            180,
        )
    ]
    commands += [
        CommandSpec(
            "coverage_combine",
            (cov, "combine", "--data-file=" + str(scratch / ".coverage"), str(scratch)),
            "owned",
        ),
        CommandSpec(
            "coverage_report",
            (
                cov,
                "report",
                "--data-file=" + str(scratch / ".coverage"),
                "--include=" + include,
                "--show-missing",
                "--fail-under=100",
            ),
            "owned",
        ),
        CommandSpec(
            "coverage_json",
            (
                cov,
                "json",
                "--data-file=" + str(scratch / ".coverage"),
                "-o",
                str(scratch / "coverage.json"),
            ),
            "owned",
        ),
        CommandSpec("ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST), "owned"),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST),
            "owned",
        ),
        CommandSpec(
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
            "owned",
        ),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py", *OWNED, TEST), "owned"),
        CommandSpec(
            "full_pytest",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            120,
        ),
    ]
    return commands


def terminal(path: Path) -> Json:
    """Check exact candidate bytes in fresh processes before the publication helper copies them."""
    commands = [
        CommandSpec(
            "cold_reduction",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / OWNED[-1]),
                "--cold-replay",
                str(path),
            ),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (str(ROOT / ".venv/bin/python"), "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (
                str(ROOT / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(path),
            ),
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
    """Run one frozen diagnostic, or replay a private or published artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261002", choices=["20261002"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin_no_pretrained_load_or_generation")
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            progress("cold_reduction_passed")
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        scratch = Path(tempfile.mkdtemp(prefix="carnot-8007-"))
        commands = validation_plan(scratch)
        code = [reference(ROOT / p) for p in OWNED + [TEST]]
        literature = args.root / "results" / "raw" / NAME / "literature_review.json"
        review = (
            json.loads(literature.read_text())
            if literature.is_file()
            else dict(method_map=[], citations=[])
        )
        atomic_json(
            raw / "configuration.json",
            dict(
                config=CONFIG,
                method_map=review["method_map"],
                code=code,
                commands=[asdict(c) for c in commands],
            ),
        )
        progress("methods_costs_gates_commands_frozen")
        if args.fixture_input:
            bundle, failures = json.loads(args.fixture_input.read_text()), []
        else:
            bundle, failures = materialize(args.root, raw)
        if not args.fixture_input and not failures and len(review["method_map"]) != 5:
            failures.append(
                dict(
                    upstream_id="V694-primary-source-review",
                    path=str(literature),
                    hash=sha256_file(literature) if literature.is_file() else None,
                    artifact_field="method_map",
                    expected="five checked primary sources",
                    observed=len(review["method_map"]),
                    passed=False,
                )
            )
        frozen_at = time.monotonic()
        value = base(failures)
        value.update(
            cited_upstream_artifacts=bundle["references"],
            preconditions_checked=bundle["upstream_checks"],
        )
        if not failures:
            measured = diagnose(bundle)
            value.update(
                measured,
                role_manifests=bundle["role_manifests"],
                original_exposure_rows=bundle["original_exposure_rows"],
            )
            atomic_json(raw / "bundle.json", bundle)
            atomic_json(raw / "measurement.json", measured)
            value["checkpoints"] = dict(
                bundle=reference(raw / "bundle.json"),
                measurement=reference(raw / "measurement.json"),
            )
            value["inference_substrate"] = "verifier_ensemble_against_cached_candidates"
            value["trained_head_specs"] = [
                dict(
                    name="intercept_diagnostic",
                    parameters=1,
                    device="cpu",
                    iterations=measured["intercept_checkpoint"]["iterations"],
                ),
                dict(
                    name="synthetic_spline",
                    parameters=109,
                    device="cpu",
                    steps=200,
                    scope="circular_fixture",
                ),
            ]
            value["genuine_headroom"] = dict(
                historical=bundle.get("historical_headroom"),
                fit_tune_changed_actions=sum(
                    r["action"]
                    != policy(
                        next(
                            s["q"]
                            for s in bundle["data"][r["role"]]
                            if s["family_id"] == r["family_id"]
                        )
                    )
                    for r in measured["rows"]
                    if r["arm"] == "intercept_only" and r["denominator"]
                ),
                natural_stream_benefit_measured=False,
            )
            slots = bundle["slots"]
            value["development_slot_rows"] = [
                dict(
                    family_id=r["family_id"],
                    role=r["role"],
                    arm="cached_capture",
                    metric="q_eligibility",
                    seed=69395,
                    numerator=int(r["eligibility"]),
                    denominator=1,
                    status="completed" if r["eligibility"] else "excluded",
                    exclusion_reason=r.get("exclusion_reason")
                    or (None if r["eligibility"] else "missing_q_or_source"),
                    censor_reason=None,
                )
                for r in slots
            ]
            value["development_role_counts"] = {
                role: dict(
                    intended=sum(r["role"] == role for r in slots),
                    eligible=sum(r["role"] == role and r["eligibility"] for r in slots),
                )
                for role in ("calibration", "stream", "retention")
            }
            fitrows = [r for rows in bundle["data"].values() for r in rows]
            valid = sparse.usable(fitrows)
            value["sample_size_budget"] = dict(
                intended=len(fitrows) + len(slots),
                eligible=len(valid) + sum(r["eligibility"] for r in slots),
                started=len(fitrows) + len(slots),
                completed=len(valid) + sum(r["eligibility"] for r in slots),
                excluded=len(fitrows) - len(valid) + sum(not r["eligibility"] for r in slots),
                failed=0,
                censored=0,
                independent=len(
                    {r["source_cluster_id"] for r in valid}
                    | {r["source_cluster_id"] for r in slots if r["eligibility"]}
                ),
                seeds_are_independent=False,
            )
            value["acceptance_gate_results"]["measurement"] = (
                measured["intercept_checkpoint"]["converged"]
                and measured["positive_control_results"]["passed"]
            )
            if args.fixture_input:
                value.update(
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_fixture_conditioning",
                )
        measured_at = time.monotonic()
        if not args.fixture_input and not failures:
            env = dict(
                PYTHONUNBUFFERED="1",
                JAX_PLATFORMS="cpu",
                OPENBLAS_NUM_THREADS="1",
                CARNOT_8007_COVERAGE_FILE=str(scratch / ".coverage"),
                COVERAGE_FILE=str(scratch / ".coverage-health"),
            )
            receipts = run_commands(
                ROOT, commands, log_dir=raw / "validation_logs", extra_env=env, heartbeat_s=30
            )
            value["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
            value["repository_health"] = [r for r in receipts if r["scope"] == "repository_health"]
            coverage = json.loads((scratch / "coverage.json").read_text())
            value["coverage_statement_counts"] = {
                k: v["summary"] for k, v in coverage["files"].items()
            }
            passed = all(r["passed"] for r in value["validation_receipts"]) and all(
                v["missing_lines"] == 0 and v["num_statements"] > 0
                for v in value["coverage_statement_counts"].values()
            )
            value["methods_ready_score"] = int(
                passed and value["acceptance_gate_results"]["measurement"]
            )
            if not value["methods_ready_score"]:
                value.update(
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_conditioning_diagnosis",
                )
        if literature.is_file():
            copied = copy_evidence(reference(literature), raw)
            for citation in review["citations"]:
                citation["evidence"] = [
                    dict(ref, **copy_evidence(ref, raw)) for ref in citation["evidence"]
                ]
                value["cited_upstream_artifacts"].extend(citation["evidence"])
            value.update(method_map=review["method_map"], citations=review["citations"])
            value["cited_upstream_artifacts"].append(copied)
        ended = time.monotonic()
        value.update(
            duration_s=ended - began,
            phase_spans=[
                dict(phase="freeze", duration_s=frozen_at - began),
                dict(phase="measurement", duration_s=measured_at - frozen_at),
                dict(phase="validation", duration_s=ended - measured_at),
            ],
            code_config_hashes=code,
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        )
        value["raw_shard_hashes"] = (
            list(value["checkpoints"].values())
            + [reference(raw / "configuration.json")]
            + value["cited_upstream_artifacts"]
        )
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=CONFIG, code=code, raw=value["raw_shard_hashes"])
        )
        value["field_principles"] = {
            k: "Bind this invocation to checked development primitives; mechanism readiness and circular controls cannot establish deployment benefit."
            for k in value
        }
        progress("publish_before")
        validator = replay if args.fixture_input else terminal
        receipt = publish_primary(output, value, validator)
        atomic_json(raw / "terminal_validation.json", receipt)
        reader = reader_receipt(
            TASK, output.parent, field="methods_ready_score", expected=value["methods_ready_score"]
        )
        atomic_json(raw / "reader_receipt.json", reader)
        if not reader["passed"] or reader["gate_sha256"] != sha256_file(output):
            raise ValueError("primary_reader")
        replay(output)
        progress("publish_after")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[exp8007] failed={error}", flush=True)
        return 1
