"""REQ-REPORT-8072: freeze independent development protocols before outcomes.

This records methods and historical custody. It neither trains the future heads
nor opens evaluator labels, so valid methods can precede a negative result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_8058_v698_sealed_evidence_methods as prior
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify.evidence_features_7980 import FEATURES, normalized

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8072_v699_sealed_methods"
TASK = "exp8072-sealed-methods"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_sealed_methods_8072.py"
OWNED = [MODULE, CLI]
BRANCHES = ["source", "learning", "service"]
PAPERS = ["2512.10461", "2609.34099", "2609.10873", "2511.12828", "2607.04223", "2512.15605"]
LITERATURE = f"results/raw/{NAME}/literature_access.json"
INPUTS = list(
    dict.fromkeys(
        prior.INPUTS
        + [
            "research-studying.md",
            "python/carnot/experiment_8058_v698_sealed_evidence_methods.py",
            "python/carnot/verify/sparse_energy_7996.py",
            "python/carnot/verify/fresh_feedback_8064.py",
            LITERATURE,
        ]
    )
)
PINS = {
    8058: "sha256:cc737701b78445d1629e63d68dea59b7407906a7f99f41305337a9699775e3b8",
    8063: "sha256:fd715b21c6c985a28ad4a8006bccf02b64e0fecd078ccc225280f84f2a1d30cf",
    8065: "sha256:2f8fb51ea604afad956112e56aae495de7edb742e6af4b5152998dcda87129ce",
    8066: "sha256:328dd7cb2ef822bc6911633ddcfcd0de10d77f1796428216cc416cb9e36489de",
}
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so waiting children cannot look like completed work."""
    print(
        f"[exp8072] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def methods() -> Json:
    """Specify comparisons before labels so selection cannot follow outcomes."""
    old = prior.methods()
    learning = deepcopy(old["learning"])
    learning.update(
        arms=["frozen", "unconditional", "ray_fresh", "projected_fresh"],
        initial_head="authenticated Exp8058 qualified historical head; independent of Exp8073",
        reused=None,
        unconditional="raw alpha1 at shared decision time",
        candidate_construction="four gradients per own incumbent; ray_fresh keeps raw endpoint; projected_fresh corrects endpoint before commitment; gradient/label budgets identical",
    )
    guard = deepcopy(old["guard"])
    guard["selection"] = (
        "largest passing alpha; projected candidates also satisfy complete memory; keep feasible incumbent else restore w0 and record reset"
    )
    source = dict(
        roles=dict(fit=64, tune=32, evaluation=96),
        inputs=9,
        features=["qualified_cached_qwen_logit", *FEATURES],
        arms=["intercept", "scalar_affine", "linear", "additive", "interaction"],
        basis="conditioned additive clamped cubic B-splines; fit-only logit centering/scaling and feature bounds; original knots; clipping counts retained",
        interactions=[
            "qwen_logit * overlap_maximum",
            "qwen_logit * uncovered_fraction_maximum",
            "numeric_mismatch_maximum * negation_mismatch_maximum",
        ],
        folds=4,
        fold_assignment="SHA256 source ID sorted, round robin; source groups never split",
        ridge_grid=[0.0001, 0.001, 0.01, 0.1, 1],
        ridge_selection="mean held-out fit log loss; ties choose largest ridge",
        geometry="fit only; four-fold geometry fit within each training fold; final geometry on full fit",
        calibration="tune-only affine calibration; evaluator labels unavailable to optimizer",
        floors=dict(fit=[48, 8], tune=[24, 4], evaluation=[72, 8]),
        budget_s=600,
        equivalent_logistic="same coefficients and basis; E0=0,E1=-f; p(unsupported)=sigmoid(f); parity tolerance1e-10",
        convergence="finite objective and parameters, gradient infinity norm<=1e-7; preserve numerical convergence receipts",
        prediction_seal="public-only process seals full evaluation predictions and input/code/head hashes before evaluator opens original targets",
        permutation="fixed seed6998072; sort original evaluation source IDs; permute all eight public features together, keep q fixed",
        ablation="zero each interaction coefficient separately; diagnostics only",
        likelihoods="qualified historical cached scalar only; no Exp8059 likelihoods",
    )
    projection = dict(
        memory_limit=64,
        memory_order="newest eligible released update-source constraints; admission labels forbidden",
        logit="f_w(x)=a*B(x)w+b; a,b frozen",
        sign="s_j=2*y_j-1",
        threshold="tau_j=0 if y_j=1 else log(9)",
        reference="c_j=min(s_j*f_w0(x_j),tau_j)",
        half_space="s_j*a*B(x_j)w >= c_j-s_j*b",
        box="w0-.5 <= w <= w0+.5",
        sample_size=8,
        relaxation=1,
        max_steps=256,
        residual_tolerance=1e-8,
        correction="v=max(0,c_j-s_j*b-s_j*a*B(x_j)w); w += v*(s_j*a*B(x_j))/||s_j*a*B(x_j)||^2; box after each correction",
        selection="seeded sample without replacement min(8,memory_count); greatest violation; ties source ID",
        termination="complete residual<=1e-8 or256 corrections; complete finite residual check afterward",
        fallback="nonfinite, zero-norm violated or unsatisfied: feasible incumbent else restore w0; record reset",
        cost_components=[
            "gradient",
            "constraint_construction",
            "projection",
            "full_validation",
            "storage",
            "fallback",
        ],
        qualification="64 seeded private feasible systems versus independent convex QP; contradictory/zero-norm controls and durable memory; circular scope only",
    )
    hypotheses = [
        dict(
            id="H1_source_interactions",
            contrast="additive cost minus interaction cost",
            margin=0.02,
            support=[72, 8],
            beneficial_changes=5,
            added_false_accepts=0,
            brier_increase=0.01,
        ),
        dict(
            id="H2_projected_learning",
            contrast="ray_fresh cost minus projected_fresh cost",
            margin=0.02,
            support=[80, 10],
            beneficial_changes=5,
            per_seed_false_accept_baselines=["ray_fresh", "frozen", "unconditional"],
            cost_noninferiority_baselines=["frozen", "unconditional"],
            cost_noninferiority_margin=0.02,
            retention_support=[48, 8],
            retention_arms=["ray_fresh", "projected_fresh"],
            retention_brier=0.01,
            retention_cost=0.02,
        ),
    ]
    return dict(
        source=source,
        learning=learning,
        guard=guard,
        projection=projection,
        costs=old["costs"],
        timeline_masks=old["timeline_masks"],
        hypotheses=hypotheses,
        statistics=dict(
            holm_family=[h["id"] for h in hypotheses],
            alpha=0.05,
            bootstrap_draws=10000,
            H1="paired source-group bootstrap",
            H2="moving blocks on original256 slots; seeds averaged within source",
            primary_block=32,
            sensitivity_blocks=[16, 64],
            confidence=0.95,
            test="center paired gains at null margin; p=(1+count(centered_mean>=observed_gain))/(1+completed_draws)",
            censored_resamples="preserve original eligibility/time masks; report incomplete draws",
            unavailable_or_safety_failure_p=1,
            support_failure="block affected hypothesis only",
            measured_noop_loss_or_safety_failure="complete null, benefit0",
            scope="conditional exposed development; no generalization",
        ),
        method_source_map=json.loads((ROOT / LITERATURE).read_text())["rows"],
    )


def service_matrix() -> Json:
    """Keep every scheduled transaction, including warmups, despite estimated cost."""
    halves = dict(
        core=["cold", "warm", "all_miss"], lifecycle=["changed_10pct", "eviction", "restart"]
    )
    pairs = [
        ["feedback_constrained", "accepted"],
        ["feedback_constrained", "rejected"],
        ["unconstrained", "accepted"],
    ]
    arms = ["python_uncached", "python_cached", "native_uncached", "native_cached"]
    rng = random.Random(6998072)
    rows = []
    for half, modes in halves.items():
        for repetition in range(-5, 30):
            order = rng.sample(modes, len(modes))
            for mode in order:
                for condition, transaction_class in pairs:
                    paired = rng.sample(arms, len(arms))
                    for arm in paired:
                        rows.append(
                            dict(
                                unit=f"{half}/{mode}/{condition}/{transaction_class}/{repetition}/{arm}",
                                partition=half,
                                mode=mode,
                                condition=condition,
                                transaction_class=transaction_class,
                                arm=arm,
                                repetition=repetition,
                                pair_order=paired,
                                warmup=repetition < 0,
                            )
                        )
    return dict(
        partitions=halves,
        condition_class_pairs=pairs,
        arms=arms,
        rows=rows,
        warmups=5,
        repetitions=30,
        rows_per_partition=1260,
        measurement_budget_s_per_partition=1800,
        capacity=256,
        parity_tolerance=1e-10,
        exact_features_actions_states=True,
        checkpoint="complete paired arm quartets; all censored slots retained",
        costs="population, hashing, extraction, guards, FFI, writes, invalidation, recovery and durable reload",
        acceptance="warm lower95% speed ratio>1.2; cold/all_miss lower95% ratio>=1/1.05; lifecycle intervals independent",
        external_costs="acquisition, original Qwen inference and external feedback unknown",
        default_enabled=False,
    )


def estimate(rows: list[Json], matrix: Json) -> Json:
    """Reduce complete historical timing cells; unknown cells remain unknown."""
    cells = []
    for half, modes in matrix["partitions"].items():
        for mode in modes:
            for condition, transaction_class in matrix["condition_class_pairs"]:
                for arm in matrix["arms"]:
                    values = [
                        r["transaction_ns"] / 1e9
                        for r in rows
                        if (
                            r.get("mode"),
                            r.get("condition"),
                            r.get("transaction_class"),
                            r.get("arm"),
                        )
                        == (mode, condition, transaction_class, arm)
                        and r.get("transaction_ns") is not None
                        and r.get("status") in ("completed", "excluded")
                    ]
                    cells.append(
                        dict(
                            partition=half,
                            mode=mode,
                            condition=condition,
                            transaction_class=transaction_class,
                            arm=arm,
                            sample_count=len(values),
                            numerator=sum(values),
                            denominator=len(values),
                            mean_s=sum(values) / len(values) if values else None,
                        )
                    )
    partitions = []
    for half in matrix["partitions"]:
        subset = [r for r in cells if r["partition"] == half]
        predicted = (
            sum(r["mean_s"] * 35 for r in subset)
            if all(r["sample_count"] for r in subset)
            else None
        )
        partitions.append(
            dict(
                partition=half,
                estimate_s=predicted,
                budget_s=1800,
                partial_observed_cell_estimate_s=sum((r["mean_s"] or 0) * 35 for r in subset),
                unknown_cells=sum(not r["sample_count"] for r in subset),
                estimated_overrun=predicted > 1800 if predicted is not None else None,
            )
        )
    return dict(
        cells=cells,
        partitions=partitions,
        historical_timing_rows=len(rows),
        scope="diagnostic timing extrapolation from censored Exp8066; overhead/drift unknown; do not shrink matrix",
    )


def seal(root: Path, raw: Path, *, mutate: bool = False) -> Json:
    """Authenticate custody before importing eligibility, without evaluator access."""
    progress("preconditions_before")
    plan: Json = dict(
        rows=[],
        role_manifests={},
        source_artifact_hashes=[],
        failures=[],
        methods=methods(),
        qualified_head={},
        qualified_head_sha256=None,
        outcome_access_ledger=[],
        service_matrix=service_matrix(),
        workload_budget_estimate={},
        validation_manifest=[],
        repository_health=[],
    )

    def require(path: Path, field: str, expected: Any, observed: Any, branches: list[str]) -> bool:
        if expected == observed:
            return True
        plan["failures"].append(
            dict(
                check=field,
                upstream=path.stem,
                path=str(path.absolute()),
                hash=sha256_file(path) if path.is_file() else None,
                field=field,
                op="==",
                expected=expected,
                observed=observed,
                branches=branches,
                passed=False,
            )
        )
        return False

    def bind(
        path: Path, expected: str | None = None, branches: list[str] = BRANCHES
    ) -> Path | None:
        if not require(path, "resource_exists", True, path.is_file(), branches):
            return None
        digest = sha256_file(path)
        if expected and not require(path, "sha256", expected, digest, branches):
            return None
        snapshot = raw / "inputs" / (digest[7:] + path.suffix)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        plan["source_artifact_hashes"].append(dict(reference(path), snapshot_path=str(snapshot)))
        return snapshot

    for relative in INPUTS:
        bind(root / relative)
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        require(
            ROOT / ".venv/bin" / tool,
            "resource_exists",
            True,
            (ROOT / ".venv/bin" / tool).is_file(),
            BRANCHES,
        )
    require(
        Path(sys.executable), "python_version>=3.11", True, sys.version_info >= (3, 11), BRANCHES
    )
    parents = {}
    for number, digest in PINS.items():
        branches = (
            ["source", "learning"]
            if number == 8058
            else ["service"]
            if number == 8066
            else ["learning"]
        )
        paths = list((root / "results").glob(f"experiment_{number}_*.json"))
        path = (
            paths[0] if len(paths) == 1 else root / "results" / f"experiment_{number}_missing.json"
        )
        snapshot = bind(path, digest, branches)
        if snapshot is None:
            continue
        try:
            value = json.loads(snapshot.read_text())
            require(path, "flagged_adversarial", False, value.get("flagged_adversarial"), branches)
            require(
                path, "required_checks_passed", True, value.get("required_checks_passed"), branches
            )
            terminal = bind(Path(value["terminal_validation_sidecar_path"]), branches=branches)
            if terminal is None:
                continue
            publication = json.loads(terminal.read_text())["publication"]
            side = bind(Path(publication["sidecar_path"]), branches=branches)
            if side is None:
                continue
            report = read_bound_sidecar(path, Path(publication["sidecar_path"]))
            require(side, "report.passed", True, report["report"]["passed"], branches)
            require(side, "primary_path", str(path.absolute()), report["primary_path"], branches)
            require(terminal, "primary_sha256", digest, publication["primary_sha256"], branches)
            parents[number] = value
        except (OSError, ValueError, KeyError, TypeError) as error:
            require(path, "authenticated_terminal", True, str(error), branches)
    progress("preconditions_after", len(plan["source_artifact_hashes"]), 0)
    if 8058 in parents:
        upstream = parents[8058]
        plan["qualified_head"] = upstream["qualified_head"]
        plan["qualified_head_sha256"] = canonical_hash(plan["qualified_head"])
        require(
            root / "results",
            "qualified_head_sha256",
            upstream["qualified_head_sha256"],
            plan["qualified_head_sha256"],
            ["learning"],
        )
        source_seen: dict[str, str] = {}
        for role, count in prior.ROLE_COUNTS.items():
            branches = ["learning"] if role in ("stream", "retention") else ["source"]
            ref = upstream["role_manifests"][role]
            bound = bind(Path(ref["path"]), ref["sha256"], branches)
            if bound is None:
                continue
            originals = json.loads(bound.read_text())["rows"]
            if mutate and role == "tune" and "fit" in plan["role_manifests"]:
                fit = json.loads(Path(plan["role_manifests"]["fit"]["path"]).read_text())["rows"][0]
                originals[0].update({k: fit[k] for k in ("source_bytes", "source_cluster_id")})
                atomic_json(raw / "role_mutation_control.json", dict(role=role, rows=originals))
            require(
                bound,
                "original_slots",
                list(range(count)),
                [r["slot"] for r in originals],
                branches,
            )
            plan["role_manifests"][role] = dict(path=str(bound), sha256=sha256_file(bound))
            for original in originals:
                identity = normalized(bytes.fromhex(original["source_bytes"]))
                require(
                    bound,
                    "source_cluster_id",
                    identity,
                    original["source_cluster_id"],
                    branches,
                )
                if branches == ["source"]:
                    require(
                        bound,
                        "source_cluster_overlap",
                        role,
                        source_seen.get(identity, role),
                        branches,
                    )
                    source_seen[identity] = role
            plan["rows"].extend(
                dict(r, condition=r["role"], source_hash=r["source"])
                for r in upstream["rows"]
                if r["role"] == role
            )
        plan["outcome_access_ledger"] = deepcopy(upstream["outcome_access_ledger"])
        plan["repository_health"] = upstream["repository_health"]
    if 8066 in parents:
        upstream = parents[8066]
        ref = next(
            r for r in upstream["raw_shard_hashes"] if Path(r["path"]).name == "measurement.json"
        )
        bind(Path(ref["path"]), ref["sha256"], ["service"])
        timing = [
            {
                k: r.get(k)
                for k in (
                    "unit",
                    "mode",
                    "condition",
                    "transaction_class",
                    "arm",
                    "status",
                    "transaction_ns",
                )
            }
            for r in upstream["rows"]
        ]
        atomic_json(raw / "historical_timing_rows.json", dict(rows=timing))
        plan["workload_budget_estimate"] = estimate(timing, plan["service_matrix"])
    plan["valid"] = {b: not any(b in f["branches"] for f in plan["failures"]) for b in BRANCHES}
    progress("roles_methods_matrix_sealed", len(plan["rows"]), 0)
    return plan


def build(
    plan: Json, raw: Path, receipts: list[Json], coverage: Json, fixture: bool, duration: float
) -> Json:
    """Reduce independent readiness; a valid protocol need not predict benefit."""
    required = [r for r in receipts if r.get("classification", "required") == "required"]
    coverage_ok = all(
        p in coverage
        and coverage[p]["summary"]["num_statements"] > 0
        and coverage[p]["summary"]["missing_lines"] == 0
        for p in OWNED
    )
    checks = (
        bool(plan["validation_manifest"])
        and [r["name"] for r in required] == plan["validation_manifest"]
        and all(r["passed"] for r in required)
        and coverage_ok
        and not fixture
    )
    kind = (
        "disqualified" if not checks and not fixture else "blocked" if plan["failures"] else "null"
    )
    gates = deepcopy(plan["failures"])
    if not checks and not fixture:
        for r in required:
            if not r["passed"]:
                gates.append(
                    dict(
                        check=r["name"],
                        upstream=TASK,
                        path=r.get("log_path", str(raw)),
                        hash=r.get("log_sha256"),
                        field="exit_code",
                        op="==",
                        expected=r.get("expected_exit", 0),
                        observed=r.get("exit_code", "failed"),
                    )
                )
        gates.append(
            dict(
                check="owned_validation",
                upstream=TASK,
                path=str(raw / "validation.json"),
                hash=None,
                field="required_checks_passed",
                op="==",
                expected=True,
                observed=False,
            )
        )
    rows = plan["rows"]
    sizes = dict(
        intended_count=512,
        eligible_count=sum(r["eligible"] for r in rows),
        independent_count=len({r["source"] for r in rows if r["eligible"]}),
        completed_count=len(rows),
        censored_count=512 - len(rows),
        excluded_count=sum(not r["eligible"] for r in rows),
        failed_count=0,
    )
    value: Json = dict(
        experiment_id=8072,
        task_id=TASK,
        milestone="2026.10.699",
        run_date="20261003",
        schema="carnot.v699.sealed_methods.v1",
        verdict_class=kind,
        honest_verdict="complete_blocked_" + Path(gates[0]["path"]).stem.replace(".", "_")
        if kind == "blocked"
        else "complete_" + kind + "_sealed_methods",
        verifier_is_oracle=False,
        flagged_adversarial=False,
        required_checks_passed=checks,
        claim_scope="Independent method readiness on historically exposed development cohorts. No current outcomes, model loads, source benefit, learning benefit or measured service speed. Private controls have no scientific credit.",
        rows=rows,
        **sizes,
        sample_size_budget=dict(
            **sizes,
            role_counts=prior.ROLE_COUNTS,
            independent_environments=0,
            seeds=list(range(101, 121)),
            service_rows=2520,
        ),
        gate_check_summary=gates,
        validation_receipts=receipts,
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        substrate_declaration=dict(
            reduction="aggregation_from_upstream_artifacts", mode="no_model_load", MODEL_SPECS=[]
        ),
        trained_head_specs=dict(
            current_fits=0,
            historical_head_sha256=plan["qualified_head_sha256"],
            parameter_count=plan["qualified_head"].get("parameter_count"),
            calibration_mapping="historical calibration=[b,a]; f=a*B*w+b; unchanged numerical head",
        ),
        generalized_learning_benefit_score=0,
        random_seed=6998072,
        source_artifact_hashes=plan["source_artifact_hashes"],
        raw_shard_hashes=[
            reference(raw / p)
            for p in [
                "seal.json",
                "work.json",
                "validation.json",
                "validation_commands.json",
                "historical_timing_rows.json",
                "role_mutation_control.json",
            ]
            if (raw / p).is_file()
        ],
        code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
        reproducibility_checksum=canonical_hash(
            dict(plan=plan, receipts=receipts, coverage=coverage)
        ),
        duration_s=duration,
        phase_spans=plan.get("phase_spans", []),
        role_manifests=plan["role_manifests"],
        exposure_rows=[
            dict(
                unit=r["unit"],
                source=r["source"],
                role=r["role"],
                historical_development_exposure=True,
            )
            for r in rows
        ],
        eligibility_rules="Original public completeness AND complete-response annotation AND known target; unknowns excluded without replacement, role changes or timeline compression.",
        method_source_map=plan["methods"]["method_source_map"],
        method_freeze=dict(
            sha256=canonical_hash(plan["methods"]),
            methods=plan["methods"],
            current_outcomes_opened=0,
        ),
        statistical_plan=dict(
            plan["methods"]["statistics"], hypotheses=plan["methods"]["hypotheses"]
        ),
        frozen_hypothesis_family=[h["id"] for h in plan["methods"]["hypotheses"]],
        candidate_equations=plan["methods"]["projection"],
        service_matrix=plan["service_matrix"],
        workload_budget_estimate=plan["workload_budget_estimate"],
        qualified_head=plan["qualified_head"],
        qualified_head_sha256=plan["qualified_head_sha256"],
        outcome_access_ledger=plan["outcome_access_ledger"]
        + [
            dict(
                event="v699_protocols_sealed",
                current_outcomes_opened=0,
                private_evaluator_files_opened=0,
            )
        ],
        repository_health=plan["repository_health"]
        + [r for r in receipts if r.get("classification") == "diagnostic"],
        methodology_note="Readiness measures authenticated protocol validity; no current scientific hypothesis executed. Existing disqualified cache timings inform feasibility only. All benefits remain unmeasured.",
    )
    value.update({b + "_protocol_ready_score": int(checks and plan["valid"][b]) for b in BRANCHES})
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to exact sealed evidence; prevent historical observations or private controls becoming current scientific credit."
        for k in value
    }
    for b in BRANCHES:
        value["field_principles"][b + "_protocol_ready_score"] = (
            "Qualify this branch's protocol independently of expected benefit or another branch's external failure. Owned failures disqualify readiness."
        )
    value["field_principles"]["generalized_learning_benefit_score"] = (
        "Exposed development cohorts cannot establish generalized lifelong improvement."
    )
    value["field_principles"]["gate_check_summary"] = (
        "Exact operands distinguish missing prerequisites from measured zeros and prevent accidental retries of terminal external failures."
    )
    value["field_principles"]["workload_budget_estimate"] = (
        "Unknown timing cells stay unknown; expected overruns cannot justify shrinking the registered matrix."
    )
    return value


def replay(path: Path) -> bool:
    """Recompute reductions from frozen custody, detecting edits without new labels."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        if any(
            sha256_file(ROOT / p) != digest for p, digest in value["code_config_hashes"].items()
        ):
            return False
        for receipt in value["validation_receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        plan = json.loads((raw / "seal.json").read_text())
        if plan["methods"] != methods() or plan["service_matrix"] != service_matrix():
            return False
        primitives = []
        for role, ref in plan["role_manifests"].items():
            originals = json.loads(Path(ref["path"]).read_text())["rows"]
            for original in originals:
                identity = normalized(bytes.fromhex(original["source_bytes"]))
                primitives.append((role, original["slot"], identity, original["family_id"]))
        if sorted(primitives) != sorted(
            [(r["role"], r["slot"], r["source"], r["family_id"]) for r in plan["rows"]]
        ):
            return False
        if plan["rows"]:
            ref = next(
                r
                for r in value["source_artifact_hashes"]
                if Path(r["path"]).name == "experiment_8058_v698_sealed_evidence_methods.json"
            )
            upstream = json.loads(Path(ref["snapshot_path"]).read_text())
            expected = [
                dict(r, condition=r["role"], source_hash=r["source"])
                for r in upstream["rows"]
                if r["role"] in plan["role_manifests"]
            ]
            if sorted(map(canonical_hash, expected)) != sorted(map(canonical_hash, plan["rows"])):
                return False
        if (
            plan["qualified_head_sha256"] is not None
            and canonical_hash(plan["qualified_head"]) != plan["qualified_head_sha256"]
        ):
            return False
        if (raw / "historical_timing_rows.json").is_file():
            timing = json.loads((raw / "historical_timing_rows.json").read_text())["rows"]
            if estimate(timing, plan["service_matrix"]) != plan["workload_budget_estimate"]:
                return False
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        return (
            build(
                plan,
                raw,
                validation["receipts"],
                validation["coverage"],
                work["fixture"],
                work["duration_s"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def manifest(private: Path) -> list[Json]:
    """Reuse qualified bounded checks, with coverage limited to the new code."""
    specs = prior.manifest(private)
    replacements = dict(
        zip([prior.MODULE, prior.CLI, prior.TEST], [MODULE, CLI, TEST], strict=True)
    )
    for spec in specs:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        spec["deadline_s"] = (
            max(spec["deadline_s"], 180)
            if spec["name"] == "focused_unit_and_cli"
            else spec["deadline_s"]
        )
        spec["argv"] = [
            a.replace(str(ROOT / prior.MODULE), str(ROOT / MODULE)).replace(
                str(ROOT / prior.CLI), str(ROOT / CLI)
            )
            for a in spec["argv"]
        ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    specs.append(
        dict(
            name="repository_full_suite",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "--basetemp=" + str(private / "full_suite"),
            ],
            deadline_s=900,
            expected_exit=0,
            classification="diagnostic",
        )
    )
    return specs


def terminal(path: Path) -> Json:
    """Check cold reconstruction and both artifact auditors before publication."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    py = str(ROOT / ".venv/bin/python")
    specs = [
        dict(name=n, argv=a, deadline_s=60, expected_exit=0)
        for n, a in [
            ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
            ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
            (
                "strict_rows",
                [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
            ),
        ]
    ]
    with tempfile.TemporaryDirectory(prefix="carnot-8072-terminal-") as tmp:
        receipts = []
        for spec in specs:
            progress("subprocess_before_" + spec["name"], len(receipts), len(specs) - len(receipts))
            receipts.append(run_check(ROOT, spec, Path(tmp), raw / "terminal_logs"))
            progress("subprocess_after_" + spec["name"], len(receipts), len(specs) - len(receipts))
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Seal in an exited child, validate owned work, then publish checked bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--seal-output", type=Path)
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("cold_replay_passed" if passed else "cold_replay_rejected")
        return 0 if passed else 1
    if args.seal_output:
        plan = seal(args.root, args.seal_output.parent, mutate=args.mutate)
        plan["phase_spans"] = [
            dict(name="preconditions_and_seal", duration_s=time.monotonic() - START)
        ]
        atomic_json(args.seal_output, plan)
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if (raw / "seal.json").exists():
        progress("existing_seal_preserved")
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="carnot-8072-") as temp:
            private = Path(temp)
            specs = manifest(private)
            cmd = [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--root",
                str(args.root),
                "--seal-output",
                str(raw / "seal.json"),
            ]
            if args.mutate:
                cmd.append("--mutate")
            config = os.environ.get("CARNOT_8072_COVERAGE_CONFIG")
            if config:
                cmd = [
                    cmd[0],
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + config,
                    "--data-file=" + str(Path(config).parent / ".coverage"),
                    *cmd[2:],
                ]
            worker = dict(
                name="seal_child_normal_exit",
                argv=cmd,
                deadline_s=120,
                expected_exit=0,
                classification="required",
            )
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=[worker, *specs],
                    code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
                    terminal_commands=[
                        dict(name=n, argv=a, deadline_s=60, expected_exit=0)
                        for n, a in [
                            (
                                "cold_replay",
                                [
                                    str(ROOT / ".venv/bin/python"),
                                    "-u",
                                    str(ROOT / CLI),
                                    "--cold-replay",
                                    str(raw / "terminal_candidate.json"),
                                ],
                            ),
                            (
                                "adversarial",
                                [
                                    str(ROOT / ".venv/bin/python"),
                                    str(ROOT / "scripts/adversarial_verify.py"),
                                    "--json",
                                    str(raw / "terminal_candidate.json"),
                                ],
                            ),
                            (
                                "strict_rows",
                                [
                                    str(ROOT / ".venv/bin/python"),
                                    str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                                    "--strict",
                                    str(raw / "terminal_candidate.json"),
                                ],
                            ),
                        ]
                    ],
                    terminal_deadline_s=60,
                ),
            )
            progress("subprocess_before_seal", 0, 1)
            worker_receipt = run_check(ROOT, worker, private, raw / "validation_logs")
            progress("subprocess_after_seal", int(worker_receipt["passed"]), 0)
            if not worker_receipt["passed"]:
                raise ValueError("seal_child_failed")
            plan = json.loads((raw / "seal.json").read_text())
            plan["validation_manifest"] = [worker["name"]] + [
                s["name"] for s in specs if s["classification"] == "required"
            ]
            plan["sealed_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(raw / "seal.json", plan)
            os.environ["CARNOT_8072_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            receipts = [worker_receipt]
            for spec in [] if args.fixture_output else specs:
                progress(
                    "subprocess_before_" + spec["name"],
                    len(receipts),
                    len(specs) + 1 - len(receipts),
                )
                receipts.append(run_check(ROOT, spec, private, raw / "validation_logs"))
                progress(
                    "subprocess_after_" + spec["name"],
                    len(receipts),
                    len(specs) + 1 - len(receipts),
                )
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            duration = time.monotonic() - START
            atomic_json(
                raw / "work.json", dict(fixture=bool(args.fixture_output), duration_s=duration)
            )
            atomic_json(raw / "validation.json", dict(receipts=receipts, coverage=coverage))
            value = build(plan, raw, receipts, coverage, bool(args.fixture_output), duration)
            progress("publication_before", len(value["rows"]), 1)
            publication = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, measurement_exit_receipt=worker_receipt),
            )
            progress("complete", len(value["rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("rejected_" + str(error))
        return 1
