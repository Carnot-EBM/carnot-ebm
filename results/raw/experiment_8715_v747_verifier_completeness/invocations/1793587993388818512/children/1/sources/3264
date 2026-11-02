"""REQ-REPORT-8063: separate candidate quality, guard reuse and evidence limits.

Historical alternatives stay fixed. Evaluator outcomes diagnose the finite
exposed stream and never change the learner or the sealed future experiment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any

import numpy as np
from scipy.stats import beta

from carnot import experiment_8058_v698_sealed_evidence_methods as prior
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import learning_benefit_8052 as audit

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8063_v698_admission_opportunity_audit"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_admission_opportunity_8063.py"
OWNED = [MODULE, CLI]
PARENTS = {
    "experiment_8051_v697_feedback_constrained_learning": "sha256:229b54014662596ab5d3f82095d316a33cd852a15e6c2ae0c48a5006f3031837",
    "experiment_8052_v697_learning_benefit_audit": "sha256:6124c1da1854394ff6a17122ba9aaa549cbd99616bcc868da72ecfac5c4223d8",
    "experiment_8058_v698_sealed_evidence_methods": "sha256:cc737701b78445d1629e63d68dea59b7407906a7f99f41305337a9699775e3b8",
}
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual units so waiting does not imply progress."""
    print(
        f"[exp8063] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def ratio(numerator: int, denominator: int) -> float | None:
    """An empty eligible population cannot supply a rate."""
    return numerator / denominator if denominator else None


def pool_summary(rows: list[Json]) -> Json:
    """Count a choice once even when several alternatives satisfy the contract."""
    useful = [r for r in rows if r["useful"]]
    return dict(
        useful_count=len(useful),
        selected_useful=any(r["selected"] for r in useful),
        missed=not any(r["selected"] for r in useful) if useful else None,
        harmful_admissions=sum(r["selected"] and r["harmful"] for r in rows),
    )


def feasibility(n: int, plus: int, minus: int, attempt: int) -> Json:
    """Fixed-size bounds describe funding limits, never certify this dependent stream."""
    alpha = 6 * 0.05 / (math.pi**2 * attempt**2)
    comparisons = 15  # Three obligations for each of five committed alternatives.
    tail = alpha / (4 * comparisons)

    def interval(k: int) -> tuple[float, float]:
        return (
            float(beta.ppf(tail, k, n - k + 1)) if k else 0.0,
            float(beta.ppf(1 - tail, k + 1, n - k)) if k < n else 1.0,
        )

    lower = upper = radius = None
    if n:
        lp, up = interval(plus)
        lm, um = interval(minus)
        lower, upper = lp - um, up - lm
        radius = math.sqrt(2 * math.log(2 * comparisons / alpha) / n)
    return dict(
        independent_source_count=n,
        positive_disagreements=plus,
        negative_disagreements=minus,
        attempt=attempt,
        alpha_spent=alpha,
        tail_alpha=tail,
        simultaneous_comparisons=comparisons,
        paired_lower=lower,
        paired_upper=upper,
        zero_disagreement_upper=1 - tail ** (1 / n) if n else None,
        paired_margin_feasible=bool(n and 1 - tail ** (1 / n) <= 0.02),
        brier_radius=radius,
        cost_radius=5 * radius if radius is not None else None,
        brier_margin_feasible=bool(radius is not None and radius <= 0.01),
        cost_margin_feasible=bool(radius is not None and 5 * radius <= 0.02),
        minimum_paired_rows=math.ceil(math.log(tail) / math.log(0.98)),
        minimum_brier_rows=math.ceil(2 * math.log(2 * comparisons / alpha) / 0.01**2),
        minimum_cost_rows=math.ceil(50 * math.log(2 * comparisons / alpha) / 0.02**2),
        iid_certificate_valid=False,
        scope="hypothetical iid fixed-size feasibility only",
    )


def empty_plan() -> Json:
    """Missing evidence keeps explicit empty populations instead of invented observations."""
    return dict(
        failures=[],
        source_artifact_hashes=[],
        candidates=[],
        rows=[],
        candidate_pool_rows=[],
        harmful_admission_rows=[],
        certificate_feasibility_rows=[],
        validation_manifest=[],
        repository_health=[],
        guard_reuse_rows=[],
    )


def collect(root: Path, raw: Path, *, mutate: bool = False) -> Json:
    """Authenticate primitives and replay their chronology before sealing alternatives."""
    plan = empty_plan()

    def bind(path: Path, expected: str | None = None) -> None:
        observed = sha256_file(path) if path.is_file() else None
        if observed is None or (expected is not None and observed != expected):
            plan["failures"].append(
                dict(
                    check="input_authentication",
                    upstream=path.stem,
                    path=str(path),
                    hash=observed,
                    field="sha256" if observed else "resource_exists",
                    op="==",
                    expected=expected if observed else True,
                    observed=observed if observed else False,
                )
            )
        else:
            plan["source_artifact_hashes"].append(dict(path=str(path), sha256=observed))

    progress("preconditions_before")
    for label in [*prior.INPUTS, "python/carnot/verify/feedback_constrained_8051.py", MODULE, CLI]:
        bind(root / label)
    values = []
    for name, digest in PARENTS.items():
        path = root / "results" / (name + ".json")
        bind(path, "mutated" if mutate else digest)
        if plan["failures"]:
            continue
        value = json.loads(path.read_text())
        sidecar = Path(value["terminal_validation_sidecar_path"])
        bind(sidecar)
        if plan["failures"]:
            continue
        publication = json.loads(sidecar.read_text())["publication"]
        bound = Path(publication["sidecar_path"])
        bind(bound)
        if plan["failures"]:
            continue
        try:
            report = read_bound_sidecar(path, bound)
            audit.equal("terminal_report.passed", True, report["report"]["passed"])
            audit.equal("flagged_adversarial", False, value["flagged_adversarial"])
            audit.equal("publication.primary_sha256", digest, publication["primary_sha256"])
        except (ValueError, KeyError) as error:
            operand = getattr(error, "operand", {})
            plan["failures"].append(
                dict(
                    check="terminal_authentication",
                    upstream=name,
                    path=str(bound),
                    hash=sha256_file(bound),
                    field=operand.get("artifact_field", "terminal_report.passed"),
                    op="==",
                    expected=operand.get("expected", True),
                    observed=operand.get("observed", str(error)),
                )
            )
            continue
        for ref in value.get("raw_shard_hashes", []):
            bind(Path(ref["path"]), ref["sha256"])
        values.append(value)
    progress("preconditions_after", len(plan["source_artifact_hashes"]), len(plan["failures"]))
    if plan["failures"]:
        return plan
    learner, historical, sealed = values
    plan["repository_health"] = sealed["repository_health"]
    bundle = json.loads(audit.checked(historical["audit_bundle"]).read_text())
    labels = {
        r["family_id"]: r["eligible_y"]
        for r in json.loads(audit.checked(bundle["target_reference"]).read_text())["rows"]
    }
    progress("historical_reconstruction_before")
    first = audit.replay(Path(learner["trajectory_directory"]), labels)
    second = audit.replay(Path(bundle["trajectory"]), labels)
    audit.equal("historical_copy_replay", first, second)
    audit.equal("historical_original_predictions", historical["rows"], first["rows"])
    progress("historical_reconstruction_after", len(first["rows"]), 0)
    sources = bundle["sources"]
    head = bundle["head"]
    public = []
    for role, rows in [("later", sources), ("retention", bundle["retention_public"])]:
        for slot, r in enumerate(rows):
            public.append(
                dict(
                    role=role,
                    slot=slot,
                    unit=r["family_id"],
                    source=r["source_cluster_id"],
                    vector=audit.math.design(head, r).tolist() if r["public_eligible"] else None,
                    in_later_mask=slot in historical["later_support"]["eligible_slots"]
                    if role == "later"
                    else False,
                )
            )
    for seed in bundle["seeds"]:
        db = sqlite3.connect(
            f"file:{Path(learner['trajectory_directory']) / f'seed-{seed}' / 'ledger.sqlite'}?mode=ro",
            uri=True,
        )
        events = db.execute("select seq,kind,identity,payload from events order by seq").fetchall()
        db.close()
        attempts: Json = {}
        for seq, kind, identity, payload in events:
            if kind != "acceptance":
                continue
            r = json.loads(payload)
            arm = r["arm"]
            attempts[arm] = attempts.get(arm, 0) + 1
            guard_ids = r["guard_ids"]
            diagnostics = {d["alpha"]: d for d in r["diagnostics"]}
            for alpha in [1, 0.5, 0.25, 0.125, 0]:
                before = np.asarray(r["before_coefficients"])
                theta = before + alpha * (np.asarray(r["proposed_coefficients"]) - before)
                plan["candidates"].append(
                    dict(
                        pool=identity,
                        seq=seq,
                        seed=seed,
                        arm=arm,
                        slot=r["slot"],
                        opportunity=r["reason"],
                        alpha=alpha,
                        coefficients=theta.tolist(),
                        incumbent=before.tolist(),
                        selected=alpha == r["alpha"],
                        reset=r["reset"],
                        guard_admissible=diagnostics.get(alpha, {}).get("admissible"),
                        guard_count=len(set(guard_ids)),
                        head_hash=r["head_hash"],
                    )
                )
            diag = diagnostics.get(r["alpha"], diagnostics.get(1))
            plus = (
                len(
                    set(diag["candidate"]["false_accepts"]) - set(diag["baseline"]["false_accepts"])
                )
                if diag
                else 0
            )
            minus = (
                len(
                    set(diag["baseline"]["false_accepts"]) - set(diag["candidate"]["false_accepts"])
                )
                if diag
                else 0
            )
            plan["certificate_feasibility_rows"].append(
                dict(
                    pool=identity,
                    arm=arm,
                    seed=seed,
                    evidence="reused_exposed_guard",
                    **feasibility(len(set(guard_ids)), plus, minus, attempts[arm]),
                )
            )
            plan["guard_reuse_rows"].append(
                dict(
                    pool=identity,
                    seed=seed,
                    arm=arm,
                    guard_ids=guard_ids,
                    guard_count=len(guard_ids),
                    selected_alpha=r["alpha"],
                    reset=r["reset"],
                    opportunity=r["reason"],
                )
            )
        progress("candidate_seed_sealed", seed - 100, 120 - seed)
    fresh_sources = [
        r
        for r in sealed["rows"]
        if r["role"] == "stream" and r["eligible"] and r["feedback_role"] == "admission"
    ]
    for attempt, slot in enumerate([64, 128, 192], 1):
        available = [r for r in fresh_sources if slot < r["release_slot"] < min(slot + 64, 256)]
        plan["certificate_feasibility_rows"].append(
            dict(
                evidence="future_fresh_block_optimistic_zero_disagreement",
                commitment_slot=slot,
                available_source_count=len(available),
                **feasibility(min(12, len(available)), 0, 0, attempt),
            )
        )
    plan["historical_closed_loop"] = dict(
        hypotheses=historical["primary_hypothesis_results"],
        retention_passed=historical["retention_passed"],
        sample_size_budget=historical["sample_size_budget"],
        scope="authenticated historical trajectories; alternative scores do not simulate changed future proposals",
    )
    plan["sealed_method_sha256"] = sealed["method_freeze"]["sha256"]
    stream_ids = {r["source_cluster_id"] for r in sources}
    retained_ids = {r["source_cluster_id"] for r in bundle["retention_public"]}
    plan["source_overlap"] = dict(
        stream_retention_overlap=sorted(stream_ids & retained_ids),
        original_stream_sources=len(stream_ids),
        original_retention_sources=len(retained_ids),
        later_source_count=historical["later_support"]["independent_count"],
        retention_source_count=historical["retention_support"]["independent_count"],
        algorithm_seed_independence=False,
        independent_environments=1,
    )
    atomic_json(
        raw / "committed_candidates.json",
        dict(
            head=head,
            candidates=plan.pop("candidates"),
            public=public,
            stream_target=bundle["target_reference"],
            retention_target=bundle["retention_target"],
            contract=dict(
                cost_gain=0.02,
                incumbent_brier_nonincrease=True,
                later_no_added_false_accepts=True,
                retention_brier_drift=0.01,
                retention_cost_drift=0.02,
            ),
        ),
    )
    progress("alternatives_committed_before_evaluator")
    return plan


def evaluate(raw: Path, *, cold: bool = False) -> Json:
    """Read targets only after candidate commitment; persist every probability operand."""
    progress("isolated_evaluator_before")
    seal = json.loads((raw / "committed_candidates.json").read_text())
    labels = {}
    for role, ref in [("later", seal["stream_target"]), ("retention", seal["retention_target"])]:
        labels[role] = {
            r["family_id"]: r["eligible_y"]
            for r in json.loads(audit.checked(ref).read_text())["rows"]
        }
    observations = [dict(r, y=labels[r["role"]][r["unit"]]) for r in seal["public"]]
    eligible = [r for r in observations if r["vector"] is not None and r["y"] is not None]
    x = np.asarray([r["vector"] for r in eligible])
    y = np.asarray([r["y"] for r in eligible])
    a, b = seal["head"]["calibration"]
    initial = np.asarray(seal["head"]["parameters"]) * seal["head"]["decay_scale"]

    def operands(parameters: Any) -> tuple[Any, Any, Any]:
        ps = audit.math.expit(a + b * (x @ parameters))
        costs = np.where(ps < 0.1, 5 * y, np.where(ps > 0.5, 1 - y, 0.5))
        return ps, costs, (ps < 0.1) & (y == 1)

    p0, c0, f0 = operands(initial)
    metrics, probabilities, pools = [], [], {}
    for index, r in enumerate(seal["candidates"]):
        pc, cc, fc = operands(np.asarray(r["coefficients"]))
        pi, ci, fi = operands(np.asarray(r["incumbent"]))
        later = np.asarray(
            [
                z["role"] == "later" and z["in_later_mask"] and z["slot"] > r["slot"]
                for z in eligible
            ]
        )
        retained = np.asarray([z["role"] == "retention" for z in eligible])
        n = int(sum(later))
        gain = float(np.mean((ci - cc)[later])) if n else None
        # Require rowwise safety against each reference, not their union.
        added = int(sum((fc & (~fi | ~f0))[later]))
        brier = float(np.mean(((pc - y) ** 2 - (pi - y) ** 2)[later])) if n else None
        rd = float(np.mean((cc - c0)[retained]))
        rb = float(np.mean(((pc - y) ** 2 - (p0 - y) ** 2)[retained]))
        harmful = bool(n and (gain < 0 or added or rd > 0.02 or rb > 0.01))
        useful = bool(
            n and gain >= 0.02 and added == 0 and brier <= 1e-12 and rd <= 0.02 and rb <= 0.01
        )
        metric = dict(
            pool=r["pool"],
            arm=r["arm"],
            seed=r["seed"],
            alpha=r["alpha"],
            opportunity=r["opportunity"],
            slot=r["slot"],
            selected=r["selected"],
            useful=useful,
            harmful=harmful,
            guard_admissible=r["guard_admissible"],
            later_count=n,
            retention_count=int(sum(retained)),
            cost_gain=gain,
            brier_drift=brier,
            added_false_accepts=added,
            retention_cost_drift=rd,
            retention_brier_drift=rb,
            probability_row=index,
            candidate_sha256=canonical_hash(r["coefficients"]),
            exclusion_reason=None if n else "no_later_rows",
        )
        metrics.append(metric)
        probabilities.append(pc)
        pools.setdefault(r["pool"], []).append(metric)
        if index % 500 == 0:
            progress("alternatives_scored", index + 1, len(seal["candidates"]) - index - 1)
    rows = []
    for identity, alternatives in pools.items():
        r = alternatives[0]
        summary = pool_summary(alternatives)
        rows.append(
            dict(
                unit=identity,
                source="historically_exposed_stream",
                arm=r["arm"],
                seed=r["seed"],
                slot=r["slot"],
                opportunity=r["opportunity"],
                numerator=int(summary["missed"] is True),
                denominator=int(summary["missed"] is not None),
                status="completed",
                exclusion_reason=None,
                later_count=r["later_count"],
                **summary,
            )
        )
    result = dict(
        candidate_pool_rows=rows,
        rows=rows,
        harmful_admission_rows=[r for r in metrics if r["selected"] and r["harmful"]],
        alternative_metric_rows=metrics,
    )
    if cold:
        audit.equal(
            "probability_operands",
            True,
            np.array_equal(
                np.asarray(probabilities), np.load(raw / "probabilities.npz")["probabilities"]
            ),
        )
        audit.equal(
            "evaluation_reduction", result, json.loads((raw / "evaluation.json").read_text())
        )
    else:
        atomic_json(raw / "observations.json", dict(rows=observations))
        np.savez_compressed(raw / "probabilities.npz", probabilities=np.asarray(probabilities))
        atomic_json(raw / "evaluation.json", result)
    progress("isolated_evaluator_after", len(metrics), 0)
    return result


def build(
    plan: Json, raw: Path, receipts: list[Json], coverage: Json, fixture: bool, duration: float
) -> Json:
    """Readiness means complete owned diagnosis; empirical gain does not control it."""
    complete = (
        bool(plan["validation_manifest"])
        and [r["name"] for r in receipts] == plan["validation_manifest"]
    )
    checks = (
        complete
        and all(r["passed"] for r in receipts)
        and all(
            coverage.get(p, {}).get("summary", {}).get("num_statements", 0) > 0
            and coverage[p]["summary"]["missing_lines"] == 0
            for p in OWNED
        )
    )
    failures = deepcopy(plan["failures"])
    if not fixture and not checks and not failures:
        failures.append(
            dict(
                check="required_validation",
                upstream="exp8063",
                path=str(raw / "validation.json"),
                hash=None,
                field="required_checks_passed",
                op="==",
                expected=True,
                observed=False,
            )
        )
    kind = "blocked" if plan["failures"] else "null" if checks or fixture else "disqualified"
    rows = plan["candidate_pool_rows"]
    categories = []
    for arm in sorted({r["arm"] for r in rows}):
        for opportunity in ["gradient", "guard_expansion"]:
            group = [
                r for r in rows if r["arm"] == arm and r["opportunity"].startswith(opportunity)
            ]
            categories.append(
                dict(
                    arm=arm,
                    opportunity=opportunity,
                    pool_count=len(group),
                    useful_pool_count=sum(r["denominator"] for r in group),
                    missed_pool_count=sum(r["numerator"] for r in group),
                    weak_pool_count=sum(r["useful_count"] == 0 for r in group),
                    harmful_admission_count=sum(r["harmful_admissions"] for r in group),
                )
            )
    numerator = sum(r["numerator"] for r in rows if r["arm"] == "feedback_constrained")
    denominator = sum(r["denominator"] for r in rows if r["arm"] == "feedback_constrained")
    receipt = build_current_work_receipt(
        run_id="exp8063",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(retrospective=True),
        inference_substrate_class="no_model_load",
        execution_venue="cpu",
        started_monotonic_ns=0,
        ended_monotonic_ns=int(duration * 1e9),
        phase_spans=[dict(name="authenticated_audit_validation", duration_s=duration)],
    )
    historical = plan.get("historical_closed_loop", {}).get("sample_size_budget", {})
    counts = {
        k + "_count": historical.get(k, 0)
        for k in [
            "intended",
            "eligible",
            "independent",
            "completed",
            "censored",
            "excluded",
            "failed",
        ]
    }
    value = dict(
        schema="carnot.v698.admission_opportunity_audit.v1",
        experiment_id=8063,
        task_id="exp8063-admission-opportunity-audit",
        run_date="20261003",
        milestone="2026.10.698",
        honest_verdict="complete_"
        + kind
        + "_"
        + (str(failures[0]["field"]) + "_" if kind == "blocked" else "")
        + "admission_opportunity_audit",
        verdict_class=kind,
        verifier_is_oracle=False,
        flagged_adversarial=False,
        required_checks_passed=bool(checks),
        admission_audit_ready_score=int(checks and kind == "null" and not fixture),
        claim_scope="Finite retrospective common-candidate diagnosis on historically exposed text. Closed-loop causality, iid safety and generalized benefit are unsupported. No Exp8064 tuning.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        substrate_declaration=dict(
            mode="aggregation_from_upstream_artifacts", reduction="no_model_load", MODEL_SPECS=[]
        ),
        model_invocation_counts=receipt["invocation_counts"],
        current_work_receipt=receipt,
        duration_s=duration,
        random_seed=6988063,
        generalized_learning_benefit_score=0,
        iid_assumptions_satisfied=False,
        validation_receipts=receipts,
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        gate_check_summary=failures,
        source_artifact_hashes=plan["source_artifact_hashes"],
        code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
        raw_shard_hashes=[
            dict(path=str(p), sha256=sha256_file(p))
            for p in sorted(raw.iterdir())
            if p.is_file()
            and p.name
            not in {"terminal_candidate.json", "terminal_validation.json", "publication.lock"}
        ],
        phase_spans=receipt["phase_spans"],
        reproducibility_checksum=canonical_hash(plan),
        rows=plan["rows"],
        candidate_pool_rows=rows,
        harmful_admission_rows=plan["harmful_admission_rows"],
        missed_opportunity_numerator=numerator,
        missed_opportunity_denominator=denominator,
        missed_opportunity_rate=ratio(numerator, denominator),
        certificate_feasibility_rows=plan["certificate_feasibility_rows"],
        sample_size_budget=dict(
            **counts,
            algorithm_seeds=20,
            independent_environments=1 if historical else 0,
            seeds_independent=False,
            audit_pool_count=len(rows),
            model_calls=0,
        ),
        **counts,
        guard_reuse_rows=plan["guard_reuse_rows"],
        pool_category_rows=categories,
        source_overlap=plan.get("source_overlap"),
        audit_contract=dict(
            cost_gain=0.02,
            later_brier_nonincrease=True,
            later_false_accept_increase=0,
            retention_brier_drift=0.01,
            retention_cost_drift=0.02,
            scope="Empirical fixed-alternative contract; does not amend V698 H3 acceptance or Exp8064.",
        ),
        historical_closed_loop=plan.get("historical_closed_loop"),
        sealed_method_sha256=plan.get("sealed_method_sha256"),
        repository_health=plan["repository_health"],
        diagnosis=dict(
            candidate_quality="Useful pool support and harmful selections are empirical finite-cohort findings.",
            validation_reuse="Growing guard snapshots are reused across attempts; guard acceptance is not later safety.",
            evidence_shortage="Counts, dependence and exposure prevent iid certification; seeds add no environments.",
            causal_limit="Arm-specific proposals arise from different historical incumbents; no closed-loop winner is inferred.",
        ),
        methodology_note="Pool controls test counting only. The diagnostic cost margin .02 and retention drift .01/.02 are fixed; population guarantees remain absent.",
    )
    principles = {
        "candidate_pool_rows": "One selected useful alternative prevents a miss regardless of pool size.",
        "missed_opportunity_denominator": "Only pools with a useful audit-contract candidate enter this denominator.",
        "certificate_feasibility_rows": "Hypothetical iid funding calculations cannot certify exposed dependent observations.",
        "guard_reuse_rows": "Repeated IDs reveal reuse without inflating independent sources.",
        "historical_closed_loop": "Original trajectory outcomes are distinct from fixed alternative evaluations.",
        "admission_audit_ready_score": "Owned checks qualify the diagnosis independently of scientific benefit.",
        "repository_health": "Preserved full-suite failures are health evidence outside current scientific acceptance.",
        "current_work_receipt": "Current counters cannot absorb cited historical model work.",
        "sample_size_budget": "Unique sources, repetitions and environments have distinct denominators.",
        "source_overlap": "Repeated exposure and overlapping candidate evaluations do not create new independent observations.",
        "pool_category_rows": "Guard expansions and gradient proposals have different opportunity denominators.",
        "audit_contract": "Empirical alternative usefulness cannot replace the preregistered trajectory hypothesis.",
    }
    value["field_principles"] = {
        k: principles.get(
            k,
            "Preserve the exact audit operand and its scope so readers can detect changed or missing evidence.",
        )
        for k in value
    }
    return value


def replay(path: Path) -> bool:
    """Cold recomputation rejects edited rows, sources, logs and reduction counters."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        plan = json.loads((raw / "plan.json").read_text())
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            audit.equal("source_hash", ref["sha256"], sha256_file(Path(ref["path"])))
        for r in value["validation_receipts"]:
            audit.equal("validation_log", r["log_sha256"], sha256_file(Path(r["log_path"])))
        for p, digest in value["code_config_hashes"].items():
            audit.equal("code_hash", digest, sha256_file(ROOT / p))
        for r in plan["certificate_feasibility_rows"]:
            recalculated = feasibility(
                r["independent_source_count"],
                r["positive_disagreements"],
                r["negative_disagreements"],
                r["attempt"],
            )
            audit.equal("fixed_size_bounds", recalculated, {k: r[k] for k in recalculated})
        if not plan["failures"]:
            evaluated = evaluate(raw, cold=True)
            audit.equal(
                "pool_reduction", plan["candidate_pool_rows"], evaluated["candidate_pool_rows"]
            )
        validation = json.loads((raw / "validation.json").read_text())
        work = json.loads((raw / "work.json").read_text())
        rebuilt = build(
            plan,
            raw,
            validation["receipts"],
            validation["coverage"],
            work["fixture"],
            value["duration_s"],
        )
        for field in [
            "candidate_pool_rows",
            "harmful_admission_rows",
            "missed_opportunity_numerator",
            "missed_opportunity_denominator",
            "certificate_feasibility_rows",
            "admission_audit_ready_score",
            "verdict_class",
            "sample_size_budget",
            "reproducibility_checksum",
        ]:
            audit.equal(field, value[field], rebuilt[field])
        return True
    except (ValueError, KeyError, TypeError, OSError):
        return False


def manifest(private: Path) -> list[Json]:
    """Reuse bounded check infrastructure while restricting coverage to this task."""
    commands = prior.manifest(private)
    mapping = {prior.MODULE: MODULE, prior.CLI: CLI, prior.TEST: TEST}
    for spec in commands:
        spec["argv"] = [replace_paths(item, mapping) for item in spec["argv"]]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    return commands


def replace_paths(item: str, mapping: dict[str, str]) -> str:
    """Keep inherited validation argv exact except for explicitly owned paths."""
    for before, after in mapping.items():
        item = item.replace(before, after)
    return item


def terminal_commands(path: Path) -> list[Json]:
    """Freeze independent reader commands against the exact candidate pathname."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    return [dict(name=n, argv=a, deadline_s=120, expected_exit=0) for n, a in commands]


def terminal(path: Path) -> Json:
    """Publish only after independent reductions and shared readers exit normally."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    with tempfile.TemporaryDirectory(prefix="carnot-8063-terminal-") as temp:
        receipts = [
            run_check(ROOT, s, Path(temp), raw / "terminal_logs") for s in terminal_commands(path)
        ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Keep private test runs isolated and preserve already sealed report bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic_ns()
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    if args.evaluate:
        evaluate(args.evaluate)
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if (raw / "plan.json").exists():
        progress("existing_seal_preserved")
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="carnot-8063-") as temp:
            private = Path(temp)
            specs = manifest(private)
            evaluator = dict(
                name="isolated_evaluator",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--evaluate",
                    str(raw),
                ],
                deadline_s=180,
                expected_exit=0,
            )
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=specs,
                    evaluator=evaluator,
                    terminal_commands=terminal_commands(raw / "terminal_candidate.json"),
                    code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
                ),
            )
            plan = collect(args.root, raw, mutate=args.mutate)
            evaluation_receipts = []
            if not plan["failures"]:
                progress("evaluator_subprocess_before")
                evaluation_receipts.append(
                    run_check(ROOT, evaluator, private, raw / "evaluator_logs")
                )
                progress("evaluator_subprocess_after", int(evaluation_receipts[0]["passed"]), 0)
                if evaluation_receipts[0]["passed"]:
                    result = json.loads((raw / "evaluation.json").read_text())
                    plan.update(
                        {
                            k: result[k]
                            for k in ["rows", "candidate_pool_rows", "harmful_admission_rows"]
                        }
                    )
            plan["validation_manifest"] = (
                ["isolated_evaluator"] if evaluation_receipts else []
            ) + [s["name"] for s in specs]
            atomic_json(raw / "plan.json", plan)
            progress("owned_validation_before", 0, len(specs))
            os.environ["CARNOT_8063_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            receipts = evaluation_receipts + (
                []
                if args.fixture_output
                else [run_check(ROOT, s, private, raw / "validation_logs") for s in specs]
            )
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            duration = (time.monotonic_ns() - started) / 1e9
            atomic_json(raw / "validation.json", dict(receipts=receipts, coverage=coverage))
            atomic_json(raw / "work.json", dict(fixture=bool(args.fixture_output)))
            progress("owned_validation_after", len(receipts), 0)
            value = build(plan, raw, receipts, coverage, bool(args.fixture_output), duration)
            progress("publication_before", len(value["rows"]), 1)
            publication = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, owned_invocation_exit=0),
            )
            progress("complete", len(value["rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("rejected_" + str(error))
        return 1
