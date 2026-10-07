"""REQ-REPORT-8237: expose fit mechanics while development benefit stays unclaimed.

Original input snapshots define the training roles and acceptance permissions.
Neither a successful fit nor a development cost change establishes generalization.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

from carnot.reporting import decision_margin_methods_8234 as methods
from carnot.reporting.current_work_receipt import canonical_hash, atomic_json, sha256_file
from carnot.reporting.v710_contract_replay import snapshot
from carnot.verify import margin_energy_training_8237 as n
from carnot.verify import utility_fit_8222 as historical
from carnot.verify import utility_kernel_8221 as kernel

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8237_v712_margin_energy_training"
TASK = "exp8237-margin-energy-training"
MILESTONE = methods.MILESTONE
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_margin_energy_training_8237.py"
OWNED = [
    "python/carnot/verify/margin_energy_training_8237.py",
    "python/carnot/reporting/margin_energy_training_8237.py",
    "python/carnot/reporting/margin_energy_runner_8237.py",
    CLI,
]
METHODS = "results/experiment_8234_v712_decision_margin_methods.json"
METHODS_PIN = "sha256:018d57444d13ec1402de739b5141bc7df57a6edc4004e097c9caa31968418f37"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts show that a bounded fitting or validation phase is still active."""
    print(f"[exp8237] phase={phase} completed={completed} pending={pending}", flush=True)


def original(work: Json, ref: Json) -> Json:
    """Read the authenticated copy so concurrent upstream writes cannot change a fit."""
    key = str(Path(work["root"]) / Path(ref["path"]).relative_to(ROOT))
    saved = next(r for r in work["refs"] if r["path"] == key)
    value: Json = json.loads(Path(saved["snapshot_path"]).read_bytes())
    return value


def global_rows(work: Json) -> tuple[Json, Json]:
    """The historical global correction remains frozen while new heads learn weights."""
    evidence = original(work, work["protocol"]["fit_measurement"])["evidence"]
    model = original(work, work["protocol"]["v711_energy_global_measurement"])["models"][
        work["protocol"]["v711_energy_global_model_key"]
    ]
    rows = historical.base_rows(evidence, "energy")
    return {r["unit_id"]: kernel.predict(model, r) for r in rows}, dict(
        base_head=next(h for h in evidence["heads"] if h["arm"] == "energy"), correction=model
    )


def measure(root: Path, raw: Path) -> Json:
    """External absence blocks fitting; numerical failures retain their actual evidence."""
    with patch.object(methods, "OWNED", OWNED):
        work: Json = methods.measure(root, raw)
    work.update(
        root=str(root),
        fitted={},
        owned_errors=[],
        numerical_checks=n.numerical_checks(),
        global_probabilities={},
        global_head={},
    )
    ref = snapshot(root / METHODS, raw / "inputs", "methods")
    work["refs"].append(ref)
    if ref["sha256"] != METHODS_PIN:
        work["failures"].append(
            methods.failure(root / METHODS, "sha256", METHODS_PIN, ref["sha256"], ref["sha256"])
        )
    else:
        value = json.loads(Path(ref["snapshot_path"]).read_bytes())
        if value.get("margin_protocol_ready_score") != 1:
            work["failures"].append(
                methods.failure(
                    root / METHODS,
                    "margin_protocol_ready_score",
                    1,
                    value.get("margin_protocol_ready_score"),
                    ref["sha256"],
                )
            )
    progress("before_benchmark_training", 0, 6)
    if not work["failures"]:
        try:
            if not work["numerical_checks"]["passed"]:
                raise ValueError("numerical_checks")
            work["global_probabilities"], work["global_head"] = global_rows(work)
            development = [r for r in work["public_rows"] if r["role"] != "reserved"]
            work["fitted"] = n.fit(
                development, work["protocol"]["role_manifest"], raw / "checkpoints"
            )
        except (ValueError, TimeoutError) as error:
            work["owned_errors"].append(str(error))
    progress(
        "after_benchmark_training",
        len(work["fitted"].get("heads", [])),
        6 - len(work["fitted"].get("heads", [])),
    )
    work["trained_heads_path"] = str(raw / "trained_heads.json")
    atomic_json(
        Path(work["trained_heads_path"]),
        dict(
            schema="carnot.v712.margin-heads.v1",
            heads=work["fitted"].get("heads", []),
            global_head=work["global_head"],
            scoring_api="carnot.verify.margin_energy_training_8237.score",
            protocol_sha256=methods.PIN,
            labels_opened=False,
        ),
    )
    work["trained_heads_sha256"] = sha256_file(Path(work["trained_heads_path"]))
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Validation gates readiness; scientific gains remain outside this producer's scope."""
    value: Json = methods.reduce(work, receipts)
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_errors"]
    fitted = work["fitted"]
    heads = fitted.get("heads", [])
    rows = []
    scores = []
    for row in work["public_rows"]:
        if row["role"] == "reserved":
            continue
        for head in heads:
            prediction = n.score(head, {k: row[k] for k in ["x", "p0", "baseline_action"]})
            complete = prediction["p"] is not None and row["y"] in (0, 1)
            record = dict(
                prediction,
                unit_id=row["unit_id"],
                source_cluster_id=row["source_cluster_id"],
                arm=head["arm"],
                condition=row["role"],
                role=row["role"],
                y=row["y"],
                baseline_action=row["baseline_action"],
                status="completed" if complete else "excluded",
                missing_status=not complete,
                completed=complete,
                failed=False,
                censored=False,
                excluded=not complete,
                metric="development_typed_cost",
                numerator=kernel.rule.base.loss(prediction["action"], row["y"])
                if row["y"] in (0, 1)
                else None,
                denominator=1,
                brier=(prediction["p"] - row["y"]) ** 2 if complete else None,
                effective_independent_groups=1 if complete else 0,
            )
            rows.append(record)
            if row["role"] == "calibration" and head["arm"] in methods.n.COMPARATORS:
                scores.append(record)
        if heads and row["role"] == "calibration":
            p = work["global_probabilities"][row["unit_id"]]
            scores.append(
                dict(
                    row,
                    arm="energy_global",
                    p=p,
                    action=kernel.rule.action(p, row["baseline_action"]),
                )
            )
    if not heads:
        rows = [
            dict(
                unit_id=slot["unit_id"],
                source_cluster_id=slot["source_cluster_id"],
                arm=arm,
                condition=role,
                role=role,
                status="censored",
                missing_status=True,
                completed=False,
                failed=False,
                censored=True,
                excluded=False,
                metric="development_typed_cost",
                numerator=None,
                denominator=1,
                effective_independent_groups=0,
                energy_probability_error=None,
            )
            for role in ["head_fit", "temperature_fit", "calibration"]
            for slot in work["protocol"]["role_manifest"][role]
            for arm in work["protocol"]["arms"]
        ]
    comparator = (
        methods.n.select_comparator(scores, work["protocol"]["role_manifest"]) if heads else {}
    )
    chosen = next((h for h in heads if h["arm"] == comparator.get("arm")), work["global_head"])
    parity = max((r["energy_probability_error"] for r in rows), default=0.0) if heads else None
    ready = (
        owned
        and not work["failures"]
        and len(heads) == 6
        and parity is not None
        and parity <= 1e-10
    )
    verdict = "disqualified" if not owned else "blocked" if work["failures"] else "null"
    suffix = "required_checks" if not owned else "margin_energy_training"
    if verdict == "blocked":
        f = work["failures"][0]
        suffix = re.sub(
            "[^a-z0-9_]",
            "_",
            (Path(f.get("path", f.get("artifact_path"))).stem + "_" + f["artifact_field"]).lower(),
        )
    value.update(
        experiment_id=8237,
        task_id=TASK,
        schema="carnot.v712.margin-energy-training.v1",
        honest_verdict="complete_" + verdict + "_" + suffix,
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        owned_errors=work["owned_errors"],
        rows=rows,
        intended_count=192 * 6,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=0,
        censored_count=0 if heads else 192 * 6,
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=len({r["source_cluster_id"] for r in rows if r["completed"]}),
        required_checks_passed=owned,
        margin_fit_ready_score=int(ready),
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if heads
        else "aggregation_from_upstream_artifacts",
        trained_head_specs=[
            dict(arm=a, coefficients=17, generator=False, fitted_here=bool(heads))
            for a in work["protocol"]["arms"]
        ],
        trained_heads_path=work["trained_heads_path"],
        trained_heads_sha256=work["trained_heads_sha256"],
        fit_fold_rows=fitted.get("fit_fold_rows", []),
        optimizer_receipts=fitted.get("optimizer_receipts", []),
        margin_weight_rows=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                role=r["role"],
                p0=r["p0"],
                baseline_action=r["baseline_action"],
                weight=r["weight"],
                status=r["status"],
            )
            for r in work["public_rows"]
        ],
        comparator_name=comparator.get("arm"),
        comparator_sha256=canonical_hash(chosen) if chosen else None,
        comparator_selection=comparator,
        energy_probability_error=parity,
        numerical_checks=work["numerical_checks"],
        equally_weighted_simple_comparisons=[
            s
            for s in comparator.get("candidates", [])
            if s["arm"] in ["additive_margin", "logistic_margin"]
        ],
        acceptance_gates=dict(
            owned_checks=dict(passed=owned),
            authenticated_methods=dict(passed=not work["failures"]),
            frozen_fits=dict(passed=bool(ready)),
        ),
        methodology_note="Six matched small heads fit original development labels. Native probabilities alone define margin weights. Separate unweighted temperature calibration and calibration-role cost/Brier select frozen heads. No reserved labels, generator calls, utility patches or independent benefit measurement occur. Zero development gains do not block valid fitting.",
        labels_opened=False,
        generator_weight_updates=0,
    )
    value.pop("current_contract_ready_score", None)
    return value
