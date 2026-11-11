"""REQ-REPORT-8987-ADMISSION: expose development gates without claiming generalization."""

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from carnot.reporting import spline_basis_methods_8986 as base
from carnot.reporting import shared_feature_methods_8973 as historical
from carnot.reporting import relational_methods_8959 as graphs
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
n, ROOT = base.n, base.ROOT
NAME, TASK, SEED = (
    "experiment_8987_v768_spline_development_admission",
    "exp8987-spline-development-admission",
    768087,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/spline_development_admission_8987.py",
    "python/carnot/reporting/spline_development_runner_8987.py",
    CLI,
]
TESTS = ["tests/python/test_spline_development_admission_8987.py"]
INPUT = "results/experiment_8986_v768_spline_basis_methods.json"
INPUT_PIN = "sha256:bf91b5f34041dcb903d48c477920836d1c98a17e381bf71c7dd19b8d7eaaee8c"
reference, retain, seal, checksum, preflight = (
    base.reference,
    base.retain,
    base.seal,
    base.checksum,
    base.preflight,
)
THRESHOLDS = dict(
    auroc_min=0.65,
    prevalence_brier_gain_min=0.005,
    shared_logistic_brier_slack=0.01,
    converged_linear_brier_slack=0.01,
    temperature=1.0,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so a waiting operator can identify a stalled phase."""
    print(f"[exp8987] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Require the pinned numerical gate; retain historical rejection without retrying it."""
    evidence = base.authenticate(root, raw)
    path = root / INPUT
    observed = reference(path)["sha256"] or "missing_file"
    checks = [base.gate(path, "sha256", INPUT_PIN, observed, "spline_methods_bytes", "exp8986")]
    if checks[0]["passed"]:
        value = json.loads(path.read_bytes())
        checks += [
            base.gate(
                path,
                field,
                expected,
                value.get(field, "missing_field"),
                "spline_methods_gate",
                "exp8986",
            )
            for field, expected in [
                ("spline_methods_ready_score", 1),
                ("required_checks_passed", True),
            ]
        ]
        evidence["references"].append(retain(reference(path), raw))
        evidence["spline_methods"] = value
        for ref in value.get("code_config_hashes", []):
            if ref.get("executed_source"):
                current = Path(ref["source_path"])
                checks.append(
                    base.gate(
                        current,
                        "sha256",
                        ref["sha256"],
                        reference(current)["sha256"] or "missing_file",
                        "registered_code",
                        "exp8986",
                    )
                )
    context = evidence.get("context", {})
    for ref in context.get("raw_shard_hashes", []):
        actual = reference(Path(ref["path"]))["sha256"] or "missing_file"
        checks.append(
            base.gate(
                Path(ref["path"]), "sha256", ref["sha256"], actual, "historical_custody", "exp8973"
            )
        )
        if actual == ref["sha256"]:
            evidence["references"].append(retain(ref, raw))
    evidence["checks"].extend(checks)
    evidence["failures"].extend(r for r in checks if not r["passed"])
    return evidence


def traced_minimize(fun: Any, x0: Any, *, trace: list[Json] | None = None, **kwargs: Any) -> Any:
    """Record the initial state and each accepted optimizer iterate with its actual gradient."""
    rows = trace if trace is not None else []

    def record(coefficients: Any) -> None:
        loss, gradient = fun(coefficients, *kwargs["args"])
        rows.append(
            dict(
                iteration=len(rows),
                loss=loss,
                gradient_inf=float(np.abs(gradient).max()),
                gradient_l2=float(np.linalg.norm(gradient)),
                coefficients=coefficients.tolist(),
            )
        )
        if len(rows) % 25 == 0:
            progress("optimizer_iterations", len(rows) - 1, max(0, 500 - len(rows) + 1))

    record(x0)
    return minimize(fun, x0, callback=record, **kwargs)


def admission(rows: list[Json]) -> tuple[Json, list[Json]]:
    """Independently reduce graph losses with stdlib sums before applying immutable margins."""
    guard = [r for r in rows if r["condition"] == "exposed_development_guard"]
    if any(
        abs(r["value"] - (r["p_error"] - r["error"]) ** 2) > 1e-15 or r["numerator"] != r["value"]
        for r in guard
    ):
        raise ValueError("primitive_loss_drift")
    losses = {
        arm: math.fsum(r["value"] for r in guard if r["arm"] == arm)
        / sum(r["arm"] == arm for r in guard)
        for arm in base.ARMS
        if any(r["arm"] == arm for r in guard)
    }
    spline = [r for r in guard if r["arm"] == "spline"]
    metrics = dict(
        auroc=float(roc_auc_score([r["error"] for r in spline], [r["p_error"] for r in spline])),
        energy_brier=losses["spline"],
        prevalence_brier=losses["prevalence"],
        logistic_brier=losses["shared_logistic"],
        converged_linear_brier=losses["converged_linear"],
        arm_brier=losses,
    )
    gates = historical.admission(metrics)
    gates[0]["metric"] = "spline_auroc"
    value = losses["spline"] - losses["converged_linear"]
    gates.append(
        dict(
            metric="brier_excess_over_converged_linear",
            value=value,
            operator="<=",
            threshold=0.01,
            passed=value <= 0.01,
            temperature=1.0,
            independent_count=len(spline),
            scope="exposed_development_only",
        )
    )
    return metrics, gates


def evaluate(evidence: Json, raw: Path, replay: bool = False) -> Json:
    """Fit six registered convex heads once, then freeze a new source budget without generation."""
    traces: list[list[Json]] = []

    def fit(*args: Any, **kwargs: Any) -> Any:
        trace: list[Json] = []
        traces.append(trace)
        fitted = traced_minimize(*args, trace=trace, **kwargs)
        seal(raw / "optimizer_traces" / f"fit-{len(traces)}.json", trace, replay)
        return fitted

    with patch.object(n, "minimize", fit):
        result = base.evaluate(evidence, raw, replay)
    result["optimizer_rows"] = [
        dict(r, seed=h["seed"], arm=h["arm"])
        for h, trace in zip(result["head_rows"], traces)
        for r in trace
    ]
    result["development_metrics"], result["development_admission_rows"] = admission(result["rows"])
    result["development_metric_rows"] = [
        r for r in result["rows"] if r["condition"] == "exposed_development_guard"
    ]
    result["historical_head_metrics"] = evidence["context"]["development_metrics"]
    progress("before_graph_construction", 0, 1024)
    history = historical.history(evidence)
    seal(raw / "historical_graphs.json", history, replay)
    with patch.object(graphs, "SEED", SEED):
        construction = graphs.graph_protocol(history)
    for graph in construction["graph_manifest"]:
        graph["graph_id"] = graph["graph_id"].replace("v766", "v768")
        graph["questions"][0]["unit_id"] = graph["questions"][0]["unit_id"].replace("v766", "v768")
        graph["trained_state_sha256"] = result["trained_state_reference"]["sha256"]
    result["construction"] = construction
    result["graph_manifest"] = construction["graph_manifest"]
    result["role_manifest"] = [
        {
            k: g[k]
            for k in (
                "graph_id",
                "graph_hash",
                "shard",
                "role",
                "stream_id",
                "stream_position",
                "trained_state_sha256",
            )
        }
        | dict(unit_id=g["questions"][0]["unit_id"], query_hash=g["questions"][0]["prompt_hash"])
        for g in result["graph_manifest"]
    ]
    result["source_manifest"] = dict(
        seed=SEED,
        population=historical.protocol()["source_population"],
        graph_manifest_hash=canonical_hash(result["graph_manifest"]),
        role_manifest_hash=canonical_hash(result["role_manifest"]),
        historical_reference=reference(raw / "historical_graphs.json"),
        responses_generated=0,
        support_planning=dict(
            final_test=128,
            streams=24,
            queries_per_stream=32,
            precision="limited at128 final-test graphs; no statistical power claim",
        ),
    )
    for name in ("optimizer_rows", "graph_manifest", "role_manifest", "source_manifest"):
        seal(raw / (name + ".json"), result[name], replay)
    seal(raw / "admission.json", result, replay)
    progress(
        "after_graph_construction",
        len(result["graph_manifest"]),
        1024 - len(result["graph_manifest"]),
    )
    return result


def verify_result(evidence: Json, result: Json) -> None:
    """Cold replay checks actual stationarity and primitive reductions without refitting heads."""
    metrics, gates = admission(result["rows"])
    if metrics != result["development_metrics"] or gates != result["development_admission_rows"]:
        raise ValueError("reduction_drift")
    states = json.loads(Path(result["trained_state_reference"]["path"]).read_bytes())
    if states["heads"] != result["head_rows"]:
        raise ValueError("state_drift")
    predictions: Json = {arm: [] for arm in ("spline", "converged_linear")}
    for head in result["head_rows"]:
        ref = next(
            r
            for r in result["feature_references"]
            if r["seed"] == head["seed"] and r["role"] == "fit"
        )
        features = json.loads(Path(ref["path"]).read_bytes())["scaled"]
        phi = n.design(features, head["arm"] == "spline")
        y = np.asarray(evidence["work"]["evidence"]["views"]["fit"]["labels"], dtype=float)
        loss, gradient = n.objective(np.asarray(head["coefficients"]), phi, y)
        if abs(loss - head["loss"]) > 1e-12 or np.abs(gradient).max() > 1e-6:
            raise ValueError("state_stationarity")
        guard_ref = next(
            r
            for r in result["feature_references"]
            if r["seed"] == head["seed"] and r["role"] == "guard"
        )
        guard = json.loads(Path(guard_ref["path"]).read_bytes())
        probabilities = expit(
            n.design(guard["scaled"], head["arm"] == "spline") @ np.asarray(head["coefficients"])
        )
        predictions[head["arm"]].append(probabilities)
    for arm, seed_values in predictions.items():
        observed = [
            r["p_error"]
            for r in result["rows"]
            if r["arm"] == arm and r["condition"] == "exposed_development_guard"
        ]
        if not np.allclose(np.mean(seed_values, axis=0), observed, rtol=0, atol=1e-14):
            raise ValueError("state_prediction_drift")
    if result["source_manifest"]["graph_manifest_hash"] != canonical_hash(
        result["graph_manifest"]
    ) or result["source_manifest"]["role_manifest_hash"] != canonical_hash(result["role_manifest"]):
        raise ValueError("manifest_drift")
    counts = Counter((r["shard"], r["role"]) for r in result["role_manifest"])
    if result["construction"]["complete"] and counts != Counter(
        {
            (s, r): count
            for s in ("A", "B")
            for r, count in [("calibration", 64), ("test", 64), ("online", 384)]
        }
    ):
        raise ValueError("role_counts")
    for graph, role in zip(result["graph_manifest"], result["role_manifest"]):
        if (
            graph["trained_state_sha256"] != result["trained_state_reference"]["sha256"]
            or any(
                role[k] != graph[k]
                for k in (
                    "graph_id",
                    "graph_hash",
                    "shard",
                    "role",
                    "stream_id",
                    "stream_position",
                    "trained_state_sha256",
                )
            )
            or role["unit_id"] != graph["questions"][0]["unit_id"]
            or role["query_hash"] != graph["questions"][0]["prompt_hash"]
        ):
            raise ValueError("role_binding_drift")
    original = []
    for graph in result["graph_manifest"]:
        restored = dict(graph)
        restored.pop("trained_state_sha256")
        restored["graph_id"] = graph["graph_id"].replace("v768", "v766")
        restored["questions"] = [
            dict(q, unit_id=q["unit_id"].replace("v768", "v766")) for q in graph["questions"]
        ]
        original.append(restored)
    history = json.loads(
        Path(result["source_manifest"]["historical_reference"]["path"]).read_bytes()
    )
    with patch.object(graphs, "SEED", SEED):
        if not graphs.verify_graphs(dict(result["construction"], graph_manifest=original), history):
            raise ValueError("graph_construction_drift")


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Numerically ready states remain usable after a terminal development signal failure."""
    with patch.object(base, "OWNED", OWNED):
        value = base.build(work, validation, fixture)
    result = work["result"]
    fitted = bool(value["spline_methods_ready_score"] and value["verdict_class"] != "disqualified")
    value["spline_methods_ready_score"] = int(fitted)
    signal = bool(result) and all(r["passed"] for r in result["development_admission_rows"])
    constructed = result.get("construction", {}).get("complete", False)
    collection = bool(fitted and signal and constructed)
    if value["verdict_class"] == "null":
        reason = (
            "development_admitted"
            if collection
            else "development_admission_failed"
            if not signal
            else "construction_exhausted"
        )
        value["honest_verdict"] = f"complete_null_{reason}"
    value.update(
        experiment_id=8987,
        task_id=TASK,
        random_seed=SEED,
        fitted_states_ready_score=int(fitted),
        collection_ready_score=int(collection),
        fresh_collection_permitted=collection,
        development_signal_gate=int(signal),
        original_thresholds=THRESHOLDS,
        unchanged_protocol_retired=bool(result and not signal),
        claim_scope="Exposed development feasibility only; no independent generalization, no oracle-distinct verifier win.",
        optimizer_rows=result.get("optimizer_rows", []),
        development_metric_rows=result.get("development_metric_rows", []),
        development_admission_rows=result.get("development_admission_rows", []),
        graph_manifest=result.get("graph_manifest", []),
        role_manifest=result.get("role_manifest", []),
        source_manifest=result.get("source_manifest", {}),
        construction=result.get("construction", {}),
        trained_state_reference=result.get("trained_state_reference"),
        convergence_failure_residual=work.get("owned_failure"),
        historical_head_metrics=work["evidence"].get("context", {}).get("development_metrics"),
        methodology_note="Fit registered convex spline and equally optimized linear coefficients on original256 fit graphs. Average three frozen encoder probabilities on64 exposed guard graphs at T=1. Apply all four fixed development margins without retuning. Freeze1024 new graphs and24 streams before generation. Readiness permits only future collection; these observations establish no independent scientific benefit.",
    )
    value["acceptance_gates"].update(
        fitted_states=fitted, development_signal=signal, frozen_population=constructed
    )
    value["cited_upstream_artifacts"].append(
        dict(
            experiment_id=8986,
            sha256=INPUT_PIN,
            fields_imported=["spline_methods_ready_score", "required_checks_passed"],
        )
    )
    value["field_principles"].update(
        {
            k: "Actual retained states qualify numerical readiness; fixed development margins alone admit collection, never independent benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value
