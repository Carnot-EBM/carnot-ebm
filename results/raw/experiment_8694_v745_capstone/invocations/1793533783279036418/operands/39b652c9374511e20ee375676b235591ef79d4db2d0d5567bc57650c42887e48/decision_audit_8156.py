"""REQ-VERIFY-8156: independently reduce sealed exposed source decisions.

Byte custody precedes human targets. The audit measures decision costs while
keeping administrative readiness separate from evidence of useful decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.verify import development_methods_8098 as human
from carnot.verify import learning_audit_8144 as audit
from carnot.verify import reserved_evidence_capture_8155 as producer

Json = dict[str, Any]
ROOT = producer.ROOT
NAME = "experiment_8156_v705_decision_audit"
TASK = "exp8156-decision-audit"
MODULE = "python/carnot/verify/decision_audit_8156.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_decision_audit_8156.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261005"
MODEL_SPECS: list[str] = []
ARMS = producer.fit.energy.ARMS
CONFIG = dict(
    seed=70556,
    draws=10000,
    valid_draws=9500,
    sources=96,
    per_class=12,
    family_alpha=0.05,
    one_sided_alpha=0.025,
    gain=0.02,
    improved_sources=5,
    brier_increase=0.01,
    control_disadvantage=0.02,
)
UPSTREAM = "results/experiment_8155_v705_reserved_evidence_capture.json"
PIN = "sha256:6a4311de73e3fca6e8fbc756f4e80824eabba4ab962936e08767359b93713673"
COHORT = "results/experiment_8098_v701_development_methods.json"
COHORT_PIN = "sha256:4d28524a53f7eed96c2ce814da5a798828594a7e540d70d0de2f6a25ca5d6f76"
execution = producer.execution
reference = producer.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counters make CPU work and pending validation children visible."""
    print(f"[exp8156] phase={phase} completed={completed} pending={pending}", flush=True)


def reconstruct(calls: list[Json], heads: list[Json]) -> list[Json]:
    """Reparse transcripts and apply frozen geometry without producer predictions."""
    if any(
        r.get("human_target") is not None or r.get("entailment_label") is not None for r in calls
    ):
        raise ValueError("prediction_label_leakage")
    pairs = producer.pairs(calls)
    spans = {r["unit_id"]: r for r in calls if r["arm"] == "source_span"}
    if len(calls) != 256 or len(pairs) != 128 or len(spans) != 128:
        raise ValueError("original_prediction_rows")
    rows = []
    for row in pairs:
        probabilities: Json = {}
        status, reason = row["status"], row["exclusion_reason"]
        span = spans[row["unit_id"]]
        if status == "completed":
            public = dict(
                family_id=row["unit_id"],
                source_bytes=span["source_bytes"],
                answer_bytes=span["answer_bytes"],
            )
            features = producer.fit.lexical.extract(public)
            if features["values"] is None:
                status, reason = "excluded", features["abstention"]
            else:
                logits = [
                    math.log(p / (1 - p))
                    for p in np.clip(
                        [row["holistic_probability"], row["span_probability"]], 1e-6, 1 - 1e-6
                    )
                ]
                parsed = producer.capture.protocol.parse(
                    span["transcript"], bytes.fromhex(span["source_bytes"]), "source_span"
                )
                x = np.asarray(
                    [
                        [
                            logits[0],
                            *features["values"],
                            logits[1],
                            parsed["valid_quote"],
                            parsed["quote_source_byte_ratio"],
                        ]
                    ]
                )
                for head in heads:
                    columns = producer.fit.energy.design(head["arm"], x, head["geometry"])[0]
                    z = math.fsum(
                        float(a) * float(b) for a, b in zip(columns, head["weights"], strict=True)
                    )
                    intercept, slope = head["calibration"]
                    z = intercept + slope * z
                    probabilities[head["arm"]] = (
                        1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                    )
                probabilities["equivalent_logistic"] = probabilities["radial16"]
        for arm in ARMS:
            p = probabilities.get(arm)
            rows.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm=arm,
                    condition="complete_original_source",
                    metric="sealed_probability",
                    numerator=p,
                    denominator=int(status == "completed"),
                    status=status,
                    exclusion_reason=reason,
                    p=p,
                    action=producer.fit.energy.action(p),
                    human_target=None,
                    source_sha256=canonical_hash(span["source_bytes"]),
                )
            )
    return rows


def scored(row: Json, y: int | None) -> Json:
    """Compute typed harm directly; missing sources remain explicit excluded rows."""
    eligible = row["status"] == "completed" and y in (0, 1)
    p = None if row["arm"] == "always_escalate" else row["p"]
    action = "escalate" if p is None or 0.1 <= p <= 0.5 else "accept" if p < 0.1 else "reject"
    cost = (
        0.5
        if action == "escalate"
        else float(5 * y)
        if action == "accept" and y is not None
        else float(1 - y)
        if y is not None
        else None
    )
    clipped = min(1 - 1e-6, max(1e-6, p)) if p is not None else None
    return dict(
        row,
        metric="typed_decision_cost",
        numerator=cost if eligible else None,
        denominator=int(eligible),
        status="completed" if eligible else "excluded",
        exclusion_reason=None if eligible else row["exclusion_reason"] or "missing_target",
        p=p,
        y=y,
        action=action,
        brier=(p - y) ** 2 if eligible and p is not None else None,
        log_loss=-y * math.log(clipped) - (1 - y) * math.log1p(-clipped)
        if eligible and clipped is not None
        else None,
        false_accept=int(eligible and action == "accept" and y == 1),
        coverage=int(eligible and action != "escalate"),
    )


def reduce(data: Json) -> Json:
    """Reject custody drift before human targets influence costs or aggregates."""
    clock = data["clock"]
    if not 0 < clock["predictions_sealed_ns"] < clock["labels_opened_ns"]:
        raise ValueError("prediction_timestamp_order")
    predictions = reconstruct(data["calls"], data["heads"])
    audit.equal(predictions, data["predictions"])
    roster = {r["unit_id"]: r for r in data["roster"]}
    targets = {r["unit_id"]: r for r in data["targets"]}
    if len(roster) != 128 or len(targets) != 128 or set(roster) != set(targets):
        raise ValueError("original_target_join")
    if len({r["source_cluster_id"] for r in roster.values()}) != 128:
        raise ValueError("source_cluster_identity")
    originals = {str(r["id"]): r for r in data["original_response_records"]}
    public = {r["family_id"]: r for r in data["public"]}
    for unit, r in roster.items():
        target = targets[unit]
        if str(originals[r["response_id"]]["source_id"]) != r["source_id"]:
            raise ValueError("original_human_source")
        if any(
            r[k] != target[k] for k in ("unit_id", "source_cluster_id", "source_id", "response_id")
        ):
            raise ValueError("source_label_join")
        expected, _ = human.target(
            originals[r["response_id"]], bytes.fromhex(public[unit]["answer_bytes"])
        )
        if any(target[k] != expected[k] for k in ("y", "status", "exclusion_reason")):
            raise ValueError("independent_human_target")
    for call in data["calls"]:
        r, p = roster[call["unit_id"]], public[call["unit_id"]]
        if call["source_cluster_id"] != r["source_cluster_id"] or any(
            call[k] != p[k] for k in ("source_bytes", "answer_bytes")
        ):
            raise ValueError("original_source_join")
    rows = [scored(r, targets[r["unit_id"]]["y"]) for r in predictions]
    return dict(rows=rows, **statistics(rows))


def statistics(rows: list[Json]) -> Json:
    """Pair source clusters before resampling; repeated arms add no sample units."""
    grouped: Json = {}
    for row in rows:
        grouped.setdefault(row["source_cluster_id"], {})[row["arm"]] = row
    complete = [
        r
        for r in grouped.values()
        if set(r) == set(ARMS) and all(v["denominator"] == 1 for v in r.values())
    ]
    support = Counter(r["radial16"]["y"] for r in complete)
    gains = np.asarray(
        [r["additive_cubic"]["numerator"] - r["radial16"]["numerator"] for r in complete]
    )
    interval: Json = dict(
        valid_draws=0,
        requested_draws=10000,
        mean_gain=None,
        lower_one_sided_975=None,
        upper_descriptive_975=None,
        scope="descriptive_exposed_development",
        cluster_unit="original_source",
    )
    if len(gains):
        progress("before_source_cluster_bootstrap", 0, 10000)
        rng = np.random.default_rng(CONFIG["seed"])
        draws = rng.integers(0, len(gains), size=(10000, len(gains)))
        means = gains[draws].mean(axis=1)
        interval.update(
            valid_draws=len(means),
            mean_gain=float(gains.mean()),
            lower_one_sided_975=float(np.quantile(means, 0.025)),
            upper_descriptive_975=float(np.quantile(means, 0.975)),
            bootstrap_checksum=canonical_hash(means.tolist()),
        )
        progress("after_source_cluster_bootstrap", 10000, 0)
    summaries = []
    for arm in ARMS:
        selected = [r[arm] for r in complete]
        summaries.append(
            dict(
                arm=arm,
                count=len(selected),
                cost=float(np.mean([r["numerator"] for r in selected])) if selected else None,
                brier=float(np.mean([r["brier"] for r in selected]))
                if selected and arm != "always_escalate"
                else None,
                log_loss=float(np.mean([r["log_loss"] for r in selected]))
                if selected and arm != "always_escalate"
                else None,
                false_accepts=sum(r["false_accept"] for r in selected),
                coverage=float(np.mean([r["coverage"] for r in selected])) if selected else None,
            )
        )
    by_arm = {r["arm"]: r for r in summaries}
    treatment, control = by_arm["radial16"], by_arm["additive_cubic"]
    supported = len(complete) >= 96 and all(support[y] >= 12 for y in (0, 1))
    safety = (
        bool(complete)
        and treatment["false_accepts"] <= control["false_accepts"]
        and treatment["brier"] - control["brier"] <= 0.01
        and all(
            treatment["cost"] - by_arm[a]["cost"] <= 0.02 for a in producer.fit.energy.FITTED_ARMS
        )
    )
    signal = int(
        supported
        and interval["valid_draws"] >= 9500
        and interval["lower_one_sided_975"] > 0.02
        and int(np.sum(gains > 0)) >= 5
        and safety
    )
    comparisons = [
        dict(
            arm=a,
            control=c,
            control_cost_minus_arm_cost=by_arm[c]["cost"] - by_arm[a]["cost"] if complete else None,
        )
        for a in ARMS
        for c in ("scalar_holistic", "scalar_span", "always_escalate")
    ]
    sources = [
        dict(
            unit_id=r["radial16"]["unit_id"],
            source_cluster_id=s,
            status="completed" if r in complete else "excluded",
            exclusion_reason=r["radial16"]["exclusion_reason"],
            h1_gain=r["additive_cubic"]["numerator"] - r["radial16"]["numerator"]
            if r in complete
            else None,
        )
        for s, r in grouped.items()
        if "radial16" in r
    ]
    return dict(
        paired_intervals=interval,
        arm_metrics=summaries,
        control_comparisons=comparisons,
        per_source_results=sources,
        h1_development_signal_score=signal,
        class_support={str(y): support[y] for y in (0, 1)},
        eligible_count=len(complete),
        improved_sources=int(np.sum(gains > 0)),
        support_passed=supported,
        safety_passed=safety,
    )


def bind(ref: Json, raw: Path, refs: list[Json]) -> Json:
    """Retain only authenticated bytes, so private mutations cannot refresh custody."""
    path = Path(ref["path"])
    if not path.is_file() or sha256_file(path) != ref["sha256"]:
        raise ValueError("input_sha256")
    target = raw / "inputs" / (ref["sha256"].split(":")[-1] + "-" + path.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    refs.append(reference(target))
    return dict(json.loads(target.read_text()))


def measure(root: Path, raw: Path, *, fixture: bool = False, mutation: str = "") -> Json:
    """Authenticate prediction custody before opening any evaluator-only bytes."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        evidence={},
        checks=[],
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        historical_model_provenance={},
        trained_head_specs=[],
        source_diagnostics=[],
    )
    path = root / UPSTREAM

    def require(p: Path, field: str, expected: Any, observed: Any) -> None:
        work["checks"].append(
            dict(
                check=field,
                upstream=UPSTREAM,
                path=str(p),
                hash=sha256_file(p) if p.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )
        if expected != observed:
            raise ValueError(field)

    def read(ref: Json) -> Json:
        p = Path(ref["path"])
        require(p, "input_sha256", ref["sha256"], sha256_file(p) if p.is_file() else None)
        return bind(ref, raw, work["source_artifact_hashes"])

    progress("before_input_custody")
    try:
        if fixture:
            if mutation:
                require(path, "evaluation_capture_ready_score", 1, 0)
            private_capture = producer.measure(root, raw / "private_capture", fixture=True)
            calls = private_capture["result"]["rows"]
            heads = private_capture["plan"]["heads"]
            public = [
                dict(
                    family_id=r["unit_id"],
                    source_bytes=r["source_bytes"],
                    answer_bytes=r["answer_bytes"],
                )
                for r in calls
                if r["arm"] == "holistic"
            ]
            roster = [
                dict(
                    unit_id=r["family_id"],
                    source_cluster_id="cluster-" + str(i),
                    source_id=str(i),
                    response_id=str(i),
                )
                for i, r in enumerate(public)
            ]
            sealed = producer.predict(calls, heads)
            work["source_artifact_hashes"] = private_capture["raw_shard_hashes"]
        else:
            require(
                Path(sys.executable), "python_runtime_supported", True, sys.version_info >= (3, 11)
            )
            for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
                require(
                    ROOT / ".venv/bin" / tool,
                    "runtime_executable",
                    True,
                    os.access(ROOT / ".venv/bin" / tool, os.X_OK),
                )
            require(path, "upstream_exists", True, path.is_file())
            value = read(dict(path=str(path), sha256=PIN))
            for field, expected in [
                ("evaluation_capture_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                require(path, field, expected, value.get(field))
            require(
                path,
                "terminal_passed",
                True,
                producer.fit.read_bound_sidecar(path, producer.fit.publication_sidecar(value))[
                    "report"
                ]["passed"],
            )
            require(path, "prediction_cold_replay", True, producer.replay(path))
            seal = read(value["sealed_prediction_manifest"])
            require(path, "evaluator_targets_opened", False, seal["evaluator_targets_opened"])
            heads = read(seal["heads"])["heads"]
            calls = read(seal["calls"])["rows"]
            sealed = read(seal["predictions"])["rows"]
            public_doc = read(value["plan"]["manifests"]["evaluation"])
            public, roster = public_doc["request_rows"], public_doc["roster"]
            fitted_path = root / producer.UPSTREAM
            fitted = read(dict(path=str(fitted_path), sha256=producer.PIN))
            require(fitted_path, "energy_fit_ready_score", 1, fitted["energy_fit_ready_score"])
            method = ROOT / producer.fit.METHOD
            require(method, "immutable_v704_methods", producer.fit.METHOD_PIN, sha256_file(method))
            work["source_artifact_hashes"].append(reference(method))
            work["historical_model_provenance"] = dict(
                MODEL_SPECS=value["MODEL_SPECS"],
                imported_invocation_counts=value["model_invocation_counts"],
                scope="historical_only",
            )
            work["trained_head_specs"] = value["trained_head_specs"]
        independent = reconstruct(calls, heads)
        audit.equal(independent, sealed)
        prediction_path = raw / "independently_sealed_predictions.json"
        atomic_json(prediction_path, dict(rows=independent))
        clock = dict(predictions_sealed_ns=time.time_ns())
        work["raw_shard_hashes"].append(reference(prediction_path))
        progress("prediction_custody_authenticated_before_labels", 128, 0)
        clock["labels_opened_ns"] = time.time_ns()
        if fixture:
            originals = [
                dict(
                    id=str(i),
                    source_id=str(i),
                    response="fact",
                    quality="good",
                    labels=[]
                    if i % 2 == 0
                    else [
                        dict(
                            start=0,
                            end=4,
                            text="fact",
                            label_type="hallucination",
                            implicit_true=False,
                            due_to_null=False,
                            meta=None,
                        )
                    ],
                )
                for i in range(128)
            ]
            targets = [
                dict(r, **human.target(originals[i], b"fact")[0]) for i, r in enumerate(roster)
            ]
        else:
            original = read(dict(path=str(root / COHORT), sha256=COHORT_PIN))
            evaluator = read(original["evaluator_label_manifests"]["evaluation"])
            targets, originals = evaluator["rows"], evaluator["original_response_records"]
            fitted_capture = read(
                dict(path=str(root / producer.fit.UPSTREAM), sha256=producer.fit.PIN)
            )
            work["source_diagnostics"] = dict(
                rows=fitted_capture["source_intervention_rows"],
                scope="upstream_fit_diagnostics_only_no_inherited_intervention_targets",
            )
        work["evidence"] = dict(
            calls=calls,
            heads=heads,
            predictions=independent,
            roster=roster,
            public=public,
            targets=targets,
            original_response_records=originals,
            clock=clock,
        )
        reduction = reduce(work["evidence"])
        evidence = raw / "decision_evidence.json"
        atomic_json(evidence, work["evidence"])
        work["raw_shard_hashes"].append(reference(evidence))
        atomic_json(raw / "independent_reduction.json", reduction)
        work["raw_shard_hashes"].append(reference(raw / "independent_reduction.json"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not work["checks"] or work["checks"][-1]["passed"]:
            work["checks"].append(
                dict(
                    check="input_custody",
                    upstream=UPSTREAM,
                    path=str(path),
                    hash=sha256_file(path) if path.is_file() else None,
                    artifact_field="input_custody",
                    op="==",
                    expected="authenticated_original_rows",
                    observed=str(error),
                    passed=False,
                )
            )
        work["evidence"] = {}
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="custody_and_independent_decision_reduction",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                producer.fit.NUMERIC,
                producer.fit.MODULE,
                producer.MODULE,
                human.__file__,
                audit.__file__,
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("after_measurement", int(bool(work["evidence"])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness needs normal owned validation; supported nulls remain terminal."""
    reduced = reduce(work["evidence"]) if work["evidence"] else dict(rows=[], **statistics([]))
    failures = [r for r in work["checks"] if not r["passed"]]
    checked = all(r.get("passed") is True for r in receipts) and bool(receipts)
    ready = int(checked and not failures and bool(work["evidence"]))
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "positive"
        if reduced["h1_development_signal_score"]
        else "null"
    )
    suffix = (
        "owned_validation"
        if not checked
        else failures[0]["check"]
        if failures
        else "fixture"
        if fixture
        else "decision_development_signal"
        if reduced["h1_development_signal_score"]
        else "decision_development_null"
    )
    rows = reduced["rows"]
    n = reduced["eligible_count"]
    masks = producer.pairs(work["evidence"]["calls"]) if work["evidence"] else []
    failed = sum(r["status"] == "failed" for r in masks)
    censored = sum(r["status"] == "censored" for r in masks)
    result: Json = dict(
        experiment_id=8156,
        task_id=TASK,
        milestone="2026.10.705",
        run_date=RUN_DATE,
        honest_verdict=f"complete_{verdict}_{suffix}",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        claim_scope="independently reconstructed exposed source decision costs",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=work["trained_head_specs"],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        historical_model_provenance=work["historical_model_provenance"],
        intended_count=128,
        independent_count=n,
        completed_count=n,
        excluded_count=128 - n - failed - censored,
        censored_count=censored,
        failed_count=failed,
        sample_size_budget=dict(original_sources=128, minimum_pairs=96, minimum_per_class=12),
        duration_s=work["duration_s"],
        random_seed=CONFIG["seed"],
        reproducibility_checksum=canonical_hash(
            dict(config=CONFIG, refs=work["source_artifact_hashes"], rows=rows)
        ),
        source_artifact_hashes=work["source_artifact_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=CONFIG,
        field_principles=dict(
            readiness="Normal owned validation is separate from H1 benefit.",
            scope="Exposed development and fixtures provide no independent generalization.",
            rows="Every original source and missing reason remains reconstructable.",
            inference="Zero current model calls; captured model provenance is historical.",
        ),
        decision_audit_ready_score=ready,
        calibration_rows=[r for r in rows if r["arm"] != "always_escalate"],
        typed_cost_rows=rows,
        leakage_checks=dict(
            predictions_authenticated_before_labels=bool(work["evidence"]),
            independent_original_source_join=bool(work["evidence"]),
            independent_annotation_targets=bool(work["evidence"]),
        ),
        source_diagnostics=work["source_diagnostics"],
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("repository_health", {}),
        methodology_note="Original128 slots; cached frozen heads; independent scalar probability and typed harm reduction; paired original-source bootstrap10000 draws; H1/H2 Bonferroni alpha.05; intervals descriptive for exposed development.",
        **reduced,
    )
    result["h1_development_signal_score"] *= ready
    return result


def replay(path: Path) -> bool:
    """Cold readers rehash all primitives and recompute headlines from those rows."""
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        return bool(
            build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
                fixture=value["fixture_protocol_only"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze explicit test and coverage paths before measurement begins."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 600
    return specs


def main(argv: list[str] | None = None) -> int:
    """Run dated no-model audit or private fixtures through the same publisher."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE], default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--mutation", choices=["", "block"], default="")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "replay_rejected")
        return 0 if passed else 1
    fixture = args.fixture_output is not None
    if args.mutation and not fixture:
        parser.error("mutations require private fixture output")
    output = (args.fixture_output or args.output).absolute()
    if fixture and output.is_relative_to(ROOT / "results"):
        parser.error("private fixture output required")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8156-validation-") as directory:
        private = Path(directory)
        specs = manifest(private, output.parent / "raw" / output.stem / "terminal_candidate.json")
        atomic_json(raw / "validation_commands.json", specs)
        work = measure(args.root, raw, fixture=fixture, mutation=args.mutation)
        receipts = [dict(name="private_fixture_normal_exit", passed=True)] if fixture else []
        if not fixture:
            for spec in specs["commands"]:
                progress(
                    "before_subprocess_" + spec["name"],
                    len(receipts),
                    len(specs["commands"]) - len(receipts),
                )
                receipts.append(producer.checked(spec, private, raw / "logs"))
                progress(
                    "after_subprocess_" + spec["name"],
                    len(receipts),
                    len(specs["commands"]) - len(receipts),
                )
            progress("before_subprocess_repository_full_suite", 0, 1)
            work["repository_health"] = producer.checked(
                specs["repository_health"], private, raw / "repository_health"
            )
            progress("after_subprocess_repository_full_suite", 1, 0)
            coverage = private / "coverage.json"
            if coverage.is_file():
                saved = raw / "changed_code_coverage.json"
                saved.write_bytes(coverage.read_bytes())
                work["raw_shard_hashes"].append(reference(saved))
            atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, fixture=fixture)
        if output.exists():
            (raw / "preserved_historical_primary.json").write_bytes(output.read_bytes())
        with patch.object(execution, "e", sys.modules[__name__]):
            execution.publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
