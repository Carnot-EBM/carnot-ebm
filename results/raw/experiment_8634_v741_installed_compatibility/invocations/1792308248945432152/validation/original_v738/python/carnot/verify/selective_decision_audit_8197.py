"""REQ-VERIFY-8197: audit frozen decisions against original human annotations.

Prediction custody precedes label access. Source bootstrap and separate error
denominators prevent sentence counts or abstention from implying safe utility.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from scipy.stats import beta  # type: ignore[import-untyped]
import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import selective_sealed_evaluation_8196 as sealed
from carnot.verify import sentence_decision_audit_8185 as old

Json = dict[str, Any]
ROOT = sealed.ROOT
NAME = "experiment_8197_v708_selective_decision_audit"
TASK = "exp8197-selective-decision-audit"
MODULE = "python/carnot/verify/selective_decision_audit_8197.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_selective_decision_audit_8197.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
n, fit, execution, reference = sealed.n, sealed.fit, sealed.execution, sealed.reference
UPSTREAM = "results/experiment_8196_v708_selective_sealed_evaluation.json"
HISTORICAL = "results/experiment_8185_v707_sentence_decision_audit.json"
PINS = {
    UPSTREAM: "sha256:9cda537f61052c800f9520e151dd20d7eef750b0a5cb587dc3e90bec9bfd8e5a",
    HISTORICAL: "sha256:85e67f7b39e7d89e621326a0bc8572468c33b43e7caee781628c69ff67cf1083",
}
CONFIG = dict(
    seed=n.SEED,
    intended=128,
    minimum_pairs=96,
    minimum_per_class=12,
    draws=10000,
    valid_draws=9500,
    alpha=0.025,
    lower_gain_gt=0.02,
    improved_sources_min=5,
    extra_false_accepts_max=0,
    brier_increase_max=0.01,
    costs=dict(correct=0, escalate=0.5, false_accept=5, false_reject=1),
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counters keep bounded validation waits visible to the operator."""
    print(f"[exp8197] phase={phase} completed={completed} pending={pending}", flush=True)


def binomial(k: int, count: int) -> Json:
    """Exact two-sided intervals describe observations without promising safety."""
    return dict(
        numerator=k,
        denominator=count,
        rate=k / count if count else None,
        interval=[
            float(beta.ppf(0.025, k, count - k + 1)) if k else 0.0,
            float(beta.ppf(0.975, k + 1, count - k)) if k < count else 1.0,
        ]
        if count
        else [None, None],
        confidence=0.95,
        method="Clopper_Pearson",
    )


def statistics(rows: list[Json]) -> Json:
    """Keep the frozen H1 primary and add descriptive source-level denominators."""
    result = n.statistics(rows)
    groups: Json = {}
    for r in rows:
        groups.setdefault(r["source_cluster_id"], {})[r["arm"]] = r
    paired = [g for g in groups.values() if set(g) == set(n.ARMS)]
    complete = [
        g
        for g in paired
        if all(
            g[a]["p"] is not None and g[a]["y"] in (0, 1)
            for a in ("local_set", "radial_set", "frozen_v707_radial")
        )
    ]
    coverage = []
    for arm in n.ARMS:
        rs = [g[arm] for g in paired]
        accepts = [r for r in rs if r["action"] == "accept" and r["y"] in (0, 1)]
        unsupported = [r for r in rs if r["y"] == 1]
        item = dict(
            arm=arm,
            singleton_coverage=binomial(sum(r["coverage"] for r in rs), len(rs)),
            false_accept_count=sum(r["false_accept"] for r in rs),
            error_among_accepted_outputs=binomial(sum(r["y"] == 1 for r in accepts), len(accepts)),
            error_conditional_on_unsupported_sources=binomial(
                sum(r["false_accept"] for r in unsupported), len(unsupported)
            ),
        )
        item["classes"] = []
        for y in (0, 1):
            observed = [
                r
                for r in rs
                if r["y"] == y and r["p"] is not None and r["prediction_set"] is not None
            ]
            k = sum(y in r["prediction_set"] for r in observed)
            item["classes"].append(
                dict(
                    label=y,
                    set_coverage=binomial(k, len(observed)),
                    conditional_set_error=binomial(len(observed) - k, len(observed)),
                )
            )
        coverage.append(item)
    gains = lambda panel: [
        g["frozen_v707_radial"]["numerator"] - g["local_set"]["numerator"] for g in panel
    ]
    with patch.dict(old.CONFIG, seed=n.SEED):
        cc_interval = old.interval(gains(complete))
    h1 = result["H1"]
    metrics = {r["arm"]: r for r in result["all_slot_metrics"]}
    stronger = (
        bool(paired)
        and metrics["local_set"]["typed_cost"] <= metrics["always_escalate"]["typed_cost"]
        and metrics["local_set"]["singleton_count"] > 0
    )
    signal = int(h1["passed"] and len(paired) == 128 and stronger)
    result.update(
        h1_development_signal_score=signal,
        paired_intervals=dict(all_slot=h1["interval"], complete_case=cc_interval),
        complete_case_metrics=[
            dict(
                arm=a,
                count=len(complete),
                typed_cost=sum(g[a]["numerator"] for g in complete) / len(complete)
                if complete
                else None,
            )
            for a in n.ARMS
        ],
        class_coverage_rows=coverage,
        secondary_comparisons=result["secondary_contrasts"],
        acceptance_operands=dict(
            complete_sources=len(complete),
            minimum_sources=96,
            class_support=result["class_support_reserved"],
            minimum_per_class=12,
            all_slot_count=len(paired),
            intended_slots=128,
            lower_cost_gain=h1["interval"]["lower_one_sided_975"],
            minimum_gain_exclusive=0.02,
            improved_sources=h1["improved_sources"],
            minimum_improved_sources=5,
            extra_false_accepts=h1["extra_false_accepts"],
            maximum_extra_false_accepts=0,
            brier_increase=h1["brier_increase"],
            maximum_brier_increase=0.01,
            valid_draws=h1["interval"]["valid_draws"],
            minimum_valid_draws=9500,
            stronger_cost_control_passed=stronger,
        ),
        per_source_results=[
            dict(
                source_cluster_id=s,
                unit_id=g["local_set"]["unit_id"],
                complete_pair=g in complete,
                h1_gain=gains([g])[0],
            )
            for s, g in groups.items()
            if set(g) == set(n.ARMS)
        ],
    )
    result["H1"]["passed"] = bool(signal)
    result["source_class_counts"] = {
        str(y): dict(
            intended=sum(g["local_set"]["y"] == y for g in paired),
            complete=sum(g["local_set"]["y"] == y for g in complete),
            missing=sum(g["local_set"]["y"] == y for g in paired if g not in complete),
        )
        for y in (0, 1)
    }
    result["secondary_comparisons"] += [
        dict(
            treatment=a,
            comparator="always_escalate",
            primary=False,
            mean_cost_gain=metrics["always_escalate"]["typed_cost"] - metrics[a]["typed_cost"]
            if paired
            else None,
        )
        for a in n.ARMS
        if a not in ("local_set", "always_escalate")
    ]
    return result


def reduce(data: Json) -> Json:
    """Reconstruct sealed probabilities, human targets and costs independently."""
    if not 0 < data["clock"]["predictions_sealed_ns"] < data["clock"]["labels_opened_ns"]:
        raise ValueError("prediction_timestamp_order")
    predicted = sealed.reduce(data["sealed"])
    old.base.audit.equal(predicted["prediction_rows"], data["predictions"])
    targets = {r["unit_id"]: r for r in data["targets"]}
    slots = {r["unit_id"]: r for r in data["slots"]}
    originals = {str(r["id"]): r for r in data["original_response_records"]}
    roster = {r["unit_id"] for r in data["sealed"]["features"]}
    if (
        len(targets) != 128
        or len(data["targets"]) != 128
        or set(targets) != roster
        or set(slots) != roster
    ):
        raise ValueError("original_target_join")
    for feature in data["sealed"]["features"]:
        s, t = slots[feature["unit_id"]], targets[feature["unit_id"]]
        original = originals[s["response_id"]]
        if (
            str(original["source_id"]) != s["source_id"]
            or any(t[k] != s[k] for k in ("source_cluster_id", "source_id", "response_id"))
            or feature["source_cluster_id"] != s["source_cluster_id"]
        ):
            raise ValueError("original_source_identity")
        expected, _ = old.base.human.target(original, bytes.fromhex(s["answer_bytes"]))
        if any(t[k] != expected[k] for k in ("y", "status", "exclusion_reason")):
            raise ValueError("original_annotation_target")
        for field, content in (
            ("answer_sha256", s["answer_bytes"]),
            ("source_sha256", s["source_bytes"]),
        ):
            if feature[field] != canonical_hash(content):
                raise ValueError("original_byte_identity")
    rows, shifts = [], []
    costs = data["costs"]
    if costs != CONFIG["costs"]:
        raise ValueError("original_cost_matrix")
    for prediction in predicted["prediction_rows"]:
        y = targets[prediction["unit_id"]]["y"]
        row = old.score(dict(prediction, p=prediction["p"] if y in (0, 1) else None), y)
        action = row["action"]
        cost = (
            costs["escalate"]
            if action == "escalate"
            else costs["false_accept"] * y
            if action == "accept"
            else costs["false_reject"] * (1 - y)
        )
        row.update(numerator=cost, denominator=1)
        rows.append(row)
    for source in data["sealed"]["features"]:
        if source["x"] is None:
            continue
        for head in data["sealed"]["frozen"]["heads"]:
            phi = n.basis(head, source["x"])
            z = math.fsum(float(x) * float(w) for x, w in zip(phi, head["weights"], strict=True))
            p = n.logit_probability([0, z], head["temperature"])
            shifted = n.logit_probability([7, 7 + z], head["temperature"])
            same = abs(p - shifted) <= 1e-10 and n.prediction_set(
                p, head["quantiles"]
            ) == n.prediction_set(shifted, head["quantiles"])
            shifts.append(
                dict(
                    unit_id=source["unit_id"],
                    arm=head["arm"],
                    probability=p,
                    shifted_probability=shifted,
                    passed=same,
                )
            )
    return dict(
        rows=rows,
        **statistics(rows),
        equivalent_logistic_parity=predicted["equivalent_logistic_parity"],
        common_logit_shift_control=dict(
            passed=bool(shifts) and all(r["passed"] for r in shifts), rows=shifts
        ),
        source_dependence=dict(
            independent_unit="original_source_cluster", sentence_independent_count=0
        ),
    )


def fixture() -> Json:
    """Original annotations in private fixtures qualify mechanics only."""
    data = sealed.fixture()
    slots, originals, targets = [], [], []
    for i, feature in enumerate(data["features"]):
        s = dict(
            unit_id=feature["unit_id"],
            source_cluster_id=feature["source_cluster_id"],
            source_id=str(i),
            response_id=str(i),
            answer_bytes=b"First fact.".hex(),
            source_bytes=b"Evidence.".hex(),
        )
        original = dict(
            id=str(i),
            source_id=str(i),
            response="First fact.",
            quality="good",
            labels=[]
            if i % 2 == 0
            else [
                dict(
                    start=0,
                    end=5,
                    text="First",
                    label_type="hallucination",
                    implicit_true=False,
                    due_to_null=False,
                    meta=None,
                )
            ],
        )
        feature.update(
            answer_sha256=canonical_hash(b"First fact.".hex()),
            source_sha256=canonical_hash(b"Evidence.".hex()),
        )
        slots.append(s)
        originals.append(original)
        targets.append(dict(s, **old.base.human.target(original, b"First fact.")[0]))
    return dict(sealed=data, slots=slots, targets=targets, original_response_records=originals)


def measure(root: Path, raw: Path, *, fixture: bool = False, mutation: str = "") -> Json:
    """Authenticate gates and predictions before opening evaluator-only bytes."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        raw_shard_hashes=[],
        precondition_receipts=[],
        trained_head_specs=[],
        historical_model_provenance={},
        cited_upstream_artifacts=[],
        owned_failure="",
        seal_receipt={},
        historical_v707_null={},
    )
    gate = lambda p, f, e, o: fit.gate(work, p, f, e, o)
    bind = lambda r: fit.bind(work, r, raw)
    progress("before_preconditions")
    try:
        gate(Path(sys.executable), "python_runtime_supported", True, sys.version_info >= (3, 11))
        for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
            gate(
                ROOT / ".venv/bin" / tool,
                "runtime_tool_executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        probe = raw / ".storage_probe"
        probe.write_bytes(b"private writable custody")
        gate(
            raw, "private_writable_storage", True, probe.read_bytes() == b"private writable custody"
        )
        probe.unlink()
        retired_path = ROOT / "ops/exclusion_manifest.yaml"
        retired = yaml.safe_load(retired_path.read_text())
        ids = {
            r.get("experiment_id")
            for k in ("retired", "retired_experiments")
            for r in retired.get(k, [])
        }
        gate(retired_path, "upstream_not_retired", True, not bool({8185, 8195, 8196} & ids))
        if fixture:
            data = globals()["fixture"]()
            if mutation:
                gate(root / UPSTREAM, "sealed_evaluation_ready_score", 1, 0)
        else:
            path = root / UPSTREAM
            gate(path, "upstream_exists", True, path.is_file())
            progress("before_subprocess_summarize_sealed")
            spec = dict(
                name="summarize_sealed",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/summarize_artifact.py",
                    str(path),
                ],
                deadline_s=60,
                expected_exit=0,
            )
            with (
                (raw / "summarize_sealed_supervision.log").open("x") as stream,
                redirect_stdout(stream),
            ):
                receipt = execution.run_check(
                    ROOT, spec, raw, raw / "precondition_logs", heartbeat_s=20
                )
            work["precondition_receipts"].append(receipt)
            progress("after_subprocess_summarize_sealed", 1, 0)
            gate(path, "upstream_summary_normal_exit", 0, receipt["actual_exit"])
            value = bind(dict(path=str(path), sha256=PINS[UPSTREAM]))
            for field, expected in (
                ("sealed_evaluation_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ):
                gate(path, field, expected, value.get(field))
            gate(
                path,
                "upstream_terminal_passed",
                True,
                read_bound_sidecar(path, fit.historical.publication_sidecar(value))["report"][
                    "passed"
                ],
            )
            gate(path, "sealed_prediction_replay", True, sealed.replay(path))
            saved = bind(value["measurement_reference"])
            predictions = bind(value["seal_receipt"]["predictions"])
            gate(path, "labels_opened", False, predictions["labels_opened"])
            old.base.audit.equal(
                sealed.reduce(saved["evidence"])["prediction_rows"], predictions["rows"]
            )
            data = dict(sealed=saved["evidence"])
            work["trained_head_specs"] = [
                dict(h, trained_in_current_run=False) for h in value["trained_head_specs"]
            ]
            work["historical_model_provenance"] = value["historical_model_provenance"]
            for ref in value["code_config_hashes"]:
                p = Path(ref["path"])
                gate(p, "versioned_method_bytes", ref["sha256"], sha256_file(p))
                target = raw / "inputs" / (ref["sha256"][7:] + "-" + p.name)
                target.write_bytes(p.read_bytes())
                work["refs"].append(reference(target))
            protocol = bind(
                dict(
                    path=str(ROOT / sealed.upstream.methods.PROTOCOL),
                    sha256=sealed.upstream.methods.PROTOCOL_PIN,
                )
            )
            gate(path, "frozen_cost_matrix", CONFIG["costs"], protocol["costs"])
            gate(
                path,
                "frozen_h1_protocol",
                dict(
                    alpha=0.025,
                    brier_increase_max=0.01,
                    comparator="frozen_v707_radial",
                    denominator=128,
                    draws=10000,
                    extra_false_accepts_max=0,
                    improved_sources_min=5,
                    lower_gain_gt=0.02,
                    treatment="local_set",
                    valid_minimum=9500,
                ),
                protocol["H1"],
            )
            work["cited_upstream_artifacts"].append(
                dict(
                    experiment_id=8196,
                    sha256=PINS[UPSTREAM],
                    fields_imported=["seal_receipt", "measurement_reference", "trained_head_specs"],
                )
            )
        predictions = sealed.reduce(data["sealed"])["prediction_rows"]
        clock = dict(predictions_sealed_ns=time.time_ns())
        atomic_json(raw / "authenticated_predictions.json", dict(rows=predictions))
        progress("predictions_verified_before_labels", 896, 0)
        clock["labels_opened_ns"] = time.time_ns()
        if not fixture:
            path = root / HISTORICAL
            gate(path, "historical_exists", True, path.is_file())
            progress("before_subprocess_summarize_historical")
            spec = dict(
                name="summarize_historical",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/summarize_artifact.py",
                    str(path),
                ],
                deadline_s=60,
                expected_exit=0,
            )
            with (
                (raw / "summarize_historical_supervision.log").open("x") as stream,
                redirect_stdout(stream),
            ):
                receipt = execution.run_check(
                    ROOT, spec, raw, raw / "precondition_logs", heartbeat_s=20
                )
            work["precondition_receipts"].append(receipt)
            progress("after_subprocess_summarize_historical", 1, 0)
            gate(path, "historical_summary_normal_exit", 0, receipt["actual_exit"])
            prior = bind(dict(path=str(path), sha256=PINS[HISTORICAL]))
            labels = bind(prior["measurement_reference"])["evidence"]
            data.update(
                slots=labels["capture"]["slots"],
                targets=labels["targets"],
                original_response_records=labels["original_response_records"],
            )
            work["historical_v707_null"] = {
                k: prior[k]
                for k in (
                    "honest_verdict",
                    "verdict_class",
                    "h1_development_signal_score",
                    "acceptance_operands",
                    "paired_intervals",
                )
            }
            work["cited_upstream_artifacts"].append(
                dict(
                    experiment_id=8185,
                    sha256=PINS[HISTORICAL],
                    fields_imported=["original_annotation_evidence", "historical_null"],
                )
            )
        data.update(predictions=predictions, clock=clock, costs=CONFIG["costs"])
        work["evidence"] = data
        progress("before_benchmark_independent_reduction", 0, 128)
        reduced = reduce(data)
        progress("after_benchmark_independent_reduction", 128, 0)
        for name, content in (
            ("decision_evidence", data),
            ("independent_reduction", reduced),
            ("frozen_configuration", CONFIG),
        ):
            atomic_json(raw / (name + ".json"), content)
        work["raw_shard_hashes"] = [
            reference(raw / (name + ".json"))
            for name in (
                "authenticated_predictions",
                "decision_evidence",
                "independent_reduction",
                "frozen_configuration",
            )
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in work["checks"]):
            try:
                gate(
                    root / UPSTREAM, "input_custody", "authenticated_original_evidence", str(error)
                )
            except ValueError:
                pass
        work["evidence"] = {}
    work.update(
        duration_s=time.monotonic() - began,
        code_config_hashes=[
            reference(ROOT / p)
            for p in [*OWNED, TEST, sealed.MODULE, sealed.upstream.methods.NUMERIC, old.MODULE]
        ],
        phase_spans=[
            dict(
                phase="authenticate_predictions_then_labels_and_reduce",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", int(bool(work["evidence"])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Execution readiness must not stand in for the frozen scientific gates."""
    reduced = reduce(work["evidence"]) if work["evidence"] else dict(rows=[], **statistics([]))
    failures = [r for r in work["checks"] if not r["passed"]]
    checked = bool(receipts) and all(r.get("passed") is True for r in receipts)
    ready = int(checked and not failures and bool(work["evidence"]))
    signal = reduced["h1_development_signal_score"] * ready
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "positive"
        if signal
        else "null"
    )
    suffix = (
        "owned_validation"
        if not checked
        else failures[0]["check"]
        if failures
        else "selective_decision_signal"
        if signal
        else "selective_decision_null"
    )
    count = reduced["eligible_count"]
    features = work["evidence"].get("sealed", {}).get("features", [])
    counts = {s: sum(r["status"] == s for r in features) for s in ("failed", "censored")}
    value: Json = dict(
        experiment_id=8197,
        task_id=TASK,
        milestone="2026.10.708",
        run_date=RUN_DATE,
        honest_verdict=f"complete_{verdict}_{suffix}",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        claim_scope="All-slot selective utility on exposed development sources; no energy-specific or independent benefit",
        exposure_scope="registered_extension_on_previously_evaluated_development_sources",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        precondition_receipts=work["precondition_receipts"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        trained_head_specs=work["trained_head_specs"],
        model_invocation_counts=dict(model_loads=0, generations=0, live_model_calls=0),
        call_ledger=[],
        historical_model_provenance=work["historical_model_provenance"],
        cited_upstream_artifacts=work["cited_upstream_artifacts"],
        intended_count=128,
        independent_count=len(features),
        completed_count=count,
        missing_count=128 - count,
        excluded_count=128 - count - counts["failed"] - counts["censored"],
        censored_count=counts["censored"],
        failed_count=counts["failed"],
        sample_size_budget=dict(original_sources=128, minimum_pairs=96, minimum_per_class=12),
        duration_s=work["duration_s"],
        random_seed=n.SEED,
        reproducibility_checksum=canonical_hash(
            dict(config=CONFIG, refs=work["refs"], rows=reduced["rows"])
        ),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=CONFIG,
        field_principles=dict(
            readiness="Normal owned validation is separate from useful decisions.",
            scope="Exposed sources cannot establish independent generalization or exchangeability.",
            rows="Only original sources are independent; missing slots escalate.",
            inference="Imported Qwen calls and trained heads incur zero current model calls.",
            safety="Exact descriptive intervals do not turn chosen alpha into a safety guarantee.",
        ),
        selective_audit_ready_score=ready,
        historical_v707_null=work["historical_v707_null"],
        attribution="Any wrapper benefit is abstention or calibration; logistic equivalence rules out energy-specific evidence.",
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("repository_health", {}),
        methodology_note="Original human source annotations; frozen cost matrix; all128 missing-escalated slots;10000 paired source bootstrap draws and one-sided97.5 percent lower bound; class and accepted-output denominators separate. Historical exposure prevents independent generalization.",
        **reduced,
    )
    value["h1_development_signal_score"] = signal
    return value


def replay(path: Path) -> bool:
    """Rehash custody and rebuild aggregates so rehashed forgeries still fail."""
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
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        data = work["evidence"]
        if data:
            if not value["fixture_protocol_only"]:
                copies = {Path(r["path"]).name: r for r in work["refs"]}
                load = lambda ref: json.loads(
                    Path(
                        copies[ref["sha256"][7:] + "-" + Path(ref["path"]).name]["path"]
                    ).read_text()
                )
                upstream = load(dict(path=UPSTREAM, sha256=PINS[UPSTREAM]))
                prior = load(dict(path=HISTORICAL, sha256=PINS[HISTORICAL]))
                old.base.audit.equal(
                    data["sealed"], load(upstream["measurement_reference"])["evidence"]
                )
                original = load(prior["measurement_reference"])["evidence"]
                for key, expected in (
                    ("slots", original["capture"]["slots"]),
                    ("targets", original["targets"]),
                    ("original_response_records", original["original_response_records"]),
                ):
                    old.base.audit.equal(data[key], expected)
            reduced = reduce(data)
            for name, expected in (
                ("decision_evidence", data),
                ("independent_reduction", reduced),
                ("authenticated_predictions", dict(rows=data["predictions"])),
                ("frozen_configuration", CONFIG),
            ):
                ref = next(r for r in work["raw_shard_hashes"] if Path(r["path"]).stem == name)
                old.base.audit.equal(json.loads(Path(ref["path"]).read_text()), expected)
        return bool(
            value
            == build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
                fixture=value["fixture_protocol_only"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze validation before measurement; static tools receive file paths."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 300
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    specs["repository_health"]["deadline_s"] = 600
    return specs


def main(argv: list[str] | None = None) -> int:
    """Reuse bounded supervision and publication for the no-model audit."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(execution, "run_check", old.producer.supervise))
        for name in (
            "ROOT",
            "NAME",
            "TASK",
            "MODULE",
            "CLI",
            "TEST",
            "OWNED",
            "RUN_DATE",
            "MODEL_SPECS",
            "progress",
            "manifest",
            "measure",
            "build",
            "replay",
        ):
            stack.enter_context(patch.object(old.base, name, globals()[name]))
        return int(old.base.main(argv))
