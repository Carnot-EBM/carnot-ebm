"""REQ-VERIFY-8185: independently audit original source decisions and costs.

Cached evidence and frozen heads precede original human labels. Source masks
retain transport failures, so formatted sentences never inflate sample size.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import decision_audit_8156 as base
from carnot.verify import reserved_sentence_capture_8184 as producer

Json = dict[str, Any]
ROOT = producer.ROOT
NAME = "experiment_8185_v707_sentence_decision_audit"
TASK = "exp8185-sentence-decision-audit"
MODULE = "python/carnot/verify/sentence_decision_audit_8185.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_sentence_decision_audit_8185.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[str] = []
ARMS = producer.energy.ARMS
CONFIG = dict(
    seed=7078185,
    draws=10000,
    valid_draws=9500,
    sources=96,
    per_class=12,
    one_sided_alpha=0.025,
    gain=0.02,
    improved_sources=5,
    brier_increase=0.01,
    extra_false_accepts=0,
    intended=128,
    primary_denominator="all_intended_slots",
)
UPSTREAM = "results/experiment_8184_v707_reserved_sentence_capture.json"
PIN = "sha256:e70bc2604c31780fb61a7a13aafb11e97ad422383523a88d8ec21e416e7fbf82"
execution = base.execution
reference = base.reference
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real completed counts show work without adding artificial runtime."""
    print(f"[exp8185] phase={phase} completed={completed} pending={pending}", flush=True)


def score(row: Json, y: int | None) -> Json:
    """Missing predictions cost an escalation while missing labels stay explicit."""
    p = row["p"]
    action = row["action"] if p is not None else "escalate"
    cost = (
        0.5
        if action == "escalate"
        else float(5 * y if action == "accept" else 1 - y)
        if y in (0, 1)
        else None
    )
    clipped = min(1 - 1e-6, max(1e-6, p)) if p is not None else None
    return dict(
        row,
        metric="typed_decision_cost",
        numerator=cost,
        denominator=int(cost is not None),
        y=y,
        action=action,
        complete_pair=row["status"] == "completed" and y in (0, 1),
        brier=(p - y) ** 2 if p is not None and y in (0, 1) else None,
        log_loss=-y * math.log(clipped) - (1 - y) * math.log1p(-clipped)
        if clipped is not None and y in (0, 1)
        else None,
        false_accept=int(action == "accept" and y == 1),
        coverage=int(action != "escalate"),
    )


def interval(gains: list[float]) -> Json:
    """Resample original sources together; sentences and arms add no draws."""
    result: Json = dict(
        valid_draws=0,
        requested_draws=10000,
        mean_gain=None,
        lower_one_sided_975=None,
        cluster_unit="original_source",
        random_seed=CONFIG["seed"],
    )
    if gains:
        progress("before_benchmark_source_bootstrap", 0, 10000)
        values = np.asarray(gains)
        indices = np.random.default_rng(CONFIG["seed"]).integers(
            0, len(gains), size=(10000, len(gains))
        )
        means = values[indices].mean(axis=1)
        result.update(
            valid_draws=len(means),
            mean_gain=float(values.mean()),
            lower_one_sided_975=float(np.quantile(means, 0.025)),
            upper_descriptive_975=float(np.quantile(means, 0.975)),
            bootstrap_checksum=canonical_hash(means.tolist()),
        )
        progress("after_benchmark_source_bootstrap", 10000, 0)
    return result


def statistics(rows: list[Json]) -> Json:
    """Apply the frozen primary to all slots; complete cases remain secondary."""
    grouped: Json = {}
    for row in rows:
        group = grouped.setdefault(row["source_cluster_id"], {})
        if row["arm"] in group:
            raise ValueError("duplicate_source_arm")
        group[row["arm"]] = row
    all_slots = [r for r in grouped.values() if set(r) == set(ARMS)]
    complete = [r for r in all_slots if all(v["complete_pair"] for v in r.values())]
    counts = Counter(r["radial16"]["y"] for r in complete)
    supported = len(complete) >= 96 and all(counts[y] >= 12 for y in (0, 1))

    def summarize(sources: list[Json]) -> list[Json]:
        summaries = []
        for arm in ARMS:
            selected = [r[arm] for r in sources if r[arm]["denominator"]]

            def mean(key: str) -> float | None:
                values = [r[key] for r in selected if r[key] is not None]
                return float(np.mean(values)) if values else None

            summaries.append(
                dict(
                    arm=arm,
                    count=len(selected),
                    cost=mean("numerator"),
                    brier=mean("brier"),
                    log_loss=mean("log_loss"),
                    coverage=mean("coverage"),
                    false_accepts=sum(r["false_accept"] for r in selected),
                    probability_count=sum(r["brier"] is not None for r in selected),
                )
            )
        return summaries

    all_metrics, complete_metrics = summarize(all_slots), summarize(complete)
    gains = lambda panel: [
        r["radial16"]["numerator"] - r["local_evidence_radial16"]["numerator"]
        for r in panel
        if r["radial16"]["denominator"] and r["local_evidence_radial16"]["denominator"]
    ]
    paired = dict(all_slot=interval(gains(all_slots)), complete_case=interval(gains(complete)))
    lookup = {r["arm"]: r for r in complete_metrics}
    t, c = lookup["local_evidence_radial16"], lookup["radial16"]
    safety = (
        bool(complete)
        and t["false_accepts"] <= c["false_accepts"]
        and t["brier"] - c["brier"] <= 0.01
    )
    improved = sum(g > 0 for g in gains(all_slots))
    primary = paired["all_slot"]
    signal = int(
        supported
        and len(gains(all_slots)) == 128
        and primary["valid_draws"] >= 9500
        and primary["lower_one_sided_975"] > 0.02
        and improved >= 5
        and safety
    )
    sources = [
        dict(
            unit_id=r["radial16"]["unit_id"],
            source_cluster_id=s,
            status=r["local_evidence_radial16"]["status"],
            exclusion_reason=r["local_evidence_radial16"]["exclusion_reason"],
            complete_pair=r in complete,
            h1_gain=gains([r])[0] if gains([r]) else None,
        )
        for s, r in grouped.items()
    ]
    return dict(
        paired_intervals=paired,
        all_slot_metrics=all_metrics,
        complete_case_metrics=complete_metrics,
        class_support={str(y): counts[y] for y in (0, 1)},
        eligible_count=len(complete),
        improved_sources=improved,
        support_sufficient=supported,
        safety_passed=safety,
        acceptance_operands=dict(
            complete_sources=len(complete),
            minimum_sources=96,
            class_support={str(y): counts[y] for y in (0, 1)},
            minimum_per_class=12,
            all_slot_count=len(gains(all_slots)),
            intended_slots=128,
            lower_cost_gain=primary["lower_one_sided_975"],
            minimum_gain_exclusive=0.02,
            improved_sources=improved,
            minimum_improved_sources=5,
            extra_false_accepts=t["false_accepts"] - c["false_accepts"],
            maximum_extra_false_accepts=0,
            brier_increase=t["brier"] - c["brier"] if complete else None,
            maximum_brier_increase=0.01,
            valid_draws=primary["valid_draws"],
            minimum_valid_draws=9500,
        ),
        h1_development_signal_score=signal,
        per_source_results=sources,
        secondary_comparisons=[
            dict(
                arm=r["arm"],
                comparator="radial16",
                all_slot_cost=r["cost"],
                scope="descriptive_only",
            )
            for r in all_metrics
        ],
    )


def reduce(data: Json) -> Json:
    """Rebuild predictions and original labels before computing decision headlines."""
    if not 0 < data["clock"]["predictions_sealed_ns"] < data["clock"]["labels_opened_ns"]:
        raise ValueError("prediction_timestamp_order")
    capture = data["capture"]
    slots = capture["slots"]
    roster = {r["unit_id"]: r for r in slots}
    targets = {r["unit_id"]: r for r in data["targets"]}
    if len(roster) != 128 or len(targets) != 128 or set(roster) != set(targets):
        raise ValueError("original128_target_join")
    if len({r["source_cluster_id"] for r in slots}) != 128:
        raise ValueError("source_cluster_identity")
    originals = {str(r["id"]): r for r in data["original_response_records"]}
    for unit, slot in roster.items():
        target = targets[unit]
        original = originals[slot["response_id"]]
        if str(original["source_id"]) != slot["source_id"] or any(
            target[k] != slot[k] for k in ("source_cluster_id", "source_id", "response_id")
        ):
            raise ValueError("source_label_identity")
        expected, _ = base.human.target(original, bytes.fromhex(slot["answer_bytes"]))
        if any(target[k] != expected[k] for k in ("y", "status", "exclusion_reason")):
            raise ValueError("original_annotation_target")
    features = producer.reduce(slots, capture["calls"], capture["plan"]["baseline"])
    predicted = producer.predict(features["feature_rows"], capture["plan"]["heads"])
    base.audit.equal(predicted, data["predictions"])
    heads = {r["arm"]: r for r in capture["plan"]["heads"]}
    parity, independent = [], []
    for row in predicted:
        source = next(r for r in features["feature_rows"] if r["unit_id"] == row["unit_id"])
        if row["p"] is not None:
            arm = "local_evidence_radial16" if row["arm"] == "equivalent_logistic" else row["arm"]
            h = heads[arm]
            phi = producer.energy.design(arm, np.asarray([source["x"]]), h["geometry"])[0]
            b, a = h["calibration"]
            z = b + a * math.fsum(
                float(x) * float(w) for x, w in zip(phi, h["weights"], strict=True)
            )
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            theta = [float(w) * a for w in h["weights"]]
            theta[0] += b
            logistic_z = math.fsum(float(x) * w for x, w in zip(phi, theta, strict=True))
            lp = (
                1 / (1 + math.exp(-logistic_z))
                if logistic_z >= 0
                else math.exp(logistic_z) / (1 + math.exp(logistic_z))
            )
            action = lambda prob: (
                producer.energy.decision(prob, h["policy"]["thresholds"])
                if "policy" in h
                else producer.energy.base.action(prob)
            )
            if abs(p - row["p"]) > 1e-12 or action(p) != row["action"]:
                raise ValueError("independent_prediction_drift")
            parity.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm=row["arm"],
                    energy_probability=p,
                    logistic_probability=lp,
                    energy_action=action(p),
                    logistic_action=action(lp),
                    passed=abs(p - lp) <= 1e-10 and action(p) == action(lp),
                )
            )
        independent.append(score(row, targets[row["unit_id"]]["y"]))
    return dict(
        rows=independent,
        **statistics(independent),
        energy_logistic_parity_rows=parity,
        equivalent_logistic_parity=dict(
            passed=bool(parity) and all(r["passed"] for r in parity),
            maximum_absolute_error=max(
                (abs(r["energy_probability"] - r["logistic_probability"]) for r in parity),
                default=None,
            ),
        ),
        source_dependence=dict(
            independent_unit="original_source_cluster",
            source_count=128,
            accepted_sentence_records=len(features["sentence_rows"]),
            sentence_independent_count=0,
            invalid_indices="transport_failure_never_semantic_gold",
        ),
    )


def measure(root: Path, raw: Path, *, fixture: bool = False, mutation: str = "") -> Json:
    """Authenticate immutable inputs before opening evaluator-only labels."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "frozen_configuration.json", CONFIG)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        raw_shard_hashes=[],
        trained_head_specs=[],
        historical_model_provenance={},
        cited_upstream_artifacts=[],
    )
    path = root / UPSTREAM
    gate = lambda p, field, expected, observed: producer.trained.gate(
        work, p, field, expected, observed
    )
    read = lambda ref: producer.trained.bind(work, ref, raw)
    progress("before_input_custody")
    try:
        if fixture:
            if mutation:
                gate(path, "evaluation_capture_ready_score", 1, 0)
            capture = producer.measure(root, raw / "private_capture", fixture=True)
            for i, s in enumerate(capture["slots"]):
                s.update(source_id=str(i), response_id=str(i))
            predictions = producer.predict(
                producer.reduce(capture["slots"], capture["calls"], capture["plan"]["baseline"])[
                    "feature_rows"
                ],
                capture["plan"]["heads"],
            )
        else:
            retired_path = ROOT / "ops/exclusion_manifest.yaml"
            retired = yaml.safe_load(retired_path.read_text())
            ids = {
                r.get("experiment_id")
                for k in ("retired", "retired_experiments")
                for r in retired.get(k, [])
            }
            gate(retired_path, "upstream_not_retired", True, not ({8184, 8183, 8098} & ids))
            gate(
                Path(sys.executable), "python_runtime_supported", True, sys.version_info >= (3, 11)
            )
            gate(path, "upstream_exists", True, path.is_file())
            value = read(dict(path=str(path), sha256=PIN))
            for field, expected in [
                ("evaluation_capture_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                gate(path, field, expected, value.get(field))
            gate(
                path,
                "upstream_terminal_passed",
                True,
                read_bound_sidecar(path, producer.historical.fit.publication_sidecar(value))[
                    "report"
                ]["passed"],
            )
            gate(path, "prediction_cold_replay", True, producer.replay(path))
            seal = read(value["prediction_manifest"])
            gate(path, "labels_opened", False, seal["labels_opened"])
            predictions = seal["rows"]
            capture = read(value["measurement_reference"])
            gate(path, "comparator_id", "radial16", capture["plan"]["comparator_id"])
            for ref in [*value["raw_shard_hashes"], *value["source_artifact_hashes"]]:
                p = Path(ref["path"])
                gate(p, "primitive_sha256", ref["sha256"], sha256_file(p) if p.is_file() else None)
                work["refs"].append(ref)
            work["trained_head_specs"] = value["imported_trained_head_specs"]
            work["historical_model_provenance"] = dict(
                MODEL_SPECS=value["MODEL_SPECS"],
                imported_invocation_counts=value["model_invocation_counts"],
                scope="historical_only",
            )
            work["cited_upstream_artifacts"] = [
                dict(
                    experiment_id=8184,
                    sha256=PIN,
                    fields_imported=[
                        "prediction_manifest",
                        "frozen_head_manifest",
                        "measurement_reference",
                    ],
                )
            ]
        clock = dict(predictions_sealed_ns=time.time_ns())
        prediction_path = raw / "authenticated_predictions.json"
        atomic_json(prediction_path, dict(rows=predictions))
        work["raw_shard_hashes"].append(reference(prediction_path))
        progress("predictions_authenticated_before_labels", 128, 0)
        clock["labels_opened_ns"] = time.time_ns()
        if fixture:
            originals = [
                dict(
                    id=str(i),
                    source_id=str(i),
                    response=bytes.fromhex(s["answer_bytes"]).decode(),
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
                for i, s in enumerate(capture["slots"])
            ]
            targets = [
                dict(s, **base.human.target(originals[i], bytes.fromhex(s["answer_bytes"]))[0])
                for i, s in enumerate(capture["slots"])
            ]
            label_provenance = dict(scope="private_oracle_fixture")
        else:
            cohort = read(dict(path=str(root / base.COHORT), sha256=base.COHORT_PIN))
            label_ref = cohort["evaluator_label_manifests"]["evaluation"]
            labels = read(label_ref)
            targets, originals = labels["rows"], labels["original_response_records"]
            label_provenance = dict(
                original_manifest=label_ref,
                independently_reconstructed=True,
                target="original_answer_level_human_annotations",
                new_labels=0,
            )
            diagnostic = read(
                dict(
                    path=str(root / "results/experiment_8156_v705_decision_audit.json"),
                    sha256="sha256:65a13925f00541b524172bb9c3ea2be765e078cee8bafda3d55619b9a8a4de13",
                )
            )
            work["source_diagnostics"] = diagnostic["source_diagnostics"]
            work["cited_upstream_artifacts"].extend(
                [
                    dict(
                        experiment_id=8098,
                        sha256=base.COHORT_PIN,
                        fields_imported=["evaluator_label_manifests.evaluation"],
                    ),
                    dict(
                        experiment_id=8156,
                        sha256="sha256:65a13925f00541b524172bb9c3ea2be765e078cee8bafda3d55619b9a8a4de13",
                        fields_imported=["source_diagnostics"],
                    ),
                ]
            )
        work["evidence"] = dict(
            capture=capture,
            predictions=predictions,
            targets=targets,
            original_response_records=originals,
            clock=clock,
        )
        work["label_provenance"] = label_provenance
        reduced = reduce(work["evidence"])
        for name, value in [
            ("decision_evidence", work["evidence"]),
            ("independent_reduction", reduced),
        ]:
            atomic_json(raw / (name + ".json"), value)
            work["raw_shard_hashes"].append(reference(raw / (name + ".json")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not work["checks"] or work["checks"][-1]["passed"]:
            try:
                gate(path, "input_custody", "authenticated_original_rows", str(error))
            except ValueError:
                pass
        work["evidence"] = {}
    work.update(
        duration_s=time.monotonic() - began,
        source_artifact_hashes=work["refs"],
        code_config_hashes=[
            reference(ROOT / p) for p in [*OWNED, TEST, producer.MODULE, producer.trained.NUMERIC]
        ]
        + [reference(raw / "frozen_configuration.json")],
        phase_spans=[
            dict(
                phase="custody_and_independent_reduction",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("after_measurement", int(bool(work["evidence"])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness requires owned normal exits; decision utility has separate gates."""
    reduced = reduce(work["evidence"]) if work["evidence"] else dict(rows=[], **statistics([]))
    failures = [r for r in work["checks"] if not r["passed"]]
    checked = bool(receipts) and all(r.get("passed") is True for r in receipts)
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
        else "sentence_decision_signal"
        if reduced["h1_development_signal_score"]
        else "sentence_decision_null"
    )
    n = reduced["eligible_count"]
    masks = work["evidence"].get("capture", {}).get("slots", [])
    actual = (
        producer.reduce(
            masks,
            work["evidence"]["capture"]["calls"],
            work["evidence"]["capture"]["plan"]["baseline"],
        )
        if masks
        else {}
    )
    failed, censored = actual.get("failed_count", 0), actual.get("censored_count", 0)
    value: Json = dict(
        experiment_id=8185,
        task_id=TASK,
        milestone="2026.10.707",
        run_date=RUN_DATE,
        honest_verdict=f"complete_{verdict}_{suffix}",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        claim_scope="independent exposed source-level decision audit",
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
        cited_upstream_artifacts=work["cited_upstream_artifacts"],
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
            dict(config=CONFIG, refs=work["refs"], rows=reduced["rows"])
        ),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=CONFIG,
        field_principles=dict(
            readiness="Normal owned validation differs from H1 utility.",
            scope="Exposed development cannot close GAP-ORACLE-DISTINCT, authorize generator training or prove unseen-generator transfer.",
            rows="Original source masks and typed costs retain all128 slots.",
            inference="Zero current calls; imported model/head provenance is historical.",
        ),
        decision_audit_ready_score=ready,
        calibration_rows=[r for r in reduced["rows"] if r["arm"] != "always_escalate"],
        typed_cost_rows=reduced["rows"],
        label_provenance=work.get("label_provenance", {}),
        oracle_distinct_corrigendum=dict(
            date="2026-09-28",
            gap="GAP-ORACLE-DISTINCT",
            closed=False,
            prior_data_exposure=True,
            generator_training_authorized=False,
            unseen_generator_transfer_established=False,
        ),
        source_diagnostics=dict(
            scope="historical_fit_source_interventions_only_no_sentence_transfer_claim",
            semantic_gold=False,
            imported=work.get("source_diagnostics", {}),
        ),
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("repository_health", {}),
        methodology_note="Independent original human labels; frozen radial16 comparator; all128 original slots with missing escalated; source bootstrap10000, seed7078185, one-sided97.5 percent; complete-case calibration, coverage and logistic parity are descriptive exposed development.",
        **reduced,
    )
    value["h1_development_signal_score"] *= ready
    return value


def replay(path: Path) -> bool:
    """Rehash custody and recompute every headline so rehashed forgeries fail."""
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
        if work["evidence"]:
            primitives = {
                Path(r["path"]).stem: json.loads(Path(r["path"]).read_text())
                for r in work["raw_shard_hashes"]
                if Path(r["path"]).stem
                in ("authenticated_predictions", "decision_evidence", "independent_reduction")
            }
            base.audit.equal(primitives["decision_evidence"], work["evidence"])
            base.audit.equal(
                primitives["authenticated_predictions"]["rows"], work["evidence"]["predictions"]
            )
            base.audit.equal(primitives["independent_reduction"], reduce(work["evidence"]))
        return bool(
            value
            == build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
                fixture=value["fixture_protocol_only"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Keep static tool file paths and coverage scope frozen before measurement."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 180
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
    specs["repository_health"]["deadline_s"] = 300
    return specs


def main(argv: list[str] | None = None) -> int:
    """Reuse the qualified supervisor and publisher with this experiment identity."""
    with ExitStack() as stack:
        for name in [
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
        ]:
            stack.enter_context(patch.object(base, name, globals()[name]))
        return int(base.main(argv))
