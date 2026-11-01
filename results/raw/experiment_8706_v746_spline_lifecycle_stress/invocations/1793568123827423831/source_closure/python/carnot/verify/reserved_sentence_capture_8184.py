"""REQ-VERIFY-8184: seal reserved predictions before independent label access.

Every source remains in the comparison. Transport and frozen head execution
certify capture only; previously exposed development cannot establish benefit.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import fit_sentence_capture_8182 as fit
from carnot.verify import reserved_evidence_capture_8155 as historical
from carnot.verify import sentence_energy_8183 as energy
from carnot.verify import sentence_energy_fit_8183 as trained
from carnot.verify import sentence_transport_8179 as transport

Json = dict[str, Any]
ROOT = fit.ROOT
NAME = "experiment_8184_v707_reserved_sentence_capture"
TASK = "exp8184-reserved-sentence-capture"
MODULE = "python/carnot/verify/reserved_sentence_capture_8184.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_reserved_sentence_capture_8184.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261006"
MODEL_SPECS = fit.MODEL_SPECS
UPSTREAM = "results/experiment_8183_v707_sentence_energy_fit.json"
CONTROL = "results/experiment_8155_v705_reserved_evidence_capture.json"
PINS = {
    UPSTREAM: "sha256:a446ca982fbe34ac06ecd68d19c31da0c305bd9d53da61598019a30b6310d7ff",
    fit.canary.UPSTREAM: fit.canary.PIN,
    fit.UPSTREAM: fit.PIN,
    CONTROL: "sha256:6a4311de73e3fca6e8fbc756f4e80824eabba4ab962936e08767359b93713673",
}
CONFIG = dict(fit.canary.CONFIG, maximum_calls=256)
canary = fit.canary
reference = fit.reference
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counters so model and child work stays observable."""
    print(f"[exp8184] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(rows: list[Json]) -> list[Json]:
    """Keep original slots and reject labels before dispatch to the worker."""
    slots = [deepcopy(r) for r in rows if r["role"] == "evaluation"]
    if len(slots) != 128 or len({s["source_cluster_id"] for s in slots}) != 128:
        raise ValueError("original128_reserved_slots")
    for i, slot in enumerate(slots):
        if (
            slot["slot"] != i + 1
            or any(slot.get(k) is not None for k in ("y", "human_target", "entailment_label"))
            or len(slot["requests"]) > 2
        ):
            raise ValueError("slot_or_label_access")
        slot.update(arms=["grammar"], condition="original", human_target=None)
    return slots


def baseline(calls: list[Json]) -> list[Json]:
    """Recover twelve public signals without opening historical human targets."""
    spans = {c["unit_id"]: c for c in calls if c["arm"] == "source_span"}
    rows = historical.pairs(calls)
    for row in rows:
        span = spans[row["unit_id"]]
        row.update(source_bytes=span["source_bytes"], answer_bytes=span["answer_bytes"], x=None)
        if row["status"] == "completed":
            lexical = historical.fit.lexical.extract(
                dict(
                    family_id=row["unit_id"],
                    source_bytes=span["source_bytes"],
                    answer_bytes=span["answer_bytes"],
                )
            )
            if lexical["values"] is not None:
                h, s = np.clip(
                    [row["holistic_probability"], row["span_probability"]], 1e-6, 1 - 1e-6
                )
                row["x"] = [
                    float(np.log(h / (1 - h))),
                    *lexical["values"],
                    float(np.log(s / (1 - s))),
                    span["parsed"]["valid_quote"],
                    span["parsed"]["quote_source_byte_ratio"],
                ]
    return rows


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Seal responses and features before heads, with zero evaluator access."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    plan = (
        inputs(root, raw)
        if not fixture
        else dict(checks=[], refs=[], upstream=[], slots=[], baseline=[], heads=[])
    )
    result: Json = dict(rows=[], checks=[])
    if fixture:
        prepared = transport.requests(
            dict(
                source_bytes=b"Evidence. More evidence.".hex(),
                answer_bytes=b"First claim. Second claim.".hex(),
            ),
            lambda _: 30,
        )
        plan["slots"] = freeze(
            [
                dict(
                    prepared,
                    unit_id=f"evaluation{i}",
                    source_cluster_id=f"cluster{i}",
                    role="evaluation",
                    slot=i + 1,
                    source_bytes=b"Evidence. More evidence.".hex(),
                    answer_bytes=b"First claim. Second claim.".hex(),
                )
                for i in range(128)
            ]
        )
        plan["baseline"] = [dict(s, x=[0.0] * 12, status="completed") for s in plan["slots"]]
        g = energy.geometry(np.zeros((2, 16)), ["fit-a", "fit-b"])
        plan["heads"] = [
            dict(
                arm=a,
                geometry=g if a in energy.NEW_ARMS else g["ablation_geometry"],
                weights=[0.0]
                * energy.design(
                    a, np.zeros((1, 16)), g if a in energy.NEW_ARMS else g["ablation_geometry"]
                ).shape[1],
                calibration=[0.0, 1.0],
                policy=dict(thresholds=[0.2, 0.6]),
            )
            for a in [*energy.NEW_ARMS, *energy.CONTROL_ARMS]
        ]
        result["rows"] = canary.capture(
            plan["slots"], fit.FixtureRuntime(), raw / "slots", dict(fixture=True)
        )
    elif all(c["passed"] for c in plan["checks"]):

        def pulse(phase: str, completed: int = 0, pending: int = 0) -> None:
            progress(
                phase,
                completed,
                sum(len(s["requests"]) for s in plan["slots"]) - completed
                if phase == "after_CPU_grammar_preflight"
                else pending,
            )

        with patch.object(canary, "progress", pulse):
            canary.preflight(plan, raw)
            canary.gate(
                plan,
                root / UPSTREAM,
                "qualified_transport_identity",
                plan["qualified_identity"],
                plan.get("identity"),
            )
            if all(c["passed"] for c in plan["checks"]):
                plan["started"] = began
                progress("before_live_capture")
                with patch.object(canary, "TASK", TASK):
                    result = canary.live(plan, raw)
                progress("after_live_capture", len(result["rows"]), 0)
    if mutation:
        canary.gate(plan, root, "private_" + mutation, "unchanged", mutation)
    atomic_json(raw / "primitive_calls.json", dict(rows=result["rows"]))
    reduced = reduce(plan["slots"], result["rows"], plan["baseline"])
    atomic_json(raw / "complete_features.json", reduced)
    progress("responses_and_features_sealed", len(reduced["feature_rows"]), 0)
    atomic_json(
        raw / "frozen_heads.json",
        dict(heads=plan["heads"], comparator_id=plan.get("comparator_id", "radial16")),
    )
    predictions = predict(reduced["feature_rows"], plan["heads"])
    atomic_json(raw / "sealed_predictions.json", dict(rows=predictions, labels_opened=False))
    progress("all_predictions_sealed_before_evaluation", len(predictions), 0)
    work = dict(
        plan=plan,
        slots=plan["slots"],
        calls=result["rows"],
        checks=[*plan["checks"], *result["checks"]],
        refs=plan["refs"],
        upstream=plan["upstream"],
        fixture=fixture,
        live_result=result,
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_capture_feature_predict_seal",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                fit.MODULE,
                canary.MODULE,
                trained.NUMERIC,
                historical.fit.NUMERIC,
                "python/carnot/verify/sentence_transport_8179.py",
            ]
        ],
        raw_shard_hashes=[
            reference(raw / p)
            for p in [
                "primitive_calls.json",
                "complete_features.json",
                "frozen_heads.json",
                "sealed_predictions.json",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Certify custody separately from the later independent decision audit."""
    plan = work["plan"]
    reduced = reduce(work["slots"], work["calls"], plan["baseline"])
    predictions = predict(reduced["feature_rows"], plan["heads"])
    failed = [c["check"] for c in work["checks"] if not c["passed"]]
    valid = bool(receipts) and all(r["passed"] for r in receipts)
    live = work["live_result"]
    ready = int(
        valid
        and not failed
        and not work["fixture"]
        and len(predictions) == 128 * 8
        and live.get("model_loads_completed") == 1
        and work["duration_s"] >= 10
    )
    verdict = "disqualified" if not valid else "blocked" if failed else "null"
    value = dict(
        experiment_id=8184,
        task_id=TASK,
        milestone="2026.10.707",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (failed[0] if verdict == "blocked" else "reserved_sentence_capture"),
        verdict_class=verdict,
        verifier_is_oracle=work["fixture"],
        claim_scope="Sealed reserved sentence predictions only; decision benefit awaits independent reader",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=valid,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        gate_check_summary=work["checks"],
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=[],
        imported_trained_head_specs=plan.get("trained_head_specs", []),
        inference_substrate="live_llm_inference"
        if live.get("model_loads_attempted")
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation"
        if live.get("model_loads_attempted")
        else "no_model_load",
        planned_inference_substrate_class="model_bounded_generation",
        model_invocation_counts=dict(
            model_loads_attempted=live.get("model_loads_attempted", 0),
            model_loads_completed=live.get("model_loads_completed", 0),
            generate=0 if work["fixture"] else len(live.get("runtime_receipts", [])),
        ),
        call_ledger=work["calls"],
        cited_upstream_artifacts=work["upstream"],
        sample_size_budget=dict(
            intended=128, bounded_calls=256, input_tokens=6000, output_tokens=256
        ),
        duration_s=work["duration_s"],
        random_seed=CONFIG["seed"],
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=dict(
            capture="normal owned validation and qualified frozen predictions",
            support=96,
            budgets=CONFIG,
        ),
        evaluation_capture_ready_score=ready,
        evaluation_support_score=int(ready and reduced["completed_count"] >= 96),
        prediction_manifest=work["raw_shard_hashes"][3],
        frozen_head_manifest=work["raw_shard_hashes"][2],
        label_access_ledger=[],
        evaluator_targets_opened=False,
        comparator_id=plan.get("comparator_id", "radial16"),
        prediction_rows=predictions,
        model_receipt={k: v for k, v in live.items() if k != "rows"},
        fixture_protocol_only=work["fixture"],
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("global_health", {}),
        **reduced,
    )
    value["methodology_note"] = (
        "Frozen indexed grammar transport, source-matched historical twelve signals plus four local predictions, unchanged calibrated heads and typed policies; no labels or decision benefit measured."
    )
    value["field_principles"] = {
        k: "Bind actual source custody and sealed predictions; exposed development grants no independent benefit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Independently rebuild all predictions and counts from immutable primitives."""
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
        for r in value["validation_receipts"]:
            if r.get("log_path") and sha256_file(Path(r["log_path"])) != r["log_sha256"]:
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        calls, features, heads, predictions = [
            json.loads(Path(r["path"]).read_text()) for r in work["raw_shard_hashes"]
        ]
        if calls["rows"] != work["calls"] or heads != dict(
            heads=work["plan"]["heads"], comparator_id=work["plan"].get("comparator_id", "radial16")
        ):
            return False
        for slot in work["slots"]:
            counts = {r["payload"]: r["input_tokens"] for r in slot["requests"]}
            if (
                transport.requests(slot, lambda text: counts.get(text, 0))["requests"]
                != slot["requests"]
            ):
                return False
        actual = reduce(work["slots"], work["calls"], work["plan"]["baseline"])
        if features != actual or predictions != dict(
            rows=predict(actual["feature_rows"], heads["heads"]), labels_opened=False
        ):
            return False
        return bool(
            value
            == build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze real argv and owned statement scope before measurement begins."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
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
    """Use the qualified child supervisor and checked publication from any cwd."""
    original_publish = execution.publish

    def preserve(
        value: Json, output: Path, private: Path, raw: Path, terminal: list[Json], fixture: bool
    ) -> None:
        if output.exists():
            (raw / "preserved_historical_primary.json").write_bytes(output.read_bytes())
        original_publish(value, output, private, raw, terminal, fixture)

    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", supervise),
        patch.object(execution, "publish", preserve),
    ):
        return execution.main(argv)


def supervise(
    root: Path, spec: Json, private: Path, durable: Path, *, heartbeat_s: float = 20
) -> Json:
    """Retain the qualified supervisor and make its real pending child explicit."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    def pulse(name: str, event: str, started: float, detail: str = "") -> None:
        complete = int(event == "after_subprocess")
        progress(event + "_" + name + "_" + detail, complete, 1 - complete)

    with patch.object(scope, "_progress", pulse):
        return canary.supervise(root, spec, private, durable, heartbeat_s=heartbeat_s)


def inputs(root: Path, raw: Path) -> Json:
    """Bind immutable heads and public manifests, never evaluator label paths."""
    plan: Json = dict(checks=[], refs=[], upstream=[], slots=[], heads=[], baseline=[])
    progress("before_input_authentication")
    try:
        retired_path = ROOT / "ops/exclusion_manifest.yaml"
        retired = yaml.safe_load(retired_path.read_text())
        ids = {
            r.get("experiment_id")
            for k in ("retired", "retired_experiments")
            for r in retired.get(k, [])
        }
        trained.gate(
            plan, retired_path, "upstream_not_retired", True, not ({8155, 8179, 8181, 8183} & ids)
        )
        values = {}
        for name, pin in PINS.items():
            path = root / name
            value = trained.bind(plan, dict(path=str(path), sha256=pin), raw)
            for field, expected in (
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ):
                trained.gate(plan, path, field, expected, value.get(field))
            trained.gate(
                plan,
                path,
                "upstream_terminal_passed",
                True,
                read_bound_sidecar(path, historical.fit.publication_sidecar(value))["report"][
                    "passed"
                ],
            )
            values[name] = value
            plan["upstream"].append(
                dict(
                    experiment_id=value["experiment_id"],
                    sha256=pin,
                    fields_imported=["frozen_public_capture_evidence"],
                )
            )
            if name == UPSTREAM:
                trained.gate(
                    plan, path, "energy_fit_ready_score", 1, value.get("energy_fit_ready_score")
                )
        head, methods, qualified, control = (
            values[n] for n in (UPSTREAM, fit.canary.UPSTREAM, fit.UPSTREAM, CONTROL)
        )
        sealed = trained.bind(plan, head["frozen_head_manifest"], raw)
        for ref in sealed["prediction_code_hashes"]:
            fit.bind(plan, ref, raw)
        for ref in head["code_config_hashes"]:
            if ref["path"].endswith((trained.NUMERIC, historical.fit.NUMERIC)):
                trained.gate(
                    plan,
                    Path(ref["path"]),
                    "prediction_runtime_sha256",
                    ref["sha256"],
                    sha256_file(Path(ref["path"])),
                )
        for name, digest in control["code_config_hashes"].items():
            ref = dict(path=str(ROOT / name), sha256=digest)
            if ref["path"].endswith(
                (
                    historical.capture.MODULE,
                    "python/carnot/verify/evidence_features_7980.py",
                    "python/carnot/verify/evidence_protocol_8124.py",
                )
            ):
                trained.gate(
                    plan,
                    Path(ref["path"]),
                    "historical_feature_code_sha256",
                    ref["sha256"],
                    sha256_file(Path(ref["path"])),
                )
        plan.update(
            heads=sealed["heads"], comparator_id=sealed["comparator_id"], frozen_heads=sealed
        )
        trained.gate(
            plan,
            root / UPSTREAM,
            "six_frozen_heads",
            [*energy.NEW_ARMS, *energy.CONTROL_ARMS],
            [h["arm"] for h in plan["heads"]],
        )
        trained.gate(
            plan,
            root / UPSTREAM,
            "comparator_identity",
            head["comparator_id"],
            sealed["comparator_id"],
        )
        plan["config"] = trained.bind(
            plan, dict(path=methods["protocol_path"], sha256=methods["protocol_sha256"]), raw
        )
        public = trained.bind(plan, methods["source_manifest"]["evaluation"], raw)
        requests = trained.bind(plan, methods["raw_shard_hashes"][0], raw)["rows"]
        plan["slots"] = freeze(requests)
        trained.gate(
            plan,
            root / UPSTREAM,
            "reserved_roster",
            [r["unit_id"] for r in public["roster"]],
            [s["unit_id"] for s in plan["slots"]],
        )
        for slot, original in zip(plan["slots"], public["request_rows"], strict=True):
            trained.gate(
                plan,
                root / UPSTREAM,
                "public_source_identity",
                [original["family_id"], original["source_bytes"], original["answer_bytes"]],
                [slot["unit_id"], slot["source_bytes"], slot["answer_bytes"]],
            )
        plan["baseline"] = baseline(trained.bind(plan, control["raw_shard_hashes"][1], raw)["rows"])
        plan["tokenizer_receipt"] = methods["tokenizer_receipt"]
        plan["qualified_identity"] = qualified["qualified_transport_configuration"]["identity"]
        plan["trained_head_specs"] = head["trained_head_specs"]
        fit.bind(
            plan,
            dict(path=str(ROOT / fit.historical.METHOD), sha256=fit.historical.METHOD_HASH),
            raw,
        )
        reduce(plan["slots"], [], plan["baseline"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in plan["checks"]):
            canary.gate(
                plan,
                root / UPSTREAM,
                "input_structure",
                "authenticated_reserved_public_evidence",
                str(error),
            )
    progress("after_input_authentication", len(plan["slots"]), 128 - len(plan["slots"]))
    return plan


def reduce(slots: list[Json], calls: list[Json], controls: list[Json]) -> Json:
    """Reparse exact responses and preserve every original source mask."""
    rows, sentences, features = [], [], []
    by_id = {r["unit_id"]: r for r in controls}
    for slot in slots:
        group = [c for c in calls if c["unit_id"] == slot["unit_id"]]
        if any(
            c["request"] != r or c["source_cluster_id"] != slot["source_cluster_id"]
            for c, r in zip(group, slot["requests"])
        ):
            raise ValueError("request_or_source_identity")
        old = by_id.get(slot["unit_id"])
        if old and any(
            old[k] != slot[k] for k in ("source_cluster_id", "source_bytes", "answer_bytes", "role")
        ):
            raise ValueError("historical_source_identity")
        predicted = [p for c in group for p in fit.accepted(slot, c)]
        expected = [i for r in slot["requests"] for i in r["sentence_indices"]]
        complete = (
            bool(expected)
            and len(group) == len(slot["requests"])
            and [p["sentence_index"] for p in predicted] == expected
            and old is not None
            and len(old.get("x") or []) == 12
        )
        status = (
            "completed"
            if complete
            else "excluded"
            if not expected
            else "censored"
            if group and all(c["status"] == "censored" for c in group)
            else "failed"
        )
        row = dict(
            unit_id=slot["unit_id"],
            source_cluster_id=slot["source_cluster_id"],
            role="evaluation",
            slot=slot["slot"],
            arm="grammar",
            condition="original",
            metric="complete_paired_source",
            numerator=int(complete),
            denominator=1,
            status=status,
            exclusion_reason=None
            if complete
            else slot.get("exclusion_reason") or "incomplete_paired_source",
        )
        rows.append(row)
        sentences.extend(
            dict(
                p,
                unit_id=slot["unit_id"],
                source_cluster_id=slot["source_cluster_id"],
                offsets=slot["sentences"][p["sentence_index"]],
                source_complete=complete,
            )
            for p in predicted
        )
        local = (
            [
                sum(p["p_unsupported"] for p in predicted) / len(predicted),
                max(p["p_unsupported"] for p in predicted),
                sum(p["relation"] == "C" for p in predicted) / len(predicted),
                sum(p["relation"] == "B" for p in predicted) / len(predicted),
            ]
            if complete
            else []
        )
        feature = dict(
            row,
            x=[*old["x"], *local] if complete else None,
            source_sha256=canonical_hash(slot["source_bytes"]),
            answer_sha256=canonical_hash(slot["answer_bytes"]),
        )
        features.append(dict(feature, feature_sha256=canonical_hash(feature)))
    counts = Counter(r["status"] for r in rows)
    return dict(
        rows=rows,
        sentence_rows=sentences,
        feature_rows=features,
        intended_count=128,
        independent_count=len({s["source_cluster_id"] for s in slots}),
        eligible_count=sum(bool(s["requests"]) for s in slots),
        completed_count=counts["completed"],
        excluded_count=counts["excluded"] if slots else 128,
        failed_count=counts["failed"],
        censored_count=counts["censored"],
    )


def predict(features: list[Json], heads: list[Json]) -> list[Json]:
    """Apply frozen heads with no fitting, label lookup or comparator selection."""
    rows = []
    for source in features:
        for head in [*heads, dict(arm="equivalent_logistic"), dict(arm="always_escalate")]:
            arm = head["arm"]
            h = (
                next(h for h in heads if h["arm"] == "local_evidence_radial16")
                if arm == "equivalent_logistic"
                else head
            )
            p = None
            if source["x"] is not None and arm != "always_escalate":
                phi = energy.design(h["arm"], np.asarray([source["x"]]), h["geometry"])
                b, a = h["calibration"]
                p = float(energy.base.expit(b + a * float((phi @ np.asarray(h["weights"]))[0])))
            action = (
                energy.decision(p, h["policy"]["thresholds"])
                if "policy" in h
                else energy.base.action(p)
            )
            row = dict(
                unit_id=source["unit_id"],
                source_cluster_id=source["source_cluster_id"],
                arm=arm,
                condition="original",
                metric="sealed_probability",
                numerator=p,
                denominator=int(p is not None),
                status=source["status"],
                exclusion_reason=source["exclusion_reason"],
                p=p,
                action=action,
                feature_sha256=source["feature_sha256"],
                head_sha256=canonical_hash(h),
            )
            rows.append(dict(row, prediction_sha256=canonical_hash(row)))
    return rows
