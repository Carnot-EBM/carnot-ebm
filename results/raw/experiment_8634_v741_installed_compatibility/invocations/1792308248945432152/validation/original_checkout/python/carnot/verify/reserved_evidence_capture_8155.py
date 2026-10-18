"""REQ-VERIFY-8155: seal reserved evidence without opening evaluator targets.

The qualified worker owns CUDA, deadlines and cleanup. This adapter applies
already frozen heads; exposed development cannot establish generalization.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import evidence_energy_fit_8154 as fit
from carnot.verify import fit_evidence_capture_8153 as capture

Json = dict[str, Any]
ROOT = capture.ROOT
NAME = "experiment_8155_v705_reserved_evidence_capture"
TASK = "exp8155-reserved-evidence-capture"
MODULE = "python/carnot/verify/reserved_evidence_capture_8155.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_reserved_evidence_capture_8155.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261005"
MODEL_SPECS = capture.MODEL_SPECS
CONFIG = dict(capture.CONFIG, seed=70555, main_calls=256, diagnostic_calls=0)
UPSTREAM = "results/experiment_8154_v705_evidence_energy_fit.json"
PIN = "sha256:5ab0405b7eb9d06fb461bbde2038f88e2c44c32a29ab02b2f650983521200350"
reference = capture.reference
execution = capture.execution


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counters so child waits cannot masquerade as model work."""
    print(f"[exp8155] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(rows: list[Json]) -> list[Json]:
    """Keep original evaluation order and reject prompt or role substitutions."""
    selected = [r for r in rows if r["role"] == "evaluation"]
    if len(selected) != 256:
        raise ValueError("original128_evaluation_roles")
    seen: set[str] = set()
    units: set[str] = set()
    slots = []
    for i in range(0, 256, 2):
        pair = selected[i : i + 2]
        if (
            len({r["unit_id"] for r in pair}) != 1
            or len({r["source_cluster_id"] for r in pair}) != 1
            or pair[0]["unit_id"] in units
            or pair[0]["source_cluster_id"] in seen
            or {r["arm"] for r in pair} != {"holistic", "source_span"}
            or [r["order"] for r in pair] != [0, 1]
        ):
            raise ValueError("pair_identity")
        seen.add(pair[0]["source_cluster_id"])
        units.add(pair[0]["unit_id"])
        for row in pair:
            expected = capture.protocol.protocol()["prompt_prefixes"][row["arm"]] + json.dumps(
                dict(
                    source=bytes.fromhex(row["source_bytes"]).decode(),
                    answer=bytes.fromhex(row["answer_bytes"]).decode(),
                ),
                ensure_ascii=False,
            )
            if (
                row["prompt"] != expected
                or row["prompt_sha256"] != canonical_hash(expected)
                or row.get("entailment_label") is not None
                or row.get("human_target") is not None
            ):
                raise ValueError("prompt_or_target_identity")
            slots.append(
                dict(
                    row,
                    slot=i // 2 + 1,
                    call_id=row["unit_id"] + ":" + row["arm"],
                    condition="original",
                    human_target=None,
                    request=dict(
                        model=MODEL_SPECS[0],
                        messages=[dict(role="user", content=expected)],
                        temperature=0,
                        top_p=1,
                        seed=CONFIG["seed"],
                        max_tokens=128,
                        cache_prompt=False,
                        chat_template_kwargs=dict(enable_thinking=False),
                    ),
                )
            )
    return slots


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate all public roles and heads before touching the live runtime."""
    plan: Json = dict(
        checks=[],
        refs=[],
        slots=[],
        heads=[],
        protocol={},
        manifests={},
        original_role_mask=[],
        trained_head_specs=[],
        evaluator_targets_opened=False,
    )

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        capture.gate(plan, path, field, expected, observed)
        plan["checks"][-1].update(
            upstream=str(path), hash=sha256_file(path) if path.is_file() else None
        )
        if expected != observed:
            raise ValueError(field)

    def bind(ref: Json) -> Json:
        path = Path(ref["path"])
        require(path, "input_sha256", ref["sha256"], sha256_file(path) if path.is_file() else None)
        target = raw / "inputs" / (ref["sha256"].split(":")[-1] + "-" + path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        plan["refs"].append(reference(target))
        return dict(json.loads(target.read_text()))

    progress("before_input_authentication")
    try:
        values = []
        for name, pin, readiness in [
            (UPSTREAM, PIN, "energy_fit_ready_score"),
            (capture.UPSTREAM, capture.PIN, "source_protocol_ready_score"),
            (fit.UPSTREAM, fit.PIN, "fit_capture_ready_score"),
        ]:
            path = root / name
            require(path, "upstream_exists", True, path.is_file())
            value = bind(dict(path=str(path), sha256=pin))
            for field, expected in [
                (readiness, 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                require(path, field, expected, value.get(field))
            require(
                path,
                "terminal_passed",
                True,
                fit.read_bound_sidecar(path, fit.publication_sidecar(value))["report"]["passed"],
            )
            values.append(value)
        trained, source, historical = values
        sealed = bind(trained["frozen_head_manifest"])
        require(
            root / UPSTREAM, "frozen_cost_rule", fit.energy.CONFIG["costs"], sealed["decision_rule"]
        )
        require(root / UPSTREAM, "frozen_tie_rule", "escalate", sealed["tie_rule"])
        plan.update(
            heads=sealed["heads"],
            trained_head_specs=trained["trained_head_specs"],
            protocol=source["expected_runtime_identity"],
            manifests=source["source_role_manifests"],
            original_role_mask=source["source_role_masks"]["evaluation"],
            fit_runtime_libraries=historical["model_receipt"]["resolved_library"]["libraries"],
        )
        for field in ("gguf_sha256", "model_revision", "model_path"):
            require(
                root / fit.UPSTREAM,
                "fit_runtime_" + field,
                plan["protocol"][field],
                historical["model_receipt"][field],
            )
        require(
            root / capture.UPSTREAM,
            "original_role_mask_count",
            128,
            len(plan["original_role_mask"]),
        )
        for ref in [source["method_freeze"], *source["pinned_method_paths"]]:
            path = Path(ref["path"])
            require(path, "immutable_method_sha256", ref["sha256"], sha256_file(path))
            plan["refs"].append(ref)
        public = {role: bind(ref) for role, ref in plan["manifests"].items()}
        roster = public["evaluation"]["roster"]
        reserved = {r["source_cluster_id"] for r in roster}
        other = {r["source_cluster_id"] for role in ("fit", "tune") for r in public[role]["roster"]}
        require(root / capture.UPSTREAM, "role_overlap", [], sorted(reserved & other))
        rows = bind(source["capture_manifest"])["rows"]
        plan["slots"] = freeze(rows)
        require(
            root / capture.UPSTREAM,
            "original_evaluation_unit_order",
            [r["unit_id"] for r in roster],
            [r["unit_id"] for r in plan["slots"][::2]],
        )
        for slot, request in zip(
            plan["slots"][::2], public["evaluation"]["request_rows"], strict=True
        ):
            require(
                root / capture.UPSTREAM,
                "matched_public_source",
                [request["family_id"], request["source_bytes"], request["answer_bytes"]],
                [slot["unit_id"], slot["source_bytes"], slot["answer_bytes"]],
            )
        require(
            root / UPSTREAM,
            "six_frozen_heads",
            fit.energy.FITTED_ARMS,
            [h["arm"] for h in plan["heads"]],
        )
        require(
            root / UPSTREAM,
            "head_role_overlap",
            [],
            sorted(
                reserved
                & {
                    s
                    for h in plan["heads"]
                    for s in h["geometry"]["fit_source_ids"] + h["tune_source_ids"]
                }
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(r["passed"] for r in plan["checks"]):
            capture.gate(plan, root / UPSTREAM, "authenticated_inputs", True, str(error))
    plan["capture_identity"] = canonical_hash([plan["slots"], CONFIG, plan["heads"]])
    progress("after_input_authentication", len(plan["slots"]), 256 - len(plan["slots"]))
    return plan


def pairs(calls: list[Json]) -> list[Json]:
    """Use the qualified parser and retain128 original slots even without a run."""
    reduced = capture.reduce(calls)["rows"]
    return reduced if calls else [dict(r, role="evaluation") for r in reduced[:128]]


def predict(calls: list[Json], heads: list[Json]) -> list[Json]:
    """Apply immutable fit transforms and costs without consulting human targets."""
    spans = {r["unit_id"]: r for r in calls if r["arm"] == "source_span"}
    records = []
    for row in pairs(calls):
        probabilities: Json = {}
        status, reason = row["status"], row["exclusion_reason"]
        if status == "completed":
            span = spans[row["unit_id"]]
            public = dict(
                family_id=row["unit_id"],
                source_bytes=span["source_bytes"],
                answer_bytes=span["answer_bytes"],
            )
            features = fit.lexical.extract(public)
            if features["values"] is None:
                status, reason = "excluded", features["abstention"]
            else:
                h, s = np.clip(
                    [row["holistic_probability"], row["span_probability"]], 1e-6, 1 - 1e-6
                )
                x = np.asarray(
                    [
                        [
                            float(np.log(h / (1 - h))),
                            *features["values"],
                            float(np.log(s / (1 - s))),
                            span["parsed"]["valid_quote"],
                            span["parsed"]["quote_source_byte_ratio"],
                        ]
                    ]
                )
                for head in heads:
                    z = float(
                        (
                            fit.energy.design(head["arm"], x, head["geometry"])
                            @ np.asarray(head["weights"])
                        )[0]
                    )
                    b, a = head["calibration"]
                    z = b + a * z
                    weights = np.exp(np.array([0.0, z]) - max(0.0, z))
                    probabilities[head["arm"]] = float(weights[1] / weights.sum())
                probabilities["equivalent_logistic"] = probabilities["radial16"]
        for arm in fit.energy.ARMS:
            p = probabilities.get(arm)
            records.append(
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
                    action=fit.energy.action(p),
                    human_target=None,
                    source_sha256=canonical_hash(spans[row["unit_id"]]["source_bytes"])
                    if row["unit_id"] in spans
                    else None,
                )
            )
    return records


def fixture_heads() -> list[Json]:
    """Public deterministic geometry tests transport, with no natural inference credit."""
    x = np.asarray([r["x"] for r in fit.fixture_rows()])
    geometry = fit.energy.geometry(x, [f"fit-{i}" for i in range(len(x))])
    return [
        dict(
            arm=arm,
            geometry=geometry,
            calibration=[0.0, 1.0],
            weights=[0.0] * fit.energy.design(arm, x[:1], geometry).shape[1],
        )
        for arm in fit.energy.FITTED_ARMS
    ]


def fixture_public() -> list[Json]:
    """Fixed public sources exercise the real CLI without touching production data."""
    rows = []
    for i in range(128):
        for order, arm in enumerate(("holistic", "source_span")):
            source = f"é fact evaluation {i}"
            prompt = capture.protocol.protocol()["prompt_prefixes"][arm] + json.dumps(
                dict(source=source, answer="fact"), ensure_ascii=False
            )
            rows.append(
                dict(
                    unit_id=f"evaluation-{i}",
                    source_cluster_id=f"cluster-{i}",
                    role="evaluation",
                    arm=arm,
                    order=order,
                    source_bytes=source.encode().hex(),
                    answer_bytes=b"fact".hex(),
                    prompt=prompt,
                    prompt_sha256=canonical_hash(prompt),
                    entailment_label=None,
                )
            )
    return rows


def measure(root: Path, raw: Path, *, fixture: bool = False, mutation: str = "") -> Json:
    """Seal label-free calls and predictions before any evaluator can consume them."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    if fixture:
        plan = dict(
            checks=[],
            refs=[],
            slots=freeze(fixture_public()),
            heads=fixture_heads(),
            protocol={},
            manifests={},
            original_role_mask=[r["unit_id"] for r in fixture_public()[::2]],
            trained_head_specs=[],
            capture_identity="private_fixture",
            evaluator_targets_opened=False,
        )
        if mutation:
            capture.gate(
                plan,
                raw,
                "energy_fit_ready_score" if mutation == "block" else "evaluator_targets_opened",
                1 if mutation == "block" else False,
                0 if mutation == "block" else True,
            )
            plan["evaluator_targets_opened"] = mutation == "labels"
        runtime: Any = capture.FixtureRuntime()
    else:
        plan = inputs(root, raw)
        runtime = None
    plan["capture_started_monotonic"] = began
    result: Json = dict(rows=[], checks=[], ledger=[])
    progress("before_reserved_capture", 0, 256)
    with TemporaryDirectory(prefix="carnot8155-model-") as directory:
        private = Path(directory)
        with (
            patch.object(capture, "CONFIG", CONFIG),
            patch.object(capture, "TASK", TASK),
            patch.object(
                capture,
                "progress",
                lambda phase, c=0, p=0: progress(phase, c, max(0, 256 - c) if p else 0),
            ),
        ):
            if not fixture and all(r["passed"] for r in plan["checks"]):
                capture.runtime_preflight(plan, raw, private)
                for ref in plan["fit_runtime_libraries"]:
                    path = Path(ref["path"])
                    capture.gate(
                        plan,
                        path,
                        "fit_runtime_library_sha256",
                        ref["sha256"],
                        sha256_file(path) if path.is_file() else None,
                    )
            failures = [r for r in plan["checks"] if not r["passed"]]
            if not fixture and not failures:
                result = capture.live(plan, raw, private)
            if fixture or not result["rows"]:
                blocked = failures + [r for r in result["checks"] if not r["passed"]]
                ledger = capture.Ledger(raw / "unstarted_ledger.json")
                result["rows"] = capture.capture(
                    plan["slots"],
                    runtime,
                    raw / "slots",
                    plan["capture_identity"],
                    ledger=ledger,
                    blocked_reason=blocked[0]["check"] if blocked else None,
                )
    progress("after_reserved_capture", len(result["rows"]), 256 - len(result["rows"]))
    heads = reference_write(raw / "frozen_heads.json", dict(heads=plan["heads"]))
    calls = reference_write(raw / "primitive_calls.json", dict(rows=result["rows"]))
    predictions = reference_write(
        raw / "sealed_predictions.json", dict(rows=predict(result["rows"], plan["heads"]))
    )
    seal = reference_write(
        raw / "sealed_prediction_manifest.json",
        dict(
            predictions=predictions,
            calls=calls,
            heads=heads,
            capture_identity=plan["capture_identity"],
            evaluator_targets_opened=plan["evaluator_targets_opened"],
            source_hashes=plan["refs"],
            original_role_mask=plan["original_role_mask"],
            config=CONFIG,
        ),
    )
    progress("predictions_sealed_before_evaluator_access", len(result["rows"]), 0)
    work = dict(
        plan=plan,
        result=result,
        frozen_head_manifest=heads,
        sealed_prediction_manifest=seal,
        raw_shard_hashes=[heads, calls, predictions, seal],
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(phase="reserved_capture_and_seal", start_s=0, duration_s=time.monotonic() - began)
        ],
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [*OWNED, TEST, capture.MODULE, fit.NUMERIC, capture.protocol.MODULE]
        },
        owned_failure=False,
    )
    atomic_json(raw / "measurement.json", work)
    return work


def reference_write(path: Path, value: Json) -> Json:
    """Atomic immutable bytes make a cold restart independent of mutable inputs."""
    return dict(
        capture.protocol.capture_seal(
            path, value, fixture=not path.resolve().is_relative_to((ROOT / "results").resolve())
        )
    )


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness certifies sealed transport, never a favorable class distribution."""
    plan, result = work["plan"], work["result"]
    rows = pairs(result["rows"])
    predictions = predict(result["rows"], plan["heads"])
    checks = [*plan["checks"], *result["checks"]]
    failures = [r for r in checks if not r["passed"]]
    ledger = capture.Ledger(raw / "unused_ledger.json")
    ledger.rows = [] if fixture else result["ledger"]
    calls = {r["call_id"]: r for r in result["rows"] if r["started"]}
    for call in ledger.rows:
        if call["operation"] == "generation":
            row = calls[call["call_id"]]
            if call["request_sha256"] != canonical_hash(row["request"]) or call[
                "response_sha256"
            ] != canonical_hash(row["raw_response"]):
                raise ValueError("ledger_binding")
    counts = ledger.counts()
    generated = counts["generation_calls_completed"] > 0
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and not work["owned_failure"]
        and not any(
            str(r["check"]).startswith(("owned_runtime_", "owned_cleanup_", "owned_cuda_"))
            for r in failures
        )
    )
    klass = (
        "disqualified"
        if not owned
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    receipt = {k: v for k, v in result.items() if k not in ("rows", "checks", "ledger")}
    cuda = bool(
        receipt.get("model_identity_receipt", {}).get("authenticated")
        and receipt.get("resolved_library")
        and receipt.get("gpu_lease_receipt")
    )
    completed = sum(r["status"] == "completed" for r in rows)
    ready = int(
        owned
        and not failures
        and not plan["evaluator_targets_opened"]
        and completed >= 96
        and sum(r["status"] == "completed" and r["arm"] == "radial16" for r in predictions) >= 96
        and (fixture or (generated and cuda and 10 <= work["duration_s"] <= 3120))
    )
    substrate = (
        "model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load"
    )
    value = dict(
        work,
        experiment_id=8155,
        task_id=TASK,
        milestone="2026.10.705",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + klass
        + "_"
        + (
            failures[0]["check"]
            if klass == "blocked"
            else "owned_validation"
            if klass == "disqualified"
            else "reserved_evidence_sealed"
        ),
        verdict_class=klass,
        evaluation_capture_ready_score=ready,
        required_checks_passed=owned,
        flagged_adversarial=False,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        claim_scope="sealed original reserved source evidence and frozen typed predictions; no evaluated correctness or learning benefit",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=checks,
        preconditions_checked=checks,
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=plan["trained_head_specs"],
        model_invocation_counts=counts,
        call_ledger=ledger.rows,
        inference_substrate="live_llm_inference"
        if generated
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class=substrate,
        planned_inference_substrate_class="model_bounded_generation",
        inference_mode="fixture" if fixture else "live_gpu" if generated else substrate,
        model_receipt=receipt,
        rows=rows,
        source_pair_rows=rows,
        original_role_mask=plan["original_role_mask"],
        intended_count=128,
        eligible_count=sum(r["status"] != "excluded" for r in rows),
        independent_count=128 if result["rows"] else 0,
        completed_count=completed,
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        sample_size_budget=dict(
            intended=128,
            main_calls=256,
            max_tokens=128,
            input_tokens=6000,
            independent_unit="original_source_cluster",
        ),
        source_artifact_hashes=plan["refs"],
        validation_receipts=receipts,
        random_seed=CONFIG["seed"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        measurement_reference=reference(raw / "measurement.json"),
        acceptance_gates=dict(
            readiness="96 paired sources and sealed frozen predictions before evaluator access",
            owned="normal validation and100% new statements",
            runtime="same immutable fit cache, tokenizer, template, native libraries and CUDA offload",
        ),
        methodology_note="Two original complete-source prompts and unchanged parser. Frozen fit normalization, transforms, weights, calibrators and typed cost rule. No target access or refitting; class support deferred to independent evaluator.",
        field_principles=dict(
            verdict="External blocks terminate; owned failures disqualify.",
            independence="Original source clusters only; exposed development earns no generalization credit.",
            inference="Current calls are separate from imported fit provenance.",
            duration="Actual elapsed time without padding.",
            targets="Predictions seal before any evaluator target access.",
        ),
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute predictions independently and reject changed evidence or timing."""
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for name, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        refs = work["raw_shard_hashes"]
        heads, calls, predictions, seal = [
            json.loads(Path(r["path"]).read_text()) for r in refs[:4]
        ]
        if (
            heads["heads"] != work["plan"]["heads"]
            or calls["rows"] != work["result"]["rows"]
            or predictions["rows"] != predict(calls["rows"], heads["heads"])
            or seal["heads"] != refs[0]
            or seal["calls"] != refs[1]
            or seal["predictions"] != refs[2]
            or seal["evaluator_targets_opened"] != work["plan"]["evaluator_targets_opened"]
            or (seal["evaluator_targets_opened"] and any(r["started"] for r in calls["rows"]))
        ):
            return False
        slots = freeze(work["plan"]["slots"]) if work["plan"]["slots"] else []
        for slot, call in zip(slots, calls["rows"], strict=True):
            if any(
                slot[k] != call[k]
                for k in (
                    "request",
                    "prompt",
                    "prompt_sha256",
                    "unit_id",
                    "source_cluster_id",
                    "source_bytes",
                    "answer_bytes",
                )
            ):
                return False
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
    """Freeze validation argv and new-code coverage before model measurement."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 600
    specs["commands"][1].update(
        name="qualified_parser_capture_and_E2E016",
        argv=[
            str(ROOT / ".venv/bin/pytest"),
            "-n0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            "tests/python/test_fit_evidence_capture_8153.py",
            "tests/python/test_evidence_protocol_8124.py",
            "tests/python/test_experiment_7868_v683_intervention_protocol.py",
        ],
    )
    for name, flag in [("E2E016_fixture", "--fixture-e2e"), ("E2E016_replay", "--cold-replay")]:
        specs["commands"].append(
            dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                    "--date",
                    "20260929",
                    flag,
                    str(private / "intervention.json"),
                ],
                expected_exit=0,
                deadline_s=60,
            )
        )
    return specs


def checked(spec: Json, private: Path, logs: Path) -> Json:
    """Report one pending child while its normal exit and log receipt are awaited."""
    stop = threading.Event()

    def pulse() -> None:
        while not stop.wait(20):
            progress("subprocess_pending_" + spec["name"], 0, 1)

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        return dict(execution.run_check(ROOT, spec, private, logs, heartbeat_s=20))
    finally:
        stop.set()
        thread.join(timeout=1)


def main(argv: list[str] | None = None) -> int:
    """Run one capture and publish only after owned checks have normal receipts."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE], default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--mutation", choices=["", "block", "labels"], default="")
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
    with TemporaryDirectory(prefix="carnot8155-validation-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
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
                receipts.append(checked(spec, private, raw / "logs"))
                progress(
                    "after_subprocess_" + spec["name"],
                    len(receipts),
                    len(specs["commands"]) - len(receipts),
                )
            progress("before_subprocess_repository_full_suite")
            work["repository_health"] = checked(
                specs["repository_health"], private, raw / "repository_health"
            )
            progress("after_subprocess_repository_full_suite", 1, 0)
            coverage = private / "coverage.json"
            if coverage.is_file():
                target = raw / "changed_code_coverage.json"
                target.write_bytes(coverage.read_bytes())
                work["raw_shard_hashes"].append(reference(target))
            atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, fixture=fixture)
        if output.exists():
            (raw / "preserved_historical_primary.json").write_bytes(output.read_bytes())
        with patch.object(execution, "e", sys.modules[__name__]):
            execution.publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
