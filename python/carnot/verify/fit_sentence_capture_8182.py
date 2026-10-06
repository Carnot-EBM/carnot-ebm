"""REQ-VERIFY-8182: collect fixed-source evidence without fitting a head.

Syntax and source addresses remain predictions. Historical calls retain their
original clocks; neither transport nor exposed labels establish semantic benefit.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import sentence_methods_8166 as historical
from carnot.verify import sentence_transport_8179 as transport
from carnot.verify import sentence_transport_canary_8181 as canary

Json = dict[str, Any]
ROOT = canary.ROOT
NAME = "experiment_8182_v707_fit_sentence_capture"
TASK = "exp8182-fit-sentence-capture"
MODULE = "python/carnot/verify/fit_sentence_capture_8182.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_fit_sentence_capture_8182.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261006"
ROLES = dict(fit=128, tune=64)
MODEL_SPECS = canary.MODEL_SPECS
UPSTREAM = "results/experiment_8181_v707_sentence_transport_canary.json"
PIN = "sha256:a963620e48696c8b0b933b34f52d0bf925273a7802e5b75b739dd27171577a44"
CONTROL = "results/experiment_8154_v705_evidence_energy_fit.json"
CONFIG = dict(
    seed=70781,
    main_calls=384,
    diagnostic_calls=16,
    input_tokens=6000,
    output_tokens=256,
    latest_launch_s=3000,
    closure_s=4800,
    checkpoint_sources=8,
    duration_floor_s=10,
)
reference = canary.reference
BASE_CAPTURE = canary.capture
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counters so an operator can distinguish work from a stall."""
    print(f"[exp8182] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(sources: list[Json]) -> list[Json]:
    """Preserve every original slot; outcomes cannot choose replacement sources."""
    selected = [deepcopy(s) for s in sources if s["role"] in ROLES]
    if Counter(s["role"] for s in selected) != ROLES:
        raise ValueError("role_count")
    expected = [(role, i + 1) for role, n in ROLES.items() for i in range(n)]
    if [(s["role"], s["slot"]) for s in selected] != expected or len(
        {s["source_cluster_id"] for s in selected}
    ) != 192:
        raise ValueError("original_slot_or_source_cluster")
    return [dict(s, condition="original", arms=["grammar"], human_target=None) for s in selected]


def bind(plan: Json, ref: Json, raw: Path) -> Json:
    """Copy authenticated bytes into this invocation without changing history."""
    path = Path(ref["path"])
    actual = sha256_file(path) if path.is_file() else None
    canary.gate(plan, path, "primitive_sha256", ref["sha256"], actual)
    if actual != ref["sha256"]:
        raise ValueError("primitive_sha256")
    target = raw / "inputs" / (actual[7:] + "-" + path.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    result = reference(target)
    plan["refs"].append(result)
    return result


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate readiness, public source custody and immutable methods first."""
    plan = canary.inputs(root, raw)
    plan.update(slots=[], reusable=[], controls_ref=None, qualified_identity={})
    try:
        ref = bind(plan, dict(path=str(root / UPSTREAM), sha256=PIN), raw)
        value = json.loads(Path(ref["path"]).read_text())
        for field, expected in [
            ("transport_canary_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            canary.gate(plan, root / UPSTREAM, field, expected, value.get(field))
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
        sidecar = Path(terminal["publication"]["sidecar_path"])
        canary.gate(
            plan,
            root / UPSTREAM,
            "canary_terminal_passed",
            True,
            read_bound_sidecar(root / UPSTREAM, sidecar)["report"]["passed"],
        )
        plan["refs"].append(reference(sidecar))
        for primitive in value["raw_shard_hashes"]:
            bind(plan, primitive, raw)
        plan["reusable"] = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())[
            "rows"
        ]
        plan["qualified_identity"] = value["qualified_transport_configuration"]["identity"]
        methods = json.loads((root / canary.UPSTREAM).read_text())
        source_ref = methods["raw_shard_hashes"][0]
        plan["slots"] = freeze(json.loads(Path(source_ref["path"]).read_text())["rows"])
        bind(plan, methods["source_manifest"]["tune"], raw)
        control_ref = bind(
            plan, dict(path=str(root / CONTROL), sha256=historical.PINS[CONTROL]), raw
        )
        control = json.loads(Path(control_ref["path"]).read_text())
        plan["controls_ref"] = bind(plan, control["raw_shard_hashes"][0], raw)
        bind(plan, dict(path=str(root / historical.METHOD), sha256=historical.METHOD_HASH), raw)
        plan["upstream"] += [
            dict(
                experiment_id=8181,
                sha256=PIN,
                fields_imported=["qualified_transport_configuration", "raw_shard_hashes"],
            ),
            dict(
                experiment_id=8154,
                sha256=historical.PINS[CONTROL],
                fields_imported=["matched_features", "historical_paired_controls"],
            ),
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in plan["checks"]):
            canary.gate(
                plan, root / UPSTREAM, "input_structure", "qualified primitives", str(error)
            )
    return plan


def payload(request: Json) -> Json:
    """Keep canary generation parameters frozen so historical reuse is meaningful."""
    return dict(
        model=MODEL_SPECS[0],
        messages=[dict(role="user", content=request["payload"])],
        temperature=0,
        seed=CONFIG["seed"],
        max_tokens=256,
        stream=False,
        chat_template_kwargs=dict(enable_thinking=False),
        grammar=request["grammar"],
    )


def cache_key(slot: Json, request: Json, identity: Json) -> str:
    """Bind full source and answer bytes as well as every generation operand."""
    return canonical_hash(
        dict(
            payload=payload(request),
            identity=identity,
            source_bytes=slot["source_bytes"],
            answer_bytes=slot["answer_bytes"],
        )
    )


def diagnostics(slots: list[Json]) -> list[Json]:
    """Alter the fixed first group only; interventions have no inherited target."""
    first = [s for s in slots if s["role"] == "fit" and s["requests"]][:8]
    rows = []
    for i, original in enumerate(first):
        for condition in ("source_removed", "source_mismatched"):
            row = deepcopy(original)
            row.update(
                condition=condition,
                source_bytes=""
                if condition == "source_removed"
                else first[(i + 1) % len(first)]["source_bytes"],
                human_target=None,
            )
            if condition == "source_removed":
                request = deepcopy(original["requests"][0])
                content = json.loads(request["payload"])
                content.update(source="", source_segments=[])
                request["payload"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
                request["grammar"] = (
                    "\n".join(
                        'refs ::= "[]"' if line.startswith("refs ::=") else line
                        for line in request["grammar"].splitlines()
                        if not line.startswith("source ::=")
                    )
                    + "\n"
                )
                row.update(source_segments=[], requests=[request])
            else:
                prepared = transport.requests(row, lambda _: 0)
                row.update(
                    source_segments=prepared["source_segments"], requests=prepared["requests"][:1]
                )
            rows.append(row)
    return rows


def accepted(slot: Json, call: Json) -> list[Json]:
    """Reparse primitive bytes; successful HTTP transport alone is insufficient."""
    response = call.get("response") or {}
    choice = (response.get("choices") or [{}])[0]
    parsed = transport.parse(
        choice.get("message", {}).get("content") or "",
        call["request"]["sentence_indices"],
        len(slot["source_segments"]),
    )
    usage = response.get("usage", {})
    return (
        parsed["rows"]
        if (
            call["status"] == "completed"
            and choice.get("finish_reason") == "stop"
            and parsed["status"] == "completed"
            and 0 < usage.get("completion_tokens", 0) <= 256
            and 0 <= usage.get("prompt_tokens", 6001) <= 6000
        )
        else []
    )


def capture(
    slots: list[Json],
    runtime: Any,
    raw: Path,
    identity: Json,
    reused: list[Json],
    *,
    started: float | None = None,
) -> list[Json]:
    """Reuse exact qualified calls, then visit original slots once within budget."""
    began = time.monotonic() if started is None else started
    raw.mkdir(parents=True, exist_ok=True)
    slots = [dict(s, condition=s.get("condition", "original")) for s in slots]
    all_slots = [*slots, *diagnostics(slots)]
    total = sum(len(s["requests"]) for s in all_slots)
    by_key = {
        (c.get("cache_key"), c["unit_id"], c["source_cluster_id"]): c
        for c in reused
        if c.get("condition", "original") == "original"
    }
    calls = []
    for number, slot in enumerate(all_slots):
        for request in slot["requests"]:
            key = cache_key(slot, request, identity)
            old = by_key.get((key, slot["unit_id"], slot["source_cluster_id"]))
            if (
                slot["condition"] == "original"
                and old
                and old["arm"] == "grammar"
                and old["identity"] == identity
                and old["request"] == request
                and old["unit_id"] == slot["unit_id"]
                and old["source_cluster_id"] == slot["source_cluster_id"]
                and accepted(slot, old)
            ):
                call = dict(
                    deepcopy(old),
                    historical=True,
                    provenance_task="exp8181-sentence-transport-canary",
                )
            else:
                with patch.object(canary, "progress", progress):
                    call = BASE_CAPTURE(
                        [dict(slot, requests=[request], arms=["grammar"])],
                        runtime,
                        raw / f"attempt-{len(calls):03d}",
                        identity,
                        started=began,
                    )[0]
                call.update(historical=False, provenance_task=TASK)
            call.update(condition=slot["condition"], cache_key=key)
            calls.append(call)
            atomic_json(raw / f"call-{len(calls):03d}.json", call)
            progress("call_sealed", len(calls), total - len(calls))
        if (number + 1) % 8 == 0:
            atomic_json(raw / f"checkpoint-{number + 1:03d}.json", dict(rows=calls))
            progress("source_checkpoint", number + 1, len(all_slots) - number - 1)
    return calls


def reduce(slots: list[Json], calls: list[Json], controls: list[Json]) -> Json:
    """Count original sources once and join historical features by source identity."""
    rows, sentence_rows, feature_rows, diagnostic_rows = [], [], [], []
    slots = [dict(s, condition=s.get("condition", "original")) for s in slots]
    by_id = {r["unit_id"]: r for r in controls}
    for slot in [*slots, *diagnostics(slots)]:
        group = [
            c
            for c in calls
            if c["unit_id"] == slot["unit_id"] and c["condition"] == slot["condition"]
        ]
        if any(
            c["request"] != r
            or c["source_cluster_id"] != slot["source_cluster_id"]
            or c.get("cache_key") != cache_key(slot, r, c["identity"])
            for c, r in zip(group, slot["requests"])
        ):
            raise ValueError("request_or_source_identity")
        predictions = [p for c in group for p in accepted(slot, c)]
        expected = [i for r in slot["requests"] for i in r["sentence_indices"]]
        complete = (
            bool(expected)
            and len(group) == len(slot["requests"])
            and [p["sentence_index"] for p in predictions] == expected
        )
        status = (
            "completed"
            if complete
            else "excluded"
            if not expected
            else "censored"
            if (group and all(c["status"] == "censored" for c in group))
            else "failed"
        )
        row = dict(
            unit_id=slot["unit_id"],
            source_cluster_id=slot["source_cluster_id"],
            role=slot["role"],
            slot=slot["slot"],
            arm="grammar",
            condition=slot["condition"],
            metric="complete_source_transport",
            numerator=int(complete),
            denominator=1,
            status=status,
            exclusion_reason=None
            if complete
            else slot.get("exclusion_reason") or "incomplete_source_transport",
            human_target=None,
        )
        if slot["condition"] != "original":
            diagnostic_rows.append(dict(row, predictions=predictions))
            continue
        rows.append(row)
        for p in predictions:
            sentence_rows.append(
                dict(
                    p,
                    unit_id=slot["unit_id"],
                    source_cluster_id=slot["source_cluster_id"],
                    offsets=slot["sentences"][p["sentence_index"]],
                    source_complete=complete,
                )
            )
        control = by_id.get(slot["unit_id"], {})
        if complete and control.get("status") == "completed" and len(control.get("x") or []) == 12:
            if (
                control["source_cluster_id"] != slot["source_cluster_id"]
                or control["role"] != slot["role"]
            ):
                raise ValueError("paired_control_identity")
            local = [
                sum(p["p_unsupported"] for p in predictions) / len(predictions),
                max(p["p_unsupported"] for p in predictions),
                sum(p["relation"] == "C" for p in predictions) / len(predictions),
                sum(p["relation"] == "B" for p in predictions) / len(predictions),
            ]
            feature_rows.append(
                dict(
                    row,
                    x=[*control["x"], *local],
                    y=control["y"],
                    historical_paired_control=control,
                )
            )
    support = {
        role: {str(y): sum(r["role"] == role and r["y"] == y for r in feature_rows) for y in (0, 1)}
        for role in ROLES
    }
    counts = Counter(r["status"] for r in rows)
    ready = all(
        sum(r["role"] == role and r["status"] == "completed" for r in rows) >= minimum
        for role, minimum in [("fit", 96), ("tune", 48)]
    )
    trainable = ready and all(
        sum(support[role].values()) >= minimum and min(support[role].values()) >= 12
        for role, minimum in [("fit", 96), ("tune", 48)]
    )
    return dict(
        rows=rows,
        sentence_rows=sentence_rows,
        feature_rows=feature_rows,
        diagnostic_rows=diagnostic_rows,
        class_support=support,
        transport_ready=int(ready),
        fit_trainable_score=int(trainable),
        intended_count=192,
        eligible_count=sum(bool(s["requests"]) for s in slots),
        independent_count=len({s["source_cluster_id"] for s in slots}),
        completed_count=counts["completed"],
        excluded_count=counts["excluded"] if slots else 192,
        censored_count=counts["censored"],
        failed_count=counts["failed"],
    )


class FixtureRuntime:
    """Exercise transport privately; these responses have no live model credit."""

    def __init__(self) -> None:
        self.worker = self

    def post_json(self, endpoint: str, value: Json, timeout: float) -> Json:
        return (
            dict(prompt=value["messages"][0]["content"])
            if endpoint == "/apply-template"
            else dict(tokens=[0] * 30)
        )

    def generate(self, value: Json) -> Json:
        indices = json.loads(value["messages"][0]["content"])["answer_sentence_indices"]
        return dict(
            choices=[
                dict(
                    message=dict(content="\n".join(f"{i}|B|0.20|[]" for i in indices)),
                    finish_reason="stop",
                )
            ],
            usage=dict(prompt_tokens=30, completion_tokens=20),
        )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Seal predictions before reading original fit/tune labels and controls."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    progress("before_input_authentication")
    plan = inputs(root, raw)
    result: Json = dict(rows=[], checks=[])
    controls = []
    if fixture and (root / "fit-fixture.json").is_file():
        ref = reference(root / "fit-fixture.json")
        private = json.loads(Path(ref["path"]).read_text())
        plan.update(
            slots=freeze(private["rows"]), checks=[], refs=[ref], identity=dict(fixture=True)
        )
        result["rows"] = capture(
            plan["slots"], FixtureRuntime(), raw / "slots", plan["identity"], []
        )
        controls = private["controls"]
    elif not fixture and all(c["passed"] for c in plan["checks"]):
        canary.preflight(plan, raw)
        canary.gate(
            plan,
            root / UPSTREAM,
            "qualified_transport_identity",
            plan["qualified_identity"],
            plan["identity"],
        )
        if all(c["passed"] for c in plan["checks"]):
            plan["started"] = began
            original_capture = capture

            def adapter(
                slots: list[Json],
                runtime: Any,
                path: Path,
                identity: Json,
                *,
                started: float | None = None,
            ) -> list[Json]:
                return original_capture(
                    slots, runtime, path, identity, plan["reusable"], started=began
                )

            def pulse(phase: str, completed: int = 0, pending: int = 0) -> None:
                done = len(list((raw / "slots").glob("call-*.json")))
                total = sum(
                    len(s["requests"]) for s in [*plan["slots"], *diagnostics(plan["slots"])]
                )
                progress(phase, done, total - done)

            progress("before_live_capture")
            with (
                patch.object(canary, "capture", adapter),
                patch.object(canary, "TASK", TASK),
                patch.object(canary, "progress", pulse),
            ):
                result = canary.live(plan, raw)
            progress("after_live_capture", len(result["rows"]), 0)
    atomic_json(raw / "primitive_calls.json", dict(rows=result["rows"]))
    progress("predictions_sealed_before_labels", len(result["rows"]), 0)
    if (
        not fixture
        and plan["controls_ref"]
        and all(c["passed"] for c in [*plan["checks"], *result["checks"]])
    ):
        controls = json.loads(Path(plan["controls_ref"]["path"]).read_text())["rows"]
    atomic_json(
        raw / "source_plan.json", dict(rows=plan["slots"], controls=controls, config=CONFIG)
    )
    work = dict(
        slots=plan["slots"],
        calls=result["rows"],
        controls=controls,
        checks=[*plan["checks"], *result["checks"]],
        refs=plan["refs"],
        upstream=plan["upstream"],
        fixture=fixture,
        live_result=result,
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(phase="authenticate_and_capture", start_s=0, duration_s=time.monotonic() - began)
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                canary.MODULE,
                "python/carnot/verify/sentence_transport_8179.py",
                "ops/exclusion_manifest.yaml",
            ]
        ],
        raw_shard_hashes=[reference(raw / p) for p in ["primitive_calls.json", "source_plan.json"]],
    )
    if mutation:
        canary.gate(work, root, "private_tamper", "unchanged", mutation)
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(result["rows"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Require normal owned exits and observed live support for readiness."""
    reduced = reduce(work["slots"], work["calls"], work["controls"])
    failures = [c["check"] for c in work["checks"] if not c["passed"]]
    valid = bool(receipts) and all(r["passed"] for r in receipts)
    result = work["live_result"]
    ready = int(
        valid
        and not failures
        and not work["fixture"]
        and reduced["transport_ready"]
        and result.get("model_loads_completed") == 1
        and work["duration_s"] >= 10
    )
    verdict = (
        "disqualified"
        if not valid
        else "blocked"
        if failures
        else "positive"
        if ready and reduced["fit_trainable_score"]
        else "null"
    )
    reused = [c for c in work["calls"] if c["historical"]]
    current = [c for c in work["calls"] if not c["historical"]]
    value = dict(
        experiment_id=8182,
        task_id=TASK,
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (failures[0] if verdict == "blocked" else "fit_sentence_capture"),
        verdict_class=verdict,
        verifier_is_oracle=work["fixture"],
        claim_scope="Fixed fit/tune sentence transport and paired feature support; semantic benefit untested until independent decision audit",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=valid,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        gate_check_summary=work["checks"],
        preconditions_checked=work["checks"],
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=[],
        inference_substrate="live_llm_inference"
        if result.get("model_loads_attempted")
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation"
        if result.get("model_loads_attempted")
        else "no_model_load",
        planned_inference_substrate_class="model_bounded_generation",
        model_invocation_counts=dict(
            model_loads_attempted=result.get("model_loads_attempted", 0),
            model_loads_completed=result.get("model_loads_completed", 0),
            generate=0 if work["fixture"] else len(result.get("runtime_receipts", [])),
        ),
        call_ledger=work["calls"],
        cited_upstream_artifacts=work["upstream"],
        fit_capture_ready_score=ready,
        reused_call_receipts=reused,
        acquisition_cost_ledger=dict(
            current_calls=len(current) if not work["fixture"] else 0,
            historical_reused_calls=len(reused),
            historical_cost_charged_once=True,
            remaining_main_call_budget=384
            - len(reused)
            - sum(c["condition"] == "original" for c in current),
            semantic_headroom="unmeasured; transport success supplies no benefit evidence",
        ),
        sample_size_budget=dict(
            intended=192,
            fit=128,
            tune=64,
            main_calls=384,
            authenticated_reuse=len(reused),
            diagnostic_calls=16,
            input_tokens=6000,
            output_tokens=256,
        ),
        duration_s=work["duration_s"],
        random_seed=70782,
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=dict(fit=96, tune=48, per_class=12, budgets=CONFIG),
        measurement_reference=reference(raw / "measurement.json"),
        fixture_protocol_only=work["fixture"],
        model_receipt={k: v for k, v in result.items() if k != "rows"},
        repository_health=work.get("global_health", {}),
        **reduced,
    )
    value["fit_trainable_score"] *= int(valid and not failures and (ready or work["fixture"]))
    value["field_principles"] = {
        k: "Bind actual source custody and normal validation; exposed predictions grant no independent benefit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild headlines from hash-bound primitive responses without inference."""
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
        primitive, source = [
            json.loads(Path(r["path"]).read_text()) for r in work["raw_shard_hashes"]
        ]
        if primitive["rows"] != work["calls"] or source != dict(
            rows=work["slots"], controls=work["controls"], config=CONFIG
        ):
            return False
        for slot in work["slots"]:
            counts = {r["payload"]: r["input_tokens"] for r in slot["requests"]}
            if (
                transport.requests(slot, lambda text: counts.get(text, 0))["requests"]
                != slot["requests"]
            ):
                return False
        return bool(
            build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze actual file paths, owned checks and private E2E before measurement."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 300
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        TEST,
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    specs["commands"][4]["argv"][-1] = str(candidate.parent / "coverage.json")
    specs["repository_health"]["deadline_s"] = 300
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use the qualified supervisor and checked primary publication unchanged."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", canary.supervise),
    ):
        return execution.main(argv)
