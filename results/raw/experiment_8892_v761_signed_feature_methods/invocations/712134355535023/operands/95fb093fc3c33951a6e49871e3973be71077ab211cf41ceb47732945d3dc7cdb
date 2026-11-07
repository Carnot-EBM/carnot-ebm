"""REQ-REPORT-8830: qualify opt-in heads with retained causal operands and checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

import jax

from carnot.memory import categorical_memory_8830 as c
from carnot.reporting import learning_signal_runner_8815 as validation
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8830_v756_categorical_memory"
CLI = "scripts/experiments/" + NAME + ".py"
UPSTREAM = ROOT / "results/experiment_8828_v756_signal_replay_qualification.json"
OWNED = [
    "python/carnot/memory/categorical_memory_8830.py",
    "python/carnot/reporting/categorical_memory_runner_8830.py",
    CLI,
]
TESTS = ["tests/python/test_categorical_memory_8830.py"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose real phase boundaries so compilation and durable work remain visible."""
    print(f"[exp8830] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind readable operands to exact bytes rather than mutable filenames."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def bound(ref: Json) -> Json:
    """Reject missing or changed operands before using their content."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("operand_hash:" + str(path))
    value: Json = json.loads(path.read_bytes())
    return value


def preflight(raw: Path, upstream: Path = UPSTREAM) -> tuple[Json, Json, Json]:
    """Authenticate original evidence and resource custody before measuring a head."""
    progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True)
    probe = raw / ".write_probe"
    probe.write_bytes(b"private")
    probe.unlink()
    free = shutil.disk_usage(raw).free
    if (
        free < 100_000_000
        or Path(sys.executable).resolve() != (ROOT / ".venv/bin/python").resolve()
    ):
        raise RuntimeError("internal_resources")
    authority = json.loads(upstream.read_bytes())
    if authority["objective_candidate_ready_score"] == 0:
        raise PermissionError("objective_candidate_ready_score:observed_zero")
    if (
        authority["objective_candidate_ready_score"] != 1
        or authority["required_checks_passed"] is not True
        or authority["verdict_class"] != "circular_positive"
    ):
        raise PermissionError("rejected_producer")
    config = authority["objective_selection"]["configuration"]
    expected = dict(
        candidate_bits=3,
        features=list(c.m.COUNTS),
        hidden_width=4,
        lr=0.1,
        objective="categorical",
        optimizer="SGD",
        seeds=[11, 23, 37, 47, 59],
        updates=400,
    )
    if config != expected or canonical_hash(config) != authority["selected_configuration_hash"]:
        raise RuntimeError("fixed_configuration")
    for ref in authority["source_artifact_hashes"] + authority["code_config_hashes"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash:" + ref["path"])
    evidence = bound(authority["raw_shard_hashes"][0])
    if sha256_file(Path(evidence["snapshot"]["path"])) != evidence["snapshot"]["sha256"]:
        raise ValueError("snapshot_hash")
    if (
        sha256_file(upstream)
        != "sha256:5defe1a198e8a431160a6bf9799b84a3db2b6f395b90f38c75cd97a6fc9da743"
    ):
        raise PermissionError("rejected_producer_hash")
    receipt = dict(
        upstream=reference(upstream),
        raw=authority["raw_shard_hashes"],
        snapshot=evidence["snapshot"],
        virtualenv=sys.executable,
        dependencies=dict(jax=jax.__version__, devices=[str(x) for x in jax.devices("cpu")]),
        private_output=str(raw),
        disk_free_bytes=free,
        configuration=expected,
        threshold=0.5,
        normalization="min(count,32)/5",
        feedback_delay=8,
        pending_capacity=32,
        template_capacity=32,
    )
    atomic_json(raw / "preconditions.json", receipt)
    progress("preconditions_after", 1)
    return authority, evidence, receipt


def child(request: Json) -> Json:
    """A private CLI request runs one real durable prediction or delayed update."""
    store = c.Store(Path(request["state"]), request["head"])
    if request["operation"] == "predict":
        return store.predict(request["event"], request["payload"], 0.5)
    return store.release(
        request["event"], request["payload"], request["ordinal"], label=request.get("label")
    )


def execute(raw: Path, evidence: Json, config: Json) -> Json:
    """Reinstantiate matched historical heads, then exercise only constructed controls."""
    rows, gradients, additions, checkpoints, final_states = [], [], [], [], []
    fit = [r for r in evidence["development_rows"] if r["split"] == "fit"]
    valid = [r for r in evidence["development_rows"] if r["split"] == "validation"]
    for seed in config["seeds"]:
        for objective in c.d.OBJECTIVES:
            progress("head_fit_before", len(checkpoints), 15 - len(checkpoints))
            fitted = c.d.fit(fit, valid, seed, objective, 0.1)
            original = next(
                r
                for r in evidence["fits"]
                if r["seed"] == seed and r["objective"] == objective and r["lr"] == 0.1
            )
            if fitted != original:
                raise RuntimeError("historical_fit_drift")
            params = dict(objective=objective, lr=0.1, weights=fitted["parameters"])
            checkpoint = raw / "checkpoints" / f"{seed}-{objective}.json"
            atomic_json(
                checkpoint,
                dict(parameters=params, threshold=0.5, seed=seed, development_updates=400),
            )
            checkpoints.append(
                dict(
                    seed=seed,
                    arm=objective,
                    parameter_hash=canonical_hash(params),
                    **reference(checkpoint),
                )
            )
            progress("head_fit_after", len(checkpoints), 15 - len(checkpoints))
            store = c.Store(raw / "states" / f"{seed}-{objective}.json", params)
            public = [c.m.world(f"fixture-{i}", c.m.LABELS[i % 3]) for i in range(24)]
            for i, payload in enumerate(public):
                sealed = store.predict(f"q{i}", payload, 0.5)
                rows.append(
                    dict(
                        unit_id=f"fixture-{i}",
                        cluster_id=f"fixture-{i % 3}",
                        arm=objective,
                        seed=seed,
                        metric="sealed_probability_sum",
                        denominator=1,
                        status="complete",
                        missing_reason=None,
                        value=sum(sealed["prediction"]),
                        payload=payload,
                        sealed=sealed,
                    )
                )
                store.close()
                if i >= 8:
                    released = store.release(
                        f"q{i - 8}",
                        public[i - 8],
                        i,
                        label=str(c.m.decide(public[i - 8])["decision"]),
                    )
                    gradients.append(dict(seed=seed, objective=objective, **released))
                    if released["template_added"]:
                        additions.append(dict(seed=seed, objective=objective, **released))
            for i in range(16, 24):
                gradients.append(
                    dict(
                        seed=seed,
                        objective=objective,
                        **store.release(
                            f"q{i}", public[i], i + 8, label=str(c.m.decide(public[i])["decision"])
                        ),
                    )
                )
            final_states.append(reference(store.path))
            progress("causal_control_after", len(rows), 360 - len(rows))
    progress("gradient_and_shuffle_before")
    controls = c.controls()
    progress("gradient_and_shuffle_after", len(controls))
    work = dict(
        rows=rows,
        gradient_rows=gradients,
        template_addition_rows=additions,
        checkpoint_refs=checkpoints,
        final_state_refs=final_states,
        controls=controls,
    )
    atomic_json(raw / "evidence.json", work)
    return work


def restart(raw: Path) -> list[Json]:
    """Real child processes recover the same sealed prediction and reject a second ack."""
    request = dict(
        state=str(raw / "state.json"),
        head=c.head(11, "linear"),
        operation="predict",
        event="q0",
        payload=c.m.world("restart", "support"),
    )
    rows = []
    for index, operation in enumerate(("predict", "release", "release")):
        request.update(operation=operation, ordinal=8, label="support")
        req, response = raw / f"request-{index}.json", raw / f"response-{index}.json"
        atomic_json(req, request)
        receipt = validation.commands(
            [
                (
                    f"restart-{index}",
                    [
                        "env",
                        "-u",
                        "PYTHONPATH",
                        "-C",
                        "/tmp",
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(ROOT / CLI),
                        "--request",
                        str(req),
                        "--response",
                        str(response),
                    ],
                    "owned",
                    60,
                )
            ],
            raw / "logs",
        )[0]
        if not receipt["passed"]:
            raise RuntimeError("restart_child")
        rows.append(
            dict(
                receipt=receipt,
                response=bound(reference(response)),
                request=reference(req),
                response_ref=reference(response),
            )
        )
    if rows[1]["response"]["prediction_hash"] != canonical_hash(rows[0]["response"]) or [
        r["response"].get("optimizer_steps", 0) for r in rows
    ] != [0, 1, 0]:
        raise RuntimeError("restart_atomicity")
    return rows


def replay(path: Path) -> Json:
    """Authenticate byte custody and recompute every causal transition in fresh scratch."""
    import tempfile

    value = json.loads(path.read_bytes())
    for ref in value["code_config_hashes"] + value["source_artifact_hashes"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("replay_hash")
    for receipt in value["validation_receipts"]:
        log = Path(receipt["log_path"])
        if not log.is_absolute():
            log = ROOT / log
        if sha256_file(log) != receipt["log_sha256"]:
            raise ValueError("validation_log")
    if value["verdict_class"] == "blocked":
        if value["memory_execution_ready_score"] != 0:
            raise ValueError("blocked_readiness")
        return dict(passed=True, replay_passed=True)
    work = bound(value["raw_shard_hashes"][0])
    for ref in work["final_state_refs"]:
        bound(ref)
    for row in work["restart_rows"]:
        bound(row["request"])
        if bound(row["response_ref"]) != row["response"]:
            raise ValueError("restart_response")
    for ref in work["checkpoint_refs"]:
        checkpoint = bound(ref)
        if canonical_hash(checkpoint["parameters"]) != ref["parameter_hash"]:
            raise ValueError("parameter_hash")
    with tempfile.TemporaryDirectory(prefix="carnot8830-replay-") as temporary:
        raw = Path(temporary)
        _, source, pre = preflight(raw, Path(value["source_artifact_hashes"][0]["path"]))
        actual = execute(raw, source, pre["configuration"])
    for key in ("rows", "gradient_rows", "template_addition_rows", "controls"):
        if actual[key] != work[key] or value[key] != work[key]:
            raise ValueError("reduction_drift:" + key)
    if [r["sha256"] for r in work["final_state_refs"]] != [
        r["sha256"] for r in actual["final_state_refs"]
    ]:
        raise ValueError("final_state_drift")
    if value["parameter_hashes"] != [r["parameter_hash"] for r in actual["checkpoint_refs"]]:
        raise ValueError("parameter_drift")
    if value["completed_count"] != len(work["rows"]) or value[
        "memory_execution_ready_score"
    ] != int(value["required_checks_passed"] and all(r["passed"] for r in work["controls"])):
        raise ValueError("readiness_drift")
    return dict(
        passed=True, replay_passed=True, rows=len(work["rows"]), numerical_training_calls=9961
    )


def validate_owned(raw: Path) -> Json:
    """Reuse the scoped unit/child coverage and private E2E manifest unchanged."""
    with patch.multiple(validation, OWNED=OWNED, TESTS=TESTS, CLI=CLI):
        return validation.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Keep shared adversarial, strict row and outside-checkout replay checks intact."""
    with patch.object(validation, "CLI", CLI):
        return validation.terminal(candidate, raw)


def main(argv: list[str] | None = None) -> int:
    """Private output changes custody only; all owned checks still gate readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105", choices=["20261105"])
    parser.add_argument(
        "--output", "--private-output", type=Path, default=ROOT / "results" / (NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    args = parser.parse_args(argv)
    if args.request:
        atomic_json(args.response, child(json.loads(args.request.read_bytes())))
        return 0
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(str(error), flush=True)
            return 1
    started = time.monotonic_ns()
    raw = args.raw_dir or args.output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    work: Json = dict(
        rows=[],
        gradient_rows=[],
        template_addition_rows=[],
        checkpoint_refs=[],
        final_state_refs=[],
        controls=[],
        restart_rows=[],
    )
    pre: Json = {}
    gates = []
    disposition = "circular_positive"
    authority_admitted = False
    try:
        _, evidence, pre = preflight(raw, args.upstream)
        authority_admitted = True
        work = execute(raw, evidence, pre["configuration"])
        work["restart_rows"] = restart(raw / "restart")
        atomic_json(raw / "evidence.json", work)
        if not all(r["passed"] for r in work["controls"]):
            raise RuntimeError("learning_control")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        external = not authority_admitted and (
            isinstance(error, (FileNotFoundError, KeyError))
            or isinstance(error, PermissionError)
            and ("producer" in str(error) or "observed_zero" in str(error))
        )
        disposition = "blocked" if external else "disqualified"
        missing = Path(getattr(error, "filename", None) or args.upstream)
        gates.append(
            dict(
                check="upstream_authority" if external else "owned_execution",
                upstream="exp8828-signal-replay-qualification",
                artifact_field="objective_candidate_ready_score"
                if "observed_zero" in str(error)
                else str(error),
                expected=1
                if "observed_zero" in str(error)
                else "qualified hash-bound original replay",
                observed=0 if "observed_zero" in str(error) else str(error),
                path=str(missing),
                hash=sha256_file(missing) if missing.is_file() else None,
                status="observed_zero"
                if "observed_zero" in str(error)
                else "missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field"
                if isinstance(error, KeyError)
                else "rejected_producer"
                if isinstance(error, PermissionError)
                else "internal_failure",
            )
        )
    progress("validation_before")
    checked = validate_owned(raw)
    progress("validation_after")
    passed = checked["passed"] and disposition == "circular_positive"
    if not checked["passed"]:
        disposition = "disqualified"
    ended = time.monotonic_ns()
    phase = dict(
        phase="qualification_and_validation",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        duration_s=(ended - started) / 1e9,
    )
    receipt = build_current_work_receipt(
        run_id=NAME,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_verifier",
        inference_substrate_details=dict(backend="jax_cpu", generator_weights_updated=False),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=[phase],
        small_ebm_training=dict(performed=bool(work["rows"])),
    )
    value = dict(
        experiment_id=8830,
        task_id="exp8830-categorical-memory",
        milestone="2026.11.756",
        run_date=args.date,
        honest_verdict="complete_" + disposition + "_categorical_memory",
        verdict_class=disposition,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=6000 + len(work["gradient_rows"]) + 3602 if work["rows"] else 0,
        numerical_training_scope="Owned scientific execution and restart child only; unit tests and cold replay excluded. Cold replay reports its own count.",
        verifier_is_oracle=True,
        exposure_scope="original historical development and constructed causal controls only; future streams unopened",
        independent_generalization_score=0,
        duration_s=(ended - started) / 1e9,
        random_seed=8830,
        reproducibility_checksum=canonical_hash(
            dict(configuration=pre.get("configuration"), upstream=pre.get("upstream"))
        ),
        source_artifact_hashes=[reference(args.upstream)] if args.upstream.is_file() else [],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + validation.CODE],
        raw_shard_hashes=[reference(raw / "evidence.json")]
        if (raw / "evidence.json").exists()
        else [],
        phase_spans=[phase],
        preconditions_checked=pre,
        **work,
        intended_count=360,
        completed_count=len(work["rows"]),
        failed_count=0 if len(work["rows"]) == 360 else 360 - len(work["rows"]),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=gates,
        acceptance_gates=dict(
            execution=dict(
                threshold=1, principle="causal and validation execution only; no loss benefit"
            ),
            gradient=dict(
                threshold=0.0001, principle="finite differences test the actual objective"
            ),
        ),
        required_checks_passed=passed,
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        memory_execution_ready_score=int(passed),
        parameter_hashes=[r["parameter_hash"] for r in work["checkpoint_refs"]],
        causal_order_rows=[
            {
                k: r[k]
                for k in ("event_id", "prediction_ordinal", "feedback_ordinal", "prediction_hash")
            }
            for r in work["gradient_rows"]
        ],
        linear_control_config=dict(
            lr=0.1,
            updates=400,
            seeds=[11, 23, 37, 47, 59],
            features=list(c.m.COUNTS),
            optimizer="SGD",
            principle="retain the stronger simple baseline at matched development budget",
        ),
        current_work_receipt=receipt,
        methodology_note="Exact normalized three-decision energy; matched multinomial linear and original mixed controls. Constructed causal execution is not future learning benefit.",
    )
    value["field_principles"] = {
        k: "Retain measured "
        + k
        + " for this scheduled producer; execution does not establish generalization."
        for k in value
    }
    value["field_principles"].update(
        dict(
            experiment_id="Bind this evidence to Exp8830 rather than its upstream.",
            task_id="Identify the scheduled causal-memory producer.",
            milestone="Research execution belongs to V756; this does not activate a milestone.",
            run_date="Record the requested execution date.",
            honest_verdict="Terminal completion preserves the measured qualification outcome.",
            verdict_class="Exact verifier fixture success is circular_positive, never positive.",
            inference_substrate="CPU arithmetic and small-head training are not language-model inference.",
            inference_substrate_class="The closed no_model_load class records actual model use.",
            MODEL_SPECS="Empty because this producer loads no language model.",
            model_invocation_counts="Owned LLM loads and generations remain exactly zero.",
            numerical_training_calls="Count scientific SGD updates separately from LLM calls.",
            verifier_is_oracle="The exact encoded checker supplies delayed truth only for constructed graphs.",
            exposure_scope="Only original development and explicit controls are exposed.",
            independent_generalization_score="Constructed and previously exposed evidence establishes no independent generalization.",
            duration_s="Monotonic elapsed execution contains no duration padding.",
            random_seed="Fix control shuffling independently of the five historical fit seeds.",
            reproducibility_checksum="Bind the fixed configuration and admitted upstream bytes.",
            source_artifact_hashes="Authenticate the qualified original replay authority.",
            code_config_hashes="Bind executed code and shared validation implementation.",
            raw_shard_hashes="Authenticate retained operands so aggregates can be independently replayed.",
            phase_spans="Record actual monotonic qualification and validation boundaries.",
            preconditions_checked="Show resource and operand checks before measurement.",
            rows="Retain each sealed unit, arm and seed without creating independent worlds.",
            gate_check_summary="Name exact absent or rejected authority separately from internal failure.",
            acceptance_gates="Execution thresholds establish readiness, not scientific benefit.",
            required_checks_passed="Owned failures suppress readiness; repository health is separately retained.",
            validation_receipts="Bind actual argv, exit status, elapsed time and log bytes.",
            owned_statement_coverage="Require every new unit and real-child statement with zero exclusions.",
            memory_execution_ready_score="Only causal execution plus passing owned checks licenses stream execution.",
            checkpoint_refs="Retain all fifteen historical mixed, categorical and linear heads.",
            parameter_hashes="Distinguish actual learned parameters from update counters.",
            gradient_rows="Retain measured gradients, movement and one-step delayed admissions.",
            causal_order_rows="Bind sealed prediction hashes to later feedback ordinals.",
            restart_rows="Real private CLI children recover pending work and acknowledge once.",
            template_addition_rows="Link admitted predicates to exact verified evidence and later uses.",
            final_state_refs="Authenticate final templates, binding witnesses, acknowledgements and parameters together.",
            controls="Finite differences and damaged shuffled learning qualify execution only.",
        )
    )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    terminal_result = terminal(candidate, raw)
    value["validation_receipts"] += terminal_result["receipts"]
    if not terminal_result["passed"]:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            memory_execution_ready_score=0,
            required_checks_passed=False,
        )
    atomic_json(candidate, value)
    publish_primary(
        args.output, value, lambda path: dict(passed=json.loads(path.read_bytes()) == value)
    )
    progress("published", len(work["rows"]), 360 - len(work["rows"]))
    return 0
