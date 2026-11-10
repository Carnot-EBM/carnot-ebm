"""REQ-REPORT-8966: measure causal calibration readiness without a benefit claim."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.memory import online_response_worker_8966 as w
from carnot.reporting import relational_controls_8965 as controls
from carnot.reporting import online_calibration_methods_8960 as calibration
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.depth_opportunity_8945 import Rejected, require
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]
ROOT, prior, b = controls.ROOT, controls.prior, controls.b
NAME, TASK, SEED = (
    "experiment_8966_v766_online_response_streams",
    "exp8966-online-response-streams",
    766066,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/memory/online_response_worker_8966.py",
    "python/carnot/reporting/online_response_streams_8966.py",
    "python/carnot/reporting/online_response_stream_runner_8966.py",
    CLI,
]
TESTS = ["tests/python/test_online_response_streams_8966.py"]
INPUTS = [
    (
        f"results/{calibration.NAME}.json",
        "sha256:1af36d37457cd5b7cda9be1b6cc80bc89c96f29dcf6427c4fbb2946728b3b09b",
        "online_methods_ready_score",
        calibration.TASK,
    ),
    (
        "results/experiment_8963_v766_corpus_admission.json",
        "sha256:586a8f5bff118cd9416f0f03cf9606bc35db0eb4068d946bbf17105329547d24",
        "corpus_ready_score",
        "exp8963-corpus-admission",
    ),
    (controls.INPUT, controls.PIN, "detector_rows_ready_score", prior.TASK),
    (
        f"results/{controls.NAME}.json",
        "sha256:188c29a7ee88ae6ca363a7e9137d4fe11c2308c1b3b9f037ce21d51259d1ad7d",
        "relational_controls_ready_score",
        controls.TASK,
    ),
]
ref = calibration.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts distinguish long computation from a stalled worker."""
    print(f"[exp8966] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path) -> Json:
    """Original producers and their retained operands are inputs, not runtime task gates."""
    result: Json = dict(failures=[], references=[], producers={})
    seen: set[str] = set()
    for name, pin, field, task in INPUTS:
        path = root / name
        result["references"].append(ref(path))
        try:
            require(ref(path)["sha256"], pin, path, "sha256", task)
            value = json.loads(path.read_text())
            for key, expected in ((field, 1), ("task_id", task), ("required_checks_passed", True)):
                require(value.get(key), expected, path, key, task)
            operands = (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            )
            for i, operand in enumerate(operands):
                if operand["path"] not in seen:
                    target = Path(operand["path"])
                    require(ref(target)["sha256"], operand["sha256"], target, "sha256", task)
                    result["references"].append(ref(target))
                    seen.add(operand["path"])
                if i % 500 == 0:
                    progress("authenticate_retained_bytes", i, len(operands) - i)
            result["producers"][task] = value
        except Rejected as error:
            result["failures"].append(dict(error.receipt, check=error.receipt["failure_kind"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["failures"].append(
                dict(
                    upstream=task,
                    path=str(path),
                    hash=ref(path)["sha256"],
                    check="rejected_producer",
                    artifact_field="producer_json",
                    expected="original complete object",
                    observed=str(error),
                )
            )
    return result


def wait_midpoint(process: subprocess.Popen[bytes]) -> None:
    """Bound the midpoint wait and forward every child line before terminating it."""
    assert process.stdout is not None
    started, output = time.monotonic(), b""
    while b"READY16\n" not in output:
        if time.monotonic() - started > 120:
            raise TimeoutError("midpoint_timeout")
        ready, _, _ = select.select([process.stdout], [], [], 15)
        if not ready:
            progress("midpoint_child_pending", 0, 1)
            continue
        chunk = os.read(process.stdout.fileno(), 8192)
        if not chunk:
            raise ValueError("midpoint_child_closed")
        output += chunk
        print(chunk.decode(), end="", flush=True)


def restart(request: Json, raw: Path) -> tuple[Json, Json]:
    """Terminate an owned child at16 and start a fresh interpreter from durable bytes."""
    atomic_json(raw / "request.json", request)
    command = [
        sys.executable,
        "-u",
        "-m",
        "carnot.memory.online_response_worker_8966",
        str(raw / "request.json"),
        str(raw),
    ]
    progress("before_midpoint_child", 0, 1)
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'python'}:{ROOT}")
    process = subprocess.Popen(
        command + ["--stop16"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    try:
        wait_midpoint(process)
        midpoint = json.loads((raw / "state.json").read_text())
        atomic_json(raw / "midpoint.json", midpoint)
    finally:
        process.terminate()
        process.wait(timeout=30)
        assert process.stdout is not None and process.stdin is not None
        process.stdout.close()
        process.stdin.close()
    progress("after_midpoint_child", 1, 0)
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                "resume_child", tuple(command + ["--resume"]), "dependent_restart_control", 120
            )
        ],
        log_dir=raw / "logs",
        heartbeat_s=15,
    )
    if not receipts[0]["passed"]:
        raise ValueError("resume_child_failed")
    state = json.loads((raw / "state.json").read_text())
    receipt = dict(
        query=16,
        pid=process.pid,
        exit_code=process.returncode,
        terminated=process.returncode == -15,
        midpoint_hash=canonical_hash(midpoint),
        resumed_state_hash=canonical_hash(state),
        receipts=receipts,
        passed=process.returncode == -15 and midpoint["query"] == 16,
    )
    atomic_json(raw / "restart.json", receipt)
    return state, receipt


def empty_result() -> Json:
    """Every scheduled history remains visible when external evidence is absent."""
    return dict(
        intended_count=24,
        completed_count=0,
        failed_count=0,
        censored_count=24,
        excluded_count=0,
        independent_count=0,
        rows=[],
        prediction_rows=[],
        retention_rows=[],
        feedback_rows=[],
        promotion_rows=[],
        restart_rows=[],
        adapter_state_hashes=[],
        control_rows=[],
        streams=[],
        numerical_training_calls=0,
    )


def metric_row(row: Json, label: int, stream: str, condition: str) -> Json:
    """Brier measures probability quality; classification error measures a hard decision."""
    loss = (row["p_error"] - label) ** 2
    error = int((row["p_error"] >= 0.5) != label)
    return dict(
        row,
        label=label,
        stream_id=stream,
        cluster_id=stream,
        retention_cluster_id=row["unit_id"] if condition == "retention" else None,
        condition=condition,
        metric="brier",
        value=loss,
        numerator=loss,
        denominator=1,
        status="completed",
        censor_reason=None,
        metrics={
            "brier": dict(metric="brier", value=loss, numerator=loss, denominator=1),
            "classification_error": dict(
                metric="classification_error", value=error, numerator=error, denominator=1
            ),
        },
    )


def evaluate(evidence: Json, raw: Path, *, replay: bool = False) -> Json:
    """Only frozen inference views enter scoring; truth opens after prediction seals."""
    producers = evidence["producers"]
    corpus = producers["exp8963-corpus-admission"]
    detector = producers[prior.TASK]
    methods = producers[calibration.TASK]
    service = prior.read_reference(detector["calibrated_state_reference"])
    service_hash = canonical_hash(service)
    views = {
        role: prior.read_reference(corpus["inference_views"][role])
        for role in ("online", "test", "guard")
    }
    roster = corpus["broker_manifest"]["online"]["streams"]
    manifest = {r["unit_id"]: r for r in corpus["corpus_manifest"]}
    online_ids = [u for s in roster for u in s["unit_ids"]]
    if (
        len(roster) != 24
        or len(online_ids) != 768
        or len(set(online_ids)) != 768
        or set(online_ids) != set(views["online"]["unit_ids"])
        or len(views["test"]["unit_ids"]) != 128
        or views["guard"]["unit_ids"] != [r["unit_id"] for r in methods["guard_manifest"]["rows"]]
        or len({manifest[u]["graph_hash"] for u in online_ids}) != 768
        or len({manifest[u]["response_id"] for u in online_ids}) != 768
    ):
        raise ValueError("real_response_roster")
    for unit in online_ids:
        prior.read_reference(manifest[unit]["first_response_reference"])
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8966-replay-")) if replay else raw
    progress("before_frozen_small_head_scoring", 0, 960)
    scored = controls.child(
        dict(
            service=service,
            denial_paths=[
                corpus["broker_manifest"][role]["label_reference"]["path"]
                for role in ("online", "test")
            ],
            conditions={k: v["rows"] for k, v in views.items()},
        ),
        scratch / "scoring",
    )
    progress("after_frozen_small_head_scoring", 960, 0)
    if not all(scored["denial_probes"].values()) or not scored["labeling_helper_absent"]:
        raise ValueError("scoring_custody")
    probabilities = {
        role: scored["predictions"][role]["relational_energy"]["mean"] for role in views
    }
    guard = list(zip(probabilities["guard"], views["guard"]["labels"], strict=True))
    p_by_id = dict(zip(views["online"]["unit_ids"], probabilities["online"], strict=True))
    result = empty_result()
    for index, stream in enumerate(roster):
        sid = stream["stream_id"]
        progress("before_stream", index, 24 - index)
        private = raw / "streams" / sid
        request = dict(
            stream_id=sid,
            unit_ids=stream["unit_ids"],
            probabilities=[p_by_id[u] for u in stream["unit_ids"]],
            label_path=stream["label_reference"]["path"],
            retention_ids=views["test"]["unit_ids"],
            retention_probabilities=probabilities["test"],
            guard_rows=[list(r) for r in guard],
        )
        if replay and json.loads((private / "request.json").read_text()) != request:
            raise ValueError("request_drift")
        control = w.trajectory(request, scratch / "controls" / sid)
        if replay:
            resumed = json.loads((private / "state.json").read_text())
            receipt = json.loads((private / "restart.json").read_text())
        else:
            resumed, receipt = restart(request, private)
        midpoint = json.loads((private / "midpoint.json").read_text())
        exact = (
            resumed == control
            and receipt["terminated"]
            and receipt["exit_code"] == -15
            and receipt["query"] == midpoint["query"] == 16
            and receipt["midpoint_hash"] == canonical_hash(midpoint)
            and receipt["resumed_state_hash"] == canonical_hash(resumed)
            and midpoint["adapter_state_hashes"] == control["adapter_state_hashes"][:16]
        )
        result["restart_rows"].append(
            dict(
                receipt,
                stream_id=sid,
                passed=bool(exact),
                dependent_check=True,
                control_state_hash=canonical_hash(control),
            )
        )
        labels = prior.read_reference(stream["label_reference"])["labels"]
        changed_path = scratch / "mutations" / sid / "future.json"
        atomic_json(changed_path, dict(labels=labels[:16] + [1 - y for y in labels[16:]]))
        future = w.trajectory(
            dict(request, label_path=str(changed_path)), scratch / "mutations" / sid / "future"
        )
        # A changed label17 first releases after prediction24, so seals1–24 cannot change.
        future_same = future["seals"][:24] == control["seals"][:24]
        retention_copy = scratch / "mutations" / sid / "retention_labels.json"
        truth = prior.read_reference(corpus["broker_manifest"]["test"]["label_reference"])
        atomic_json(
            retention_copy, dict(rows=[dict(r, label=1 - r["label"]) for r in truth["rows"]])
        )
        retention_control = w.trajectory(request, scratch / "mutations" / sid / "retention")
        result["control_rows"].append(
            dict(
                stream_id=sid,
                future_label_seals_unchanged=future_same,
                retention_label_seals_unchanged=retention_control == control,
                future_labels_original_hash=canonical_hash(labels),
                future_labels_mutated_hash=canonical_hash(
                    labels[:16] + [1 - y for y in labels[16:]]
                ),
                retention_original_hash=canonical_hash(truth),
                retention_mutated_hash=canonical_hash(json.loads(retention_copy.read_text())),
                passed=bool(future_same and retention_control == control and exact),
            )
        )
        retention_truth = {r["unit_id"]: r["label"] for r in truth["rows"]}
        online_truth = dict(zip(stream["unit_ids"], labels, strict=True))
        result["prediction_rows"].extend(
            dict(
                metric_row(r, online_truth[r["unit_id"]], sid, "online"),
                graph_hash=manifest[r["unit_id"]]["graph_hash"],
                first_response_reference=manifest[r["unit_id"]]["first_response_reference"],
            )
            for r in control["rows"]
        )
        result["retention_rows"].extend(
            metric_row(r, retention_truth[r["unit_id"]], sid, "retention")
            for r in control["retention_rows"]
        )
        for key in ("feedback_rows", "promotion_rows", "adapter_state_hashes"):
            result[key].extend(dict(r, stream_id=sid) for r in control[key])
        result["streams"].append(
            dict(
                stream_id=sid,
                unit_ids=stream["unit_ids"],
                pending=control["pending"],
                pending_ids=control["pending_ids"],
                released_through=control["released_through"],
                final_params=control["params"],
                prediction_seals=control["seals"],
                final_state_hash=canonical_hash(control),
            )
        )
        progress("after_stream", index + 1, 23 - index)
    result.update(
        rows=result["prediction_rows"] + result["retention_rows"],
        completed_count=24,
        censored_count=0,
        independent_count=24,
        scheduled_adapter_training_calls=24 * 7 * 3,
        dependent_control_training_calls=3 * 24 * 7 * 3,
        numerical_training_calls=4 * 24 * 7 * 3,
        numerical_training_call_scope="measured trajectories including restart and two mutation controls; qualification and cold replay are separate validation",
        calibrated_service_hash=service_hash,
        frozen_service_unchanged=canonical_hash(service) == service_hash,
        guard_ids=views["guard"]["unit_ids"],
        retention_ids=views["test"]["unit_ids"],
        frozen_scoring_denial_probes=scored["denial_probes"],
        retention_scope="shared128 final-test graphs; evaluation only; previously opened by Exp8964",
        future_mutation_boundary="labels17–32 inverted; seals1–24 unchanged before first eligible release",
    )
    return result


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Causal readiness is a custody gate; only the later benefit audit can claim gain."""
    result = work["result"]
    failures = deepcopy(work["evidence"]["failures"])
    owned = prior.admission.shard.qualified(validation, OWNED)
    causal = (
        result["completed_count"] == result["independent_count"] == 24
        and len(result["prediction_rows"]) == 3072
        and len(result["retention_rows"]) == 36864
        and len(result["control_rows"]) == len(result["restart_rows"]) == 24
        and all(r["passed"] for r in result["control_rows"] + result["restart_rows"])
        and all(s["pending"] == [29, 30, 31, 32] for s in result["streams"])
        and result.get("frozen_service_unchanged", False)
    )
    cls = (
        "disqualified"
        if not fixture and (not owned or work["owned_failure"] or not failures and not causal)
        else "blocked"
        if failures
        else "null"
    )
    if cls == "disqualified":
        failures.append(
            dict(
                upstream=TASK,
                path=work["raw"],
                hash=None,
                check="owned_controls",
                artifact_field="causal_custody_and_validation",
                expected=True,
                observed=dict(causal=causal, owned=owned, owned_failure=work["owned_failure"]),
            )
        )
    verdict = f"complete_{cls}_sealed_real_error_online_calibration"
    previous = work.get("prior_terminal", {})
    if (
        cls in ("blocked", "disqualified")
        and previous.get("verdict_class") == cls
        and previous.get("gate_check_summary") == failures
    ):
        verdict = previous["honest_verdict"]
    value: Json = dict(
        result,
        experiment_id=8966,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261110",
        random_seed=SEED,
        honest_verdict=verdict,
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if failures else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(
            historical_tasks_are_retained_inputs=True, failures=work["evidence"]["failures"]
        ),
        verifier_is_oracle=False,
        exposure_scope="released_real_graph_errors_and_exposed_development_guard",
        independent_generalization_score=0,
        continuous_self_learning_task=True,
        online_rows_ready_score=int(owned and causal and not failures and not fixture),
        online_benefit_score=None,
        scientific_benefit_evaluated=False,
        generator_weights_frozen=True,
        required_checks_passed=owned,
        gate_check_summary=failures,
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[ref(Path(work["raw"]) / "work.json"), *work["raw_references"]],
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        fixture_only=fixture,
        flagged_adversarial=False,
        methodology_note="Frozen relational scores; four adapter arms; seven delayed blocks. "
        "Exact two-scalar block-mean updates and exposed64-graph development guard from Exp8960. "
        "Last four labels stay pending for scientific scoring. Actual child termination at16 "
        "is a dependent check. Retention uses shared128 final-test graphs and never promotes. "
        "Readiness without demonstrated gain is null; Exp8967 owns benefit. No generator or ARC change.",
    )
    value["acceptance_gates"] = {
        k: dict(
            passed=v,
            principle="Custody, complete causal evidence and owned checks gate readiness; benefit is separate.",
        )
        for k, v in dict(
            authenticated_inputs=not work["evidence"]["failures"],
            causal_controls=causal,
            owned_checks=owned,
        ).items()
    }
    value["field_principles"] = {
        k: "Bind this invocation to retained bytes and scoped causal evidence."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    for fields, principle in (
        (
            "experiment_id task_id milestone run_date random_seed reproducibility_checksum",
            "Bind the result to the scheduled task, input bytes and reproducible randomness.",
        ),
        (
            "honest_verdict verdict_class",
            "An honest terminal disposition avoids wasted retries; failed acceptance gates forbid positive.",
        ),
        (
            "inference_substrate inference_substrate_class execution_venue",
            "Use the actual legal substrate and venue; host is valid and no_model_load never claims a new GGUF run.",
        ),
        (
            "MODEL_SPECS model_specs model_invocation_counts numerical_training_calls",
            "Distinguish two-scalar training, retained model outputs and actual generator calls; count each separately.",
        ),
        (
            "duration_s phase_spans preconditions_checked",
            "Actual clocks and resource receipts prevent fabricated execution and duration padding.",
        ),
        (
            "verifier_is_oracle exposure_scope independent_generalization_score",
            "Separate exact labels from learned inference; graph-domain success does not establish the no-cheap-oracle north star.",
        ),
        (
            "intended_count completed_count failed_count censored_count excluded_count independent_count rows",
            "Retain every scheduled stream and typed metric; dependent checks never enlarge the independent denominator.",
        ),
        (
            "gate_check_summary",
            "Name upstream, path/hash, check, exact field, expected and observed values for each failed gate.",
        ),
        (
            "source_artifact_hashes code_config_hashes raw_shard_hashes field_principles",
            "Retain content custody and echo the reasons these evidence fields exist.",
        ),
        (
            "acceptance_gates required_checks_passed owned_statement_coverage validation_receipts global_test_debt",
            "Record each gate with its principle; failed owned checks disqualify and global debt cannot waive them.",
        ),
        (
            "online_rows_ready_score online_benefit_score",
            "Complete causal evidence gates readiness; scientific benefit belongs to Exp8967 and is unmeasured here.",
        ),
        (
            "continuous_self_learning_task prediction_rows retention_rows feedback_rows promotion_rows restart_rows adapter_state_hashes",
            "Show how released real errors change future decisions without accessing future or retention labels.",
        ),
    ):
        value["field_principles"].update({k: principle for k in fields.split()})
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
