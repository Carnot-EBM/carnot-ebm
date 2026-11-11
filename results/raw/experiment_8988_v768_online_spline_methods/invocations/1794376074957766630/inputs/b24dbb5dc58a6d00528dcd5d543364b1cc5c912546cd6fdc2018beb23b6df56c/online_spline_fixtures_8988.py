"""REQ-REPORT-8988-CAUSAL: private informative, causal and real recovery controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot.memory import online_spline_8988 as a
from carnot.reporting import online_spline_methods_8988 as d
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]


def crash(request: Json, raw: Path) -> Json:
    """Kill an owned waiting child with SIGKILL, then resume from the durable query16."""
    atomic_json(raw / "request.json", request)
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [
        sys.executable,
        "-u",
        "-m",
        "carnot.memory.online_spline_8988",
        str(raw / "request.json"),
        str(raw / "restarted"),
    ]
    d.progress("before_kill_child", 0, 1)
    with (raw / "kill-child.log").open("w+") as log:
        child = subprocess.Popen(
            [*argv, "--stop16"],
            cwd="/tmp",
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        started, heartbeat = time.monotonic(), time.monotonic()
        try:
            while "READY16" not in (raw / "kill-child.log").read_text():
                if child.poll() is not None or time.monotonic() - started > 90:
                    raise ValueError("child_midpoint_failed")
                if time.monotonic() - heartbeat >= 15:
                    d.progress("waiting_child_query16", 0, 1)
                    heartbeat = time.monotonic()
                time.sleep(0.02)
            midpoint = json.loads((raw / "restarted/state.json").read_text())
        finally:
            child.kill()
            child.wait(timeout=10)
            if child.stdin is not None:
                child.stdin.close()
    d.progress("after_kill_child", 1, 0)
    atomic_json(raw / "midpoint.json", midpoint)
    d.progress("before_resume_child", 0, 1)
    with (raw / "resume-child.log").open("w") as log:
        resumed = subprocess.run(
            [*argv, "--resume"],
            cwd="/tmp",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
    d.progress("after_resume_child", 1, 0)
    full = a.trajectory(request, raw / "uninterrupted")
    recovered = json.loads((raw / "restarted/state.json").read_text())
    return dict(
        condition="real_sigkill_query16",
        killed_returncode=child.returncode,
        resume_returncode=resumed.returncode,
        midpoint_query=midpoint["query"],
        state_bit_exact=full == recovered,
        prediction_bit_exact=full["rows"] == recovered["rows"],
        passed=child.returncode == -9 and resumed.returncode == 0 and full == recovered,
        final_state_sha256=canonical_hash(full),
        argv=argv,
        duration_s=time.monotonic() - started,
        logs=[d.reference(raw / "kill-child.log"), d.reference(raw / "resume-child.log")],
    )


def fixtures(evidence: Json, raw: Path, replay: bool = False) -> Json:
    """All fixture labels stay in private raw sidecars and supply no independent observations."""
    if replay:
        result: Json = json.loads((raw / "fixtures.json").read_bytes())
        request = result["fixture_request"]
        if d.request(evidence, Path(request["label_path"])) != request:
            raise ValueError("fixture_operand_drift")
        with tempfile.TemporaryDirectory(prefix="carnot8988-replay-") as scratch:
            recomputed = a.trajectory(request, Path(scratch) / "informative")
            no_signal = d.request(
                evidence,
                Path(request["label_path"]).with_name("no-signal-labels.json"),
                no_signal=True,
            )
            constant = a.trajectory(no_signal, Path(scratch) / "no-signal")
        expected_rows = [
            dict(r, group_id=condition)
            for condition, state in [("informative", recomputed), ("no_signal", constant)]
            for r in state["rows"]
        ]
        if result["rows"] != expected_rows:
            raise ValueError("primitive_rows_drift")
        if canonical_hash(recomputed) != result["crash_restart_rows"][0]["final_state_sha256"]:
            raise ValueError("trajectory_drift")
        rows = result["rows"]
        if sum(r["value"] for r in rows) != result["summary_recomputation"]["prediction_sum"]:
            raise ValueError("primitive_reduction_drift")
        return result
    raw.mkdir(parents=True, exist_ok=True)
    d.progress("before_private_benchmark", 0, 2)
    request = d.request(evidence, raw / "informative-labels.json")
    x = np.asarray(request["online_design"]["spline"])
    labels = [
        int(a.probability(request["initial"]["adaptive_spline"], x[:, i]) >= 0.5) for i in range(32)
    ]
    atomic_json(
        Path(request["label_path"]), dict(labels=labels, scope="constructed_private_fixture")
    )
    full = a.trajectory(request, raw / "informative")
    no_signal = d.request(evidence, raw / "no-signal-labels.json", no_signal=True)
    atomic_json(
        Path(no_signal["label_path"]),
        dict(labels=[i % 2 for i in range(32)], scope="constructed_private_fixture"),
    )
    constant = a.trajectory(no_signal, raw / "no-signal")
    causal = deepcopy(request)
    causal["label_path"] = str(raw / "sentinel-labels.json")
    atomic_json(Path(causal["label_path"]), dict(labels=labels[:12] + [1 - y for y in labels[12:]]))
    sentinel = a.trajectory(causal, raw / "sentinel", stop=16)
    midpoint = a.trajectory(request, raw / "midpoint-direct", stop=16)
    duplicate = deepcopy(full)
    duplicate_result = a.feedback(duplicate, request, list(range(25, 29)), labels[24:28])
    t = np.zeros((3, 2))
    _, _, clipped = a.step(t, t, t, np.full((3, 2), 100.0), 1, False)
    rejected: Json = dict(
        query=8,
        coefficients={arm: t.tolist() for arm in a.ARMS},
        accumulator={arm: t.tolist() for arm in a.ARMS},
        acknowledgments=[],
        promotion_rows=[],
        feedback_rows=[],
        prediction_hash="private-rejection",
    )
    reject_request = dict(
        initial={arm: t.tolist() for arm in a.ARMS},
        online_design={k: np.ones((3, 32, 2)).tolist() for k in ["spline", "linear"]},
        guard_design={k: np.ones((3, 64, 2)).tolist() for k in ["spline", "linear"]},
        guard_labels=[0] * 64,
    )
    a.feedback(rejected, reject_request, [1, 2, 3, 4], [1] * 4)
    checks = [
        dict(
            condition="informative",
            passed=any(r["candidate_hash"] != r["before_hash"] for r in full["promotion_rows"]),
        ),
        dict(
            condition="no_signal",
            passed=len(set(r["value"] for r in constant["rows"] if r["arm"] == "frozen_spline"))
            == 1,
        ),
        dict(condition="causal_sentinel", passed=sentinel["rows"] == midpoint["rows"]),
        dict(condition="duplicate_feedback", passed=not duplicate_result and duplicate == full),
        dict(
            condition="clipping",
            passed=clipped["unclipped_norm"] > 1 and clipped["clipped_norm"] <= 1 + 1e-14,
        ),
        dict(
            condition="rejected_state",
            passed=all(
                not r["passed"] and r["before_hash"] == r["after_hash"]
                for r in rejected["promotion_rows"]
            )
            and rejected["acknowledgments"] == [1, 2, 3, 4],
        ),
        dict(
            condition="causal_budget",
            passed=full["acknowledgments"] == list(range(1, 29))
            and full["pending"] == [29, 30, 31, 32],
        ),
    ]
    restart = crash(request, raw)
    rows = [
        dict(r, group_id=condition)
        for condition, s in [("informative", full), ("no_signal", constant)]
        for r in s["rows"]
    ]
    operations = dict(
        labels_released_per_stream=28,
        pending_per_stream=4,
        arms=5,
        ensemble_states=3,
        proposals_per_stream=28,
        label_gradient_steps_per_stream=112,
        coefficient_slots_anchored_per_stream=28 * (3 * 3 * 216 + 3 * 27),
        guard_graph_evaluations_per_stream=7 * 4 * 64,
        guard_seed_predictions_per_stream=7 * 4 * 64 * 3 * 2,
        cost_boundary=d.protocol()["cost_boundary"],
    )
    result = dict(
        rows=rows,
        fixture_request=request,
        causal_fixture_rows=checks,
        crash_restart_rows=[restart],
        promotion_guard_rows=full["promotion_rows"]
        + constant["promotion_rows"]
        + rejected["promotion_rows"],
        operation_counts=operations,
        summary_recomputation=dict(
            prediction_sum=sum(r["numerator"] / r["denominator"] for r in rows),
            passed=all(r["numerator"] / r["denominator"] == r["value"] for r in rows),
        ),
    )
    atomic_json(raw / "rejected-state.json", rejected)
    atomic_json(raw / "fixtures.json", result)
    d.progress("after_private_benchmark", 2, 0)
    return result
