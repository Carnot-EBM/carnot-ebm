"""REQ-REPORT-8112: bind bounded Qwen source measurements to owned receipts.

Reuse the qualified publication and runtime paths so historical failures remain
visible while this invocation declares only the work it actually performed.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot import experiment_8099_v701_fit_source_capture as qualified
from carnot.reporting.current_work_receipt import sha256_file, canonical_hash
from carnot.verify import qwen_fit_source_capture_8112 as c

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8112_v702_fit_source_capture"
TASK = "exp8112-fit-source-capture"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_fit_source_capture_8112.py",
    f"scripts/experiments/{NAME}.py",
]
TEST = "tests/python/test_fit_source_capture_8112.py"
HISTORY = "results/experiment_8102_v701_learning_stream_capture.json"
METHODS = "results/experiment_8111_v702_methods_and_stream_custody.json"
PINS = {
    qualified.UPSTREAM: qualified.PINS[qualified.UPSTREAM],
    HISTORY: "sha256:32ca6d4b55322e3b31c8cf69fbc95dbb5a7fa065be8b84c5e59ce07c899623e8",
    METHODS: "sha256:9f0c89a0d205488ceb545ca93497e6341bbc3fd8d01a94d69c773ed6535e9ece",
}
CONSUMERS = [
    "test_primary_publication_7928.py",
    "test_current_work_receipt.py",
    "test_qwen_development_capture_7995.py",
    "test_fit_source_capture_8099.py",
    "test_learning_stream_capture_8102.py",
    "test_source_boundary_7852.py",
    "test_experiment_7942_v689_sentence_labels.py",
]
INPUTS = [
    *qualified.INPUTS,
    "python/carnot/experiment_8099_v701_fit_source_capture.py",
    "python/carnot/experiment_8102_v701_learning_stream_capture.py",
    "python/carnot/verify/qwen_fit_source_capture_8099.py",
    "python/carnot/verify/qwen_learning_stream_capture_8102.py",
    "results/experiment_8099_v701_fit_source_capture.json",
    "openspec/change-proposals/v702-methods-and-stream-protocol.md",
    *["tests/python/" + p for p in CONSUMERS],
]
operand = qualified.operand
_preconditions, _build, _replay, _commands, _live = (
    qualified.preconditions,
    qualified.build,
    qualified.replay_value,
    qualified.commands,
    qualified.live_capture,
)
_run_check = qualified.run_check


def preconditions(root: Path) -> Json:
    """Authenticate public custody and evaluator references before admission."""
    with scope():
        plan: Json = _preconditions(root)
    plan["evaluator_refs"] = {}
    history = root / HISTORY
    if history.is_file() and sha256_file(history) == PINS[HISTORY]:
        runtime = json.loads(history.read_text())
        plan["protocol"]["model_revision"] = runtime["runtime_identity"]["model_revision"]
        plan["protocol"].update({k: runtime[k] for k in ("runtime_sha256", "chat_template_sha256")})
        native = runtime["runtime_identity"]["native_binary"]
        binary = Path(native["path"])
        plan["checks"].extend(
            [
                operand(
                    "qualified_runtime", binary, "executable", True, os.access(binary, os.X_OK)
                ),
                operand(
                    "qualified_runtime",
                    binary,
                    "sha256",
                    native["sha256"],
                    sha256_file(binary) if binary.is_file() else None,
                ),
            ]
        )
        plan["references"].append(native)
    path = root / METHODS
    if path.is_file() and sha256_file(path) == PINS[METHODS]:
        value = json.loads(path.read_text())
        plan["checks"].append(
            operand(METHODS, path, "methods_ready_score", 1, value.get("methods_ready_score"))
        )
        plan["evaluator_refs"] = {r: value["evaluator_label_manifests"][r] for r in c.ROLES}
        for ref in plan["evaluator_refs"].values():
            p = Path(ref["path"])
            observed = sha256_file(p) if p.is_file() else None
            plan["checks"].append(
                operand("original_class_custody", p, "sha256", ref["sha256"], observed)
            )
            plan["references"].append(ref)
    plan["capture_identity"] = canonical_hash(
        [plan["capture_identity"], plan["protocol"], PINS[HISTORY], c.config()]
    )
    return plan


def live_capture(plan: Json, raw: Path, scratch: Path) -> Json:
    """Use the qualified owned worker and disqualify owned runtime failures."""

    def runtime_progress(phase: str, *args: Any) -> None:
        path = raw / "ledger.json"
        rows = json.loads(path.read_text())["rows"] if path.is_file() else []
        completed = sum(r["operation"] == "generation" and r["status"] != "running" for r in rows)
        progress(phase, completed, 288 - completed)

    with scope(), patch.object(qualified.legacy, "progress", runtime_progress):
        result: Json = _live(plan, raw, scratch)
    identity = result.get("model_identity_receipt", {})
    if identity.get("authenticated"):
        for field, observed in dict(
            runtime_sha256=result["native_binary"]["sha256"],
            chat_template_sha256=canonical_hash(identity["props"]["chat_template"]),
        ).items():
            result["checks"].append(
                operand(
                    "owned_runtime",
                    Path(result.get("model_path", scratch)),
                    field,
                    plan["protocol"][field],
                    observed,
                )
            )
    attempted = any(r["operation"] == "model_load" for r in result["ledger"])
    result["owned_failure"] = attempted and any(not r["passed"] for r in result["checks"])
    return result


def build(
    plan: Json, result: Json, validation: Json, raw: Path, duration: float, *, fixture: bool
) -> Json:
    """Join original labels after inference; altered evidence has no target."""
    labels = dict(plan.get("labels", {}))
    if not fixture:
        for role, ref in plan.get("evaluator_refs", {}).items():
            path = Path(ref["path"])
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("evaluator_hash_drift")
            labels[role] = json.loads(path.read_text())["rows"]
        targets = {r["unit_id"]: r["y"] for values in labels.values() for r in values}
        for row in result["rows"]:
            row["human_target"] = (
                targets.get(row["unit_id"]) if row["arm"] == "full_source" else None
            )
    with scope():
        value: Json = _build(plan, result, validation, raw, duration, fixture=fixture)
    value.update(
        experiment_id=8112,
        milestone="2026.10.702",
        planned_MODEL_SPECS=[c.risk.MODEL],
        verifier_is_oracle=fixture,
        claim_scope="source-use diagnostics on exposed original development sources",
        exposure_scope="exposed_development_within_run_disjoint",
        execution_owner=dict(pid=os.getpid(), argv=sys.argv),
        capture_protocol_override="operator: first24, cyclic modulo24 and exact reverse sentence blocks; historical Exp8111 preserved",
    )
    value["acceptance_gates"].update(fit_per_class=16, tune_per_class=8, load_only_floor_s=2)
    value["phase_spans"].extend(
        dict(
            phase=r["operation"],
            call_id=r["call_id"],
            started_monotonic_ns=r["started_monotonic_ns"],
            ended_monotonic_ns=r["ended_monotonic_ns"],
        )
        for r in value["call_ledger"]
    )
    value["honest_verdict"] = value["honest_verdict"].replace(
        "fit_source_capture", "v702_fit_source_capture"
    )
    value["gate_check_summary"] = [
        dict(r, hash=r.get("observed") if r.get("artifact_field") == "sha256" else None)
        for r in [*plan["checks"], *result["checks"]]
    ]
    value["field_principles"].update(
        {
            k: "Binds this invocation to exact evidence without independent-generalization credit."
            for k in value
        }
    )
    return value


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose work boundaries without extending the measurement duration."""
    print(
        f"[exp8112] phase={phase} completed_units={completed} pending_units={pending}", flush=True
    )


def scope() -> ExitStack:
    """Parameterize the qualified helpers without changing historical modules."""
    stack = ExitStack()
    for key, value in dict(
        c=c,
        NAME=NAME,
        TASK=TASK,
        OWNED=OWNED,
        TEST=TEST,
        INPUTS=INPUTS,
        HISTORY=HISTORY,
        PINS=PINS,
        progress=progress,
    ).items():
        stack.enter_context(patch.object(qualified, key, value))
    return stack


def replay_value(value: Json) -> bool:
    """Cold replay rejects a live declaration unsupported by owned calls."""
    with scope():
        _replay(value)
    counts = value["model_invocation_counts"]
    klass = (
        "model_bounded_generation"
        if counts["generation_calls_attempted"]
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load"
    )
    specs = [c.risk.MODEL] if klass != "no_model_load" else []
    if value["inference_substrate_class"] != klass or value["MODEL_SPECS"] != specs:
        raise ValueError("invocation_substrate_drift")
    return True


def commands(private: Path) -> list[Json]:
    """Collect actual consumers first and freeze private validation before calls."""
    with scope():
        specs: list[Json] = _commands(private)
    consumer = next(r for r in specs if r["name"] == "consumer_tests")
    consumer["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        *[str(ROOT / "tests/python" / p) for p in CONSUMERS],
    ]
    consumer["deadline_s"] = 300
    collect = dict(
        consumer,
        name="collect_consumers",
        argv=[*consumer["argv"], "--collect-only"],
        deadline_s=60,
    )
    return [collect, *specs]


def run_check(
    root: Path, spec: Json, private: Path, durable: Path, *, heartbeat_s: float = 30
) -> Json:
    """Record normal exit and visible boundaries around every owned subprocess."""
    progress("before_subprocess_" + spec["name"])
    receipt: Json = _run_check(root, spec, private, durable, heartbeat_s=min(heartbeat_s, 30))
    receipt["normal_exit"] = receipt["actual_exit"] >= 0 and not receipt["timed_out"]
    progress("after_subprocess_" + spec["name"])
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run the established bounded CLI with the new frozen source protocol."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    with scope(), ExitStack() as stack:
        for key, value in dict(
            preconditions=preconditions,
            build=build,
            replay_value=replay_value,
            commands=commands,
            live_capture=live_capture,
            run_check=run_check,
        ).items():
            stack.enter_context(patch.object(qualified, key, value))
        return int(qualified.main(argv))
