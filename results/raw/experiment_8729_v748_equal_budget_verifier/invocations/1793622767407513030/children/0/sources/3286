"""REQ-REPORT-8118: publish current new-key costs with qualified CUDA custody.

The raw timing panel is independent of label support and fitted-head readiness.
Historical model evidence authenticates versions but never supplies current calls.
"""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot import experiment_8102_v701_learning_stream_capture as transport
from carnot import experiment_8099_v701_fit_source_capture as qualified
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.verify import fresh_acquisition_cost_8118 as c

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8118_v702_fresh_acquisition_cost"
TASK = "exp8118-fresh-acquisition-cost"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/fresh_acquisition_cost_8118.py",
    f"scripts/experiments/{NAME}.py",
]
TEST = "tests/python/test_fresh_acquisition_cost_8118.py"
METHODS = "results/experiment_8111_v702_methods_and_stream_custody.json"
HISTORY = "results/experiment_8102_v701_learning_stream_capture.json"
PINS = {
    METHODS: "sha256:9f0c89a0d205488ceb545ca93497e6341bbc3fd8d01a94d69c773ed6535e9ece",
    HISTORY: "sha256:32ca6d4b55322e3b31c8cf69fbc95dbb5a7fa065be8b84c5e59ce07c899623e8",
}
INPUTS = [
    *qualified.INPUTS,
    TEST,
    "tests/python/test_primary_publication_7928.py",
    "python/carnot/experiment_8102_v701_learning_stream_capture.py",
    "python/carnot/verify/qwen_learning_stream_capture_8102.py",
    "python/carnot/experiment_8106_v701_radial_service_cost.py",
    "results/experiment_8106_v701_radial_service_cost.json",
]
_build, _live, _validate = qualified.build, qualified.live_capture, transport.validate
_run_check = qualified.run_check
_commands = qualified.commands


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counters let the supervisor distinguish bounded work from silence."""
    print(f"[exp8118] phase={phase} completed={completed} pending={pending}", flush=True)


def preconditions(root: Path) -> Json:
    """Read only authenticated public roles and runtime provenance, never labels."""
    checks, refs, values = [], [], {}
    for name in [*INPUTS, *PINS]:
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            qualified.operand(
                name,
                path,
                "sha256" if name in PINS else "exists",
                PINS.get(name, True),
                observed if name in PINS else path.is_file(),
            )
        )
        if observed:
            refs.append(dict(path=str(path), sha256=observed))
        if name not in PINS or observed != PINS[name]:
            continue
        value = json.loads(path.read_text())
        values[name] = value
        try:
            terminal = Path(value["terminal_validation_sidecar_path"])
            publication = json.loads(terminal.read_text())["publication"]
            bound = read_bound_sidecar(path, Path(publication["sidecar_path"]))
            checks.extend(
                qualified.operand(name, terminal, k, expected, actual)
                for k, expected, actual in [
                    ("terminal_primary_sha256", observed, publication["primary_sha256"]),
                    ("terminal_passed", True, bound["report"]["passed"]),
                    ("required_checks_passed", True, value.get("required_checks_passed")),
                    ("flagged_adversarial", False, value.get("flagged_adversarial")),
                ]
            )
            refs.extend(
                dict(path=str(p), sha256=sha256_file(p))
                for p in [terminal, Path(publication["sidecar_path"])]
            )
        except (OSError, ValueError, KeyError) as error:
            checks.append(qualified.operand(name, path, "authenticated_terminal", True, str(error)))
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / tool
        checks.append(
            qualified.operand("required_tool", path, "executable", True, os.access(path, os.X_OK))
        )
    checks.append(
        qualified.operand(
            "live_environment", root, "CARNOT_FORCE_LIVE", "1", os.getenv("CARNOT_FORCE_LIVE")
        )
    )
    views, manifests = {}, {}
    for role in ("fit", "tune"):
        ref = values.get(METHODS, {}).get("role_manifests", {}).get(role)
        if ref:
            path = Path(ref["path"])
            observed = sha256_file(path) if path.is_file() else None
            checks.append(
                qualified.operand("exp8111_public", path, "sha256", ref["sha256"], observed)
            )
            if observed == ref["sha256"]:
                views[role], manifests[role] = json.loads(path.read_text()), ref
                refs.append(ref)
    history = values.get(HISTORY, {})
    protocol = {
        k: history.get(k) for k in ("gguf_sha256", "runtime_sha256", "chat_template_sha256")
    }
    protocol["model_revision"] = history.get("runtime_identity", {}).get("model_revision")
    binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    checks.extend(
        [
            qualified.operand("runtime", binary, "executable", True, os.access(binary, os.X_OK)),
            qualified.operand(
                "runtime",
                binary,
                "sha256",
                protocol["runtime_sha256"],
                sha256_file(binary) if binary.is_file() else None,
            ),
        ]
    )
    return dict(
        checks=checks,
        references=refs,
        slots=c.freeze(views) if len(views) == 2 else [],
        manifests=manifests,
        protocol=protocol,
        capture_identity=canonical_hash([manifests, protocol, c.config()]),
    )


def live_capture(plan: Json, raw: Path, scratch: Path) -> Json:
    """Use one qualified CUDA lease and reject version changes before generation."""
    runtime = qualified.legacy.QwenRuntime

    class VersionedRuntime(runtime):  # type: ignore[misc,valid-type]
        def load(self) -> Json:
            identity: Json = super().load()
            if (
                canonical_hash(identity["props"]["chat_template"])
                != plan["protocol"]["chat_template_sha256"]
            ):
                raise ValueError("chat_template_drift")
            return identity

    def pulse(phase: str, *args: Any) -> None:
        path = raw / "ledger.json"
        rows = json.loads(path.read_text())["rows"] if path.is_file() else []
        completed = sum(r["operation"] == "generation" and r["status"] != "running" for r in rows)
        progress(phase, completed, 48 - completed)

    adapter = SimpleNamespace(
        capture=partial(
            c.capture,
            key_identity=plan["protocol"],
            historical_keys=plan.get("historical_keys", []),
        ),
        config=c.config,
        risk=c.risk,
    )
    with (
        patch.object(qualified, "c", adapter),
        patch.object(qualified, "TASK", TASK),
        patch.object(qualified.legacy, "QwenRuntime", VersionedRuntime),
        patch.object(qualified.legacy, "progress", pulse),
    ):
        result: Json = _live(plan, raw, scratch)
    result["owned_failure"] = bool(
        result["ledger"] and any(not r["passed"] for r in result["checks"])
    )
    return result


def build(
    plan: Json, result: Json, validation: Json, raw: Path, duration: float, *, fixture: bool
) -> Json:
    """Keep acquisition readiness separate from learning and historical model work."""
    with (
        patch.object(qualified, "c", c),
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "TASK", TASK),
        patch.object(qualified, "INPUTS", INPUTS),
    ):
        value: Json = _build(plan, result, validation, raw, duration, fixture=fixture)
    value.update(
        experiment_id=8118,
        milestone="2026.10.702",
        acquisition_cost_ready_score=value.pop("fit_capture_ready_score"),
        planned_MODEL_SPECS=[c.risk.MODEL],
        verifier_is_oracle=fixture,
        claim_scope="current new-key acquisition costs; no accuracy or learning claims",
        exposure_scope="exposed public development fit sources",
        execution_owner=dict(pid=os.getpid(), argv=sys.argv),
        exact_judgment_keys=[r["judgment_key"] for r in value["rows"]],
        current_acquisition_rows=[]
        if fixture
        else [r for r in value["rows"] if r["current_acquisition"]],
        reuse_rows=[
            dict(unit_id=r["unit_id"], arm=r["arm"], judgment_key=r["judgment_key"], **r["reuse"])
            for r in value["rows"]
        ],
        component_cost_rows=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                arm=r["arm"],
                **r["component_costs"],
            )
            for r in value["rows"]
        ],
        cold_load_rows=[r for r in value["call_ledger"] if r["operation"] == "model_load"],
        cuda_telemetry={
            k: result[k]
            for k in ("capacity_receipt", "resident_gpu_receipt", "unloaded_gpu_receipt")
            if k in result
        },
        gpu_memory_delta_mb=None,
        gate_check_summary=[
            dict(r, hash=r.get("observed") if r.get("artifact_field") == "sha256" else None)
            for r in [*plan["checks"], *result.get("checks", [])]
        ],
        acceptance_gates=dict(
            current_calls=32,
            independent_sources=16,
            intended_slots=48,
            current_cuda_required=True,
            owned_checks_required=True,
            duration_floor_s=10,
            load_only_floor_s=2,
            fitted_head_required=False,
        ),
    )
    if result.get("resident_gpu_receipt"):
        resident = int(result["resident_gpu_receipt"]["output_tail"].strip())
        unloaded = int(result.get("unloaded_gpu_receipt", {}).get("output_tail", "0").strip())
        value["gpu_memory_delta_mb"] = resident - unloaded
    value["honest_verdict"] = value["honest_verdict"].replace(
        "fit_source_capture", "fresh_acquisition_cost"
    )
    manifest = raw / "acquisition_manifest.json"
    atomic_json(
        manifest,
        dict(
            schema="carnot.exp8118.acquisition.v1",
            consumer="exp8119",
            current_acquisition_rows=value["current_acquisition_rows"],
            component_cost_rows=value["component_cost_rows"],
            cold_load_rows=value["cold_load_rows"],
            exact_judgment_keys=value["exact_judgment_keys"],
            reuse_rows=value["reuse_rows"],
            runtime_identity=plan["protocol"],
            capture_budget=c.config(),
            acquisition_cost_ready_score=value["acquisition_cost_ready_score"],
            fixture_protocol_only=fixture,
        ),
    )
    value["acquisition_manifest"] = dict(path=str(manifest), sha256=sha256_file(manifest))
    value["raw_shard_hashes"].append(value["acquisition_manifest"])
    value["field_principles"].update(
        {
            k: "Binds actual current work to exact source bytes; grants no accuracy or independent learning credit."
            for k in value
        }
    )
    value["field_principles"].update(
        acquisition_cost_ready_score="Needs32 owned current calls across16 sources, current CUDA and owned checks, independently of head readiness.",
        acquisition_manifest="Consumers reopen measured primitive costs instead of inserting timing constants.",
        reuse_rows="Exact hits do no model work; changed keys escalate and cannot borrow a completion.",
        sample_size_budget="Two calls per source do not create two independent source clusters.",
    )
    return value


def replay_value(value: Json) -> bool:
    """Independently reopen primitive bytes and reduce all counters and custody."""
    for ref in [*value["raw_shard_hashes"], *value["source_artifact_hashes"]]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("raw_hash_drift")
    raw = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())["rows"]
    if raw != value["rows"]:
        raise ValueError("raw_rows_drift")
    reduced = c.reduce(raw)
    if any(value[k] != v for k, v in reduced.items() if k != "fit_capture_ready_score"):
        raise ValueError("reduction_drift")
    ledger = c.prior.Ledger(Path("/tmp/carnot-8118-unwritten-ledger.json"))
    ledger.rows = value["call_ledger"]
    counts = ledger.counts()
    substrate = (
        "model_bounded_generation"
        if counts["generation_calls_attempted"]
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load"
    )
    if (
        counts != value["model_invocation_counts"]
        or substrate != value["inference_substrate_class"]
        or value["MODEL_SPECS"] != ([c.risk.MODEL] if substrate != "no_model_load" else [])
    ):
        raise ValueError("invocation_substrate_drift")
    if not value["fixture_protocol_only"]:
        qualified.provenance(ledger, raw, live=substrate != "no_model_load")
    manifest = json.loads(Path(value["acquisition_manifest"]["path"]).read_text())
    for k in (
        "exact_judgment_keys",
        "current_acquisition_rows",
        "reuse_rows",
        "component_cost_rows",
        "cold_load_rows",
    ):
        if value[k] != manifest[k]:
            raise ValueError("manifest_drift")
    if value["acquisition_cost_ready_score"] and (
        not reduced["fit_capture_ready_score"]
        or value["fixture_protocol_only"]
        or value["verdict_class"] in ("blocked", "disqualified")
        or not value["required_checks_passed"]
        or not value["gpu_lease"]
        or not value["cuda_offload_receipts"].get("authenticated")
    ):
        raise ValueError("unsafe_acquisition_readiness")
    for receipt in value["validation_receipts"]:
        if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
            raise ValueError("validation_log_drift")
    return True


def commands(private: Path) -> list[Json]:
    """Freeze scoped checks and real private CLI routes before any model work."""
    with (
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "NAME", NAME),
    ):
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
        *[
            str(ROOT / "tests/python" / p)
            for p in [
                "test_primary_publication_7928.py",
                "test_current_work_receipt.py",
                "test_qwen_development_capture_7995.py",
                "test_source_boundary_7852.py",
                "test_experiment_7942_v689_sentence_labels.py",
            ]
        ],
    ]
    return [
        dict(
            consumer,
            name="collect_consumers",
            argv=[*consumer["argv"], "--collect-only"],
            deadline_s=60,
        ),
        *specs,
    ]


def run_check(
    root: Path, spec: Json, private: Path, durable: Path, *, heartbeat_s: float = 15
) -> Json:
    """Every normally exiting child leaves exact argv, elapsed time and log bytes."""
    progress("before_subprocess_" + spec["name"])
    row: Json = _run_check(root, spec, private, durable, heartbeat_s=min(heartbeat_s, 30))
    row["normal_exit"] = row["actual_exit"] >= 0 and not row["timed_out"]
    progress("after_subprocess_" + spec["name"])
    return row


def terminal_publish(output: Path, value: Json, raw: Path) -> None:
    """Publish checked bytes atomically; owned auditor failures grant zero readiness."""
    candidate = raw / "audit_candidate.json"
    atomic_json(candidate, value)
    receipts = []
    with tempfile.TemporaryDirectory(prefix="carnot-8118-terminal-") as private:
        for name, script, flags in [
            ("adversarial_verify", "adversarial_verify.py", ["--json"]),
            ("strict_row_consistency", "verdict_row_consistency_lint.py", ["--strict"]),
        ]:
            spec = dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts" / script),
                    *flags,
                    str(candidate),
                ],
                expected_exit=0,
                deadline_s=120,
            )
            receipts.append(run_check(ROOT, spec, Path(private), raw / "terminal_logs"))
    value["validation_receipts"].extend(receipts)
    if not all(r["passed"] for r in receipts):
        value.update(
            acquisition_cost_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=not receipts[0]["passed"],
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_validation",
        )
    replay_value(value)
    publication = publish_primary(
        output,
        value,
        lambda p: dict(passed=replay_value(json.loads(p.read_text())), receipts=receipts),
    )
    atomic_json(
        raw / "terminal_validation.json",
        dict(publication=publication, receipts=receipts, normal_exit=True),
    )


def main(argv: list[str] | None = None) -> int:
    """Parameterize the qualified terminal workflow without rewriting history."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    with ExitStack() as stack:
        for name, value in dict(
            c=c,
            NAME=NAME,
            TASK=TASK,
            OWNED=OWNED,
            TEST=TEST,
            preconditions=preconditions,
            build=build,
            replay_value=replay_value,
            commands=commands,
            live_capture=live_capture,
            progress=progress,
            terminal_publish=terminal_publish,
            run_check=run_check,
        ).items():
            stack.enter_context(patch.object(transport, name, value))
        stack.enter_context(patch.object(qualified, "run_check", run_check))
        return int(transport.main(argv))
