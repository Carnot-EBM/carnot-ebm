"""REQ-REPORT-8099-PUBLICATION: publish owned bounded Qwen source diagnostics.

Current inference remains separate from the historical qualified transport.
Only public role manifests and an aggregate support gate reach the worker.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot import experiment_7969_v691_qwen_calibration_capture as legacy
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import qwen_fit_source_capture_8099 as c
from carnot.verify.qwen_development_capture_7995 import Ledger, provenance

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8099_v701_fit_source_capture"
TASK = "exp8099-fit-source-capture"
MODEL_SPECS = [c.risk.MODEL]
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_fit_source_capture_8099.py",
    f"scripts/experiments/{NAME}.py",
]
TEST = "tests/python/test_fit_source_capture_8099.py"
UPSTREAM = "results/experiment_8098_v701_development_methods.json"
HISTORY = "results/experiment_7995_v693_qwen_development_capture.json"
PINS = {
    UPSTREAM: "sha256:4d28524a53f7eed96c2ce814da5a798828594a7e540d70d0de2f6a25ca5d6f76",
    HISTORY: "sha256:df6eb8b559181455348b1d806f23c36d13c7f49202c2d49b2233aa80edebe02d",
}
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "python/carnot/verify/qwen_development_capture_7995.py",
    "python/carnot/verify/qwen_response_risk_7958.py",
    "python/carnot/inference/sota_models.py",
]


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name the failed check and preserve the exact observed operand."""
    return dict(
        legacy.operand(upstream, path, field, expected, observed),
        upstream=upstream,
        check=Path(upstream).stem.replace("-", "_") + "_" + field,
    )


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show phase changes and real counters without extending run duration."""
    print(
        f"[exp8099] phase={phase} completed_units={completed} pending_units={pending}", flush=True
    )


def preconditions(root: Path) -> Json:
    """Authenticate terminals and public bytes without opening evaluator shards."""
    checks: list[Json] = []
    refs: list[Json] = []
    values: Json = {}
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / tool
        checks.append(operand("required_tool", path, "executable", True, os.access(path, os.X_OK)))
    for name in [*INPUTS, *PINS]:
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            operand(
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
            document = json.loads(terminal.read_text())
            publication = document.get("publication", document)
            bound = read_bound_sidecar(path, Path(publication["sidecar_path"]))
            checks.append(
                operand(
                    name,
                    terminal,
                    "terminal_primary_sha256",
                    observed,
                    publication["primary_sha256"],
                )
            )
            checks.append(
                operand(name, terminal, "terminal_passed", True, bound["report"]["passed"])
            )
            refs.extend(
                dict(path=str(p), sha256=sha256_file(p))
                for p in (terminal, Path(publication["sidecar_path"]))
            )
        except (OSError, ValueError, KeyError) as error:
            checks.append(operand(name, path, "authenticated_terminal", True, str(error)))
        checks.append(
            operand(name, path, "flagged_adversarial", False, value.get("flagged_adversarial"))
        )
        checks.append(operand(name, path, "verdict_class", "null", value.get("verdict_class")))
    manifests: Json = {}
    slots = []
    if UPSTREAM in values:
        methods = values[UPSTREAM]
        for key in ("cohort_ready_score", "fit_support_ready_score", "required_checks_passed"):
            checks.append(operand("exp8098", root / UPSTREAM, key, 1, methods.get(key)))
        views = {}
        for role in c.ROLES:
            ref = methods["role_manifests"][role]
            path = Path(ref["path"])
            observed = sha256_file(path) if path.is_file() else None
            checks.append(operand("exp8098", path, "public_sha256", ref["sha256"], observed))
            if observed == ref["sha256"]:
                views[role] = json.loads(path.read_text())
                manifests[role] = ref
                refs.append(ref)
        if all(r["passed"] for r in checks):
            slots = c.freeze(views)
    protocol = {k: values.get(HISTORY, {}).get(k) for k in ("gguf_sha256", "model_revision")}
    return dict(
        checks=checks,
        references=refs,
        slots=slots,
        manifests=manifests,
        protocol=protocol,
        capture_identity=canonical_hash(
            [manifests, c.config(), {p: sha256_file(ROOT / p) for p in OWNED}]
        ),
    )


def live_capture(plan: Json, raw: Path, scratch: Path) -> Json:
    """Reuse the qualified lease, loaded CUDA library and owned worker path."""
    ledger = Ledger(raw / "ledger.json")
    runtime_class = legacy.QwenRuntime

    class RecordedRuntime:
        def __init__(self, model: Path, scratch: Path, gpu: int) -> None:
            self.model = model
            self.runtime = runtime_class(model, scratch, gpu)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.runtime, name)

        def load(self) -> Json:
            ledger.start("model_load", "owned-model-load", dict(model=str(self.model)))
            try:
                result = dict(self.runtime.load())
            except (OSError, RuntimeError, TimeoutError, ValueError):
                ledger.finish("owned-model-load", "failed", {})
                raise
            ledger.finish("owned-model-load", "completed", result)
            return result

    adapter = SimpleNamespace(
        freeze=lambda _: plan["slots"], capture=partial(c.capture, ledger=ledger)
    )
    with (
        patch.object(legacy, "TASK", TASK),
        patch.object(legacy, "capture", adapter),
        patch.object(legacy, "load_public", lambda _: {}),
        patch.object(legacy, "QwenRuntime", RecordedRuntime),
    ):
        result = dict(
            legacy.live_capture(dict(plan, public_role_manifests=plan["manifests"]), raw, scratch)
        )
    result["ledger"] = ledger.rows
    for check in result["checks"]:
        check.update(
            check=check["upstream_id"] + "_" + check["field"], upstream=check["upstream_id"]
        )
    return result


def build(
    plan: Json, result: Json, validation: Json, raw: Path, duration: float, *, fixture: bool
) -> Json:
    """Readiness means usable transport, with no independent science credit."""
    reduced = c.reduce(result.get("rows", []))
    failures = [r for r in [*plan["checks"], *result.get("checks", [])] if not r["passed"]]
    owned = not validation["passed"] or result.get("owned_failure", False)
    klass = (
        "disqualified"
        if owned
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    identity = result.get("model_identity_receipt", {})
    cuda = bool(
        identity.get("authenticated")
        and identity.get("offload_layers", [0])[0] > 0
        and result.get("gpu_lease_receipt")
    )
    ready = int(
        not fixture
        and not owned
        and not failures
        and cuda
        and result.get("measured_duration_s", 0) >= 10
        and reduced["fit_capture_ready_score"]
    )
    ledger = result.get("ledger", [])
    current = Ledger(raw / "unused-ledger.json")
    current.rows = [] if fixture else ledger
    invocation_counts = current.counts()
    generation_started = invocation_counts["generation_calls_attempted"] > 0
    load_started = invocation_counts["model_loads_attempted"] > 0
    rows = result.get("rows", [])
    shard = raw / "primitive_rows.json"
    atomic_json(shard, dict(rows=rows))
    manifest = raw / "capture_manifest.json"
    atomic_json(
        manifest,
        dict(rows=plan["slots"], config=c.config(), capture_identity=plan.get("capture_identity")),
    )
    value = dict(
        reduced,
        experiment_id=8099,
        task_id=TASK,
        run_date="20261004",
        execution_date="20261004",
        milestone="2026.10.701",
        honest_verdict="complete_"
        + klass
        + "_"
        + (failures[0]["check"] if failures else "fit_source_capture"),
        verdict_class=klass,
        verifier_is_oracle=False,
        claim_scope=0,
        exposure_scope=0,
        generalized_learning_benefit_score=0,
        independent_generalization_score=0,
        flagged_adversarial=False,
        required_checks_passed=validation["passed"],
        validation_receipts=validation["receipts"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=plan["checks"],
        gate_check_summary=failures,
        inference_substrate="live_llm_inference"
        if generation_started or load_started
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation"
        if generation_started
        else "model_load_no_generation"
        if load_started
        else "no_model_load",
        MODEL_SPECS=MODEL_SPECS if generation_started or load_started else [],
        model_invocation_counts=invocation_counts,
        trained_head_specs=[],
        rows=rows,
        duration_s=duration,
        random_seed=c.config()["seed"],
        reproducibility_checksum=canonical_hash([plan["references"], c.config()]),
        source_artifact_hashes=plan["references"],
        raw_shard_hashes=[
            dict(path=str(shard), sha256=sha256_file(shard)),
            dict(path=str(manifest), sha256=sha256_file(manifest)),
        ],
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [*OWNED, TEST, *[x for x in INPUTS if x.endswith(".py")]]
        },
        phase_spans=result.get("phase_spans", []),
        fit_capture_ready_score=ready,
        capture_manifest=dict(path=str(manifest), sha256=sha256_file(manifest)),
        raw_completion_hashes=[canonical_hash(r["raw_response"]) for r in rows],
        context_complete=all(r.get("failure_kind") != "context" for r in rows),
        call_ledger=[] if fixture else ledger,
        fixture_transport_ledger=ledger if fixture else [],
        gpu_lease=result.get("gpu_lease_receipt", {}),
        cuda_offload_receipts=identity,
        gguf_sha256=result.get("gguf_sha256"),
        runtime_sha256=result.get("native_binary", {}).get("sha256"),
        chat_template_sha256=canonical_hash(identity.get("props", {}).get("chat_template")),
        runtime_receipts=result.get("runtime_receipts", []),
        runtime_identity={k: v for k, v in result.items() if k not in {"rows", "ledger", "checks"}},
        capture_budget=c.config(),
        fixture_protocol_only=fixture,
        acceptance_gates=dict(
            valid_fit=96,
            valid_tune=48,
            null_source_effect_allowed=True,
            current_cuda_required=True,
            owned_checks_required=True,
            duration_floor_s=10,
        ),
        methodology_note="Previously exposed V701 development sources. Frozen Qwen weights. Changed source evidence has no inherited human correctness labels. Source effects are mechanistic diagnostics only.",
    )
    value["field_principles"] = {
        k: "Binds the stated observation to its exact scope and prevents unmeasured science credit."
        for k in value
    }
    value["field_principles"].update(
        fit_capture_ready_score="Valid transport can have a null signal; readiness does not imply prediction benefit.",
        source_effect_diagnostics="Changed evidence has no inherited correctness label.",
        duplicate_variation="Repeated calls do not add independent sources.",
        model_invocation_counts="Only this invocation's owned events count as current model work.",
    )
    return value


def replay_value(value: Json) -> bool:
    """Recompute denominators and authenticate raw bytes before reader use."""
    for ref in value["raw_shard_hashes"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("raw_hash_drift")
    if json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())["rows"] != value["rows"]:
        raise ValueError("raw_rows_drift")
    reduced = c.reduce(value["rows"])
    independently_completed = sum(int(r["parsed"]["completed"]) for r in value["rows"])
    independently_grouped = len(
        {
            (r["role"], r["source_cluster_id"])
            for r in value["rows"]
            if r["arm"] == "full_source" and r["parsed"]["completed"]
        }
    )
    if (
        value["completed_count"] != independently_completed
        or value["independent_count"] != independently_grouped
    ):
        raise ValueError("independent_reduction_drift")
    for key, observed in reduced.items():
        if key != "fit_capture_ready_score" and value[key] != observed:
            raise ValueError("reduction_drift:" + key)
    ready = int(
        not value["fixture_protocol_only"]
        and value["verdict_class"] == "null"
        and value["required_checks_passed"]
        and bool(value["cuda_offload_receipts"].get("authenticated"))
        and bool(value["gpu_lease"])
        and value["runtime_identity"].get("measured_duration_s", 0) >= 10
        and reduced["fit_capture_ready_score"]
    )
    if value["fit_capture_ready_score"] != ready:
        raise ValueError("readiness_drift")
    ledger = Ledger(Path("/tmp") / "carnot-8099-unused-ledger")
    ledger.rows = value["call_ledger"]
    if ledger.counts() != value["model_invocation_counts"] or value["raw_completion_hashes"] != [
        canonical_hash(r["raw_response"]) for r in value["rows"]
    ]:
        raise ValueError("current_receipt_drift")
    if not value["fixture_protocol_only"]:
        provenance(ledger, value["rows"], live=True)
    for receipt in value["validation_receipts"]:
        if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
            raise ValueError("validation_log_drift")
    return True


def commands(private: Path) -> list[Json]:
    """Freeze exact required argv and diagnostic global health before inference."""
    import runpy

    fixtures = runpy.run_path(str(ROOT / TEST))["views"]()
    atomic_json(private / "public.json", fixtures)
    fixtures["fit"]["request_rows"][0]["labels"] = []
    atomic_json(private / "bad.json", fixtures)
    py, cov, pytest, ruff, mypy = (
        str(ROOT / ".venv/bin" / p) for p in ("python", "coverage", "pytest", "ruff", "mypy")
    )
    cli = str(ROOT / OWNED[2])
    include = "--include=" + ",".join(str(ROOT / p) for p in OWNED)
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    entries = [
        (
            "focused_coverage",
            [
                cov,
                "run",
                f"--data-file={private / '.coverage.unit'}",
                include,
                "-m",
                "pytest",
                *common,
                str(ROOT / TEST),
            ],
            0,
            300,
        )
    ]
    for name, file in [
        ("success", "public.json"),
        ("blocked", "absent.json"),
        ("mutation", "bad.json"),
    ]:
        entries.append(
            (
                "cli_" + name,
                [
                    cov,
                    "run",
                    f"--data-file={private / ('.coverage.' + name)}",
                    include,
                    cli,
                    "--date",
                    "20261004",
                    "--fixture",
                    str(private / file),
                    "--output",
                    str(private / name / (NAME + ".json")),
                ],
                0,
                120,
            )
        )
    entries.extend(
        [
            (
                "cli_cold_replay",
                [
                    py,
                    cli,
                    "--date",
                    "20261004",
                    "--cold-replay",
                    str(private / "success" / (NAME + ".json")),
                ],
                0,
                60,
            ),
            (
                "consumer_tests",
                [
                    pytest,
                    *common,
                    *[
                        str(ROOT / "tests/python" / p)
                        for p in (
                            "test_primary_publication_7928.py",
                            "test_current_work_receipt.py",
                            "test_qwen_development_capture_7995.py",
                            "test_experiment_7891_v685_authority_lifecycle.py",
                        )
                    ],
                ],
                0,
                180,
            ),
            (
                "coverage_combine",
                [
                    cov,
                    "combine",
                    f"--data-file={private / '.coverage.all'}",
                    *[
                        str(private / (".coverage." + p))
                        for p in ("unit", "success", "blocked", "mutation")
                    ],
                ],
                0,
                60,
            ),
            (
                "changed_code_coverage",
                [
                    cov,
                    "json",
                    f"--data-file={private / '.coverage.all'}",
                    include,
                    "--fail-under=100",
                    "-o",
                    str(private / "coverage.json"),
                ],
                0,
                60,
            ),
            ("ruff_check", [ruff, "check", *[str(ROOT / p) for p in [*OWNED, TEST]]], 0, 60),
            (
                "ruff_format",
                [ruff, "format", "--check", *[str(ROOT / p) for p in [*OWNED, TEST]]],
                0,
                60,
            ),
            (
                "strict_mypy",
                [
                    mypy,
                    "--config-file=/dev/null",
                    "--strict",
                    "--follow-imports=skip",
                    "--ignore-missing-imports",
                    *[str(ROOT / p) for p in OWNED[:2]],
                ],
                0,
                120,
            ),
            (
                "spec_coverage",
                [py, str(ROOT / "scripts/check_spec_coverage.py"), str(ROOT / TEST)],
                0,
                60,
            ),
            ("repository_full_suite", [pytest, "tests/python", "-q"], 0, 180),
        ]
    )
    return [
        dict(
            name=name,
            argv=argv,
            expected_exit=expected,
            deadline_s=deadline,
            classification="diagnostic" if name == "repository_full_suite" else "required",
            external_cwd=name.startswith("cli_"),
        )
        for name, argv, expected, deadline in entries
    ]


def validate(raw: Path, private: Path) -> Json:
    """Keep global failures separate from the frozen required command set."""
    specs = commands(private)
    atomic_json(raw / "validation_manifest.json", dict(commands=specs))
    receipts = []
    health_path = raw.parent / "global_health.json"
    for spec in specs:
        if spec["name"] == "repository_full_suite" and health_path.is_file():
            previous = json.loads(health_path.read_text())["receipt"]
            if sha256_file(Path(previous["log_path"])) != previous["log_sha256"]:
                raise ValueError("global_health_log_drift")
            receipts.append(previous)
            progress("preserved_repository_health_once")
            continue
        progress("before_subprocess_" + spec["name"])
        receipts.append(
            run_check(
                private if spec["external_cwd"] else ROOT,
                spec,
                private,
                raw / "validation_logs",
                heartbeat_s=15,
            )
        )
        progress("after_subprocess_" + spec["name"], len(receipts), len(specs) - len(receipts))
        if spec["name"] == "repository_full_suite":
            atomic_json(
                health_path, dict(receipt=receipts[-1], scope="diagnostic_repository_health_only")
            )
    if (private / "coverage.json").is_file():
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
    return dict(
        passed=all(r["passed"] for r in receipts if r["classification"] == "required"),
        receipts=[r for r in receipts if r["classification"] == "required"],
        global_health=[r for r in receipts if r["classification"] == "diagnostic"],
    )


def main(argv: list[str] | None = None) -> int:
    """Run one bounded capture and atomically publish a checked terminal result."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start_preconditions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=["20261004"])
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            replay_value(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        except (OSError, ValueError, KeyError) as error:
            progress("replay_rejected:" + str(error))
            return 1
    output = args.output.resolve()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    fixture = args.fixture is not None
    result: Json = dict(rows=[], ledger=[], checks=[])
    validation: Json = dict(passed=True, receipts=[], global_health=[])
    if fixture:
        path = args.fixture.resolve()
        plan: Json = dict(
            checks=[operand("fixture_public", path, "exists", True, path.is_file())],
            references=[],
            slots=[],
            protocol={},
            manifests={},
            capture_identity="private_stub",
        )
        if path.is_file():
            plan["references"] = [dict(path=str(path), sha256=sha256_file(path))]
            try:
                plan["slots"] = c.freeze(json.loads(path.read_text()))
                ledger = Ledger(raw / "fixture_ledger.json")
                result.update(
                    rows=c.capture(
                        plan["slots"],
                        legacy.FixtureRuntime(),
                        raw / "slots",
                        "private_stub",
                        ledger=ledger,
                    ),
                    ledger=ledger.rows,
                )
            except (ValueError, KeyError, TypeError) as error:
                result["owned_failure"] = True
                result["error"] = str(error)
    else:
        plan = preconditions(ROOT)
        progress("preconditions_completed", len(plan["checks"]), len(plan["slots"]))
        with tempfile.TemporaryDirectory(prefix="carnot-8099-validation-") as private:
            validation = validate(raw, Path(private))
            if validation["passed"] and all(r["passed"] for r in plan["checks"]):
                binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
                spec = dict(
                    name="cuda_support",
                    argv=[str(binary), "--list-devices"],
                    expected_exit=0,
                    deadline_s=15,
                )
                progress("before_subprocess_cuda_support")
                receipt = run_check(ROOT, spec, Path(private), raw / "preconditions", heartbeat_s=5)
                progress("after_subprocess_cuda_support")
                plan["checks"].append(
                    operand(
                        "cuda_support",
                        binary,
                        "native_cuda_device",
                        True,
                        receipt["passed"] and "CUDA" in receipt["output_tail"],
                    )
                )
                plan["references"].append(
                    dict(path=receipt["log_path"], sha256=receipt["log_sha256"])
                )
                if plan["checks"][-1]["passed"]:
                    progress("before_model_capture", 0, 288)
                    result = live_capture(plan, raw, raw / "owned-model")
                    progress("after_model_capture", len(result["rows"]), 288 - len(result["rows"]))
        if not result["rows"] and plan["slots"]:
            failures = [r for r in [*plan["checks"], *result["checks"]] if not r["passed"]]
            reason = failures[0]["check"] if failures else "owned_validation_failed"
            result["rows"] = c.capture(
                plan["slots"],
                legacy.FixtureRuntime(),
                raw / "slots",
                plan["capture_identity"],
                ledger=Ledger(raw / "ledger.json"),
                blocked_reason=reason,
            )
    result["phase_spans"] = [
        dict(phase="preconditions_validation_capture", start_s=0, end_s=time.monotonic() - started)
    ]
    value = build(plan, result, validation, raw, time.monotonic() - started, fixture=fixture)
    value["global_health"] = validation["global_health"]
    candidate = raw / "audit_candidate.json"
    atomic_json(candidate, value)
    progress("before_terminal_validation")
    with tempfile.TemporaryDirectory(prefix="carnot-8099-terminal-") as private:
        receipts = []
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
            receipts.append(
                run_check(ROOT, spec, Path(private), raw / "terminal_logs", heartbeat_s=15)
            )
    value["validation_receipts"].extend(receipts)
    if not all(r["passed"] for r in receipts):
        value.update(
            fit_capture_ready_score=0,
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
    progress("published_" + value["honest_verdict"], len(value["rows"]))
    return 0
