"""REQ-REPORT-8102: freeze public learning inputs with owned Qwen receipts.

This capture supplies later replay features. It cannot establish historical
novelty, label support, successful learning or independent generalization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot import experiment_8099_v701_fit_source_capture as qualified
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.verify import qwen_learning_stream_capture_8102 as c

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8102_v701_learning_stream_capture"
TASK = "exp8102-learning-stream-capture"
MODEL_SPECS = [c.risk.MODEL]
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_learning_stream_capture_8102.py",
    f"scripts/experiments/{NAME}.py",
]
TEST = "tests/python/test_learning_stream_capture_8102.py"
INPUTS = [
    *qualified.INPUTS,
    "python/carnot/verify/evidence_features_7980.py",
    "python/carnot/experiment_8099_v701_fit_source_capture.py",
    "python/carnot/verify/qwen_fit_source_capture_8099.py",
    "python/carnot/experiment_7969_v691_qwen_calibration_capture.py",
    "python/carnot/inference/qwen_sufficiency_7920.py",
    "python/carnot/inference/llama_cpp_process.py",
    "python/carnot/gpu_lease_phase_journal.py",
]
operand = qualified.operand
run_check = qualified.run_check


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose real phase boundaries and counts without extending work duration."""
    print(
        f"[exp8102] phase={phase} completed_units={completed} pending_units={pending}", flush=True
    )


def preconditions(root: Path) -> Json:
    """Read authenticated public manifests; future evaluator shards stay closed."""
    checks, refs, values = [], [], {}
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / tool
        checks.append(operand("required_tool", path, "executable", True, os.access(path, os.X_OK)))
    for name in [*INPUTS, *qualified.PINS]:
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            operand(
                name,
                path,
                "sha256" if name in qualified.PINS else "exists",
                qualified.PINS.get(name, True),
                observed if name in qualified.PINS else path.is_file(),
            )
        )
        if observed:
            refs.append(dict(path=str(path), sha256=observed))
        if name not in qualified.PINS or observed != qualified.PINS[name]:
            continue
        value = json.loads(path.read_text())
        values[name] = value
        try:
            terminal = Path(value["terminal_validation_sidecar_path"])
            document = json.loads(terminal.read_text())
            publication = document.get("publication", document)
            bound = read_bound_sidecar(path, Path(publication["sidecar_path"]))
            checks.extend(
                [
                    operand(
                        name,
                        terminal,
                        "terminal_primary_sha256",
                        observed,
                        publication["primary_sha256"],
                    ),
                    operand(name, terminal, "terminal_passed", True, bound["report"]["passed"]),
                ]
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
    views, manifests, slots = {}, {}, []
    if qualified.UPSTREAM in values:
        methods = values[qualified.UPSTREAM]
        for key in ("cohort_ready_score", "required_checks_passed"):
            checks.append(operand("exp8098", root / qualified.UPSTREAM, key, 1, methods.get(key)))
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
    return dict(
        checks=checks,
        references=refs,
        slots=slots,
        manifests=manifests,
        protocol={
            k: values.get(qualified.HISTORY, {}).get(k) for k in ("gguf_sha256", "model_revision")
        },
        capture_identity=c.canonical_hash(
            [manifests, c.config(), {p: sha256_file(ROOT / p) for p in OWNED}]
        ),
    )


def live_capture(plan: Json, raw: Path, scratch: Path) -> Json:
    """Use the qualified process lease, embedded tokenizer and CUDA worker."""
    with patch.object(qualified, "c", c), patch.object(qualified, "TASK", TASK):
        return dict(qualified.live_capture(plan, raw, scratch))


def reduced_adapter(rows: list[Json]) -> Json:
    """Translate the common publication helper's transport key at its boundary."""
    reduced = c.reduce(rows)
    ready = reduced.pop("stream_capture_ready_score")
    return dict(reduced, fit_capture_ready_score=ready)


def build(
    plan: Json, result: Json, validation: Json, raw: Path, duration: float, *, fixture: bool
) -> Json:
    """Reuse custody schema while publishing stream and retention bytes separately."""
    adapter = SimpleNamespace(reduce=reduced_adapter, config=c.config)
    with (
        patch.object(qualified, "c", adapter),
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "TASK", TASK),
        patch.object(qualified, "INPUTS", INPUTS),
    ):
        value = dict(qualified.build(plan, result, validation, raw, duration, fixture=fixture))
    value.update(
        experiment_id=8102,
        task_id=TASK,
        stream_capture_ready_score=value.pop("fit_capture_ready_score"),
        capture_budget=c.config(),
        preparation_cost_rows=[
            dict(
                phase=phase, duration_s=cost, scope="current_capture", later_cpu_learning_cost=False
            )
            for phase, cost in [
                ("model_work", result.get("measured_duration_s", 0)),
                ("preparation_total", duration),
            ]
        ],
        methodology_note="Previously exposed development sources; frozen weights and complete original sources. Feature transport readiness gives no class-support, H1 or learning-benefit credit.",
    )
    for role in c.ROLES:
        path = raw / (role + "_features.json")
        atomic_json(
            path,
            dict(
                rows=value.pop(role + "_features"),
                feature_names=["qwen_risk_logit", *c.lexical.FEATURES],
                label_scope="no_labels",
                original_slot_order=True,
            ),
        )
        ref = dict(path=str(path), sha256=sha256_file(path))
        value[role + "_feature_manifest"] = ref
        value["raw_shard_hashes"].append(ref)
    value["acceptance_gates"] = dict(
        complete_stream=224,
        complete_retention=48,
        owned_checks_required=True,
        current_cuda_required=True,
        duration_floor_s=10,
        positive_H1_required=False,
    )
    value["honest_verdict"] = value["honest_verdict"].replace(
        "fit_source_capture", "learning_stream_capture"
    )
    value["field_principles"].update(
        {
            k: "Binds measured transport to its original source and excludes future labels."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        stream_capture_ready_score="Transport only: missing slots retain clocks and no class or benefit gate is used.",
        preparation_cost_rows="Capture costs are separate from later CPU learning costs.",
    )
    return value


def replay_value(value: Json) -> bool:
    """Cold reduction authenticates feature bytes, counters, ledger and masks."""
    expanded = dict(value, fit_capture_ready_score=value["stream_capture_ready_score"])
    for role in c.ROLES:
        ref = value[role + "_feature_manifest"]
        expanded[role + "_features"] = json.loads(Path(ref["path"]).read_text())["rows"]
    adapter = SimpleNamespace(reduce=reduced_adapter)
    with patch.object(qualified, "c", adapter):
        return bool(qualified.replay_value(expanded))


def commands(private: Path) -> list[Json]:
    """Freeze required argv and private real CLI checks before measuring calls."""
    import runpy

    fixtures = runpy.run_path(str(ROOT / TEST))["views"]()
    atomic_json(private / "public.json", fixtures)
    fixtures["stream"]["request_rows"][0]["future_label"] = 1
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
                60,
            ),
            (
                "consumer_tests_e2e015_019",
                [
                    pytest,
                    *common,
                    *[
                        str(ROOT / "tests/python" / p)
                        for p in (
                            "test_primary_publication_7928.py",
                            "test_current_work_receipt.py",
                            "test_qwen_development_capture_7995.py",
                            "test_source_boundary_7852.py",
                            "test_experiment_7942_v689_sentence_labels.py",
                        )
                    ],
                ],
                300,
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
                60,
            ),
            ("ruff_check", [ruff, "check", *[str(ROOT / p) for p in [*OWNED, TEST]]], 60),
            (
                "ruff_format",
                [ruff, "format", "--check", *[str(ROOT / p) for p in [*OWNED, TEST]]],
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
                120,
            ),
            (
                "spec_coverage",
                [py, str(ROOT / "scripts/check_spec_coverage.py"), str(ROOT / TEST)],
                60,
            ),
            ("repository_full_suite", [pytest, "tests/python", "-q"], 180),
        ]
    )
    return [
        dict(
            name=name,
            argv=argv,
            expected_exit=0,
            deadline_s=deadline,
            classification="diagnostic" if name == "repository_full_suite" else "required",
            external_cwd=name.startswith("cli_"),
        )
        for name, argv, deadline in entries
    ]


def validate(raw: Path, private: Path) -> Json:
    """Reuse bounded receipts and preserve repository health as a separate check."""
    with (
        patch.object(qualified, "commands", commands),
        patch.object(qualified, "progress", progress),
    ):
        return dict(qualified.validate(raw, private))


def terminal_publish(output: Path, value: Json, raw: Path) -> None:
    """Publish a terminal verdict only after normal auditor exits and cold replay."""
    candidate = raw / "audit_candidate.json"
    atomic_json(candidate, value)
    receipts = []
    with tempfile.TemporaryDirectory(prefix="carnot-8102-terminal-") as private:
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
            progress("before_subprocess_" + name)
            receipts.append(
                run_check(ROOT, spec, Path(private), raw / "terminal_logs", heartbeat_s=15)
            )
            progress("after_subprocess_" + name)
    value["validation_receipts"].extend(receipts)
    if not all(r["passed"] for r in receipts):
        value.update(
            stream_capture_ready_score=0,
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
    """Run bounded capture or private transport checks without opening labels."""
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
                ledger = c.prior.Ledger(raw / "fixture_ledger.json")
                result.update(
                    rows=c.capture(
                        plan["slots"],
                        qualified.legacy.FixtureRuntime(),
                        raw / "slots",
                        plan["capture_identity"],
                        ledger=ledger,
                    ),
                    ledger=ledger.rows,
                )
            except (ValueError, KeyError, TypeError) as error:
                result.update(owned_failure=True, error=str(error))
    else:
        plan = preconditions(ROOT)
        progress("preconditions_completed", len(plan["checks"]), len(plan["slots"]))
        with tempfile.TemporaryDirectory(prefix="carnot-8102-validation-") as private:
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
                    progress("before_model_capture", 0, 320)
                    result = live_capture(plan, raw, raw / "owned-model")
                    progress("after_model_capture", len(result["rows"]), 320 - len(result["rows"]))
        if not result["rows"] and plan["slots"]:
            failures = [r for r in [*plan["checks"], *result["checks"]] if not r["passed"]]
            reason = failures[0]["check"] if failures else "owned_validation_failed"
            result["rows"] = c.capture(
                plan["slots"],
                qualified.legacy.FixtureRuntime(),
                raw / "slots",
                plan["capture_identity"],
                ledger=c.prior.Ledger(raw / "ledger.json"),
                blocked_reason=reason,
            )
    result["phase_spans"] = [
        dict(phase="preconditions_validation_capture", start_s=0, end_s=time.monotonic() - started)
    ]
    value = build(plan, result, validation, raw, time.monotonic() - started, fixture=fixture)
    value["global_health"] = validation["global_health"]
    progress("before_terminal_validation")
    terminal_publish(output, value, raw)
    progress("published_" + value["honest_verdict"], len(value["rows"]))
    return 0
