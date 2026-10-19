"""Publish bounded public Qwen calibration judgments with owned provenance.

REQ-REPORT-7969. The inference child sees public manifests only. Development
capture readiness says nothing about accuracy or future decision benefit.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot import experiment_7958_v690_qwen_response_risk as prior
from carnot.gpu_lease_phase_journal import GpuLease, LeaseError, proc_start_ticks
from carnot.inference.gguf_metadata import read_gguf_metadata
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded
from carnot.inference.sota_models import cached_current_model
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import qwen_calibration_capture_7969 as capture

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_7969_v691_qwen_calibration_capture"
TASK = "exp7969-qwen-calibration-capture"
MODEL_SPECS = prior.MODEL_SPECS
UPSTREAM = "results/experiment_7968_v691_response_role_targets.json"
UPSTREAM_PIN = "sha256:5978a0302945b4111afd06ee5f756ff0d8d810a3d84e2cf4fec4d88acb8f06d1"
HISTORY = "results/experiment_7958_v690_qwen_response_risk.json"
HISTORY_PIN = "sha256:3342ad4f66c5613482ac7da1da2b6f87099dfa3135647fa559b65748d7f62cad"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_calibration_capture_7969.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = [
    "tests/python/test_qwen_calibration_capture_7969.py",
    f"tests/python/test_{NAME}.py",
    *prior.TESTS,
    "tests/python/test_llama_cpp_process.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_gguf_metadata.py",
    "tests/python/test_gpu_lease_phase_journal.py",
    "tests/python/test_experiment_7920_v687_qwen_sufficiency.py",
]
IMPORTS = [
    "carnot.verify.qwen_response_risk_7958",
    "carnot.verify.qwen_completion_7932",
    "carnot.inference.qwen_sufficiency_7920",
    "carnot.inference.llama_cpp_process",
    "carnot.inference.sota_models",
    "carnot.inference.gguf_metadata",
    "carnot.gpu_lease_phase_journal",
    "carnot.reporting.primary_publication",
    "carnot.reporting.current_work_receipt",
    "carnot.reporting.experiment_7303_validation_scope",
    "scripts.conductor_gates",
    "scripts.in_process_doc_reconcile",
]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)
reference, operand = prior.reference, prior.operand


def progress(phase: str, started: float, units: int = 0) -> None:
    """Make real work and completed units visible without padding duration."""
    print(
        f"[exp7969] phase={phase} completed_units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def load_public(manifests: Json) -> Json:
    """Open only the four authenticated public role files in the model child."""
    views = {}
    for role in capture.ROLES:
        item = manifests[role]
        path = prior.targets.prior.checked_reference(item)
        views[role] = json.loads(path.read_text())
    capture.freeze(views)
    return views


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Authenticate each historical identity rather than rewriting its date."""
    checks: list[Json] = []
    upstream: Json = {}
    for label, name, pin, expected in [
        (
            "exp7968",
            UPSTREAM,
            UPSTREAM_PIN,
            dict(
                experiment_id=7968,
                task_id="exp7968-response-role-targets",
                milestone="2026.10.691",
                run_date="20261001",
                response_roles_ready_score=1,
                verdict_class="null",
                flagged_adversarial=False,
            ),
        ),
        (
            "exp7958",
            HISTORY,
            HISTORY_PIN,
            dict(
                experiment_id=7958,
                task_id=prior.TASK,
                milestone="2026.09.690",
                run_date="20261001",
                qwen_response_measurement_ready_score=1,
                verdict_class="null",
                flagged_adversarial=False,
            ),
        ),
    ]:
        path = root / name
        checks.append(
            operand(label, path, "sha256", pin, sha256_file(path) if path.is_file() else None)
        )
        value = json.loads(path.read_text()) if checks[-1]["passed"] else {}
        checks += [operand(label, path, k, v, value.get(k)) for k, v in expected.items()]
        upstream[label] = value
    import yaml

    exclusion = root / "ops/exclusion_manifest.yaml"
    retired = yaml.safe_load(exclusion.read_text()) if exclusion.is_file() else {}
    for eid in (7958, 7968):
        checks.append(
            operand(
                "exclusion_manifest",
                exclusion,
                f"exp{eid}_retired",
                False,
                any(
                    r.get("experiment_id") == eid
                    for key in ("retired", "retired_experiments")
                    for r in retired.get(key, [])
                )
                if exclusion.is_file()
                else None,
            )
        )
    if all(c["passed"] for c in checks):
        try:
            upstream["public_role_manifests"] = {
                k: upstream["exp7968"]["public_role_manifests"][k] for k in capture.ROLES
            }
            load_public(upstream["public_role_manifests"])
            history = upstream["exp7958"]
            for item in history["code_config_hashes"] + history["raw_response_shards"]:
                prior.targets.prior.checked_reference(item)
            sealed = json.loads(Path(history["raw_response_shards"][0]["path"]).read_text())["rows"]
            full = [r for r in sealed if r["arm"] == "full_source"]
            checks.append(
                operand("exp7958", root / HISTORY, "frozen_full_source_slots", 64, len(full))
            )
            checks.append(
                operand(
                    "exp7958",
                    root / HISTORY,
                    "gguf_sha256",
                    prior.MODEL_PIN,
                    history["gguf_sha256"],
                )
            )
            protocol = {
                k: capture.config()[k]
                for k in ("seed", "temperature", "max_tokens", "input_tokens", "grammar_sha256")
            }
            checks.append(
                operand(
                    "exp7958",
                    root / HISTORY,
                    "decoder_protocol",
                    protocol,
                    {k: history["token_budget"].get(k) for k in protocol},
                )
            )
            for r in full:
                original = dict(
                    family_id=r["family_id"],
                    source_bytes=json.loads(r["request"]["messages"][1]["content"])[
                        "complete_source"
                    ]
                    .encode()
                    .hex(),
                    answer_bytes=json.loads(r["request"]["messages"][1]["content"])[
                        "original_answer"
                    ]
                    .encode()
                    .hex(),
                )
                if (
                    r["request"]
                    != prior.risk.freeze([original], lambda _: 0)[0]["requests"]["full_source"]
                ):
                    raise ValueError("historical_prompt_drift")
            upstream["protocol"] = dict(
                decoder=protocol,
                system=prior.risk.SYSTEM,
                model_revision=history["model_revision"],
                gguf_sha256=history["gguf_sha256"],
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks.append(
                operand(
                    "public_protocol",
                    root / UPSTREAM,
                    "authenticated_public_protocol",
                    True,
                    str(error),
                )
            )
    upstream["checks"] = checks
    return [r for r in checks if not r["passed"]], upstream


def base(failures: list[Json]) -> Json:
    """Keep every declared field explicit even when external inputs block work."""
    value = prior.base(failures)
    for key in ("qwen_response_measurement_ready_score", "qwen_response_benefit_score"):
        value.pop(key)
    value.update(capture.reduce([]))
    value.update(
        schema="carnot.exp7969.qwen_calibration_capture.v1",
        experiment_id=7969,
        task_id=TASK,
        milestone="2026.10.691",
        run_date="20261001",
        execution_date="20261001",
        rows=[],
        request_manifest=None,
        protocol_fingerprint=None,
        grammar_sha256=capture.config()["grammar_sha256"],
        capture_budget=capture.config(),
        gpu_lease_receipt={},
        offload_evidence={},
        inference_mode="blocked_no_run",
        started_at=None,
        finished_at=None,
        scratch_root_receipt={},
        historical_evaluation_provenance={},
        acceptance_gate_results=dict(
            validity=False,
            readiness=False,
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        field_principles={},
        prior_verdicts_unchanged=[],
        resumed_invocation_counts={},
        public_role_manifests={},
        capture_identity=None,
    )
    return value


def loaded_libraries(owner: Json, raw: Path, proc: Path = Path("/proc")) -> Json:
    """Bind loaded native CUDA libraries to the worker's Linux start identity."""
    if proc_start_ticks(owner["pid"]) != owner["start_time_ticks"]:
        raise RuntimeError("owned_pid_start_identity")
    maps = (proc / str(owner["pid"]) / "maps").read_text()
    libraries = sorted(
        {
            line.split()[-1]
            for line in maps.splitlines()
            if line.split()
            and line.split()[-1].startswith("/")
            and any(
                name in line.split()[-1] for name in ("libggml", "libllama", "libcuda", "libcublas")
            )
        }
    )
    if not any("libggml-cuda" in path for path in libraries):
        raise RuntimeError("loaded_cuda_library")
    raw.mkdir(parents=True, exist_ok=True)
    snapshot = raw / "loaded_libraries.maps"
    snapshot.write_text(maps)
    return dict(
        owner_pid=owner["pid"],
        owner_start_time_ticks=owner["start_time_ticks"],
        maps=reference(snapshot),
        libraries=[reference(Path(path)) for path in libraries],
    )


def live_capture(plan: Json, raw: Path, scratch: Path) -> Json:
    """Own an idle GPU and preserve actual model, device and process receipts."""
    import threading

    started = time.monotonic()
    views = load_public(plan["public_role_manifests"])
    frozen = capture.freeze(views)
    atomic_json(
        raw / "request_manifest.json", dict(rows=frozen, capture_identity=plan["capture_identity"])
    )
    result: Json = dict(
        rows=[],
        checks=[],
        model_loads_attempted=0,
        model_loads_completed=0,
        capture_identity=plan["capture_identity"],
        runtime_receipts=[],
    )
    spec = cached_current_model() or {}
    model = Path(spec.get("model_path", scratch / "missing-model"))
    checks = result["checks"]
    checks += [
        operand("qwen_cache", model, "hf_id", prior.risk.MODEL, spec.get("hf_id")),
        operand("qwen_cache", model, "exists", True, model.is_file()),
    ]
    if not all(c["passed"] for c in checks):
        return result
    progress("before_model_hash", started)
    digest = bounded(lambda: sha256_file(model), 120)
    metadata = read_gguf_metadata(model)
    result.update(
        model_path=str(model),
        gguf_sha256=digest,
        model_revision=model.parent.name,
        gguf_metadata=metadata,
        quantization=metadata["quantization"],
    )
    checks += [
        operand("qwen_cache", model, "sha256", plan["protocol"]["gguf_sha256"], digest),
        operand(
            "qwen_cache", model, "revision", plan["protocol"]["model_revision"], model.parent.name
        ),
        operand("qwen_cache", model, "quantization", "Q4_K_M", metadata["quantization"]),
    ]
    progress("after_model_hash", started)
    if not all(c["passed"] for c in checks):
        return result
    capacity = run_commands(
        ROOT,
        [
            CommandSpec(
                "gpu_capacity",
                (
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ),
                "idle_cuda_gate",
                10,
            )
        ],
        log_dir=raw / "gpu",
        heartbeat_s=5,
    )[0]
    result["capacity_receipt"] = capacity
    devices = (
        [
            line.split(",")
            for line in capacity["output_tail"].splitlines()
            if len(line.split(",")) == 4
            and int(line.split(",")[2]) < 1000
            and int(line.split(",")[3]) >= 20000
        ]
        if capacity["passed"]
        else []
    )
    lease: Any = None
    for device in devices:
        try:
            lease = GpuLease.acquire(
                runtime_dir="/tmp/carnot-gpu-leases",
                task_id=TASK,
                device_uuid=device[1].strip(),
                expected_model=str(model),
                vram_before_mb=int(device[2]),
                ttl_s=3600,
            )
            break
        except LeaseError as error:
            checks.append(operand("gpu_lease", model, "available_owned_lease", True, str(error)))
    checks.append(
        operand("idle_cuda_gate", model, "eligible_owned_device", True, lease is not None)
    )
    if lease is None:
        return result
    # A lease refusal on another device does not reject a successfully owned device.
    checks[:] = [c for c in checks if c["field"] != "available_owned_lease"]
    gpu = int(device[0])
    result["gpu_lease_receipt"] = lease.owner_receipt()
    stop = threading.Event()

    def pulse() -> None:
        while not stop.wait(15):
            lease.heartbeat()
            progress("owned_runtime_heartbeat", started)

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    runtime = QwenRuntime(model, scratch, gpu)
    work_started = time.monotonic()
    try:
        lease.transition("admitted")
        lease.transition("loading")
        result["model_loads_attempted"] = 1
        progress("before_model_load", started)
        result["model_identity_receipt"] = runtime.load()
        result["model_loads_completed"] = 1
        result["native_binary"] = reference(Path(runtime.command[0]))
        result["resolved_library"] = loaded_libraries(
            result["model_identity_receipt"]["owner"], raw
        )
        progress("after_model_load", started)
        resident, receipt = prior.completion.prior.gpu_memory(gpu, scratch, raw)
        result["resident_gpu_receipt"] = receipt
        checks.append(
            operand("owned_cuda", model, "resident_memory_observed", True, resident >= 10000)
        )
        lease.transition("resident", vram_mb=resident)
        lease.transition("inferencing")
        if checks[-1]["passed"]:
            result["rows"] = capture.capture(
                frozen, runtime, raw / "slots", plan["capture_identity"], started=work_started
            )
        result["runtime_receipts"] = runtime.receipts
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        checks.append(
            operand(
                "owned_runtime",
                model,
                "authenticated_capture",
                True,
                f"{type(error).__name__}:{error}",
            )
        )
    finally:
        progress("before_model_unload", started)
        result["measured_duration_s"] = time.monotonic() - work_started
        result["cleanup"] = runtime.close()
        stop.set()
        thread.join(timeout=1)
        if lease.document["phase"] in ("resident", "inferencing"):
            lease.transition("unloading")
            after, receipt = prior.completion.prior.gpu_memory(gpu, scratch, raw)
            result["unloaded_gpu_receipt"] = receipt
            lease.transition(
                "validating",
                vram_mb=after,
                exit_code=0,
                unload_observed=result["cleanup"]["leak_free"],
            )
        lease.transition(
            "terminal_complete" if result["model_loads_completed"] else "terminal_blocked"
        )
        result["gpu_lease_receipt"]["release"] = lease.release()
        checks.append(
            operand("owned_cleanup", model, "leak_free", True, result["cleanup"]["leak_free"])
        )
        if runtime.log.is_file():
            shutil.copy2(runtime.log, raw / "server.log")
            result["server_log"] = reference(raw / "server.log")
        progress("after_model_unload", started)
    return result


class FixtureRuntime(prior.FixtureRuntime):
    """Scripted public transport tests the protocol without invoking a model."""

    def count(self, text: str) -> int:
        """Fixed fixture counts cannot stand in for the real embedded tokenizer."""
        return 30


def freeze_commands(raw: Path, scratch: Path, views: Json) -> Json:
    """Freeze exact commands, private scratch and dependencies before outcomes."""
    py, cov = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/coverage")
    coverage_file = str(scratch / ".coverage")
    fixture = scratch / "input.json"
    atomic_json(fixture, views)
    bad = scratch / "bad-input.json"
    atomic_json(bad, {"evaluation": {}})
    commands: list[Json] = []

    def add(
        name: str,
        argv: list[str],
        expected: int = 0,
        reason: str | None = None,
        required: bool = True,
        deadline: int = 300,
    ) -> None:
        commands.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=expected,
                failure_reason=reason,
                required=required,
                deadline_s=deadline,
            )
        )

    covered = [
        cov,
        "run",
        "--parallel-mode",
        "--data-file=" + coverage_file,
        "--include=" + INCLUDE,
    ]
    add(
        "unit_consumer_e2e015",
        [
            *covered,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=" + str(scratch / "pytest"),
            *TESTS,
            "-q",
        ],
    )
    cli = str(ROOT / OWNED[2])
    success = scratch / "success" / (NAME + ".json")
    for name, args, expected, reason in [
        ("fixture_cli", ["--fixture-input", str(fixture), "--output", str(success)], 0, None),
        ("cold_fixture_cli", ["--cold-replay", str(success)], 0, None),
        (
            "blocked_cli",
            [
                "--root",
                str(scratch / "missing"),
                "--output",
                str(scratch / "blocked" / (NAME + ".json")),
            ],
            0,
            None,
        ),
        (
            "negative_input_cli",
            ["--fixture-input", str(bad), "--output", str(scratch / "negative" / (NAME + ".json"))],
            1,
            "role_roster",
        ),
        ("negative_date_cli", ["--date", "20260930"], 2, "invalid choice"),
        ("cold_recorded_cli", ["--cold-replay", str(raw / "capture_candidate.json")], 0, None),
    ]:
        add(
            name,
            [
                *covered,
                cli,
                *([] if name == "negative_date_cli" else ["--date", "20261001"]),
                *args,
            ],
            expected,
            reason,
        )
    historical = str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    e2e = str(scratch / "e2e016" / "fixture.json")
    add("e2e016_fixture", [py, "-u", historical, "--date", "20260929", "--fixture-e2e", e2e])
    add("e2e016_cold", [py, "-u", historical, "--date", "20260929", "--cold-replay", e2e])
    add(
        "coverage_combine", [cov, "combine", "--keep", "--data-file=" + coverage_file, str(scratch)]
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            "--data-file=" + coverage_file,
            "--include=" + INCLUDE,
            "-o",
            str(scratch / "coverage.json"),
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            "--data-file=" + coverage_file,
            "--include=" + INCLUDE,
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS[:2]])
    add("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS[:2]])
    add("strict_mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=skip", *OWNED])
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS])
    add(
        "repository_health",
        [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        required=False,
        deadline=60,
    )
    manifest = dict(
        commands=commands,
        affected_files=OWNED,
        explicit_tests=TESTS,
        coverage_includes=INCLUDE,
        coverage_file=coverage_file,
        transitive_consumers=IMPORTS,
        frozen_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        current_date="20261001",
        historical_e2e016_date="20260929",
        scratch_root=str(scratch),
        terminal_commands=[
            dict(
                name=name,
                argv=[py, "-u", script, flag, str(raw / "terminal_candidate.json")],
                expected_exit=0,
                deadline_s=60,
            )
            for name, script, flag in [
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            ]
        ],
    )
    atomic_json(raw / "validation_command_manifest.json", manifest)
    return manifest


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Owned required failures disqualify, while historical failures stay open."""
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = value.get("owned_capture_commands", []) + [
        r.get("command_argv", []) for r in receipts
    ]
    value["repository_health"]["current"] = [r for r in receipts if not r.get("required", True)]
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_required_validation",
            qwen_capture_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)


def replay(value: Json) -> Json:
    """Rebuild public requests and claims from sealed raw bytes without a model."""
    for item in value.get("code_config_hashes", []) + value["source_artifact_hashes"]:
        prior.targets.prior.checked_reference(item)
    if not value["raw_response_shards"]:
        if value["qwen_capture_ready_score"]:
            raise ValueError("unsafe_readiness")
        return capture.reduce([])
    manifest = json.loads(
        prior.targets.prior.checked_reference(value["request_manifest"]).read_text()
    )
    views = (
        json.loads(prior.targets.prior.checked_reference(value["fixture_input"]).read_text())
        if value.get("fixture_input")
        else load_public(value["public_role_manifests"])
    )
    frozen = capture.freeze(views)
    if manifest != dict(rows=frozen, capture_identity=value["capture_identity"]):
        raise ValueError("request_manifest_drift")
    rows = [
        json.loads(prior.targets.prior.checked_reference(item).read_text())
        for item in value["raw_response_shards"]
    ]
    if len(rows) != len(frozen) or any(
        any(row.get(k) != v for k, v in slot.items())
        or row["capture_identity"] != value["capture_identity"]
        for row, slot in zip(rows, frozen, strict=True)
    ):
        raise ValueError("request_drift")
    reduced = capture.reduce(rows)
    for key, observed in reduced.items():
        if key == "qwen_capture_ready_score" and value["verdict_class"] != "null":
            continue
        if value[key] != observed:
            raise ValueError("reduction_drift:" + key)
    if value["rows"] != rows:
        raise ValueError("rows_drift")
    if value["qwen_capture_ready_score"]:
        if not value.get("validation_pending"):
            prior.targets.check_receipts(value)
        if (
            value["inference_mode"] != "live_gpu"
            or value["verdict_class"] != "null"
            or value["flagged_adversarial"]
        ):
            raise ValueError("unsafe_readiness")
    return reduced


def terminal_check(candidate: Path) -> Json:
    """Inspect cold claims and both required validators on exact final bytes."""
    replay(json.loads(candidate.read_text()))
    specs = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
            "terminal_candidate",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = run_commands(ROOT, specs, log_dir=candidate.parent / "terminal_logs", heartbeat_s=10)
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=any(r["name"] == "adversarial" and not r["passed"] for r in receipts),
    )


def publish(output: Path, value: Json) -> None:
    """Publish checked bytes and bind sidecars to the consumers' selected hash."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = dict(path=str(raw / "primary_resolution.json"))
    value["field_principles"] = {
        k: "Bind exact producer identity, public protocol, owned work or measured validity; capture does not establish science benefit."
        for k in value
    }
    value["field_principles"].update(
        qwen_capture_ready_score="All admitted slots accounted for plus 128 valid fit and 32 tune source clusters and passing owned checks; no accuracy threshold.",
        current_evaluation_call_count="Zero: existing evaluation responses are reserved for later analysis.",
        capture_budget="Load plus capture <=3000 seconds, <=384 calls, <=36864 output tokens; no retries.",
        protocol_fingerprint="Hash the unchanged historical whole-response prompt, grammar, decoder, GGUF and model revision.",
    )
    if value["verdict_class"] == "partial":
        raw.mkdir(parents=True, exist_ok=True)
        with (raw / "publication.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (
                output.name != NAME + ".json"
                or any(p != output for p in output.parent.glob("experiment_7969_*.json"))
                or value["qwen_capture_ready_score"]
            ):
                raise ValueError("partial_publication_identity")
            candidate = raw / "terminal_candidate.json"
            atomic_json(candidate, value)
            digest = sha256_file(candidate)
            report = terminal_check(candidate)
            if not report["passed"] or sha256_file(candidate) != digest:
                raise ValueError("candidate_rejected")
            sidecar = raw / "validators" / (digest.split(":")[1] + ".json")
            atomic_json(
                sidecar, dict(primary_path=str(output), primary_sha256=digest, report=report)
            )
            atomic_json(output, value)
            published = dict(primary_sha256=digest, sidecar_path=str(sidecar))
    else:
        published = publish_primary(output, value, terminal_check)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            primary_path=str(output),
            primary_sha256=published["primary_sha256"],
            validator=published["sidecar_path"],
        ),
    )
    atomic_json(raw / "newer_nested_sidecar.json", dict(task_id=TASK, qwen_capture_ready_score=99))
    receipt = reader_receipt(
        TASK,
        output.parent,
        field="qwen_capture_ready_score",
        expected=value["qwen_capture_ready_score"],
    )
    if not receipt["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", receipt)


def build(raw: Path, result: Json, upstream: Json, *, fixture: Path | None = None) -> Json:
    """Separate actual current invocations from scripted and historical evidence."""
    rows = result.get("rows", [])
    failures = [c for c in upstream.get("checks", []) + result.get("checks", []) if not c["passed"]]
    value = base(failures)
    value.update(capture.reduce(rows))
    value.update(
        rows=rows,
        capture_identity=result.get("capture_identity"),
        preconditions_checked=upstream.get("checks", []) + result.get("checks", []),
        raw_response_shards=[
            reference(raw / "slots" / f"slot-{i:03d}.json") for i in range(len(rows))
        ],
        request_manifest=reference(raw / "request_manifest.json")
        if (raw / "request_manifest.json").is_file()
        else None,
    )
    value["owned_capture_receipts"] = [
        result[k]
        for k in (
            "child_receipt",
            "capacity_receipt",
            "resident_gpu_receipt",
            "unloaded_gpu_receipt",
        )
        if result.get(k)
    ]
    value["owned_capture_commands"] = [
        r.get("command_argv", r.get("argv", [])) for r in value["owned_capture_receipts"]
    ]
    value["observed_child_commands"] = value["owned_capture_commands"]
    if fixture:
        value.update(
            fixture_input=reference(fixture),
            honest_verdict="complete_circular_positive_scripted_capture",
            verdict_class="circular_positive",
            qwen_capture_ready_score=0,
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            inference_mode="fixture",
        )
    else:
        value["public_role_manifests"] = upstream.get("public_role_manifests", {})
        value["source_artifact_hashes"] = [
            reference(ROOT / path) for path in (UPSTREAM, HISTORY) if (ROOT / path).is_file()
        ]
        value["cited_upstream_artifacts"] = value["source_artifact_hashes"]
        if upstream.get("protocol"):
            value["protocol_fingerprint"] = canonical_hash(upstream["protocol"])
            value["model_revision"] = upstream["protocol"]["model_revision"]
        history = upstream.get("exp7958", {})
        value["historical_required_failures"] = history.get("historical_required_failures", [])
        value["repository_health"]["historical"] = history.get("repository_health", {})
        value["historical_evaluation_provenance"] = dict(
            producer_id=7958,
            primary_path=str(ROOT / HISTORY),
            primary_sha256=HISTORY_PIN,
            model_invocation_counts=history.get("model_invocation_counts", {}),
            scope="historical_evaluation_only",
        )
        value["prior_verdicts_unchanged"] = [
            dict(
                experiment_id=eid,
                honest_verdict=upstream.get(f"exp{eid}", {}).get("honest_verdict"),
            )
            for eid in (7958, 7968)
        ]
        for key in (
            "model_identity_receipt",
            "gpu_lease_receipt",
            "gguf_sha256",
            "quantization",
            "resident_gpu_receipt",
            "unloaded_gpu_receipt",
            "server_log",
            "runtime_receipts",
            "cleanup",
            "resolved_library",
        ):
            if key in result:
                value[key] = result[key]
        loads = result.get("model_loads_attempted", 0)
        value["model_invocation_counts"].update(
            model_loads_attempted=loads,
            model_loads_completed=result.get("model_loads_completed", 0),
            model_loads_failed=loads - result.get("model_loads_completed", 0),
            generation_calls_attempted=sum(r["started"] for r in rows),
            generation_calls_completed=sum(r["status"] == "generated" for r in rows),
            generation_calls_failed=sum(r["started"] and r["status"] == "failed" for r in rows),
        )
        if loads:
            value.update(
                MODEL_SPECS=MODEL_SPECS,
                model_specs=MODEL_SPECS,
                target_model=prior.risk.MODEL,
                inference_substrate="live_llm_inference",
                inference_substrate_class="model_bounded_generation",
            )
        if result.get("model_loads_completed"):
            value["offload_evidence"] = dict(
                layers=result["model_identity_receipt"]["offload_layers"],
                resident_gpu_receipt=result["resident_gpu_receipt"],
            )
            value["inference_mode"] = "live_gpu"
        if failures:
            value["qwen_capture_ready_score"] = 0
        elif rows:
            partial = any(r["status"] == "censored" for r in rows)
            value.update(
                honest_verdict="complete_partial_owned_capture_budget"
                if partial
                else "complete_null_qwen_calibration_capture",
                verdict_class="partial" if partial else "null",
            )
        if value["qwen_capture_ready_score"] and result.get("measured_duration_s", 0) < 10:
            value.update(
                honest_verdict="complete_disqualified_duration_floor",
                verdict_class="disqualified",
                qwen_capture_ready_score=0,
            )
    value["acceptance_gate_results"].update(
        validity=value["verdict_class"] in {"null", "circular_positive"},
        readiness=bool(value["qwen_capture_ready_score"]),
    )
    return value


def main(argv: list[str] | None = None) -> int:
    """Provide isolated fixture, blocked, inference-child and cold-replay routes."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--fixture-input", type=Path)
    route.add_argument("--cold-replay", type=Path)
    route.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--runtime-scratch", type=Path)
    args = parser.parse_args(argv)
    raw = args.output.absolute().parent / "raw" / args.output.stem
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed", started)
            return 0
        if args.capture_manifest:
            if args.runtime_scratch is None:
                raise ValueError("runtime_scratch_required")
            plan = json.loads(args.capture_manifest.read_text())
            result = live_capture(plan, Path(plan["raw"]), args.runtime_scratch)
            atomic_json(Path(plan["raw"]) / "runtime_receipt.json", result)
            progress("capture_child_finished", started, len(result["rows"]))
            return 0
        with TemporaryDirectory(prefix="carnot-7969-") as workspace:
            scratch = Path(workspace)
            progress("authenticate", started)
            failures, upstream = ([], {}) if args.fixture_input else authenticate(args.root)
            views = (
                json.loads(args.fixture_input.read_text())
                if args.fixture_input
                else load_public(upstream["public_role_manifests"])
                if not failures
                else {
                    role: dict(
                        role=role,
                        request_rows=[
                            dict(
                                family_id=f"{role}-{i}",
                                source_bytes=f"Source {role} {i}.".encode().hex(),
                                answer_bytes=b"Answer.".hex(),
                            )
                            for i in range(n)
                        ],
                        boundaries=[
                            dict(
                                family_id=f"{role}-{i}", public_eligible=True, exclusion_reason=None
                            )
                            for i in range(n)
                        ],
                    )
                    for role, n in capture.ROLES.items()
                }
            )
            frozen = capture.freeze(views)
            manifest = freeze_commands(raw, scratch, views)
            imports = {
                name: str(Path(importlib.import_module(name).__file__).resolve())
                for name in IMPORTS
            }
            code_hashes = [reference(ROOT / p) for p in OWNED + TESTS] + [
                reference(Path(p)) for p in imports.values()
            ]
            identity = canonical_hash(
                dict(
                    code=code_hashes,
                    config=capture.config(),
                    public=frozen,
                    protocol=upstream.get("protocol"),
                )
            )
            result: Json = dict(rows=[], capture_identity=identity)
            if args.fixture_input:
                atomic_json(
                    raw / "request_manifest.json", dict(rows=frozen, capture_identity=identity)
                )
                result["rows"] = capture.capture(frozen, FixtureRuntime(), raw / "slots", identity)
            elif not failures:
                plan = dict(
                    public_role_manifests=upstream["public_role_manifests"],
                    protocol=upstream["protocol"],
                    capture_identity=identity,
                    raw=str(raw),
                )
                atomic_json(raw / "capture_plan.json", plan)
                progress("before_live_capture_child", started)
                receipt = run_commands(
                    ROOT,
                    [
                        CommandSpec(
                            "live_capture",
                            (
                                str(ROOT / ".venv/bin/python"),
                                "-u",
                                str(ROOT / OWNED[2]),
                                "--date",
                                "20261001",
                                "--capture-manifest",
                                str(raw / "capture_plan.json"),
                                "--runtime-scratch",
                                str(scratch),
                            ),
                            "owned_live_e2e",
                            3180,
                        )
                    ],
                    log_dir=raw / "live_logs",
                    heartbeat_s=10,
                )[0]
                result = (
                    json.loads((raw / "runtime_receipt.json").read_text())
                    if (raw / "runtime_receipt.json").is_file()
                    else dict(
                        rows=[],
                        checks=[
                            operand(
                                "owned_child",
                                raw / "runtime_receipt.json",
                                "completed",
                                True,
                                False,
                            )
                        ],
                    )
                )
                result["child_receipt"] = receipt
                progress("after_live_capture_child", started, len(result["rows"]))
            value = build(raw, result, upstream, fixture=args.fixture_input)
            value.update(
                started_at=started_at,
                scratch_root_receipt=dict(
                    path=workspace,
                    outside_checkout=True,
                    cleanup_after_children=True,
                    purpose="private pytest, coverage, mutable CLI publications and owned process state",
                ),
                resolved_imports=imports,
                code_config_hashes=code_hashes
                + [reference(raw / "validation_command_manifest.json")],
                reproducibility_checksum=identity,
                validation_command_manifest_path=str(raw / "validation_command_manifest.json"),
            )
            value["capture_identity"] = identity
            capture_end = time.monotonic() - started
            value.update(
                duration_s=capture_end,
                phase_spans=[
                    dict(phase="authentication_load_capture", start_s=0.0, end_s=capture_end)
                ],
                validation_pending=True,
            )
            atomic_json(raw / "capture_candidate.json", value)
            if not args.fixture_input and args.output.absolute().is_relative_to(ROOT / "results"):
                progress("before_required_validation", started)
                with patch.dict(
                    os.environ,
                    {
                        "PYTEST_ADDOPTS": "--basetemp=" + str(scratch / "health-pytest"),
                        "COVERAGE_FILE": str(scratch / "repository-health.coverage"),
                    },
                ):
                    receipts = prior.targets.prior.execute_commands(
                        manifest, raw / "validation_logs"
                    )
                apply_validation(value, receipts)
                coverage_path = scratch / "coverage.json"
                measured = (
                    json.loads(coverage_path.read_text())
                    if coverage_path.is_file()
                    else {"files": {}}
                )
                value["coverage_statement_counts"] = {
                    p: info["summary"] for p, info in measured["files"].items()
                }
                if not measured["files"] or any(
                    info["summary"]["missing_lines"] for info in measured["files"].values()
                ):
                    apply_validation(
                        value,
                        receipts
                        + [dict(name="nonempty_complete_coverage", passed=False, required=True)],
                    )
                for path in scratch.glob(".coverage*"):
                    shutil.copy2(path, raw / path.name)
                if coverage_path.is_file():
                    shutil.copy2(coverage_path, raw / "coverage.json")
                progress("after_required_validation", started)
            value.update(validation_pending=False, finished_at=datetime.now(UTC).isoformat())
            elapsed = time.monotonic() - started
            value.update(
                duration_s=elapsed,
                phase_spans=[
                    *value["phase_spans"],
                    dict(phase="validation", start_s=capture_end, end_s=elapsed),
                ],
            )
            progress("before_terminal_publication", started)
            publish(args.output.absolute(), value)
            progress("terminal_published", started, len(value["rows"]))
            return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7969] rejected={type(error).__name__}:{error}", flush=True)
        return 1
