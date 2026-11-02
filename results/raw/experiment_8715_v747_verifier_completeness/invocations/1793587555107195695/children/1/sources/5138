"""REQ-REPORT-8214: reuse owned validation and CUDA supervision before publication.

Private fixture CLI runs exercise the real storage and publisher. Only the
normal CLI can acquire Qwen evidence, and historical primary bytes stay intact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.verify import prospective_service_8214 as e
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.inference.qwen_sufficiency_7920 import get_json
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
execute = e.qualified.execute


def validation_plan(private: Path) -> list[CommandSpec]:
    """Use the qualified coverage/CLI framework with this experiment's owned paths."""
    with (
        patch.object(e.qualified, "MODULES", [e.MODULE, e.RUNNER]),
        patch.object(e.qualified, "TEST", e.TEST),
        patch.object(e.recorder, "CLI", e.CLI),
    ):
        plan = e.qualified.validation_plan(private)
    py = str(e.ROOT / ".venv/bin/pytest")
    plan.append(
        CommandSpec(
            "recorder_replay_e2e003_004",
            (
                py,
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "recorder"),
                "tests/python/test_prospective_request_recorder_8213.py",
                "-q",
            ),
            "private_e2e",
            180,
        )
    )
    return plan


def validators(path: Path) -> list[CommandSpec]:
    """Point the existing auditor command builder at this real replay CLI."""
    with patch.object(e.recorder, "CLI", e.CLI):
        return e.qualified.validators(path)


def live(data: Json, raw: Path, private: Path) -> Json:
    """The existing CUDA supervisor supplies genuine offload, ownership and cleanup."""
    legacy = e.prior.shared.prior.qualified.legacy
    work: Json = {}
    before = time.monotonic()
    e.prior.shared.host.old.prior.old.host.load_binding(data)
    native_startup_s = time.monotonic() - before
    binary = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    actual = e.sha256_file(binary) if binary.is_file() else None
    if actual != data["identity"]["runtime_sha256"]:
        return dict(
            checks=[e.operand("runtime_sha256", binary, data["identity"]["runtime_sha256"], actual)]
        )

    def capture(
        slots: list[Json], runtime: Any, path: Path, identity: str, *, started: float
    ) -> list[Json]:
        """Start service work only after the supervisor has authenticated CUDA residency."""
        props = get_json(f"http://127.0.0.1:{runtime.port}/props")
        if e.key(props["chat_template"]) != data["identity"]["chat_template_sha256"]:
            raise ValueError("served_chat_template_sha256")
        work.update(
            e.measure(data, runtime, path, deadline_s=max(0, 3600 - (time.monotonic() - started)))
        )
        return list(work["requests"])

    adapter = SimpleNamespace(freeze=lambda _: data["schedule"]["rows"], capture=capture)
    identity = data["identity"]
    model = legacy.cached_current_model() or {}
    protocol = dict(gguf_sha256=identity["model_sha256"], model_revision=identity["revision"])
    plan = dict(
        public_role_manifests={}, capture_identity=e.key(data["schedule"]), protocol=protocol
    )
    with (
        patch.object(legacy, "capture", adapter),
        patch.object(legacy, "load_public", lambda _: {}),
        patch.object(legacy, "TASK", e.TASK),
    ):
        result: Json = legacy.live_capture(plan, raw, private)
    if work:
        work["service_startup_s"] = work["startup_s"]
        work["startup_s"] += result["model_identity_receipt"]["duration_s"] + native_startup_s
    result["work"] = work
    result["checks"].append(
        e.operand(
            "recorded_runtime_sha256",
            Path(model.get("model_path", "/missing")),
            identity["runtime_sha256"],
            result.get("native_binary", {}).get("sha256"),
        )
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Freeze checks, measure once, cold-replay a private candidate, then publish atomically."""
    began = time.monotonic()
    e.progress("8214_start", 0, 24)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("8214_cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    fixture = args.fixture_e2e is not None
    output = (args.fixture_e2e or args.output).absolute()
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("private fixtures require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8214-validation-"))
    private.chmod(0o700)
    plan = validation_plan(private)
    candidate = private / (e.NAME + ".json")
    health = CommandSpec(
        "repository_health_once",
        (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_health_not_owned",
        600,
    )
    resources = [
        CommandSpec(
            "cuda_backend",
            (
                str(e.ROOT / ".venv/bin/python"),
                "-c",
                "from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload()); "
                "raise SystemExit(not llama_cpp.llama_supports_gpu_offload())",
            ),
            "preconditions",
            30,
        ),
        CommandSpec("rust_toolchain", ("rustc", "--version"), "preconditions", 10),
        CommandSpec("cargo_toolchain", ("cargo", "--version"), "preconditions", 10),
    ]
    e.atomic_json(
        raw / "validation_commands.json",
        dict(
            commands=[asdict(c) for c in plan],
            resources=[asdict(c) for c in resources],
            terminal=[asdict(c) for c in validators(candidate)],
            repository_health=asdict(health),
            measurement_config=e.CONFIG,
            frozen_before_measurement=True,
        ),
    )
    code_refs = [
        e.copy_bytes(e.ROOT / p, raw / "code")
        for p in e.OWNED
        + [
            e.TEST,
            "python/carnot/verify/request_recorder_8213.py",
            "python/carnot/verify/complete_request_8174.py",
            "python/carnot/verify/durable_batch_8159.py",
            "python/carnot/inference/qwen_sufficiency_7920.py",
            "python/carnot/inference/llama_cpp_process.py",
            "python/carnot/reporting/primary_publication.py",
        ]
    ]
    e.progress("8214_preconditions_before", 0, 1)
    data = e.inputs(args.root, raw)
    e.progress("8214_preconditions_after", int(data["ready"]), 0)
    resource_receipts = [] if fixture else execute(resources, raw / "resources")
    data["checks"].extend(
        e.operand(r["name"], Path(r["stdout_path"]), 0, r["exit_code"]) for r in resource_receipts
    )
    data["ready"] = data["ready"] and all(r["passed"] for r in resource_receipts)
    e.atomic_json(Path(data["input_path"]), data)
    receipts = execute(plan[:1] if fixture else plan, raw / "validation")
    result: Json = {}
    if data["ready"] and all(r["passed"] for r in receipts):
        if fixture:
            result["work"] = e.measure(data, e.FixtureRuntime(), raw / "slots")
        else:
            result = live(data, raw, private)
    e.atomic_json(raw / "result.json", result)
    value = e.build(data, result, raw, receipts, time.monotonic() - began, fixture)
    value["repository_health"] = [] if fixture else execute([health], raw / "health")
    value["raw_shard_hashes"] = [e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    value["code_config_hashes"] = code_refs
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(candidate, value)
    terminal = execute(validators(candidate), raw / "terminal")
    if not all(r["passed"] for r in terminal):
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        return 1
    publication = publish_primary(output, value, lambda _: dict(passed=True, receipts=terminal))
    e.atomic_json(
        raw / "publication_receipt.json", dict(publication=publication, terminal=terminal)
    )
    e.progress("8214_published", 24, 0)
    return 0
