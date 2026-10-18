"""REQ-REPORT-8633: installed execution preserves the original numerical authority.

These diagnostics qualify executable mechanics. They do not measure speed or
turn a research checkpoint into independently calibrated factual confidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time
from tempfile import TemporaryDirectory
from typing import Any

from carnot.inference import resident_protocol as protocol
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8633_v741_installed_lean_prototype"
TASK = "exp8633-installed-lean-prototype"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
SOURCE = "results/experiment_8622_v740_lean_resident_cost.json"
PIN = "sha256:53ad0975f8b227e921bd9164541b846d617c5c338e8216d0bc0761c5dcf7d40a"
OWNED = [
    "python/carnot/__init__.py",
    "python/carnot/inference/resident_protocol.py",
    "python/carnot/reporting/installed_lean_8633.py",
    "python/carnot/reporting/installed_runner_8633.py",
    CLI,
    "python/carnot/reporting/installed_validation_8633.py",
]
TESTS = [
    "tests/python/test_installed_lean_8633.py",
    "tests/python/test_installed_prototype_8633.py",
    "tests/python/test_installed_runner_8633.py",
]


def reference(path: Path) -> Json:
    """Retain actual byte hashes so replay never trusts a path alone."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def load(ref: Json) -> Json:
    """Refuse changed operands before interpreting their JSON fields."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("input_hash")
    return dict(json.loads(path.read_bytes()))


def inputs(root: Path) -> Json:
    """Authenticate only the original producer, retained head and64 request operands."""
    ref = dict(path=str(root / SOURCE), sha256=PIN)
    original = load(ref)
    work = load(original["work_reference"])
    bundle: Json = work["bundle"]
    refs = [ref, original["work_reference"], bundle["target_model"]["checkpoint"]]
    refs.append({k: bundle["extension"][k] for k in ["path", "sha256"]})
    for operand in refs:
        if sha256_file(Path(operand["path"])) != operand["sha256"]:
            raise ValueError("original_operand_hash")
    return dict(bundle, refs=refs)


def reduce_rows(rows: list[Json], bundle: Json) -> Json:
    """A response must retain its request, arithmetic arm and exact baseline result."""
    expected = [(arm, row["unit_id"]) for arm in ["python", "native"] for row in bundle["requests"]]
    if [(r["arm"], r["unit_id"]) for r in rows] != expected:
        raise ValueError("request_roster")
    for index, (row, (_, identifier)) in enumerate(zip(rows, expected, strict=True)):
        original = bundle["references"][index % 64]
        if (
            row["status"] != "ok"
            or not row["parity"]
            or row["response"] != row["baseline_response"]
            or row["response"].get("unit_id") != identifier
            or row["response"].get("decision") != original["decision"]
            or abs(row["response"].get("probability", 10) - original["probability"]) > 1e-12
        ):
            raise ValueError("typed_response_parity")
    return dict(
        intended_count=128,
        completed_count=128,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
    )


def session(
    command: list[str],
    config: Path,
    digest: str,
    arm: str,
    rows: list[Json],
    cwd: Path,
    log: Path,
    original_frames: list[Json] | None = None,
) -> Json:
    """Own a bounded pipe child and require a clean shutdown after actual inference."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    progress("8633_before_worker_" + arm)
    began = time.monotonic_ns()
    with (
        log.open("wb") as err,
        subprocess.Popen(
            [*command, str(config), digest, arm],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
        ) as process,
    ):
        assert process.stdin is not None and process.stdout is not None
        ready = json.loads(protocol.read_frame(process.stdout.fileno(), 30) or b"null")
        responses, frames = [], []
        for index, row in enumerate(rows):
            frame = (
                original_frames[index]
                if original_frames is not None
                else protocol.envelope(str(index), digest, row)
            )
            protocol.write_frame(process.stdin.fileno(), frame)
            response = json.loads(protocol.read_frame(process.stdout.fileno(), 30) or b"null")
            frames.append(frame)
            responses.append(response)
        shutdown = dict(protocol.envelope("shutdown", digest, {}), operation="shutdown")
        protocol.write_frame(process.stdin.fileno(), shutdown)
        stopped = json.loads(protocol.read_frame(process.stdout.fileno(), 30) or b"null")
        process.stdin.close()
        code = process.wait(timeout=30)
    progress("8633_after_worker_" + arm, len(rows), 0)
    return dict(
        command=command,
        cwd=str(cwd),
        pythonpath_unset=True,
        ready=ready,
        frames=frames,
        responses=responses,
        shutdown=stopped,
        exit_code=code,
        stderr_reference=reference(log),
        started_monotonic_ns=began,
        ended_monotonic_ns=time.monotonic_ns(),
    )


def measure(raw: Path, bundle: Json) -> Json:
    """Build offline and launch the installed module outside all checkout directories."""
    raw.mkdir(parents=True, exist_ok=True)
    receipts, parity, traces = [], [], []
    with TemporaryDirectory(prefix="carnot-8633-", dir="/var/tmp") as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        source.mkdir()
        shutil.copytree(
            ROOT / "python",
            source / "python",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        for name in ["pyproject.toml", "README.md", "LICENSE", "NOTICE"]:
            shutil.copy2(ROOT / name, source / name)
        wheels = raw / "wheels"
        receipts.append(
            child(
                "wheel_build",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-index",
                    "--no-deps",
                    "--no-build-isolation",
                    str(source),
                    "-w",
                    str(wheels),
                ],
                raw,
                deadline=180,
            )
        )
        (wheel,) = wheels.glob("*.whl")
        environment = scratch / "environment"
        receipts.append(
            child(
                "venv",
                [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
                raw,
                deadline=90,
            )
        )
        python = str(environment / "bin/python")
        (site_packages,) = environment.glob("lib/python*/site-packages")
        (site_packages / "local-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
        receipts.append(
            child(
                "install",
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--ignore-installed",
                    str(wheel),
                ],
                raw,
                deadline=90,
            )
        )
        config = raw / "head.json"
        atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))
        digest = sha256_file(config)
        for arm in ["python", "native"]:
            baseline = session(
                [sys.executable, "-u", str(ROOT / "scripts/experiments/resident_worker_8608.py")],
                config,
                digest,
                arm,
                bundle["requests"],
                scratch,
                raw / (arm + "-baseline.log"),
            )
            installed = session(
                [python, "-u", "-X", "importtime", "-m", "carnot.inference.resident_protocol"],
                config,
                digest,
                arm,
                bundle["requests"],
                scratch,
                raw / (arm + "-installed.log"),
                baseline["frames"],
            )
            atomic_json(
                raw / (arm + "-sessions.json"), dict(baseline=baseline, installed=installed)
            )
            for row, old, new in zip(
                bundle["requests"], baseline["responses"], installed["responses"], strict=True
            ):
                parity.append(
                    dict(
                        unit_id=row["unit_id"],
                        arm=arm,
                        status=new["status"],
                        response=new.get("response"),
                        baseline_response=old.get("response"),
                        parity=new.get("response") == old.get("response"),
                    )
                )
            modules = [
                line.split("|")[-1].strip()
                for line in (raw / (arm + "-installed.log")).read_text().splitlines()
                if line.startswith("import time:")
            ]
            traces.append(
                dict(
                    arm=arm,
                    modules=modules,
                    clean=not any(
                        n.startswith(("carnot.training", "carnot.reporting", "jax"))
                        for n in modules
                    ),
                    clean_shutdown=installed["exit_code"] == baseline["exit_code"] == 0
                    and installed["ready"]["config_hash"]
                    == baseline["ready"]["config_hash"]
                    == digest
                    and installed["ready"]["loaded_extension"]
                    == baseline["ready"]["loaded_extension"]
                    and installed["shutdown"]["status"]
                    == baseline["shutdown"]["status"]
                    == "shutdown"
                    and installed["frames"] == baseline["frames"],
                    log_reference=reference(raw / (arm + "-installed.log")),
                )
            )
        code = """
import importlib, importlib.metadata, json, sys
import carnot
from carnot.inference import resident_protocol
origin = resident_protocol.__file__
assert origin.startswith(sys.prefix + '/')
assert not any(n.startswith(('carnot.training','carnot.reporting','jax')) for n in sys.modules)
rows = []
for name in carnot.__all__:
 module = carnot._EXPORTS.get(name)
 if module:
  scope = {}
  exec('from carnot import ' + name, scope)
  passed = scope[name] is getattr(importlib.import_module('carnot.' + module), name)
 else:
  passed = getattr(carnot, name) is not None
 rows.append(dict(path='carnot.' + name, passed=passed))
record = importlib.metadata.distribution('carnot-ebm').read_text('RECORD')
versions = {name:importlib.metadata.version(name) for name in ['carnot-ebm','numpy','scipy','setuptools','pip']}
print(json.dumps(dict(rows=rows, origin=origin, version=carnot.__version__, record=record, versions=versions)))
"""
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        progress("8633_before_installed_exports")
        result = subprocess.run(
            [python, "-u", "-c", code],
            cwd=scratch,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        progress("8633_after_installed_exports")
        (raw / "exports.stdout").write_text(result.stdout)
        (raw / "exports.stderr").write_text(result.stderr)
        exports = json.loads(result.stdout)
        assert result.returncode == 0
        (raw / "installed-RECORD.csv").write_text(exports["record"])
    reduce_rows(parity, bundle)
    return dict(
        wheel_reference=reference(wheel),
        installed_entry_command=[
            "python",
            "-m",
            "carnot.inference.resident_protocol",
            str(config),
            digest,
            "python",
        ],
        installed_distribution_record=reference(raw / "installed-RECORD.csv"),
        dependency_versions=exports["versions"],
        import_trace_rows=traces,
        export_contract_rows=exports["rows"],
        installed_module_origin=exports["origin"],
        prototype_parity_rows=parity,
        build_receipts=receipts,
    )
