"""SCENARIO-REPORT-8702-INSTALLED: execute exact wheel bytes in private processes."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import sysconfig
import time
from typing import Any

from carnot.inference import resident_protocol as wire
from carnot.reporting import installed_batch_8702 as e
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]


def prepare(raw: Path, scratch: Path, inputs: Json) -> tuple[list[str], Path, list[Json], Json]:
    """Build current sources offline and install without importing the checkout package."""
    source = scratch / "source"
    source.mkdir()
    shutil.copytree(
        e.ROOT / "python",
        source / "python",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    for name in ["pyproject.toml", "README.md", "LICENSE", "NOTICE"]:
        shutil.copyfile(e.ROOT / name, source / name)
    receipts = [
        child(
            "build_wheel",
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
                str(raw / "wheels"),
            ],
            raw,
            deadline=180,
        )
    ]
    if not receipts[-1]["passed"]:
        raise ValueError("wheel_build")
    (wheel,) = (raw / "wheels").glob("*.whl")
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
    (site,) = environment.glob("lib/python*/site-packages")
    (site / "local-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    receipts.append(
        child(
            "install_wheel",
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
    if not all(r["passed"] for r in receipts):
        raise ValueError("wheel_install")
    probe = (
        "import os,runpy,json,importlib.metadata;from pathlib import Path;"
        f"os.chdir({str(scratch)!r});"
        "from carnot.inference import resident_protocol as p;"
        f"assert str(p.__file__).startswith({str(site)!r});"
        f"s=runpy.run_path({str(e.ROOT / 'tests/python/test_installed_lean_8633.py')!r});"
        f"[s[n](Path({str(scratch)!r})) for n in ['test_lean_import_and_no_path_mutation','test_original_export_contract','test_optional_jax_absence','test_module_cli_help_and_rejection']];"
        "s['test_legacy_contract_snapshot']();"
        "d=importlib.metadata.distribution('carnot-ebm');"
        "print(json.dumps(dict(origin=p.__file__,record=d.read_text('RECORD'),versions={n:importlib.metadata.version(n) for n in ['carnot-ebm','numpy','scipy','pip','setuptools']})))"
    )
    receipts.append(
        child(
            "installed_public_contract",
            ["env", "-u", "PYTHONPATH", python, "-u", "-c", probe],
            raw,
            deadline=90,
        )
    )
    if not receipts[-1]["passed"]:
        raise ValueError("installed_public_contract")
    identity = json.loads(Path(receipts[-1]["stdout_path"]).read_text().splitlines()[-1])
    e.atomic_json(raw / "installed_identity.json", identity)
    extension = site / "carnot_native" / Path(inputs["extension"]["path"]).name
    extension.parent.mkdir()
    shutil.copyfile(inputs["extension"]["path"], extension)
    retained = raw / extension.name
    shutil.copyfile(extension, retained)
    config = raw / "head.json"
    e.atomic_json(
        config,
        dict(
            head=inputs["head"],
            extension=e.reference(extension),
            max_batch_size=16,
            item_budget=1024,
        ),
    )
    command = [python, "-u", "-m", "carnot.inference.resident_protocol"]
    return (
        command,
        config,
        receipts,
        dict(
            installed_wheel_reference=e.reference(wheel),
            loaded_extension_reference=e.reference(retained),
            installed_extension=e.reference(extension),
            installed_identity_reference=e.reference(raw / "installed_identity.json"),
        ),
    )


def session(
    command: list[str],
    config: Path,
    arm: str,
    requests: list[Json],
    scratch: Path,
    raw: Path,
    *,
    malformed: bytes | None = None,
    deadline_ns: int = 2**63,
) -> Json:
    """A bounded real worker retains all envelopes, responses, exits and resets."""
    if time.monotonic_ns() + 90_000_000_000 >= deadline_ns:
        raise TimeoutError("measurement_admission_deadline")
    raw.mkdir(parents=True)
    digest = e.reference(config)["sha256"]
    argv = command + [str(config), digest, arm, "--idle-timeout", "2"]
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        LLVM_PROFILE_FILE=str(raw / "native-%p.profraw"),
    )
    env.pop("PYTHONPATH", None)
    start = time.monotonic_ns()
    progress("8702_before_worker_" + raw.name)
    with (
        (raw / "stderr").open("wb") as err,
        subprocess.Popen(
            argv,
            cwd=scratch,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            start_new_session=True,
        ) as proc,
    ):
        try:
            assert proc.stdin is not None and proc.stdout is not None
            out, inp = proc.stdout.fileno(), proc.stdin.fileno()
            ready = json.loads(wire.read_frame(out, 30) or b"null")
            observations = []
            if malformed is not None:
                proc.communicate(malformed, timeout=10)
            else:
                for index, request in enumerate(requests):
                    if time.monotonic_ns() + 60_000_000_000 >= deadline_ns:
                        raise TimeoutError("measurement_stop_deadline")
                    frame = deepcopy(request)
                    frame.setdefault("request_id", str(index))
                    frame.setdefault("config_hash", digest)
                    frame.setdefault("deadline_ns", time.monotonic_ns() + 15_000_000_000)
                    wire.write_frame(inp, frame)
                    response = json.loads(wire.read_frame(out, 30) or b"null")
                    if (
                        response.get("request_id") != frame["request_id"]
                        or response.get("config_hash") != digest
                    ):
                        raise ValueError("worker_identity")
                    observations.append(dict(request=frame, response=response))
                    if index % 16 == 0:
                        progress("8702_frames", index + 1, len(requests) - index - 1)
                wire.write_frame(
                    inp, dict(wire.envelope("shutdown", digest, {}), operation="shutdown")
                )
                shutdown = json.loads(wire.read_frame(out, 10) or b"null")
                proc.stdin.close()
                proc.wait(timeout=10)
                observations.append(dict(shutdown=shutdown))
            end = time.monotonic_ns()
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as failure:
            killed = proc.poll() is None
            if killed:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            e.atomic_json(
                raw / "failed_child.json",
                dict(
                    argv=argv,
                    actual_exit=proc.returncode,
                    killed=killed,
                    error=str(failure),
                    start_ns=start,
                    end_ns=time.monotonic_ns(),
                ),
            )
            raise
    progress("8702_after_worker_" + raw.name, len(requests), 0)
    result = dict(
        argv=argv,
        cwd=str(scratch),
        pythonpath_unset=True,
        ready=ready,
        observations=observations,
        actual_exit=proc.returncode,
        duration_s=(end - start) / 1e9,
        start_ns=start,
        end_ns=end,
        stderr_reference=e.reference(raw / "stderr"),
    )
    e.atomic_json(raw / "session.json", result)
    return result


def measure(raw: Path, scratch: Path, inputs: Json, *, deadline_ns: int = 2**63) -> Json:
    """Parity includes every batch size and every original request, without timing claims."""
    if time.monotonic_ns() + 450_000_000_000 >= deadline_ns:
        raise TimeoutError("measurement_prepare_deadline")
    raw.mkdir(parents=True)
    command, config, receipts, identities = prepare(raw, scratch, inputs)
    run_session = partial(session, deadline_ns=deadline_ns)
    traces, rows, rejects = [], [], []
    for arm in ["python", "native"]:
        single = run_session(
            command,
            config,
            arm,
            [dict(operation="infer", row=row) for row in inputs["requests"]],
            scratch,
            raw / (arm + "-single"),
        )
        traces.append(e.reference(raw / (arm + "-single") / "session.json"))
        requests = [
            dict(operation="batch", rows=inputs["requests"][i : i + size])
            for size in range(1, 17)
            for i in range(0, 64, size)
        ]
        trace = run_session(command, config, arm, requests, scratch, raw / arm)
        traces.append(e.reference(raw / arm / "session.json"))
        if (
            trace["actual_exit"] != 0
            or trace["ready"]["status"] != "ready"
            or single["actual_exit"] != 0
            or trace["observations"][-1]["shutdown"]["status"] != "shutdown"
            or single["observations"][-1]["shutdown"]["status"] != "shutdown"
        ):
            raise ValueError("installed_child")
        baseline = {
            r["response"]["response"]["unit_id"]: r["response"]["response"]
            for r in single["observations"][:-1]
        }
        for index, item in enumerate(trace["observations"][:-1]):
            response, request = item["response"], item["request"]
            for original, actual, feature in zip(
                request["rows"], response["responses"], response["feature_hashes"], strict=True
            ):
                expected = baseline[original["unit_id"]]
                passed = (
                    actual["unit_id"] == original["unit_id"]
                    and actual["decision"] == expected["decision"]
                    and abs(actual["probability"] - expected["probability"]) <= 1e-12
                )
                rows.append(
                    dict(
                        unit_id=f"{arm}-{index}-{original['unit_id']}",
                        source_id=original["unit_id"],
                        arm=arm,
                        batch=len(request["rows"]),
                        response=actual,
                        feature_hash=feature,
                        passed=passed,
                        status="completed" if passed else "failed",
                        absolute_metric=int(passed),
                        effective_independent_groups=0,
                        censored=False,
                        excluded=False,
                    )
                )
        row = inputs["requests"][0]
        bad = [
            dict(operation="batch", rows=items)
            for items in [
                [],
                [row] * 17,
                [row, row],
                [dict(row, q=-1)],
                [dict(row, public={})],
                None,
            ]
        ]
        bad += [
            dict(operation="batch", rows=[row], config_hash="wrong"),
            dict(operation="batch", rows=[row], deadline_ns=1),
            dict(operation="batch", rows=[row], request_id="duplicate"),
            dict(operation="batch", rows=[row], request_id="duplicate"),
        ]
        rejection = run_session(command, config, arm, bad, scratch, raw / (arm + "-rejections"))
        traces.append(e.reference(raw / (arm + "-rejections") / "session.json"))
        for i, item in enumerate(rejection["observations"][:-1]):
            response = item["response"]
            passed = response["status"] == ("ok" if i == 8 else "error") and (
                i == 8 or not {"response", "responses"} & response.keys()
            )
            rejects.append(dict(unit_id=f"{arm}-rejection-{i}", passed=passed, response=response))
        for i, payload in enumerate([b"\0", struct.pack("!I", 65537), struct.pack("!I", 5) + b"x"]):
            truncated = run_session(
                command, config, arm, [], scratch, raw / f"{arm}-frame-{i}", malformed=payload
            )
            traces.append(e.reference(raw / f"{arm}-frame-{i}" / "session.json"))
            rejects.append(dict(unit_id=f"{arm}-frame-{i}", passed=truncated["actual_exit"] == 1))
    py, native = rows[: len(rows) // 2], rows[len(rows) // 2 :]
    parity = all(
        a["source_id"] == b["source_id"]
        and a["feature_hash"] == b["feature_hash"]
        and a["response"]["decision"] == b["response"]["decision"]
        and abs(a["response"]["probability"] - b["response"]["probability"]) <= 1e-12
        for a, b in zip(py, native, strict=True)
    )
    return dict(
        **identities,
        config_reference=e.reference(config),
        receipts=receipts,
        traces=traces,
        rows=rows,
        envelope_rejection_rows=rejects,
        passed=parity and all(r["passed"] for r in rows + rejects),
    )
