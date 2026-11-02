"""REQ-REPORT-8646: fresh wheel inference uses current source and original operands."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any

from carnot.reporting import installed_compatibility_8646 as e
from carnot.reporting import installed_lean_8633 as previous
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
ROOT, reference, load = e.ROOT, e.reference, e.load


def forced_cleanup(command: list[str], config: Path, raw: Path) -> Json:
    """A stopped real worker must be killed and reaped within the cleanup deadline."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    progress("8646_before_forced_cleanup")
    with (raw / "forced-cleanup.stderr").open("wb") as log:
        process = subprocess.Popen(
            [*command, str(config), sha256_file(config), "python"],
            cwd=raw,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        try:
            assert process.stdout is not None
            ready = json.loads(e.previous.wire.read_frame(process.stdout.fileno(), 20) or b"null")
            assert ready["status"] == "ready"
            os.kill(process.pid, signal.SIGSTOP)
            process.kill()
            process.wait(timeout=10)
        finally:
            process.kill() if process.poll() is None else None
            process.wait(timeout=10)
            assert process.stdin is not None and process.stdout is not None
            process.stdin.close()
            process.stdout.close()
    progress("8646_after_forced_cleanup", 1, 0)
    return dict(
        unit_id="intentional_forced_cleanup",
        passed=process.returncode == -signal.SIGKILL,
        exit_code=process.returncode,
        forced_cleanup=True,
        child_reaped=process.poll() is not None,
        intentional_failure_control=True,
        log=reference(raw / "forced-cleanup.stderr"),
    )


def worker_command(python: str, raw: Path, entry: str) -> list[str]:
    """Authenticate loaded protocol bytes in every worker before its first request."""
    package = entry == "installed"
    setup = (
        "from carnot.inference import resident_protocol as p;"
        "assert Path(p.__file__).resolve().is_relative_to(Path(sys.prefix).resolve());"
        if package
        else f"sys.path.insert(0,{str(ROOT / 'python/carnot/inference')!r});import resident_protocol as p;"
    )
    code = (
        "import sys,os,json,hashlib,runpy;from pathlib import Path;"
        + setup
        + "rows=[dict(path=m.__file__,sha256='sha256:'+hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()) for m in [p,p.numeric]];"
        + f"assert rows[0]['sha256']=={reference(ROOT / e.PROTOCOL)['sha256']!r};"
        + f"Path({str(raw)!r},'worker-import-'+str(os.getpid())+'.json').write_text(json.dumps(rows));"
        + (
            "raise SystemExit(p.main())"
            if package
            else f"runpy.run_path({str(ROOT / 'scripts/experiments/resident_worker_8608.py')!r},run_name='__main__')"
        )
    )
    return [python, "-u", "-c", code]


def configurations() -> tuple[Json, Json]:
    """Read original controls under the sealed prior authority without legacy imports."""
    history = load(dict(path=str(ROOT / e.HISTORY), sha256=e.HISTORY_PIN))
    authority = load(history["authority_reference"])
    result = tuple(deepcopy(load(s["config_reference"])) for s in authority["sessions"][:2])
    return result[0], result[1]


def build_wheel(raw: Path, bundle: Json) -> Path:
    """Copy current sources so packaging cannot mutate the working checkout."""
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8646-current-", dir="/var/tmp"))
    source = scratch / "source"
    source.mkdir()
    shutil.copytree(
        ROOT / "python",
        source / "python",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    for name in ["pyproject.toml", "README.md", "LICENSE", "NOTICE"]:
        shutil.copyfile(ROOT / name, source / name)
    receipt = child(
        "current_wheel_build",
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
    if not receipt["passed"]:
        raise ValueError("current_wheel_build")
    (wheel,) = (raw / "wheels").glob("*.whl")
    bundle["wheel_reference"] = reference(wheel)
    atomic_json(
        raw / "current-source-manifest.json",
        dict(files=[reference(p) for p in sorted((source / "python").rglob("*.py"))]),
    )
    progress("8646_current_wheel_built")
    return scratch


def installed(raw: Path, bundle: Json) -> Json:
    """Install the producer wheel offline; keep installed paths outside the checkout."""
    from carnot.reporting.installed_worker_8634 import checks

    scratch = build_wheel(raw, bundle)
    environment = scratch / "environment"
    receipts = [
        child(
            "private_venv",
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
            raw,
            deadline=90,
        )
    ]
    (site,) = environment.glob("lib/python*/site-packages")
    (site / "local-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    bootstrap = site / "zz-coverage.pth"
    bootstrap.write_text("import coverage; coverage.process_startup()\n")
    python = str(environment / "bin/python")
    wheel = raw / Path(bundle["wheel_reference"]["path"]).name
    shutil.copyfile(bundle["wheel_reference"]["path"], wheel)
    receipts.append(
        child(
            "install_exact_wheel",
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
    assert all(r["passed"] for r in receipts)
    extension = site / "carnot_native" / Path(bundle["extension"]["path"]).name
    extension.parent.mkdir()
    shutil.copyfile(bundle["extension"]["path"], extension)
    shutil.copyfile(extension, raw / "native_extension.so")
    assert sha256_file(extension) == bundle["extension"]["sha256"]
    installed_extension = reference(extension)
    exports, public, identities = [], [], []
    for mode in ["exports", "absent", "errors", "public"]:
        argv = [
            "env",
            "-u",
            "PYTHONPATH",
            python,
            "-u",
            "-c",
            "import os,runpy,sys;os.chdir(sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')",
            str(scratch),
            str(ROOT / "python/carnot/reporting/installed_probe_8646.py"),
            mode,
        ]
        receipt = child("installed_" + mode, argv, raw, deadline=90)
        receipts.append(receipt)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
        result = json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
        identities = [{k: row[k] for k in ["path", "sha256"]} for row in result["modules"]]
        atomic_json(raw / ("probe-" + mode + ".json"), result)
        if mode == "public":
            public = result["rows"]
        else:
            exports.append(result)
    config, threshold = configurations()
    assert config["head"] == bundle["head"]
    assert all(
        {key: old[key] for key in row} == row
        for old, row in zip(config["rows"][:64], bundle["requests"], strict=True)
    )
    # Only arithmetic input fields enter either worker, including old controls.
    for control in [config, threshold]:
        control["rows"] = [
            {
                key: value
                for key, value in row.items()
                if key in {"unit_id", "q", "public", "kind", "scaled_features", "decay_multiplier"}
            }
            for row in control["rows"]
        ]
    config["extension"] = installed_extension
    threshold["extension"] = installed_extension
    parity, sessions, rejection = [], [], []
    for arm in ["python", "native"]:
        for kind, control in [("original_controls", config), ("threshold", threshold)]:
            path = raw / (arm + "-" + kind + "-head.json")
            atomic_json(path, control)
            digest = sha256_file(path)
            old = previous.session(
                worker_command(sys.executable, raw, "private"),
                path,
                digest,
                arm,
                control["rows"],
                scratch,
                raw / (arm + "-" + kind + "-private.stderr"),
            )
            new = previous.session(
                worker_command(python, raw, "installed"),
                path,
                digest,
                arm,
                control["rows"],
                scratch,
                raw / (arm + "-" + kind + "-installed.stderr"),
                old["frames"],
            )
            sessions.append(
                dict(
                    arm=arm, kind=kind, config_reference=reference(path), private=old, installed=new
                )
            )
            for row, expected, observed in zip(
                control["rows"], old["responses"], new["responses"], strict=True
            ):
                parity.append(
                    dict(
                        unit_id=arm + ":" + kind + ":" + row["unit_id"],
                        arm=arm,
                        kind=kind,
                        request=row,
                        status=observed["status"],
                        response=observed.get("response"),
                        baseline_response=expected.get("response"),
                        passed=observed["status"] == "ok"
                        and {k: v for k, v in observed.items() if k != "spans"}
                        == {k: v for k, v in expected.items() if k != "spans"}
                        and [s["operation"] for s in observed["spans"]]
                        == [s["operation"] for s in expected["spans"]],
                    )
                )
        for label, command in [
            ("installed", worker_command(python, raw, "installed")),
            (
                "private",
                worker_command(sys.executable, raw, "private"),
            ),
        ]:
            rejection.extend(
                dict(r, entry=label)
                for r in checks(command, config, raw / (label + "-" + arm), scratch, arm)
            )
    record = next(site.glob("carnot_ebm-*.dist-info/RECORD"))
    shutil.copyfile(record, raw / "installed-RECORD.csv")
    forced = forced_cleanup(
        worker_command(python, raw, "installed"), raw / "python-original_controls-head.json", raw
    )
    return dict(
        installed_parity_rows=parity,
        sessions=sessions,
        rejection_rows=rejection,
        public_surface_rows=public,
        export_compatibility_rows=exports,
        installed_module_identity=dict(
            coverage_bootstrap=reference(bootstrap),
            modules=identities,
            extension=installed_extension,
            configuration_references=[s["config_reference"] for s in sessions],
            environment=str(environment),
            outside_checkout=not environment.is_relative_to(ROOT),
            pythonpath_unset=True,
        ),
        wheel_reference=reference(wheel),
        wheel_identity=dict(reference(wheel), fresh_current_build=True, editable=False),
        installed_module_hashes=identities,
        worker_import_rows=[
            row
            for p in sorted(raw.glob("worker-import-*.json"))
            for row in json.loads(p.read_bytes())
        ],
        isolation_rows=[
            dict(
                unit_id="current_installed_root",
                passed=True,
                root=str(environment),
                pythonpath_unset=True,
            )
        ],
        cleanup_rows=[forced]
        + [
            dict(
                r,
                unit_id=r["entry"] + ":" + r["unit_id"],
                passed=r["child_reaped"] and not r["forced_cleanup"],
            )
            for r in rejection
        ],
        dependencies=load(reference(raw / "probe-public.json"))["dependencies"],
        execution_receipts=receipts,
    )
