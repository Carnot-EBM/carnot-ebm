"""REQ-REPORT-8634: authenticate and independently exercise the Exp8633 wheel.

Only the fixed small energy head performs numerical inference. No generator is
loaded. These observations qualify deployment contracts, not scientific benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any
import zipfile

from carnot.inference import resident_protocol as wire
from carnot.reporting import installed_lean_8633 as previous
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8634_v741_installed_compatibility"
TASK = "exp8634-installed-compatibility"
CLI = f"scripts/experiments/{NAME}.py"
SOURCE = "results/experiment_8633_v741_installed_lean_prototype.json"
PIN = "sha256:2ec3626df74e4f27e71f4e3c6e74e32c3249d502684ffb09e1aeec5fc641abb8"
MODEL_SPECS: list[Json] = []
OWNED = [
    f"python/carnot/reporting/installed_{part}_8634.py"
    for part in ["compatibility", "worker", "probe", "runner", "validation"]
] + [CLI]
TESTS = [
    "tests/python/test_installed_compatibility_8634.py",
    "tests/python/test_installed_validation_8634.py",
]
reference, load = previous.reference, previous.load


def inputs(root: Path) -> Json:
    """Require the original wheel, its publication and unchanged public entry bytes."""
    ref = dict(path=str(root / SOURCE), sha256=PIN)
    value = load(ref)
    sidecar = (
        root / "results/raw" / Path(SOURCE).stem / "validators" / (PIN.split(":")[1] + ".json")
    )
    if value.get("installed_prototype_ready_score") != 1 or not value["required_checks_passed"]:
        raise ValueError("prototype_not_ready")
    if read_bound_sidecar(root / SOURCE, sidecar)["report"]["passed"] is not True:
        raise ValueError("prototype_publication")
    bundle = previous.inputs(root)
    wheel = value["wheel_reference"]
    if sha256_file(Path(wheel["path"])) != wheel["sha256"]:
        raise ValueError("wheel_hash")
    with zipfile.ZipFile(wheel["path"]) as archive:
        for path in [
            "carnot/__init__.py",
            "carnot/inference/resident_protocol.py",
            "carnot/inference/resident_energy.py",
        ]:
            if archive.read(path) != (root / "python" / path).read_bytes():
                raise ValueError("wheel_entry_drift")
    history_path = root / "results/experiment_8609_v739_lean_resident_qualification.json"
    history = json.loads(history_path.read_bytes())
    return dict(
        bundle,
        wheel_reference=wheel,
        prototype_reference=ref,
        refs=bundle["refs"] + [ref, reference(sidecar), wheel, reference(history_path)],
        original_nodes=history["original_node_union"]["collected"],
        historical_8609={
            k: history[k]
            for k in [
                "verdict_class",
                "honest_verdict",
                "lean_measurement_ready_score",
                "gate_check_summary",
            ]
        },
    )


def installed(raw: Path, bundle: Json) -> Json:
    """Install the producer wheel offline; keep installed paths outside the checkout."""
    from carnot.reporting.lean_qualification_8609 import configurations, original
    from carnot.reporting.installed_worker_8634 import checks

    scratch = Path(tempfile.mkdtemp(prefix="carnot-8634-", dir="/var/tmp"))
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
            str(ROOT / "python/carnot/reporting/installed_probe_8634.py"),
            mode,
        ]
        receipt = child("installed_" + mode, argv, raw, deadline=90)
        receipts.append(receipt)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
        result = json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
        identities = result["modules"]
        if mode == "public":
            public = result["rows"]
        else:
            exports.append(result)
    config, threshold = configurations(original.admit(ROOT))
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
                [sys.executable, "-u", str(ROOT / "scripts/experiments/resident_worker_8608.py")],
                path,
                digest,
                arm,
                control["rows"],
                scratch,
                raw / (arm + "-" + kind + "-private.stderr"),
            )
            new = previous.session(
                [python, "-u", "-m", "carnot.inference.resident_protocol"],
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
                        and observed.get("response") == expected.get("response"),
                    )
                )
        for label, command in [
            ("installed", [python, "-u", "-m", "carnot.inference.resident_protocol"]),
            (
                "private",
                [sys.executable, "-u", str(ROOT / "scripts/experiments/resident_worker_8608.py")],
            ),
        ]:
            rejection.extend(
                dict(r, entry=label)
                for r in checks(command, config, raw / (label + "-" + arm), scratch, arm)
            )
    record = next(site.glob("carnot_ebm-*.dist-info/RECORD"))
    shutil.copyfile(record, raw / "installed-RECORD.csv")
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
        execution_receipts=receipts,
    )


def reduce(work: Json) -> Json:
    """Recompute arithmetic and identities before accepting any compatibility claim."""
    from carnot.inference.resident_energy import evaluate, load_module
    from carnot.reporting.installed_worker_8634 import KINDS

    rows = work["installed_parity_rows"]
    identity = work["installed_module_identity"]
    environment = Path(identity["environment"])
    if (
        not identity["outside_checkout"]
        or not identity["pythonpath_unset"]
        or environment.is_relative_to(ROOT)
    ):
        raise ValueError("checkout_shadowing")
    for ref in identity["modules"] + [identity["extension"]]:
        if not Path(ref["path"]).is_relative_to(environment) or reference(Path(ref["path"])) != ref:
            raise ValueError("installed_identity")
    if {r["surface"] for r in work["public_surface_rows"]} != {"python", "cli", "mcp", "http"} or {
        r["mode"] for r in work["export_compatibility_rows"]
    } != {"exports", "absent", "errors"}:
        raise ValueError("public_roster")
    required_controls = {
        (entry, arm, kind)
        for entry in ["installed", "private"]
        for arm in ["python", "native"]
        for kind in KINDS
    }
    if {
        (r["entry"], r["arm"], r["kind"]) for r in work["rejection_rows"]
    } != required_controls or len(work["rejection_rows"]) != len(required_controls):
        raise ValueError("rejection_roster")
    if not all(
        r["passed"]
        for r in rows
        + work["public_surface_rows"]
        + work["export_compatibility_rows"]
        + work["rejection_rows"]
    ):
        raise ValueError("failed_compatibility")
    native = load_module(Path(identity["extension"]["path"]), "_nine_input_8533")
    offset = 0
    for session in work["sessions"]:
        config = load(session["config_reference"])
        head = deepcopy(work["bundle"]["head"])
        if session["kind"] == "threshold":
            head.update(parameters=[0.0] * 109, temperature=1.0, decay_scale=1.0)
        if config["head"] != head or config["extension"] != identity["extension"]:
            raise ValueError("configuration_operands")
        for entry in ["private", "installed"]:
            receipt = session[entry]
            if (
                receipt["exit_code"] != 0
                or receipt["shutdown"]["status"] != "shutdown"
                or receipt["ready"]["config_hash"] != session["config_reference"]["sha256"]
                or receipt["ready"]["loaded_extension"]
                != (identity["extension"] if session["arm"] == "native" else None)
            ):
                raise ValueError("worker_identity_or_cleanup")
        for request in config["rows"]:
            row = rows[offset]
            offset += 1
            expected, _ = evaluate(head, request, native if session["arm"] == "native" else None)
            numerical_reference, _ = evaluate(head, request, None)
            if (
                row["request"] != request
                or row["response"] != row["baseline_response"]
                or row["status"] != "ok"
                or row["arm"] != session["arm"]
                or set(row["response"]) != set(expected)
            ):
                raise ValueError("typed_response_parity")
            for key, value in expected.items():
                actual = row["response"][key]
                if type(actual) is not type(value) or (
                    abs(actual - numerical_reference[key])
                    > 1e-12 * (1 + abs(numerical_reference[key]))
                    if isinstance(value, float)
                    else actual != value
                ):
                    raise ValueError("numerical_parity")
    if offset != len(rows) or len({r["unit_id"] for r in rows}) != len(rows) or len(rows) < 128:
        raise ValueError("parity_roster")
    return dict(
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
    )


def measure(raw: Path, run_date: str, root: Path) -> Json:
    """Check branch inputs before bounded CPU work, preserving missing-input gates."""
    progress("8634_preconditions_no_model_load_MODEL_SPECS=[]_current_language_model_calls=0")
    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic_ns()
    work: Json = dict(
        raw=str(raw),
        run_date=run_date,
        bundle={},
        source_artifact_hashes=[],
        gate_check_summary=[],
        installed_parity_rows=[],
        public_surface_rows=[],
        export_compatibility_rows=[],
        rejection_rows=[],
        installed_module_identity={},
        sessions=[],
        phase_spans=[],
    )
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8634-preconditions-", dir="/var/tmp"))
    probe = scratch / "writable"
    probe.write_bytes(b"disk-backed scratch")
    mount = subprocess.check_output(
        ["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"], text=True, timeout=5
    ).strip()
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0])
    work["preconditions_checked"] = dict(
        memory_available_kib=memory,
        disk_free_bytes=shutil.disk_usage(scratch).free,
        scratch=str(scratch),
        writable=probe.read_bytes() == b"disk-backed scratch",
        filesystem=mount,
        tools={
            name: (ROOT / ".venv/bin" / name).is_file()
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        optional_gpu_required=False,
        optional_board_required=False,
    )
    try:
        if (
            mount in {"tmpfs", "ramfs"}
            or memory < 262144
            or shutil.disk_usage(scratch).free < 268435456
            or not all(work["preconditions_checked"]["tools"].values())
        ):
            raise ValueError("memory_disk_tools")
        work["bundle"] = inputs(root)
        work["source_artifact_hashes"] = work["bundle"]["refs"]
        work["historical_8609"] = work["bundle"]["historical_8609"]
        progress("8634_inputs_authenticated")
        work.update(installed(raw, work["bundle"]))
        reduce(work)
        if (time.monotonic_ns() - start) / 1e9 > 3600:
            raise ValueError("measurement_deadline")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AssertionError,
        subprocess.SubprocessError,
    ) as error:
        work["owned_execution_failed"] = bool(work["bundle"])
        path = root / SOURCE
        work["gate_check_summary"].append(
            dict(
                check="installed_compatibility_inputs_or_execution",
                upstream="exp8633-installed-lean-prototype",
                path=str(path),
                sha256=reference(path)["sha256"] if path.exists() else None,
                field="installed_prototype_ready_score_and_contracts",
                operator="==",
                expected=1,
                observed="missing" if not path.exists() else str(error),
            )
        )
    end = time.monotonic_ns()
    work["duration_s"] = (end - start) / 1e9
    work["phase_spans"] = [
        dict(phase="preconditions_and_installed_execution", start_ns=start, end_ns=end)
    ]
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    progress("8634_measurement_complete", len(work["installed_parity_rows"]), 0)
    return work
