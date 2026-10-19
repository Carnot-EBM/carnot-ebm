"""REQ-REPORT-8661: freeze once, measure once, and publish only checked bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile
import time
from typing import Any

from carnot.reporting import installed_cost_8661 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
REQUIRED = {
    "owned_tests",
    "exact_node_union",
    "owned_coverage",
    "e2e018",
    "e2e021",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
}


def prepare(raw: Path, scratch: Path, bundle: Json) -> tuple[Json, Path, list[Json]]:
    """Install the exact qualified wheel offline; both arms load the same native bytes."""
    environment = scratch / "environment"
    receipts = [
        child(
            "venv",
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
            raw,
            deadline=90,
        )
    ]
    python = str(environment / "bin/python")
    (site,) = environment.glob("lib/python*/site-packages")
    (site / "local-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    wheel = bundle["wheel_reference"]
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
                wheel["path"],
            ],
            raw,
            deadline=90,
        )
    )
    if not all(r["passed"] for r in receipts):
        raise ValueError("installation_failed")
    extension = site / "carnot_native" / Path(bundle["extension"]["path"]).name
    extension.parent.mkdir()
    shutil.copyfile(bundle["extension"]["path"], extension)
    if c.reference(extension)["sha256"] != bundle["extension"]["sha256"]:
        raise ValueError("installed_extension_hash")
    config = raw / "head.json"
    atomic_json(config, dict(head=bundle["head"], extension=c.reference(extension)))
    commands = dict(
        private=[sys.executable, "-u", str(c.ROOT / "scripts/experiments/resident_worker_8608.py")],
        installed=[python, "-u", "-m", "carnot.inference.resident_protocol"],
    )
    modules = []
    for entry in ["private", "installed"]:
        code = (
            f"import sys,json,hashlib,importlib.metadata,os;from pathlib import Path;os.chdir({str(scratch)!r});"
            + (
                f"sys.path.insert(0,{str(c.ROOT / 'python/carnot/inference')!r});import resident_protocol as w;"
                if entry == "private"
                else "from carnot.inference import resident_protocol as w;"
            )
            + f"v,n,e=w.configuration(Path({str(config)!r}),{c.reference(config)['sha256']!r},'native');"
            + "refs=[dict(path=m.__file__,sha256='sha256:'+hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()) for k,m in list(sys.modules.items()) if k in ['resident_protocol','resident_energy','_resident_alignment','carnot.inference.resident_protocol','carnot.inference.resident_energy','numpy','scipy']];"
            + "print(json.dumps(dict(head_hash=w.numeric.canonical_hash(v['head']),modules=refs,extension=e,versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pip','setuptools']})))"
        )
        receipt = child(
            "identity_" + entry,
            ["env", "-u", "PYTHONPATH", commands[entry][0], "-u", "-c", code],
            raw,
            deadline=60,
        )
        if not receipt["passed"]:
            raise ValueError("loaded_identity_probe")
        observed = json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
        if observed["head_hash"] != canonical_hash(bundle["head"]) or observed[
            "extension"
        ] != c.reference(extension):
            raise ValueError("actual_loaded_head")
        for ref in observed["modules"] + [observed["extension"]]:
            destination = (
                raw / "loaded_modules" / entry / (ref["sha256"][7:] + "-" + Path(ref["path"]).name)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ref["path"], destination)
            if c.reference(destination)["sha256"] != ref["sha256"]:
                raise ValueError("loaded_module_snapshot")
            modules.append(
                dict(
                    c.reference(destination),
                    loaded_path=ref["path"],
                    entry=entry,
                    versions=observed["versions"],
                )
            )
        receipts.append(receipt)
    return dict(commands=commands, modules=modules), config, receipts


def measure(raw: Path, run_date: str, root: Path) -> Json:
    """Optional hardware never enters the independent CPU admission checks."""
    from carnot.reporting.installed_cost_session_8661 import session

    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic_ns()
    work = c.empty_work(raw, run_date)
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8661-", dir="/var/tmp"))
    (scratch / "writable").write_bytes(b"disk-backed probe")
    mount_receipt = child(
        "scratch_filesystem",
        ["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"],
        raw,
        deadline=10,
    )
    mount = Path(mount_receipt["stdout_path"]).read_text().strip()
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0])
    work["preconditions_checked"] = dict(
        scratch=str(scratch),
        filesystem=mount,
        writable=(scratch / "writable").read_bytes() == b"disk-backed probe",
        memory_available_kib=memory,
        disk_free_bytes=shutil.disk_usage(scratch).free,
        authenticated_tools=[
            c.reference(c.ROOT / ".venv/bin" / n)
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        ],
        optional_hardware_required=False,
        cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_contention="Retain host load before/after each cell; other work is not stopped",
    )
    try:
        if (
            not mount_receipt["passed"]
            or mount in {"tmpfs", "ramfs"}
            or memory < 262144
            or shutil.disk_usage(scratch).free < 268435456
        ):
            raise ValueError("cpu_scratch_memory")
        work["bundle"], work["gate_check_summary"], work["source_artifact_hashes"] = c.admit(root)
        if not work["gate_check_summary"]:
            for ref in work["source_artifact_hashes"][:]:
                destination = raw / "operands" / (ref["sha256"][7:] + Path(ref["path"]).suffix)
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(ref["path"], destination)
                work["source_artifact_hashes"].append(c.reference(destination))
            prepared, config, receipts = prepare(raw, scratch, work["bundle"])
            work["validation_receipts"].extend(receipts)
            work["loaded_module_hashes"] = prepared["modules"]
            work["source_artifact_hashes"].extend(prepared["modules"] + [c.reference(config)])
            work["phase_spans"].append(
                dict(
                    phase="preconditions_freeze_install", start_ns=start, end_ns=time.monotonic_ns()
                )
            )
            began = time.monotonic_ns()
            progress("8661_measurement_begin", 0, 120)
            for index, row in enumerate(work["rows"]):
                slot = {k: row[k] for k in c.roster()[0]}
                try:
                    primitive = session(
                        prepared["commands"][row["arm"].split("_")[0]],
                        config,
                        slot,
                        work["bundle"]["requests"],
                        scratch,
                        raw / "sessions" / row["unit_id"],
                        deadline_ns=start + 3600_000_000_000,
                    )
                    c.summarize(primitive, work["bundle"])
                    path = raw / "sessions" / row["unit_id"] / "primitive.json"
                    atomic_json(path, primitive)
                    row.update(
                        status="completed",
                        missing_reason=None,
                        primitive_reference=c.reference(path),
                    )
                except TimeoutError as error:
                    for pending in work["rows"][index:]:
                        pending["missing_reason"] = str(error)
                    break
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row.update(status="failed", missing_reason=str(error))
                    work["owned_error"] = str(error)
                    raise
                progress("8661_cells", index + 1, 119 - index)
            work["phase_spans"].append(
                dict(phase="measurement", start_ns=began, end_ns=time.monotonic_ns())
            )
    except (FileNotFoundError, KeyError) as error:
        work["gate_check_summary"].append(
            dict(
                upstream="frozen_operands",
                check="operand_available",
                path=getattr(error, "filename", None),
                sha256=None,
                field=str(error),
                operator="exists",
                expected=True,
                observed=None,
                observed_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field",
            )
        )
    except (OSError, ValueError, TypeError) as error:
        work["owned_error"] = str(error)
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    return work


def terminal(path: Path, authority: Json, logs: Path) -> Json:
    """Fresh bounded readers inspect exact candidate bytes and retained negative controls."""
    logs.mkdir(parents=True, exist_ok=True)
    changed = json.loads(path.read_bytes())
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = canonical_hash(changed)
    bad = logs / "rehashed_tamper.json"
    atomic_json(bad, changed)
    args = ["--authority", authority["path"], authority["sha256"]]
    receipts = []
    for name, argv, expected in [
        ("cold", [c.CLI, "--cold-replay", str(path), *args], 0),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)], 0),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("missing", [c.CLI, "--cold-replay", str(logs / "missing.json"), *args], 1),
        ("rehashed_tamper", [c.CLI, "--cold-replay", str(bad), *args], 1),
    ]:
        receipts.append(
            child(name, [sys.executable, "-u", *argv], logs, expected=expected, deadline=120)
        )
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_text())
    return dict(
        passed=all(r["passed"] for r in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, run_date: str, root: Path) -> int:
    """Owned checks finish before the independently pinned artifact is made reader-visible."""
    from carnot.reporting.installed_cost_validation_8661 import validate

    raw = output.parent / "raw" / c.NAME / "invocations" / str(time.time_ns())
    start = time.monotonic_ns()
    frozen_code = [c.reference(c.ROOT / p) for p in c.OWNED + c.TESTS]
    work = measure(raw, run_date, root)
    began = time.monotonic_ns()
    work["validation_receipts"].extend(validate(work, raw))
    end = time.monotonic_ns()
    work["phase_spans"].append(dict(phase="validation", start_ns=began, end_ns=end))
    work["duration_s"] = (end - start) / 1e9
    work["validation_receipts"].append(dict(name="runtime_cap", passed=work["duration_s"] < 4800))
    work["validation_receipts"].append(
        dict(
            name="complete_measurement",
            passed=bool(work["gate_check_summary"])
            or all(row["status"] == "completed" for row in work["rows"]),
        )
    )
    work["code_config_hashes"] = frozen_code
    work["validation_receipts"].append(
        dict(
            name="frozen_code_unchanged",
            passed=all(c.reference(Path(ref["path"])) == ref for ref in frozen_code),
        )
    )
    work["raw_shard_hashes"] = [c.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    path = raw / "authority.json"
    atomic_json(path, work)
    authority = c.reference(path)
    value = c.artifact(authority)
    result = publish_primary(output, value, lambda p: terminal(p, authority, raw / "terminal"))
    atomic_json(
        raw / "terminal_validation.json", read_bound_sidecar(output, Path(result["sidecar_path"]))
    )
    progress("8661_" + value["honest_verdict"], value["completed_count"], value["censored_count"])
    return int(value["verdict_class"] == "disqualified")


def main(argv: list[str] | None = None) -> int:
    """Private boundaries permit actual CLI replay without publishing test fixtures."""
    progress("8661_start_no_model_load_MODEL_SPECS=[]_current_language_model_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261019")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            if not args.authority:
                raise ValueError("immutable_authority_required")
            c.replay(args.cold_replay, dict(path=args.authority[0], sha256=args.authority[1]))
            return 0
        if (
            datetime.strptime(args.date, "%Y%m%d").strftime("%Y%m%d") != args.date
            or (args.input_root and not args.private_output)
            or (args.private_output and args.private_output.resolve().is_relative_to(c.ROOT))
        ):
            raise ValueError("date_or_private_boundary")
        return run(
            args.private_output or c.ROOT / "results" / (c.NAME + ".json"),
            args.date,
            args.input_root or c.ROOT,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[8661] error={error}", file=sys.stderr, flush=True)
        return 1
