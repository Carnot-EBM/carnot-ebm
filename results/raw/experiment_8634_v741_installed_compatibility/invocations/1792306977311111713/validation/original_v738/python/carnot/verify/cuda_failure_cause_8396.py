"""REQ-VERIFY-8396: device observations explain only failures the trace supports."""

from __future__ import annotations

import ctypes as c
import json
import os
from pathlib import Path
import re
import shutil
import time
from tempfile import TemporaryDirectory
from typing import Any

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.v710_contract_replay import snapshot
from carnot.reporting import v723_contract_methods as authority
from carnot.reporting.v709_execution import child
from carnot.verify import cuda_primitive_8290 as primitive
from carnot.verify import runtime_evidence_delta_8382 as prior
from carnot.verify import typed_runtime_8368 as typed

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8396_v723_cuda_failure_cause",
    "exp8396-cuda-failure-cause",
    "2026.10.723",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_cuda_failure_cause_8396.py"
OWNED = [
    "python/carnot/verify/cuda_failure_cause_8396.py",
    "python/carnot/verify/cuda_failure_runner_8396.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:2d9410490984435b6f75b67cf9239f1e8303dd1bd35f4dc50c0a2476dd9155b0"
old, checksum = prior.old, prior.checksum
TRACE_CAP = 10 * 1024 * 1024
PINS = dict(prior.PROTOCOL_PINS, **{prior.UPSTREAM: prior.PIN})
PINS.update(
    {
        "results/experiment_8290_v716_runtime_localization.json": "sha256:62875d880623865338f7d6b85e1caa82ed388cb0b94b4402f0457245361a0b1f",
        "results/experiment_8382_v722_runtime_evidence_delta.json": "sha256:6318011848533abc0b4bd14372f35fb5b6e3137ca5a14233298cfbbc5a9d0080",
        authority.old.PROTOCOL: authority.old.PROTOCOL_PIN,
    }
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so supervised waits cannot look like silent computation."""
    print(f"[exp8396] phase={phase} completed={completed} pending={pending}", flush=True)


def initialize(library: str) -> Json:
    """Load only the driver; its initialization cannot certify a usable context."""
    result: Json = dict(
        api_returns={},
        libraries=[],
        error=None,
        environment={k: os.environ[k] for k in primitive.VARIABLES if k in os.environ},
    )
    try:
        lib = c.CDLL(library)
        lib.cuInit.restype = c.c_int
        result["api_returns"]["cuInit"] = int(lib.cuInit(c.c_uint(0)))
    except (OSError, AttributeError) as error:
        result["error"] = str(error)
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and any(s in line for s in ("libcuda", "libnvidia"))
    }
    result["libraries"] = [reference(Path(p).resolve()) for p in sorted(paths) if Path(p).is_file()]
    result["kernel_driver"] = (
        Path("/proc/driver/nvidia/version").read_text()
        if Path("/proc/driver/nvidia/version").is_file()
        else None
    )
    result["device_nodes"] = [
        dict(
            path=str(p),
            mode=oct(p.stat().st_mode),
            uid=p.stat().st_uid,
            gid=p.stat().st_gid,
            read_write=os.access(p, os.R_OK | os.W_OK),
        )
        for p in sorted(Path("/dev").glob("nvidia*"))
    ]
    result.update(pid=os.getpid(), uid=os.getuid(), gid=os.getgid(), groups=os.getgroups())
    return result


def measure(
    root: Path,
    raw: Path,
    receipt: Path | None = None,
    receipt_hash: str | None = None,
    *,
    fixture: bool = False,
) -> Json:
    """Seal actual inputs before diagnosis; prior absence does not manufacture a repair."""
    progress("authenticate_inputs_before")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    from carnot.verify.runtime_evidence_execution_8382 import resources

    resource_state = resources(raw)
    current = authority.authority(root, raw / "authority")
    tasks = [t for t in current["tasks"] if t["id"] == TASK]
    gates = list(current["gate_check_summary"])
    if (
        not resource_state["private_disk_backed_scratch"]
        or resource_state["available_memory_bytes"] < resource_state["minimum_memory_bytes"]
    ):
        gates.append(old.gate(raw, "private_disk_memory_budget", True, resource_state))
    if len(tasks) != 1 or canonical_hash(tasks[0]) != TASK_PIN:
        gates.append(
            old.gate(
                root / authority.ACTIVE,
                "exact_task_hash",
                TASK_PIN,
                canonical_hash(tasks[0]) if tasks else None,
            )
        )
    work: Json = dict(
        refs=[],
        failures=gates,
        authority=current,
        historical={},
        baseline={},
        fixture=fixture,
        root=str(root),
        resources=resource_state,
        diagnostic=dict(
            rows=[],
            checks=[],
            cleanup_receipts=[],
            intervention_rows=[],
            trace_manifest={},
            **analyze("", {}),
        ),
    )
    for i, (name, pin) in enumerate(PINS.items()):
        ref = snapshot(root / name, raw / "inputs", str(i))
        work["refs"].append(ref)
        if ref["sha256"] != pin:
            gates.append(old.gate(root / name, "immutable_input_hash", pin, ref["sha256"]))
    try:
        source = next(r for r in work["refs"] if r["path"] == str(root / prior.UPSTREAM))
        if source["sha256"] != prior.PIN:
            raise ValueError("typed_runtime_primary_pin")
        work["refs"].extend(prior.historical_aliases(root, raw / "custody", source))
        work["historical"] = json.loads(typed.authenticate(source).read_bytes())
        bound = dict(path=work["historical"]["runtime_binding_path"], sha256=prior.BASELINE_PIN)
        work["baseline_reference"] = snapshot(typed.authenticate(bound), raw / "inputs", "baseline")
        work["baseline"] = json.loads(typed.authenticate(work["baseline_reference"]).read_bytes())
        failure = json.loads(
            typed.authenticate(
                next(
                    r
                    for r in work["refs"]
                    if r["path"]
                    == str(root / "results/experiment_8290_v716_runtime_localization.json")
                )
            ).read_bytes()
        )
        if failure["root_cause_status"] != "unproved" or {
            r["binding"] for r in failure["failure_layer"] if r["failed_api"].get("cuInit") == 101
        } != {"inherited", "explicit_uuid"}:
            raise ValueError("original_cuInit101_controls")
    except (OSError, ValueError, KeyError, StopIteration) as error:
        gates.append(old.gate(root / prior.UPSTREAM, "typed_runtime_custody", True, str(error)))
    progress("authenticate_inputs_after", len(work["refs"]), 0)
    if not gates and not fixture:
        progress("bounded_diagnosis_before", 0, 1)
        work["diagnostic"] = diagnose(work["baseline"], raw / "diagnosis")
        progress("bounded_diagnosis_after", 1, 0)
    atomic_json(raw / "measurement.json", work)
    return work


def reduction(work: Json, passed: bool) -> Json:
    """Custody, intervention and execution are separate gates for the future canary."""
    diagnostic = work["diagnostic"]
    reader = bool(
        passed
        and not work["fixture"]
        and not work["failures"]
        and work["authority"].get("activated")
        and work["historical"].get("runtime_reader_ready_score") == 1
    )
    changed = int(bool(reader and diagnostic["runtime_changed_score"]))
    context = int(
        bool(changed and diagnostic["cuda_context_ready_score"] and not diagnostic["checks"])
    )
    gates = list(work["failures"]) + diagnostic["checks"]
    for field, observed in [
        ("runtime_reader_ready_score", int(reader)),
        ("runtime_changed_score", changed),
        ("cuda_context_ready_score", context),
    ]:
        if observed != 1:
            gates.append(old.gate(Path(work["root"]) / CLI, field, 1, observed))
    for gate in gates:
        gate.update(check=gate.get("field", gate.get("artifact_field")), upstream_id=TASK)
    rows = [
        dict(
            unit_id=name,
            arm=name,
            status="completed" if operands else "censored",
            completed=bool(operands),
            failed=False,
            excluded=False,
            censored=not bool(operands),
            absolute_metric=len(operands),
            missing_reason=None if operands else "trace_or_qualified_intervention_absent",
        )
        for name, operands in [
            ("inherited_init", diagnostic["rows"]),
            ("child_local_correction", diagnostic["intervention_rows"][:1]),
            ("context_copy", diagnostic["intervention_rows"][1:]),
        ]
    ]
    verdict = "disqualified" if not passed else "null" if context else "blocked"
    return dict(
        honest_verdict="complete_" + verdict + "_cuda_failure_cause",
        verdict_class=verdict,
        gate_check_summary=gates,
        rows=rows,
        intended_count=3,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        runtime_reader_ready_score=int(reader),
        runtime_changed_score=changed,
        cuda_context_ready_score=context,
        root_cause_status=diagnostic["root_cause_status"],
        failure_layer=diagnostic["failure_layer"],
        acceptance_gates=dict(
            typed_custody_and_current_authority=reader,
            authenticated_causal_change=bool(changed),
            context_copy_cleanup=bool(context),
        ),
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Bind every terminal field to primitives and keep current model calls at zero."""
    passed = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    value = reduction(work, passed)
    shards = [reference(raw / "measurement.json")]
    trace = work["diagnostic"]["trace_manifest"].get("ref")
    if trace:
        shards.append(trace)
    value.update(
        experiment_id=8396,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261010",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical"].get("historical_model_provenance", {}),
        verifier_is_oracle=False,
        exposure_scope="exposed development CUDA substrate; no scientific samples",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        validation_receipts=receipts,
        sample_size_budget=dict(intended_arms=3, independent_scientific_samples=0),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        resource_preconditions=work.get("resources", {}),
        duration_s=work.get("duration_s", 0.001),
        phase_spans=work.get("phase_spans", []),
        random_seed=7238396,
        source_artifact_hashes=work["refs"],
        code_config_hashes=[
            snapshot(ROOT / p, raw / "code", str(i))
            for i, p in enumerate(
                OWNED
                + [
                    TEST,
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                ]
            )
        ],
        primitive_reference=shards[0],
        raw_shard_hashes=shards,
        trace_manifest=work["diagnostic"]["trace_manifest"],
        intervention_rows=work["diagnostic"]["intervention_rows"],
        cleanup_receipts=work["diagnostic"]["cleanup_receipts"],
        diagnostic=work["diagnostic"],
        execution_authority=work["authority"],
        owned_statement_coverage=work.get("owned_statement_coverage", {}),
        global_health=dict(
            status="not_rerun_in_task",
            prior_primary_path=str(
                ROOT / "results/experiment_8382_v722_runtime_evidence_delta.json"
            ),
            prior_primary_sha256=PINS["results/experiment_8382_v722_runtime_evidence_delta.json"],
            policy="Report inherited global health separately; conductor owns the repository suite.",
        ),
        adversarial_findings=work.get("finding_audits", []),
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=[
                    "original initialization failure, typed custody and frozen authority"
                ],
            )
            for r in work["refs"]
        ],
        operator_remedy="Investigate the recorded device-open/ioctl failure under the current kernel and device namespace; any system repair needs operator action. If no causal return is identified, obtain driver-side diagnostic evidence. This trace performs no driver repair.",
        methodology_note="One leased, owned driver initialization child with bounded strace. A distinct installed-library or process-binding error permits one reversible comparison and one 32-byte copy probe. Unknown cause blocks; no model is loaded.",
    )
    binding = raw / "runtime_binding.json"
    atomic_json(
        binding,
        dict(
            baseline=work["baseline"],
            observed=work["diagnostic"]["rows"],
            interventions=work["diagnostic"]["intervention_rows"],
        ),
    )
    value.update(runtime_binding_path=str(binding), runtime_binding_sha256=sha256_file(binding))
    value["field_principles"] = {
        k: "Bind actual invocation, sealed primitive operands and authority; absence cannot become readiness or scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Explain the evidence policy for each field.",
        reproducibility_checksum="Bind every field except this checksum.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute readiness from exact child logs so consistently rehashed claims fail."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value) or [
            value[k] for k in ("experiment_id", "task_id", "milestone", "run_date")
        ] != [8396, TASK, MILESTONE, "20261010"]:
            return False
        for ref in typed.references(value):
            typed.authenticate(ref)
        work = json.loads(typed.authenticate(value["primitive_reference"]).read_bytes())
        for ref in typed.references(work):
            typed.authenticate(ref)
        snapshots = work["authority"].get("authority_snapshots", {})
        if snapshots:
            with TemporaryDirectory(prefix="authority-replay-", dir=path.parent) as directory:
                frozen = Path(directory)
                for key, name in [
                    ("design", authority.DESIGN),
                    ("staged", authority.STAGED),
                    ("active", authority.ACTIVE),
                ]:
                    ref = snapshots.get(key, {})
                    if ref.get("exists"):
                        destination = frozen / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(
                            typed.authenticate(dict(ref, path=ref["source_path"])).read_bytes()
                        )
                rebuilt = authority.authority(frozen, frozen / "receipts")
                if any(
                    rebuilt.get(k) != work["authority"].get(k)
                    for k in ("activated", "tasks", "canonical_tasks_sha256")
                ):
                    return False
        if work["authority"].get("tasks"):
            task = next(t for t in work["authority"]["tasks"] if t["id"] == TASK)
            if canonical_hash(task) != TASK_PIN:
                return False
        for name, pin in PINS.items():
            ref = next(r for r in work["refs"] if r["path"] == str(Path(work["root"]) / name))
            if ref.get("exists") and ref["sha256"] != pin:
                return False
        if work["historical"]:
            source = next(r for r in work["refs"] if r["sha256"] == prior.PIN)
            if (
                json.loads(typed.authenticate(source).read_bytes()) != work["historical"]
                or json.loads(typed.authenticate(work["baseline_reference"]).read_bytes())
                != work["baseline"]
            ):
                return False
        diagnostic = work["diagnostic"]
        for row in diagnostic["rows"] + diagnostic["intervention_rows"]:
            receipt = row["receipt"]
            observed = (
                json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
                if receipt["passed"]
                else {}
            )
            if observed != row["primitive"]:
                return False
            for library in observed.get("libraries", []):
                typed.authenticate(library)
        if diagnostic["rows"]:
            trace = diagnostic["trace_manifest"]["ref"]
            recomputed = analyze(
                typed.authenticate(trace).read_text() if trace else "",
                diagnostic["rows"][0]["primitive"],
            )
            if any(
                diagnostic[k] != recomputed[k]
                for k in ("root_cause_status", "failure_layer", "causal_trace_line", "correction")
            ):
                return False
        changed = bool(
            diagnostic["intervention_rows"]
            and diagnostic["correction"]
            and diagnostic["intervention_rows"][0]["primitive"].get("api_returns", {}).get("cuInit")
            == 0
        )
        context = bool(
            changed
            and len(diagnostic["intervention_rows"]) == 2
            and all(
                diagnostic["intervention_rows"][1]["primitive"].get(k) is True
                for k in ("context_copy_ready", "byte_copy_parity", "cleanup_passed")
            )
            and diagnostic["intervention_rows"][1]["primitive"].get("allocation_bytes") == 32
        )
        if [diagnostic["runtime_changed_score"], diagnostic["cuda_context_ready_score"]] != [
            int(changed),
            int(context),
        ]:
            return False
        passed = bool(value["validation_receipts"]) and all(
            r["passed"] for r in value["validation_receipts"] if r.get("scope") != "global"
        )
        return all(value[k] == v for k, v in reduction(work, passed).items())
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError):
        return False


def analyze(text: str, observed: Json) -> Json:
    """Name the first supported failure without guessing a kernel repair from API 101."""
    result: Json = dict(
        root_cause_status="cause_unproved",
        failure_layer="driver_initialization",
        causal_trace_line=None,
        correction=None,
        runtime_changed_score=0,
        cuda_context_ready_score=0,
    )
    if observed.get("api_returns", {}).get("cuInit") == 101:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            target = re.search(r"/dev/nvidia(?:ctl|[0-9]+)(?=[\">])", line)
            if (
                target
                and " = -1 " in line
                and re.search(r"\b(EACCES|EPERM|ENOENT|ENODEV)\b", line)
                and not any(
                    target.group() in later and re.search(r" = [0-9]+", later)
                    for later in lines[index + 1 :]
                )
            ):
                cause = (
                    "device_ioctl_rejected"
                    if "ioctl(" in line
                    else "device_namespace_missing"
                    if "ENOENT" in line
                    else "device_open_denied"
                )
                result.update(
                    root_cause_status=cause,
                    failure_layer="device_ioctl" if "ioctl(" in line else "device_namespace",
                    causal_trace_line=line,
                )
                break
        if any("/stubs/" in r["path"] for r in observed.get("libraries", [])):
            result.update(
                root_cause_status="installed_stub_library", correction="installed_library"
            )
        elif observed.get("environment", {}).get("CUDA_VISIBLE_DEVICES") in {"-1", ""}:
            result.update(
                root_cause_status="process_visibility_disabled", correction="process_binding"
            )
    return result


def diagnose(binding: Json, raw: Path) -> Json:
    """Trace only an owned child under a lease; a correction gets one bounded comparison."""
    raw.mkdir(parents=True, exist_ok=True)
    result: Json = dict(
        rows=[],
        checks=[],
        cleanup_receipts=[],
        intervention_rows=[],
        trace_manifest={},
        **analyze("", {}),
    )
    tools = {name: shutil.which(name) for name in ("strace", "prlimit")}
    lease: Any = None
    try:
        if not all(tools.values()):
            raise ValueError("owned_child_tracing_tools_absent")
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id=TASK,
            device_uuid=binding["permitted_uuid"],
            expected_model="no_model_load:driver_initialization",
            vram_before_mb=0,
            ttl_s=900,
        )
        result["lease_receipt"] = lease.owner_receipt()
        lease.transition("admitted")
        trace = raw / "driver.strace"
        argv = [
            str(tools["prlimit"]),
            f"--fsize={TRACE_CAP}:{TRACE_CAP}",
            str(tools["strace"]),
            "-f",
            "-yy",
            "-s",
            "128",
            "-e",
            "trace=openat,open,ioctl,access,readlink,newfstatat",
            "-o",
            str(trace),
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / CLI),
            "--driver-init",
        ]
        atomic_json(
            raw / "frozen_child.json",
            dict(
                argv=argv,
                trace_cap_bytes=TRACE_CAP,
                child_deadline_s=90,
                diagnosis_deadline_s=900,
                tracing_permission="owned descendant only",
            ),
        )
        receipt = child(
            "inherited_init",
            argv,
            raw / "children",
            deadline=90,
            heartbeat=20,
            scope="external_cuda",
        )
        lines = Path(receipt["stdout_path"]).read_text().splitlines()
        observed = json.loads(lines[-1]) if receipt["passed"] and lines else {}
        text = trace.read_text() if trace.is_file() else ""
        result["rows"].append(dict(receipt=receipt, primitive=observed, arm="inherited_init"))
        result.update(analyze(text, observed))
        result["trace_manifest"] = dict(
            ref=reference(trace) if trace.is_file() else None,
            size_bytes=trace.stat().st_size if trace.is_file() else None,
            cap_bytes=TRACE_CAP,
            bounded=trace.is_file() and trace.stat().st_size <= TRACE_CAP,
            complete=receipt["passed"],
            device_returns=[
                line
                for line in text.splitlines()
                if "/dev/nvidia" in line and any(s in line for s in ("open(", "openat(", "ioctl("))
            ],
        )
        if not receipt["passed"] or not trace.is_file():
            result["checks"].append(old.gate(trace, "complete_owned_trace", True, False))
        if result["correction"] and receipt["passed"] and trace.is_file():
            library = (
                binding["driver_library"]
                if result["correction"] == "installed_library"
                else "libcuda.so.1"
            )
            if result["correction"] == "installed_library" and not Path(library).is_absolute():
                installed = {
                    str(typed.authenticate(r).resolve())
                    for r in binding.get("libraries", [])
                    if Path(r["path"]).name.startswith("libcuda.") and "/stubs/" not in r["path"]
                }
                if len(installed) != 1:
                    raise ValueError("distinct_authenticated_installed_library_absent")
                library = installed.pop()
            env = ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=" + binding["permitted_uuid"]]
            if result["correction"] == "installed_library":
                env = [
                    "/usr/bin/env",
                    "-u",
                    "LD_PRELOAD",
                    "CUDA_VISIBLE_DEVICES=" + binding["permitted_uuid"],
                ]
            fixed = child(
                "corrected_init",
                env
                + [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--driver-init",
                    "--library",
                    library,
                ],
                raw / "children",
                deadline=90,
                heartbeat=20,
                scope="external_cuda",
            )
            corrected = (
                json.loads(Path(fixed["stdout_path"]).read_text().splitlines()[-1])
                if fixed["passed"]
                else {}
            )
            result["intervention_rows"].append(
                dict(
                    receipt=fixed, primitive=corrected, changed_environment=env[1:], library=library
                )
            )
            result["runtime_changed_score"] = int(
                corrected.get("api_returns", {}).get("cuInit") == 0
            )
            if result["runtime_changed_score"]:
                copied = child(
                    "context_copy",
                    env
                    + [
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(ROOT / CLI),
                        "--cuda-probe",
                        "driver",
                        "--library",
                        library,
                        "--uuid",
                        binding["permitted_uuid"],
                    ],
                    raw / "children",
                    deadline=90,
                    heartbeat=20,
                    scope="external_cuda",
                )
                operand = (
                    json.loads(Path(copied["stdout_path"]).read_text().splitlines()[-1])
                    if copied["passed"]
                    else {}
                )
                result["intervention_rows"].append(
                    dict(receipt=copied, primitive=operand, arm="context_copy")
                )
                result["cuda_context_ready_score"] = int(
                    copied["passed"]
                    and all(
                        operand.get(k) is True
                        for k in ("context_copy_ready", "byte_copy_parity", "cleanup_passed")
                    )
                    and operand.get("allocation_bytes") == 32
                )
    except (OSError, ValueError, KeyError, LeaseError) as error:
        result["checks"].append(old.gate(raw, "bounded_owned_diagnosis", True, str(error)))
    finally:
        if lease:
            lease.transition("terminal_blocked")
            result["cleanup_receipts"].append(lease.release())
    return result
