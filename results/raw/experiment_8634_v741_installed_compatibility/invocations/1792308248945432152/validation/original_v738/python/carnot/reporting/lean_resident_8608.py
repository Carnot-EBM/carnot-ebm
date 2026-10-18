"""REQ-REPORT-8608: bind a private lifecycle prototype to original numerical authority.

Fixtures establish that the new path executes. Independent parity qualification
and cost measurement belong to the next experiments, not these diagnostics.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import progress

ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8608_v739_lean_resident_prototype"
CLI = "scripts/experiments/" + NAME + ".py"
WORKER = "scripts/experiments/resident_worker_8608.py"
OWNED = [
    "python/carnot/inference/resident_energy.py",
    "python/carnot/inference/resident_protocol.py",
    "python/carnot/reporting/lean_resident_8608.py",
    "python/carnot/reporting/lean_runner_8608.py",
    CLI,
    WORKER,
]
TESTS = ["tests/python/test_lean_resident_8608.py", "tests/python/test_lean_prototype_8608.py"]
BASELINE = [
    "python/carnot/reporting/resident_worker_8572.py",
    "python/carnot/reporting/resident_worker_8571.py",
    "python/carnot/reporting/complete_native_worker_8559.py",
    "python/carnot/verify/nine_input_8533.py",
    "python/carnot/verify/sparse_energy_7996.py",
    "python/carnot/verify/evidence_features_7980.py",
    "python/carnot/verify/source_alignment.py",
    "python/carnot/verify/v726_decision_metrics.py",
]
Json = dict[str, Any]


def reference(path: Path) -> Json:
    """Record actual paths and exact bytes rather than an inferred producer name."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def authenticate(ref: Json) -> Path:
    """Missing evidence is different from a measured zero or a changed file."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("input_hash:" + str(path))
    return path


def energy_module() -> Any:
    """Use the same direct import boundary as the actual private worker."""
    sys.path.insert(0, str(ROOT / "python/carnot/inference"))
    import resident_energy

    return resident_energy


def session(config: Path, arm: str, raw: Path, *, baseline: bool = False) -> Json:
    """Supervise one bounded real process and retain import and process diagnostics."""
    energy_module()
    import resident_protocol as wire

    raw.mkdir(parents=True, exist_ok=True)
    graph = raw / "modules.json"
    invocation = (
        "from carnot.reporting.resident_worker_8572 import serve\n"
        + f"raise SystemExit(serve(Path({str(config)!r}), {sha256_file(config)!r}, {arm!r}))\n"
        if baseline
        else f"sys.argv={['resident_worker_8608.py', str(config), sha256_file(config), arm]!r}\n"
        + f"runpy.run_path({str(ROOT / WORKER)!r}, run_name='__main__')\n"
    )
    code = "import json,sys,runpy\nfrom pathlib import Path\ntry:\n"
    code += "".join("    " + line + "\n" for line in invocation.splitlines())
    code += f"finally:\n    Path({str(graph)!r}).write_text(json.dumps(sorted(sys.modules)))\n"
    argv = [sys.executable, "-u", "-X", "importtime", "-c", code]
    started = time.monotonic_ns()
    progress("8608_before_private_child_" + arm)
    responses = []
    with (raw / "importtime.stderr").open("wb") as log:
        with subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log
        ) as proc:
            assert proc.stdin is not None and proc.stdout is not None
            atomic_json(
                raw / "process.json",
                dict(
                    pid=proc.pid, command=argv, status=Path(f"/proc/{proc.pid}/status").read_text()
                ),
            )
            try:
                ready = json.loads(wire.read_frame(proc.stdout.fileno(), 30) or b"null")
                if ready is None or ready["status"] != "ready":
                    raise ValueError("worker_ready")
                for i, row in enumerate(json.loads(config.read_bytes())["rows"]):
                    value = wire.envelope(str(i), sha256_file(config), row)
                    wire.write_frame(proc.stdin.fileno(), value)
                    responses.append(
                        json.loads(wire.read_frame(proc.stdout.fileno(), 30) or b"null")
                    )
                wire.write_frame(proc.stdin.fileno(), value)
                duplicate = json.loads(wire.read_frame(proc.stdout.fileno(), 30) or b"null")
                value = wire.envelope("stop", sha256_file(config), {})
                value["operation"] = "shutdown"
                wire.write_frame(proc.stdin.fileno(), value)
                shutdown = json.loads(wire.read_frame(proc.stdout.fileno(), 30) or b"null")
                proc.wait(timeout=30)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            receipt = dict(
                command=argv,
                exit_code=proc.returncode,
                duration_s=(time.monotonic_ns() - started) / 1e9,
                passed=proc.returncode == 0
                and all(r["status"] == "ok" for r in responses)
                and duplicate["error"] == "duplicate_request_id"
                and shutdown["status"] == "shutdown",
                ready=ready,
                duplicate=duplicate,
                shutdown=shutdown,
                log_reference=reference(raw / "importtime.stderr"),
                process_reference=reference(raw / "process.json"),
            )
    progress("8608_after_private_child_" + arm)
    return dict(
        responses=responses,
        receipt=receipt,
        modules=json.loads(graph.read_bytes()),
        graph_reference=reference(graph),
    )


def measure(raw: Path, run_date: str, *, root: Path = ROOT) -> Json:
    """Authenticate CPU inputs independently of optional GPU and board availability."""
    progress("8608_start_preconditions")
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        raw=str(raw),
        run_date=run_date,
        started_ns=time.monotonic_ns(),
        gate_check_summary=[],
        sources=[],
        sessions=[],
        target_model=None,
    )
    with TemporaryDirectory(prefix="carnot8608-", dir="/var/tmp") as scratch:
        probe = Path(scratch) / "writable"
        probe.write_bytes(b"disk-backed CPU scratch")
        work["preconditions_checked"] = dict(
            scratch=str(probe.parent),
            writable=probe.read_bytes() == b"disk-backed CPU scratch",
            disk_free_bytes=shutil.disk_usage(scratch).free,
            memory=Path("/proc/meminfo").read_text(),
            tools={name: shutil.which(name) for name in ["python", "findmnt"]},
            python=sys.version,
            optional_gpu="not required; no probe",
            optional_board="not required; no probe",
            scratch_mount=subprocess.check_output(
                ["findmnt", "-T", scratch, "-n", "-o", "FSTYPE"], text=True, timeout=5
            ).strip(),
        )
    upstream = []
    for number, suffix, field in [
        (8593, "resident_provenance_qualification", "resident_measurement_ready_score"),
        (8596, "resident_complete_cost", "resident_cost_ready_score"),
    ]:
        path = root / f"results/experiment_{number}_v738_{suffix}.json"
        gate: Json = dict(
            check="qualified_original_primitives",
            upstream=number,
            path=str(path),
            sha256=None,
            field=field,
            operator="==",
            expected=1,
            observed="missing",
        )
        try:
            value = json.loads(path.read_bytes())
            gate.update(sha256=sha256_file(path), observed=value.get(field))
            if value.get(field) != 1 or value.get("required_checks_passed") is not True:
                raise ValueError("unqualified_upstream")
            sidecar = (
                path.parent
                / "raw"
                / path.stem
                / "validators"
                / (sha256_file(path).split(":")[1] + ".json")
            )
            if read_bound_sidecar(path, sidecar)["report"]["passed"] is not True:
                raise ValueError("upstream_publication")
            refs = [
                reference(path),
                reference(sidecar),
                value["work_reference"],
                reference(Path(value["terminal_validation_sidecar_path"])),
                *value["raw_shard_hashes"],
            ]
            for ref in refs:
                authenticate(ref)
            work["sources"].extend(refs)
            upstream.append(value)
        except (OSError, ValueError, KeyError) as error:
            gate["error"] = str(error)
            work["gate_check_summary"].append(gate)
    if len(upstream) == 2:
        provenance, costs = upstream
        ref = provenance["head_descriptor_reference"]
        target = json.loads(authenticate(ref).read_bytes())
        checkpoint = authenticate(target["checkpoint"])
        head = json.loads(checkpoint.read_bytes())["heads"]["spline9"][0]
        ext = target["native_extension_build"]
        authenticate(ext)
        if (
            canonical_hash(head["parameters"]) != target["parameters_sha256"]
            or canonical_hash(head["scaler"]) != target["scaler_sha256"]
            or target != provenance["target_model"]
            or target != costs["target_model"]
        ):
            raise ValueError("head_descriptor")
        work.update(
            target_model=target,
            head=head,
            historical_model_provenance=provenance["historical_model_provenance"],
            preserved_quarantines=provenance["preserved_quarantines"],
        )
        work["sources"].extend([ref, target["checkpoint"], {k: ext[k] for k in ["path", "sha256"]}])
        work["baseline_snapshots"] = []
        for name in BASELINE:
            destination = raw / "baseline" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            work["baseline_snapshots"].append(
                dict(original=reference(ROOT / name), preserved=reference(destination))
            )
        work["module_graph_registration"] = [reference(ROOT / name) for name in OWNED]
        atomic_json(raw / "module_graph_registration.json", work["module_graph_registration"])
        rows = [
            dict(
                unit_id="private-text",
                q=0.4,
                public=dict(
                    family_id="private-fixture",
                    source_bytes=b"There are 12 apples. No pears.".hex(),
                    answer_bytes=b"There are 12 apples.".hex(),
                ),
            ),
            dict(
                unit_id="private-finite",
                kind="numerical",
                scaled_features=[0.5] * 9,
                decay_multiplier=0.125,
            ),
        ]
        config = raw / "configuration.json"
        atomic_json(
            config, dict(head=head, extension={k: ext[k] for k in ["path", "sha256"]}, rows=rows)
        )
        work["config_reference"] = reference(config)
        progress("8608_registered_graph_before_fixture_execution")
        for baseline in [True, False]:
            for arm in ["python", "native"]:
                label = ("baseline_" if baseline else "lean_") + arm
                session_row = session(config, arm, raw / label, baseline=baseline)
                work["sessions"].append(dict(label=label, **session_row))
                progress("8608_fixture_sessions", len(work["sessions"]), 4 - len(work["sessions"]))
        selected = [
            s
            for s in costs["startup_shutdown_cost"]
            if s["unit_id"].startswith("L16-B1-") and s["unit_id"].endswith("resident_python")
        ]
        work["historical_startup_shutdown"] = dict(
            producer=reference(root / "results/experiment_8596_v738_resident_complete_cost.json"),
            repeat_count=len(selected),
            startup_s=sum(s["startup_s"] for s in selected),
            shutdown_s=sum(s["shutdown_s"] for s in selected),
            interpretation="historical motivation only; native CI95 [1.0740525,1.1057774]; NFR01 unmet",
        )
    work["ended_ns"] = time.monotonic_ns()
    work["code_config_hashes"] = [reference(ROOT / p) for p in OWNED + TESTS]
    atomic_json(raw / "work.json", work)
    progress("8608_primitives_complete")
    return work


def checksum(value: Json) -> str:
    """Bind every emitted field so altered envelopes require an explicit replay."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness credits executable fixtures only; original nulls retain authority."""
    blocked = bool(work["gate_check_summary"])
    sessions = work["sessions"]
    passed = all(r["passed"] for r in receipts) and all(s["receipt"]["passed"] for s in sessions)
    lean = [s for s in sessions if s["label"].startswith("lean_")]
    rows = [
        dict(
            unit_id=s["label"] + ":" + str(i),
            arm=s["label"],
            status=r["status"],
            response=r.get("response"),
            passed=r["status"] == "ok",
        )
        for s in sessions
        for i, r in enumerate(s["responses"])
    ]
    ready = int(not blocked and passed and len(lean) == 2 and len(rows) == 8)
    verdict = "blocked" if blocked else "null" if ready else "disqualified"
    value: Json = dict(
        experiment_id=8608,
        task_id="exp8608-lean-resident-prototype",
        milestone="2026.10.739",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v739_lean_resident_prototype",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads_attempted=0,
            model_loads_completed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
        ),
        current_llm_invocations=0,
        historical_model_provenance=work.get("historical_model_provenance", []),
        target_model=deepcopy(work["target_model"]),
        head_descriptor=deepcopy(work["target_model"]),
        extension_identity=[s["receipt"]["ready"]["loaded_extension"] for s in lean],
        rows=rows,
        intended_count=8,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(fixture_requests=8, independent_questions=0),
        verifier_is_oracle=False,
        exposure_scope="private executable fixtures and historical serving diagnostics",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        acceptance_gates=receipts,
        validation_receipts=receipts,
        adversarial_findings=[],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="authentication_and_private_fixtures",
                start_ns=work["started_ns"],
                end_ns=work["ended_ns"],
            )
        ],
        random_seed=17,
        source_artifact_hashes=work["sources"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[reference(Path(work["raw"]) / "work.json")],
        cited_upstream_artifacts=work["sources"],
        lean_prototype_ready_score=ready,
        prototype_paths=OWNED,
        accepted_benchmark_claim=False,
        import_dependency_rows=[
            dict(
                arm=s["label"],
                modules=s["modules"],
                training_present=any("training" in m for m in s["modules"]),
                sklearn_present=any(m.startswith("sklearn") for m in s["modules"]),
            )
            for s in sessions
        ],
        import_trace_hashes=[s["receipt"]["log_reference"] for s in sessions],
        fixture_response_rows=rows,
        preserved_semantics=[
            "complete original text features",
            "fit scaler and clipping",
            "raw-q offset",
            "109-coefficient SciPy basis",
            "stable normalization",
            "0.2/0.8 policy",
            "1e-10 native boundary fallback",
            "sorted compact JSON",
            "framed protocol",
            "configuration/extension hashes",
            "duplicate rejection",
            "request/idle deadlines",
            "clean shutdown",
        ],
        preserved_quarantines=work.get("preserved_quarantines", {}),
        historical_startup_shutdown=work.get("historical_startup_shutdown"),
        work_reference=reference(Path(work["raw"]) / "work.json"),
        publication_output=str(output.absolute()),
    )
    value["field_principles"] = {
        k: "Invocation-bound prototype evidence; no benchmark or generalization credit."
        for k in value
    }
    value["field_principles"].update(
        {
            "lean_prototype_ready_score": "One means both lean arms executed bounded fixtures and owned validation passed; Exp8609/8610 remain future dependencies.",
            "accepted_benchmark_claim": "Always false: import diagnostics and process durations cannot establish speed.",
            "independent_count": "Zero: eight fixture/arm evaluations do not constitute independent natural questions.",
            "sample_size_budget": "Two fixed private requests in four processes; no added independent samples.",
            "rows": "Every baseline/lean and Python/native fixture response remains individually visible.",
            "target_model": "The exact Exp8593 descriptor identifies original small-head numerical inference despite zero language-model calls.",
            "head_descriptor": "Original producer parameters, scaler, checkpoint, basis and policy retain authority.",
            "extension_identity": "Actual child ready frames identify the extension file and hash; neither a mirror nor a rebuild supplies it.",
            "import_dependency_rows": "Fresh process module lists locate training/reporting/sklearn imports; they are not performance estimates.",
            "import_trace_hashes": "Immutable importtime stderr bytes bind diagnostic traces to this invocation.",
            "fixture_response_rows": "Responses recompute complete public text and numerical work without feature/prediction caching.",
            "preserved_semantics": "The arithmetic and lifecycle obligations retained at the changed dependency boundary.",
            "historical_startup_shutdown": "Original Exp8596 L16/B1 resident Python clocks motivate the prototype; NFR01 and prior scientific nulls are unchanged.",
            "gate_check_summary": "A failed external-input gate names producer, path/hash, field, operator, expected and observed; missing is not zero.",
            "required_checks_passed": "All owned command receipts and real fixture children must pass; a killed validation child fails.",
            "validation_receipts": "Actual commands, exits, durations, original node union, 100 percent owned coverage and immutable log hashes.",
            "duration_s": "Actual authentication/fixture wall duration; no padding or accepted benchmark interpretation.",
            "work_reference": "Immutable primitive work document used for fresh-process semantic reduction and tamper rejection.",
            "prototype_paths": "Only newly owned code paths; shipped baseline bytes and historical evidence stay unchanged.",
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> None:
    """Fresh reads recompute semantics; rehashing a false aggregate cannot qualify it."""
    value = json.loads(path.read_bytes())
    if value.get("reproducibility_checksum") != checksum(value):
        raise ValueError("artifact_checksum")
    work = json.loads(authenticate(value["work_reference"]).read_bytes())
    for ref in work["sources"] + work["code_config_hashes"] + value["import_trace_hashes"]:
        authenticate(ref)
    for snapshot in work.get("baseline_snapshots", []):
        if (
            authenticate(snapshot["preserved"]).read_bytes()
            != authenticate(snapshot["original"]).read_bytes()
        ):
            raise ValueError("baseline_changed")
    if work["target_model"] is not None:
        target = work["target_model"]
        original = next(
            ref
            for ref in work["sources"]
            if Path(ref["path"]).name
            == "experiment_8593_v738_resident_provenance_qualification.json"
        )
        producer = json.loads(authenticate(original).read_bytes())
        descriptor = json.loads(authenticate(producer["head_descriptor_reference"]).read_bytes())
        if target != descriptor:
            raise ValueError("descriptor_authority")
        head = json.loads(authenticate(target["checkpoint"]).read_bytes())["heads"]["spline9"][0]
        if (
            head != work["head"]
            or canonical_hash(head["parameters"]) != target["parameters_sha256"]
        ):
            raise ValueError("head_changed")
        config = json.loads(authenticate(work["config_reference"]).read_bytes())
        if config["head"] != head or config["extension"] != {
            k: target["native_extension_build"][k] for k in ["path", "sha256"]
        }:
            raise ValueError("configuration_authority")
        numeric = energy_module()
        for session_row in work["sessions"]:
            modules = json.loads(authenticate(session_row["graph_reference"]).read_bytes())
            authenticate(session_row["receipt"]["process_reference"])
            if modules != session_row["modules"]:
                raise ValueError("module_trace_changed")
            native = (
                None
                if session_row["label"].endswith("python")
                else numeric.load_module(authenticate(config["extension"]), "_nine_input_8533")
            )
            for row, response in zip(config["rows"], session_row["responses"], strict=True):
                actual, _ = numeric.evaluate(head, row, native)
                if actual != response["response"]:
                    raise ValueError("fixture_response_changed")
            if session_row["label"].startswith("lean_") and any(
                m.startswith(("carnot.reporting", "carnot.training", "sklearn"))
                for m in session_row["modules"]
            ):
                raise ValueError("serving_import_graph")
    expected = artifact(work, value["validation_receipts"], Path(value["publication_output"]))
    if value != expected:
        raise ValueError("reduction_drift")
    progress("8608_cold_replay_passed")
