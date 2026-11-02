"""REQ-REPORT-8379: finite native parity is engineering evidence, not utility.

Sealed inputs and primitive outputs let a fresh interpreter repeat the actual
binding calls. Missing operands retain explicit gates instead of invented zeros.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import tempfile
from typing import Any

import numpy as np
import yaml

from carnot.reporting import v722_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import native_direct_8379 as k

ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8379_v722_native_direct_parity"
TASK = "exp8379-native-direct-parity"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_native_direct_parity_8379.py"
OWNED = [
    "python/carnot/verify/native_direct_8379.py",
    "python/carnot/reporting/native_direct_8379.py",
    "python/carnot/reporting/native_direct_runner_8379.py",
    CLI,
]
MODEL_SPECS: list[dict[str, Any]] = []
Json = dict[str, Any]
TASK_PIN = "sha256:ff21a0c169bc59766a8d2f64d2c715c0883df844b432ceeac3a3dcf999070dda"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts at real boundaries so a supervisor can detect stalled work."""
    print(f"[exp8379] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind a measurement to exact operand bytes, independent of file names."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def bind(root: Path, raw: Path) -> tuple[Json | None, list[Json], list[Json]]:
    """Only direct protocol operands enter this study; future producers stay dependencies."""
    refs: list[Json] = []
    gates: list[Json] = []
    for relative, pin in [
        (authority.PROTOCOL, authority.PROTOCOL_PIN),
        (authority.METHODS, authority.METHODS_PIN),
    ]:
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        if observed != pin:
            gates.append(authority.failure(path, "direct_input_hash", pin, observed))
        else:
            refs.append(reference(path))
    active = root / "research-roadmap.yaml"
    tasks = yaml.safe_load(active.read_bytes()).get("tasks", []) if active.is_file() else []
    task = [t for t in tasks if t["id"] == TASK]
    if (
        len(task) != 1
        or task[0].get("milestone") != "2026.10.722"
        or task[0].get("MODEL_SPECS") != []
        or canonical_hash(task[0]) != TASK_PIN
    ):
        gates.append(authority.failure(active, "exact_task_authority", TASK, task or None))
    else:
        refs.append(dict(reference(active), task_sha256=canonical_hash(task[0])))
    if gates:
        return None, refs, gates
    protocol: Json = json.loads((root / authority.PROTOCOL).read_bytes())
    operands = [
        protocol["checkpoint"],
        protocol["scientific_protocol"],
        protocol["preserved_v721_protocol"],
        *[protocol["panels"][key] for key in ("vectors", "metadata", "natural_predictor")],
    ]
    for operand in operands:
        path = Path(operand["path"])
        observed = sha256_file(path) if path.is_file() else None
        if observed != operand["sha256"]:
            gates.append(
                authority.failure(path, "direct_operand_hash", operand["sha256"], observed)
            )
        else:
            refs.append(reference(path))
    for ref in refs:
        destination = raw / "inputs" / (ref["sha256"].split(":")[1] + Path(ref["path"]).suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref["path"], destination)
        ref["snapshot_path"] = str(destination)
    return (None if gates else protocol), refs, gates


def panels(protocol: Json) -> Json:
    """Construct witnesses before seeing either arm's measured arithmetic."""
    head = protocol["head"]
    vectors = np.load(protocol["panels"]["vectors"]["path"]).tolist()
    kinds = json.loads(Path(protocol["panels"]["metadata"]["path"]).read_bytes())["kinds"]
    rows = [
        dict(unit_id=f"frozen-{i}", kind=kind, x=x)
        for i, (x, kind) in enumerate(zip(vectors, kinds, strict=True))
    ]
    for row in rows:
        row["input_rejected"] = any(not 0 <= v <= 1 for v in row["x"][1:])
    for feature in range(4):
        for knot in sorted(set(k.kernel.KNOTS)):
            for value in [np.nextafter(knot, -np.inf), knot, np.nextafter(knot, np.inf)]:
                if 0 <= value <= 1:
                    x = [0.0, 0.5, 0.5, 0.5, 0.5]
                    x[feature + 1] = float(value)
                    rows.append(dict(unit_id=f"knot-{len(rows)}", kind="nextafter_knot", x=x))
    for threshold in (0.25, 0.75):
        x = [0.0, 0.5, 0.5, 0.5, 0.5]
        local = float(k.probabilities(head, [x])[0])
        offset = head["temperature"] * np.log(local / (1 - local))
        holistic = (head["temperature"] * np.log(threshold / (1 - threshold)) - offset) / head[
            "coefficients"
        ][0]
        for value in [np.nextafter(holistic, -np.inf), holistic, np.nextafter(holistic, np.inf)]:
            rows.append(
                dict(
                    unit_id=f"boundary-{len(rows)}",
                    kind="nextafter_action_boundary",
                    x=[float(value), 0.5, 0.5, 0.5, 0.5],
                )
            )
    natural = json.loads(Path(protocol["panels"]["natural_predictor"]["path"]).read_bytes())["rows"]
    for row in natural:
        rows.append(
            dict(
                unit_id=row["unit_id"],
                kind="natural_cached",
                source_cluster_id=row["source_cluster_id"],
                x=None if row["x"] is None else [row["x"][0], *row["x"][12:16]],
                missing_reason=row.get("exclusion_reason"),
            )
        )
    updates = []
    for seed in (11, 22, 33):
        rng = np.random.default_rng(seed)
        for slot in range(64):
            updates.append(
                dict(
                    seed=seed,
                    slot=slot,
                    y=slot % 2,
                    x=[float(rng.normal()), *rng.random(4).tolist()],
                )
            )
    from carnot.verify import direct_atomic_state_8376 as coordinator

    traces = [coordinator.trace(seed, head) for seed in (11, 22, 33)]
    for trace in traces:
        trace["events"] = [trace["events"][i] for i in (0, 1, 9, 11)]
    return dict(head=head, rows=rows, updates=updates, traces=traces, seed=7208352)


def execute(panel: Json, native: Any) -> Json:
    """Compare both arms without hiding small differences at action boundaries."""
    rows: list[Json] = []
    available = [r for r in panel["rows"] if r["x"] is not None and not r.get("input_rejected")]
    native, counts = k.tracked(native)
    for start in range(0, len(available), 128):
        batch = available[start : start + 128]
        x = [r["x"] for r in batch]
        py, rust = k.probabilities(panel["head"], x), k.probabilities(panel["head"], x, native)
        for item, left, right in zip(batch, py, rust, strict=True):
            pa, na = k.kernel.action(float(left)), k.kernel.action(float(right))
            error = abs(float(left) - float(right))
            rows.append(
                dict(
                    item,
                    arm="paired_python_native",
                    status="completed",
                    excluded=False,
                    python_probability=float(left),
                    native_probability=float(right),
                    python_action=pa,
                    native_action=na,
                    probability_error=error,
                    coefficient_error=0.0,
                    action_mismatch=int(pa != na),
                    absolute_metric=error,
                    raw_numerator=error,
                    raw_denominator=1,
                )
            )
        progress("direct_batch", len(rows), len(available) - len(rows))
    for item in panel["rows"]:
        if item.get("input_rejected"):
            rejected = []
            for arm in (None, native):
                try:
                    k.probabilities(panel["head"], [item["x"]], arm)
                    rejected.append(False)
                except ValueError:
                    rejected.append(True)
            try:
                native.direct_logits_8379(
                    panel["head"]["coefficients"], [item["x"]], panel["head"]["temperature"]
                )
                rejected.append(False)
            except ValueError:
                rejected.append(True)
            rows.append(
                dict(
                    item,
                    arm="paired_python_native",
                    status="completed",
                    excluded=False,
                    python_probability=None,
                    native_probability=None,
                    python_action="input_rejected",
                    native_action="input_rejected",
                    python_native_binding_rejected=rejected,
                    probability_error=None,
                    coefficient_error=None,
                    action_mismatch=int(not all(rejected)),
                    absolute_metric=int(all(rejected)),
                    raw_numerator=int(all(rejected)),
                    raw_denominator=1,
                )
            )
        if item["x"] is None:
            rows.append(
                dict(
                    item,
                    arm="paired_python_native",
                    status="excluded",
                    excluded=True,
                    missing_reason=item.get("missing_reason") or "natural_features_absent",
                    python_probability=None,
                    native_probability=None,
                    python_action=None,
                    native_action=None,
                    probability_error=None,
                    coefficient_error=None,
                    action_mismatch=None,
                    absolute_metric=None,
                    raw_numerator=None,
                    raw_denominator=1,
                )
            )
    py_head, native_head = deepcopy(panel["head"]), deepcopy(panel["head"])
    for index, item in enumerate(panel["updates"]):
        if item["slot"] == 0:
            py_head, native_head = deepcopy(panel["head"]), deepcopy(panel["head"])
        py_before, rust_before = list(py_head["coefficients"]), list(native_head["coefficients"])
        expected = k.optimizer.learn(py_head, item["x"], item["y"], "online_sparse")["coefficients"]
        actual = k.learn(native_head, item["x"], item["y"], native)
        error = max(abs(a - b) for a, b in zip(expected, actual, strict=True))
        py_head["coefficients"], native_head["coefficients"] = expected, actual
        py = float(k.probabilities(py_head, [item["x"]])[0])
        rust = float(k.probabilities(native_head, [item["x"]], native)[0])
        rows.append(
            dict(
                item,
                unit_id=f"update-{item['seed']}-{item['slot']}",
                kind="numeric_update",
                arm="paired_python_native",
                status="completed",
                excluded=False,
                python_before=py_before,
                native_before=rust_before,
                python_coefficients=expected,
                native_coefficients=actual,
                python_probability=py,
                native_probability=rust,
                python_action=k.kernel.action(py),
                native_action=k.kernel.action(rust),
                probability_error=abs(py - rust),
                coefficient_error=error,
                action_mismatch=int(k.kernel.action(py) != k.kernel.action(rust)),
                absolute_metric=error,
                raw_numerator=error,
                raw_denominator=1,
            )
        )
        if index % 32 == 0:
            progress("update_panel", index + 1, len(panel["updates"]) - index - 1)
    rejections = []
    bad_rows = [
        [0.0],
        [float("nan")] * 5,
        [float("inf")] * 5,
        [0.0, -1e-20, 0.0, 0.0, 0.0],
        [0.0, 1.00001, 0.0, 0.0, 0.0],
    ]
    for index, x in enumerate(bad_rows):
        rejected = []
        for arm in (None, native):
            try:
                k.probabilities(panel["head"], [x], arm)
                rejected.append(False)
            except ValueError:
                rejected.append(True)
        try:
            native.direct_design_8379(x)
            rejected.append(False)
        except ValueError:
            rejected.append(True)
        rejections.append(
            dict(case=index, python_native_binding_rejected=rejected, passed=all(rejected))
        )
    coordinator = k.coordinator(native)
    states = []
    for trace in panel["traces"]:
        progress("durable_coordinator_before_seed_" + str(trace["seed"]))
        with tempfile.TemporaryDirectory(
            dir=Path.home() / ".cache/carnot-exp8379-private"
        ) as directory:
            store = coordinator.Store(Path(directory), trace)
            store.initialize()
            for event in trace["events"]:
                store.apply(event)
            states.append(store.read())
        progress("durable_coordinator_after_seed_" + str(trace["seed"]))
    return dict(rows=rows, rejection_controls=rejections, coordinator_states=states, **counts)


def measure(root: Path, raw: Path, private: Path, supplied: str | None = None) -> Json:
    """Authenticate operands and freeze panels before native observations open."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_before")
    protocol, refs, gates = bind(root, raw)
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    preconditions = dict(
        scratch=str(private),
        private_mode=oct(private.stat().st_mode & 0o777),
        available_memory_bytes=available,
        free_disk_bytes=shutil.disk_usage(private).free,
        tools={name: shutil.which(name) for name in ("cargo", "rustc", "findmnt")},
        task_cap_s=4800,
        memory_budget_bytes=1536 * 1024 * 1024,
        no_model_load=True,
        task_sha256=next((r.get("task_sha256") for r in refs if "task_sha256" in r), None),
    )
    from carnot.reporting.v709_execution import child

    fs = child(
        "scratch_filesystem",
        ["findmnt", "-T", str(private), "-n", "-o", "FSTYPE"],
        raw,
        deadline=30,
    )
    filesystem = Path(fs["stdout_path"]).read_text().strip()
    preconditions.update(
        filesystem=filesystem, disk_backed=filesystem not in {"tmpfs", "ramfs", ""}
    )
    resource_ok = bool(
        preconditions["disk_backed"]
        and available >= 536870912
        and all(preconditions["tools"].values())
    )
    if not resource_ok:
        gates.append(authority.failure(private, "resource_preconditions", True, preconditions))
    result: Json = dict(
        rows=[],
        rejection_controls=[],
        native_invocation_count=0,
        binding_copy_bytes=0,
        binding_conversion_bytes=0,
        native_call_counts={},
        coordinator_states=[],
    )
    receipt: Json = dict(path=None, sha256=None, actual_loaded=False, receipts=[])
    failure = None
    panel_ref = None
    progress("preconditions_after", len(refs), len(gates))
    if protocol is not None and resource_ok:
        panel = panels(protocol)
        panel_path = raw / "frozen_panel.json"
        atomic_json(panel_path, panel)
        panel_ref = reference(panel_path)
        progress("native_build_before")
        try:
            native, receipt = k.extension(private, raw / "native_build", supplied)
            progress("native_build_after")
            progress("benchmark_before")
            result = execute(panel, native)
            progress("benchmark_after", len(result["rows"]), 0)
        except (OSError, ValueError, ImportError) as error:
            failure = str(error)
            progress("native_owned_failure_" + failure)
    primitive = raw / "primitive_rows.json"
    atomic_json(primitive, result)
    work = dict(
        result,
        input_ready=protocol is not None and resource_ok,
        gate_check_summary=gates,
        source_artifact_hashes=refs,
        panel_reference=panel_ref,
        primitive_reference=reference(primitive),
        extension=receipt,
        owned_failure=failure,
        preconditions_checked=preconditions,
        code_config_hashes=[
            reference(ROOT / p)
            for p in OWNED + k.SOURCES + [TEST, "crates/carnot-core/tests/direct_spline_8379.rs"]
        ],
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authentication_build_and_finite_parity", duration_s=time.monotonic() - began
            )
        ],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json]) -> Json:
    """A successful run cannot substitute for parity or independent scientific benefit."""
    rows = work["rows"] or [
        dict(
            unit_id=f"unavailable-{i}",
            arm="paired_python_native",
            status="unstarted",
            excluded=False,
            missing_reason="authenticated_direct_inputs_or_native_execution_unavailable",
            probability_error=None,
            coefficient_error=None,
            action_mismatch=None,
            absolute_metric=None,
            raw_numerator=None,
            raw_denominator=1,
        )
        for i in range(4564)
    ]
    complete = [r for r in rows if r["status"] == "completed"]
    probability_error = max(
        (r["probability_error"] for r in complete if r["probability_error"] is not None),
        default=None,
    )
    coefficient_error = max(
        (r["coefficient_error"] for r in complete if r["coefficient_error"] is not None),
        default=None,
    )
    mismatches = sum(r["action_mismatch"] for r in complete) if complete else None
    checks = bool(receipts) and all(
        r["passed"] for r in receipts if r.get("scope", "owned") == "owned"
    )
    parity = bool(
        complete
        and probability_error <= 1e-12
        and coefficient_error <= 1e-12
        and mismatches == 0
        and all(r["passed"] for r in work["rejection_controls"])
    )
    verdict = (
        "disqualified"
        if not checks or work["owned_failure"]
        else "blocked"
        if not work["input_ready"]
        else "circular_positive"
        if parity
        else "disqualified"
    )
    ready = int(verdict == "circular_positive" and work["native_invocation_count"] > 0)
    boundary = [r for r in complete if "boundary" in r.get("kind", "")]
    result = dict(
        experiment_id=8379,
        task_id=TASK,
        milestone="2026.10.722",
        run_date="20261010",
        honest_verdict="complete_" + verdict + "_finite_native_direct_parity",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        no_model_load=True,
        model_invocation_counts=dict(
            current_llm_calls=0, current_model_loads=0, numeric_heads=1 if complete else 0
        ),
        historical_model_provenance="cached source observations only; no current generation",
        rows=rows,
        intended_count=len(rows),
        completed_count=len(complete),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "unstarted" for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        sample_size_budget=dict(
            frozen_random_vectors=4096,
            frozen_panel_vectors=4174,
            nextafter_knot_vectors=64,
            explicit_threshold_vectors=6,
            natural_slots=128,
            numeric_update_vectors=192,
            independent_examples=0,
            timing_repeats_as_examples=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed cached development and constructed numerical witnesses",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checks,
        flagged_adversarial=not checks,
        acceptance_gates=dict(
            owned_validation=checks,
            direct_inputs=work["input_ready"],
            finite_parity=parity,
            native_execution=work["native_invocation_count"] > 0,
            universal_floating_point_theorem=False,
            semantic_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=work["terminal_validation_sidecar_path"],
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7208352,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[work["primitive_reference"]]
        + ([work["panel_reference"]] if work["panel_reference"] else []),
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields="direct head, feature operands, task authority and preserved protocol bytes",
            )
            for r in work["source_artifact_hashes"]
        ],
        native_parity_ready_score=ready,
        extension_path=work["extension"]["path"],
        extension_sha256=work["extension"]["sha256"],
        native_invocation_count=work["native_invocation_count"],
        max_probability_error=probability_error,
        max_coefficient_error=coefficient_error,
        action_mismatch_count=mismatches,
        boundary_rows=boundary,
        binding_copy_bytes=work["binding_copy_bytes"],
        binding_conversion_bytes=work["binding_conversion_bytes"],
        native_call_counts=work["native_call_counts"],
        coordinator_states=work["coordinator_states"],
        binding_copy_scope="logical float64 payload copied into PyO3 Vec and returned list; allocator/object overhead unmeasured",
        rejection_controls=work["rejection_controls"],
        extension_receipt=work["extension"],
        owned_failure=work["owned_failure"],
        measurement_reference=reference(
            Path(work["primitive_reference"]["path"]).parent / "measurement.json"
        ),
        panel_reference=work["panel_reference"],
        finite_domain_only=True,
        arithmetic_order="original basis stages; per-feature native dot then four ordered additions; no fast-math or native-only fallback",
        speed_claim="full transaction cost deferred to Exp8380",
        future_dependencies=[],
        durable_coordinator="python/carnot/verify/direct_atomic_state_8376.py; native arithmetic opt-in only",
    )
    result["field_principles"] = {
        key: "Bind the finite parity observation to sealed operands; readiness never implies semantic benefit."
        for key in [*result, "field_principles", "reproducibility_checksum"]
    }
    result["reproducibility_checksum"] = canonical_hash(result)
    return result


def replay(path: Path) -> bool:
    """Fresh native execution rejects self-consistently rehashed aggregate tampering."""
    try:
        value = json.loads(path.read_bytes())
        digest = value.pop("reproducibility_checksum")
        if canonical_hash(value) != digest:
            raise ValueError("result_checksum")
        ref = value["measurement_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("measurement_hash")
        work = json.loads(Path(ref["path"]).read_bytes())
        for operand in (
            work["code_config_hashes"]
            + work["source_artifact_hashes"]
            + [work["primitive_reference"]]
        ):
            source = Path(operand.get("snapshot_path", operand["path"]))
            if sha256_file(source) != operand["sha256"]:
                raise ValueError("operand_hash")
        if work["panel_reference"] is not None and work["extension"]["actual_loaded"]:
            panel_ref = work["panel_reference"]
            if sha256_file(Path(panel_ref["path"])) != panel_ref["sha256"]:
                raise ValueError("panel_hash")
            if sha256_file(Path(work["extension"]["path"])) != work["extension"]["sha256"]:
                raise ValueError("binary_hash")
            panel = json.loads(Path(panel_ref["path"]).read_bytes())
            protocol_ref = next(
                r for r in work["source_artifact_hashes"] if r["path"].endswith(authority.PROTOCOL)
            )
            protocol = json.loads(Path(protocol_ref["snapshot_path"]).read_bytes())
            if protocol_ref["sha256"] != authority.PROTOCOL_PIN or panel != panels(protocol):
                raise ValueError("frozen_panel_semantics")
            module, _ = k.extension(
                Path(ref["path"]).parent, Path(ref["path"]).parent, work["extension"]["path"]
            )
            actual = execute(panel, module)
            primitive = json.loads(Path(work["primitive_reference"]["path"]).read_bytes())
            if actual != primitive or any(work[key] != actual[key] for key in actual):
                raise ValueError("native_primitive_drift")
        rebuilt = build(work, value["validation_receipts"])
        rebuilt.pop("reproducibility_checksum")
        if rebuilt != value:
            raise ValueError("aggregate_drift")
        return True
    except (OSError, ValueError, KeyError, TypeError, ImportError, StopIteration):
        return False
