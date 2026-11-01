"""REQ-REPORT-8352: bind numerical measurements to immutable scientific operands."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import yaml

from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.verify import spline_table_fidelity_8352 as n
from carnot.verify import local_update_isolation_8306 as kernel

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8352_v720_spline_table_fidelity"
TASK = "exp8352-spline-table-fidelity"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_spline_table_fidelity_8352.py"
OWNED = [
    "python/carnot/verify/spline_table_fidelity_8352.py",
    "python/carnot/reporting/spline_table_fidelity_8352.py",
    "python/carnot/reporting/spline_table_execution_8352.py",
    CLI,
]
HEAD = "results/experiment_8334_v719_sentence_spline_fit.json"
PROTOCOL = "openspec/change-proposals/v717-local-learning-protocol.json"
PINS = {
    HEAD: "sha256:a63db57cb7f735a1171ce16b87d09aa02abe69c7f7abb6fc6ca5fa6240d59208",
    PROTOCOL: "sha256:853709123024de763e96dd688e819f0430205ae6d97d6561a2b95cca23b81c6f",
}
CHECKPOINT = "sha256:c76b68dc99911ac1dbb2471ead9f787c1ebedb2e0562ac861d8960c88bb3b97e"
TASK_PIN = "sha256:9f3f122c877e1c5d4a3f74ad26e0511e29f389ad0f7fa5321014fa41eefc4c7d"
MODEL_SPECS: list[Json] = []


class OperandError(ValueError):
    """Missing or changed external bytes need a named block, not a synthetic input."""

    def __init__(self, finding: Json):
        super().__init__(str(finding))
        self.finding = finding


def reference(path: Path) -> Json:
    """Exact hashes keep primitive files separate from their derived summaries."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def require(path: Path, field: str, expected: Any, observed: Any) -> None:
    """Record missing as null; a real measured zero remains a different operand."""
    if expected != observed:
        raise OperandError(
            dict(
                upstream=path.stem,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                field=field,
                artifact_field=field,
                operator="==",
                op="==",
                expected=expected,
                observed=observed,
                passed=False,
            )
        )


def authenticate(root: Path, raw: Path) -> Json:
    """Only the original checkpoint supplies coefficients; reserved shards stay closed."""
    refs: list[Json] = []

    def bind(path: Path, expected: str | None = None) -> Json:
        observed = sha256_file(path) if path.is_file() else None
        require(
            path,
            "sha256" if expected else "exists",
            expected or True,
            observed if expected else path.is_file(),
        )
        dest = raw / "inputs" / (str(observed)[7:] + "-" + path.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        refs.append(dict(reference(dest), source_path=str(path), source_sha256=observed))
        return dict(json.loads(dest.read_bytes()))

    primary = bind(root / HEAD, PINS[HEAD])
    protocol = bind(root / PROTOCOL, PINS[PROTOCOL])
    checkpoint_ref = primary["checkpoint_hashes"][0]
    require(root / HEAD, "checkpoint_sha256", CHECKPOINT, checkpoint_ref["sha256"])
    checkpoint = bind(Path(checkpoint_ref["path"]), CHECKPOINT)
    terminal = bind(Path(primary["terminal_validation_sidecar_path"]))
    sidepath = Path(terminal["publication"]["sidecar_path"])
    try:
        side = read_bound_sidecar(root / HEAD, sidepath)
    except (OSError, ValueError):
        side = {}
        require(sidepath, "bound_terminal_sidecar", True, False)
    bind(sidepath)
    require(sidepath, "report.passed", True, side["report"].get("passed"))
    require(
        root / HEAD,
        "qualified_heads",
        [1, True, False, "null"],
        [
            primary.get("heads_ready_score"),
            primary.get("required_checks_passed"),
            primary.get("flagged_adversarial"),
            primary.get("verdict_class"),
        ],
    )
    active_path = root / "research-roadmap.yaml"
    require(active_path, "exists", True, active_path.is_file())
    active = yaml.safe_load(active_path.read_bytes())
    design_path = root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    require(design_path, "exists", True, design_path.is_file())
    _, tasks = parse_design(design_path.read_text(), milestone="2026.10.720")
    task = next(t for t in tasks if t["id"] == TASK)
    actual = next((t for t in active["tasks"] if t["id"] == TASK), None)
    require(active_path, "milestone", "2026.10.720", active.get("milestone"))
    require(active_path, "task", task, actual)
    require(active_path, "task_sha256", TASK_PIN, canonical_hash(task))
    for path in [active_path, design_path, root / "docs/research-notes/v720-table-protocol.md"]:
        require(path, "exists", True, path.is_file())
        dest = raw / "inputs" / (sha256_file(path)[7:] + "-" + path.name)
        shutil.copyfile(path, dest)
        refs.append(dict(reference(dest), source_path=str(path), source_sha256=sha256_file(path)))
    for paper, digest in [
        ("2512.12850v3", "5ff42c9d57e123f3b21e93a813ca41e0721d4edae5281eaf12fa6a8af902054e"),
        ("2602.02056v4", "1ba2e3c7dae9bd49cf23b06a12480a4df3a56f173b049a2211761e536bbf3a26"),
    ]:
        path = root / "results/raw" / NAME / "literature" / (paper + ".html")
        require(path, "sha256", "sha256:" + digest, sha256_file(path) if path.is_file() else None)
        dest = raw / "inputs" / path.name
        shutil.copyfile(path, dest)
        refs.append(dict(reference(dest), source_path=str(path), source_sha256=sha256_file(path)))
    head = next(h for h in checkpoint["heads"] if h["arm"] == "spline34")
    require(
        root / PROTOCOL,
        "numeric_spec",
        [34, 3, list(kernel.KNOTS)],
        [protocol["head"]["parameters"], protocol["head"]["degree"], protocol["head"]["knots"]],
    )
    return dict(
        head=head,
        protocol=protocol,
        refs=refs,
        task=task,
        authority_sha256=canonical_hash(task),
        historical_model_provenance=primary["historical_model_provenance"],
    )


def measure(inputs: Json, raw: Path) -> Json:
    """Save probabilities and timings as primitives before reducing any candidate gates."""
    began = time.monotonic_ns()
    head = inputs["head"]
    x, kinds, available = n.panel(head)
    np.save(raw / "panel.npy", x)
    atomic_json(raw / "panel_metadata.json", dict(kinds=kinds, boundary_available=available))
    direct, ref = n.direct(head, x), n.reference(head, x)
    np.save(raw / "direct.npy", direct)
    np.save(raw / "reference.npy", ref)
    configs: list[Json] = []
    per_vector = raw / "per_vector_rows.jsonl"
    with per_vector.open("w") as stream:
        for order, (size, storage, interpolation) in enumerate(n.CONFIGS):
            n.progress("configuration", order, len(n.CONFIGS) - order)
            values, sat = n.table(head, size, storage)
            actual = n.lookup(head, x, values, storage, interpolation)
            row = n.summarize(ref, actual, size, storage, interpolation, order, sat)
            timings = n.evaluation_timings(head, x, values, storage, interpolation)
            name = row["configuration_id"]
            for label, data in [("table", values), ("probabilities", actual), ("timings", timings)]:
                path = raw / (name + "-" + label + ".npy")
                np.save(path, data)
                row[label + "_reference"] = reference(path)
            row["direct_ns_per_vector_median"] = float(np.median(timings[:, 1:, 0]))
            row["table_ns_per_vector_median"] = float(np.median(timings[:, 1:, 1]))
            configs.append(row)
            for i, p in enumerate(actual):
                stream.write(
                    json.dumps(
                        dict(
                            configuration_id=name,
                            vector_id=i,
                            kind=kinds[i],
                            reference_probability=float(ref[i]),
                            direct_probability=float(direct[i]),
                            table_probability=float(p),
                            probability_error=float(abs(p - ref[i])),
                            reference_action=kernel.action(float(ref[i])),
                            table_action=kernel.action(float(p)),
                            action_flip=kernel.action(float(ref[i])) != kernel.action(float(p)),
                            near_threshold=bool(
                                min(abs(ref[i] - 0.25), abs(ref[i] - 0.75)) < 0.002
                            ),
                            timing_ns=timings[i].tolist(),
                        )
                    )
                    + "\n"
                )
    refresh, finals, refresh_ok = n.refresh_measurement(head)
    atomic_json(raw / "refresh_rows.json", dict(rows=refresh))
    atomic_json(raw / "final_tables.json", dict(rows=finals))
    return dict(
        inputs=inputs,
        configurations=configs,
        direct_parity=float(np.max(np.abs(direct - ref))),
        refresh_passed=refresh_ok,
        boundary_available=available,
        vector_count=len(x),
        primitive_refs=[
            reference(raw / p)
            for p in [
                "panel.npy",
                "panel_metadata.json",
                "direct.npy",
                "reference.npy",
                "per_vector_rows.jsonl",
                "refresh_rows.json",
                "final_tables.json",
            ]
        ],
        phase_spans=[
            dict(
                phase="numerical_measurement",
                started_monotonic_ns=began,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualified measurement and a passing candidate are separate downstream decisions."""
    failures = work.get("failures", [])
    rows = work.get("configurations", [])
    coverage = work.get("coverage", {})
    plan = json.loads(Path(work["manifest_reference"]["path"]).read_bytes())["commands"]
    executed = {r["name"]: r for r in receipts}
    owned = all(
        spec["name"] in executed
        and executed[spec["name"]]["argv"] == spec["argv"]
        and executed[spec["name"]]["expected_exit"] == spec["expected"]
        for spec in plan
    )
    owned = owned and (
        bool(receipts)
        and all(
            r["passed"] and r["exit_code"] == r["expected_exit"] and not r["timed_out"]
            for r in receipts
        )
        and coverage.get("totals", {}).get("percent_covered") == 100
    )
    owned = owned and all(
        p in coverage.get("files", {}) and coverage["files"][p]["summary"]["missing_lines"] == 0
        for p in OWNED
    )
    numeric = bool(rows) and work["direct_parity"] <= 1e-10 and work["refresh_passed"]
    ready = owned and numeric and not failures
    verdict = "blocked" if failures else "circular_positive" if ready else "disqualified"
    candidate = n.select(rows) if ready else None
    inputs = work.get("inputs", {})
    refs = [
        *work.get("primitive_refs", []),
        *[
            r[k]
            for r in rows
            for k in ["table_reference", "probabilities_reference", "timings_reference"]
        ],
    ]
    value: Json = dict(
        experiment_id=8352,
        task_id=TASK,
        milestone="2026.10.720",
        run_date="20261009",
        honest_verdict="complete_" + verdict + "_spline_table_fidelity",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=inputs.get("historical_model_provenance", []),
        rows=rows,
        configurations=rows,
        intended_count=12,
        completed_count=len(rows),
        failed_count=0 if numeric or not rows else len(rows),
        censored_count=12 - len(rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            constructed_random_vectors=4096,
            actual_vectors=work.get("vector_count", 0),
            configurations=12,
            local_updates=64,
            independent_natural_sources=0,
            warmups=1,
            paired_repetitions=5,
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed_controls_from_exposed_development_coefficients",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=not all(
            r["passed"] for r in receipts if r.get("name") == "adversarial"
        ),
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            direct_parity=bool(rows) and work["direct_parity"] <= 1e-10,
            exact_refresh=work.get("refresh_passed", False),
            owned_checks=owned,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work.get("preconditions", {}),
        duration_s=work.get("duration_s", 0.0),
        phase_spans=work.get("phase_spans", [])
        + [
            dict(
                phase=r["name"],
                started_monotonic_ns=r["started_monotonic_ns"],
                ended_monotonic_ns=r["ended_monotonic_ns"],
            )
            for r in receipts
        ],
        random_seed=n.SEED,
        source_artifact_hashes={r["source_path"]: r["sha256"] for r in inputs.get("refs", [])},
        code_config_hashes={r["path"]: r["sha256"] for r in work.get("code_refs", [])},
        raw_shard_hashes={r["path"]: r["sha256"] for r in refs},
        cited_upstream_artifacts=[
            dict(
                experiment_id=8334,
                path=HEAD,
                sha256=PINS[HEAD],
                fields_imported=[
                    "spline34 coefficients",
                    "temperature",
                    "historical_model_provenance",
                ],
            )
        ],
        table_fidelity_ready_score=int(ready),
        table_candidate_score=int(candidate is not None),
        selected_candidate=candidate,
        per_vector_rows=reference(raw / "per_vector_rows.jsonl") if rows else None,
        refresh_rows=reference(raw / "refresh_rows.json") if rows else None,
        probability_error_max=max((r["probability_error_max"] for r in rows), default=None),
        action_flip_count=sum(r["action_flip_count"] for r in rows),
        boundary_flip_count=sum(r["boundary_flip_count"] for r in rows),
        saturation_count=sum(r["saturation_count"] for r in rows),
        table_bytes=candidate["table_bytes"] if candidate else None,
        direct_parity=work.get("direct_parity"),
        boundary_construction_available=work.get("boundary_available"),
        operation_manifest=dict(
            board="future_KV260_mapping",
            current_device_execution_count=0,
            rtl_created=False,
            per_vector=dict(
                feature_indices=4,
                local_reads_nearest=4,
                local_reads_linear=8,
                linear_interpolations=4,
                local_sum_additions=3,
                float64_global_multiply=1,
                float64_global_additions=2,
                float64_temperature_divide=1,
                float64_sigmoid=1,
            ),
            refresh="support-indexed edge sums and encoded writes; see refresh_rows",
            unmeasured=[
                "DMA",
                "device latency",
                "energy",
                "synthesis resources",
                "float64 hardware fallback",
            ],
        ),
        work_reference=reference(raw / "measurement.json"),
        methodology_note="Numerical construction is circular_positive, never natural accuracy. All twelve configurations are retained. Global arithmetic stays float64. Warmups and repeated timings add no independent sources. H1/H2 remain unmeasured. No full integer inference or device claim.",
    )
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to pinned inputs, replayable primitive evidence and constructed engineering scope."
        for k in value
    }
    value["field_principles"].update(
        per_vector_rows="Byte-bound JSONL contains every configuration/vector row and paired timing sample.",
        refresh_rows="Byte-bound JSON contains every storage/grid/update support entry, coefficient and paired timing.",
        table_fidelity_ready_score="Qualified measurements can be ready even if no table passes candidate criteria.",
        table_candidate_score="Only a measured table meeting both frozen gates is a candidate.",
        reproducibility_checksum="Hash the complete semantic artifact except the checksum itself.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay_numeric(work: Json, raw: Path) -> bool:
    """Reconstruct tables, updates and every per-vector row instead of trusting hashes."""
    inputs = work["inputs"]
    copies = {r["source_sha256"]: Path(r["path"]) for r in inputs["refs"]}
    if any(
        sha256_file(copies[digest]) != digest for digest in [PINS[HEAD], PINS[PROTOCOL], CHECKPOINT]
    ):
        return False
    primary = json.loads(copies[PINS[HEAD]].read_bytes())
    protocol = json.loads(copies[PINS[PROTOCOL]].read_bytes())
    checkpoint = json.loads(copies[CHECKPOINT].read_bytes())
    head = next(h for h in checkpoint["heads"] if h["arm"] == "spline34")
    if (
        inputs["head"] != head
        or inputs["protocol"] != protocol
        or primary["checkpoint_hashes"][0]["sha256"] != CHECKPOINT
    ):
        return False
    source_paths = {Path(r["source_path"]).name: Path(r["path"]) for r in inputs["refs"]}
    _, tasks = parse_design(
        source_paths["research-roadmap-vNEXT.md"].read_text(), milestone="2026.10.720"
    )
    task = next(t for t in tasks if t["id"] == TASK)
    active = yaml.safe_load(source_paths["research-roadmap.yaml"].read_bytes())
    if (
        task != inputs["task"]
        or task != next(t for t in active["tasks"] if t["id"] == TASK)
        or canonical_hash(task) != inputs["authority_sha256"]
        or canonical_hash(task) != TASK_PIN
    ):
        return False
    x, kinds, available = n.panel(head)
    direct, ref = n.direct(head, x), n.reference(head, x)
    if (
        not np.array_equal(x, np.load(raw / "panel.npy"))
        or not np.array_equal(direct, np.load(raw / "direct.npy"))
        or not np.array_equal(ref, np.load(raw / "reference.npy"))
    ):
        return False
    if (
        work["vector_count"] != len(x)
        or work["boundary_available"] != available
        or work["direct_parity"] != float(np.max(np.abs(direct - ref)))
        or not np.isfinite(ref).all()
    ):
        return False
    if (
        json.loads((raw / "panel_metadata.json").read_bytes())
        != dict(kinds=kinds, boundary_available=available)
        or len(work["configurations"]) != 12
    ):
        return False
    with (raw / "per_vector_rows.jsonl").open() as stream:
        for order, (row, config) in enumerate(zip(work["configurations"], n.CONFIGS, strict=True)):
            size, storage, interpolation = config
            values, sat = n.table(head, size, storage)
            actual = n.lookup(head, x, values, storage, interpolation)
            expected = n.summarize(ref, actual, size, storage, interpolation, order, sat)
            if (
                any(row[k] != v for k, v in expected.items())
                or not np.array_equal(values, np.load(row["table_reference"]["path"]))
                or not np.array_equal(actual, np.load(row["probabilities_reference"]["path"]))
            ):
                return False
            timings = np.load(row["timings_reference"]["path"])
            if (
                timings.shape != (len(x), 6, 2)
                or not np.all(timings > 0)
                or row["direct_ns_per_vector_median"] != float(np.median(timings[:, 1:, 0]))
                or row["table_ns_per_vector_median"] != float(np.median(timings[:, 1:, 1]))
            ):
                return False
            for i, p in enumerate(actual):
                expected_vector = dict(
                    configuration_id=row["configuration_id"],
                    vector_id=i,
                    kind=kinds[i],
                    reference_probability=float(ref[i]),
                    direct_probability=float(direct[i]),
                    table_probability=float(p),
                    probability_error=float(abs(p - ref[i])),
                    reference_action=kernel.action(float(ref[i])),
                    table_action=kernel.action(float(p)),
                    action_flip=kernel.action(float(ref[i])) != kernel.action(float(p)),
                    near_threshold=bool(min(abs(ref[i] - 0.25), abs(ref[i] - 0.75)) < 0.002),
                    timing_ns=timings[i].tolist(),
                )
                if json.loads(next(stream)) != expected_vector:
                    return False
        if stream.read():
            return False
    stored = json.loads((raw / "refresh_rows.json").read_bytes())["rows"]
    finals = json.loads((raw / "final_tables.json").read_bytes())["rows"]
    if len(stored) != 384 or len(finals) != 6:
        return False
    for group, final in enumerate(finals):
        size, storage = final["grid_points"], final["storage"]
        if (size, storage) != [(s, t) for s in [65, 257, 1025] for t in ["float64", "int16"]][
            group
        ]:
            return False
        current = head
        values, _ = n.table(current, size, storage)
        for index, xrow in enumerate(n.events()):
            row = stored[group * 64 + index]
            updated = n.update(current, xrow, index % 2)
            phi = np.asarray(kernel.scalar_design(xrow.tolist())[2:])
            gradient = (
                (float(n.reference(current, xrow[None])[0]) - index % 2) / head["temperature"] * phi
            )
            gradient *= min(1.0, 1.0 / max(float(np.linalg.norm(gradient)), 1e-300))
            oracle = np.clip(np.asarray(current["coefficients"])[2:] - 0.01 * gradient, -4, 4)
            if not np.allclose(oracle, np.asarray(updated["coefficients"])[2:], rtol=0, atol=1e-14):
                return False
            scoped, entries = n.refresh(current, updated, values, storage)
            full, _ = n.table(updated, size, storage)
            if (
                row["x"] != xrow.tolist()
                or row["target"] != index % 2
                or row["coefficients"] != updated["coefficients"]
                or row["affected_entries"] != entries
                or row["identical_bytes"] is not True
                or full.tobytes() != scoped.tobytes()
            ):
                return False
            if len(row["timings"]) != 6 or any(
                t["repetition"] != j - 1
                or t["warmup"] != (j == 0)
                or any(
                    t[k] <= 0
                    for k in [
                        "local_update_ns",
                        "full_refresh_ns",
                        "scoped_refresh_ns",
                        "full_serialization_ns",
                        "scoped_serialization_ns",
                    ]
                )
                for j, t in enumerate(row["timings"])
            ):
                return False
            current, values = updated, scoped
        if final["head"] != current or final["table_hex"] != values.tobytes().hex():
            return False
    original, _ = n.table(head, 65, "float64")
    updated = n.update(head, n.events()[0], 0)
    missed, _ = n.refresh(head, updated, original, "float64", miss=True)
    return bool(
        work["refresh_passed"] and missed.tobytes() != n.table(updated, 65, "float64")[0].tobytes()
    )


def replay(path: Path) -> bool:
    """A repaired checksum cannot authorize altered numerical values or unsafe readiness."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["work_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        work = json.loads(Path(ref["path"]).read_bytes())
        raw = Path(ref["path"]).parent
        refs = [
            *work.get("inputs", {}).get("refs", []),
            *work.get("code_refs", []),
            *work.get("primitive_refs", []),
            *[
                row[k]
                for row in work.get("configurations", [])
                for k in ["table_reference", "probabilities_reference", "timings_reference"]
            ],
            work["manifest_reference"],
        ]
        if any(sha256_file(Path(r["path"])) != r["sha256"] for r in refs):
            return False
        for receipt in value["validation_receipts"]:
            for label in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]:
                    return False
        if work.get("configurations") and not replay_numeric(work, raw):
            return False
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        return bool(
            canonical_hash(value)
            == canonical_hash(build(work, value["validation_receipts"], raw, output))
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def write_note(value: Json, path: Path) -> None:
    """Write the measured engineering limits before the primary becomes reader-visible."""
    rows = [
        "# Frozen spline table fidelity, 2026-10-09",
        "",
        "This constructed numerical study uses the original Exp8334 checkpoint and V717 rule.",
        "Current model loads and generation calls are zero. H1/H2 remain unmeasured.",
        "Global slope, intercept, temperature, interpolation and sigmoid remain float64.",
        "The panel has4096 random vectors,72 knot controls and6 action boundary controls.",
        "One warmup and five paired repetitions do not add independent scientific sources.",
        "",
        f"Verdict: {value['honest_verdict']}. Measurement readiness: {value['table_fidelity_ready_score']}.",
        f"Candidate score: {value['table_candidate_score']}. Direct probability parity: {value['direct_parity']}.",
        "",
        "| Configuration | Max probability error | Flips | Boundary flips | Saturation | Bytes | Direct ns/vector | Table ns/vector |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in value["configurations"]:
        rows.append(
            f"| {row['configuration_id']} | {row['probability_error_max']:.12g} | {row['action_flip_count']} | {row['boundary_flip_count']} | {row['saturation_count']} | {row['table_bytes']} | {row['direct_ns_per_vector_median']:.0f} | {row['table_ns_per_vector_median']:.0f} |"
        )
    rows += [
        "",
        "All per-vector errors and paired timings remain in the byte-bound JSONL evidence.",
        "All six storage/grid trajectories retain64 updates, support entry lists and six timing samples per update.",
        "Full and scoped refresh require identical encoded bytes after every update and fresh-process restart.",
        "A deliberately missed entry must fail. Update, refresh and byte serialization timings are separate.",
        "Candidate gates are maximum probability error<=.001 and no flips outside a .002 action margin.",
        "Near-threshold flips are retained. Candidate choice uses bytes, then error, then frozen order.",
        "This is circular_positive engineering evidence and provides no natural accuracy or utility claim.",
        "The board operation manifest creates no RTL and makes no device speed, resource or energy claim.",
        "[Protocol](v720-table-protocol.md) binds the methods, source versions and fixed adaptation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")
