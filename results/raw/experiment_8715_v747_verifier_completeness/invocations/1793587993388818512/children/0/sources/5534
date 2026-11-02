"""REQ-REPORT-8530: qualify an accounting adapter while keeping old failures immutable."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import brier_online_learning_8517 as old
from carnot.reporting import objective_publication_8516 as upstream
from carnot.reporting import v733_contract_methods as methods
from carnot.reporting import v733_update_fixture as fixture
from carnot.reporting.current_work_receipt import atomic_json, sha256_file, ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v733_source_diagnostics import (
    DIAGNOSTIC_FIELDS as DIAGNOSTIC_FIELDS,
    diagnostic_rows,
)
from carnot.reporting.v733_update_metrics import (
    aggregate as aggregate,
    coordinate_row as coordinate_row,
    historical_accounting as historical_accounting,
    HISTORICAL_PIN as HISTORICAL_PIN,
)

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = (
    "experiment_8530_v733_update_accounting_qualification",
    "exp8530-update-accounting-qualification",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v733_update_accounting_8530.py"
OWNED = [
    f"python/carnot/reporting/v733_{s}.py"
    for s in [
        "update_metrics",
        "update_fixture",
        "update_accounting",
        "update_accounting_runner",
        "update_validation",
    ]
] + [CLI]
MODEL_SPECS: list[Json] = []
UPSTREAM_TASK = upstream.TASK
METHODS_PIN = "sha256:5185ba15b6e23c5a8185553fe4c878921255768bd7d12d88ebf16202d3f5c004"
read, reference, checksum = old.read, old.reference, old.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries because no GPU or model activity can reveal a stalled reader."""
    print(f"[exp8530] phase={phase} completed={completed} pending={pending}", flush=True)


def diagnostics(rows: list[Json]) -> list[Json]:
    """Normalize only this experiment's copy, preserving historical rows and thresholds."""
    return list(
        diagnostic_rows(
            [dict(row, upstream=row.get("upstream", row.get("upstream_id", TASK))) for row in rows]
        )
    )


def inspect_adapter(root: Path, raw: Path, private: Path) -> Json:
    """Retain the historical inspect result while making every failure safe to name.

    This is a local reporting adapter. It does not change the original producer,
    original assertion or the zero learning qualification observation.
    """
    result = dict(old.inspect(root, raw, private))
    result["failures"] = diagnostics(result["failures"])
    return result


def accounting_controls() -> list[Json]:
    """Fixed independent operands expose assignment and decay counting mistakes."""
    return [
        coordinate_row(
            dict(parameters=[1.0, 2.0, 0.0], decay_scale=1.0),
            after,
            writes,
            event=name,
            namespace="control",
        )
        for name, after, writes in [
            ("zero_gradient", dict(parameters=[1.0, 2.0, 0.0], decay_scale=1.0), [0, 1]),
            ("repeated_coordinate", dict(parameters=[2.0, 2.0, 0.0], decay_scale=1.0), [0, 0]),
            ("decay_only", dict(parameters=[1.0, 2.0, 0.0], decay_scale=0.9), []),
        ]
    ]


def qualification_diagnostics(path: Path, pin: str, sidecar: Path) -> list[Json]:
    """Name the actual failed operands; a missing observation never means zero."""
    failures = []
    try:
        value = json.loads(path.read_bytes())
        digest = sha256_file(path)
        for field, expected, observed in [
            ("sha256", pin, digest),
            ("task_id", UPSTREAM_TASK, value.get("task_id")),
            (
                "learning_qualification_ready_score",
                1,
                value.get("learning_qualification_ready_score"),
            ),
            ("required_checks_passed", True, value.get("required_checks_passed")),
            ("flagged_adversarial", False, value.get("flagged_adversarial")),
        ]:
            if expected != observed:
                failures.append(old.gate(path, digest, field, expected, observed, UPSTREAM_TASK))
        bound = read_bound_sidecar(path, sidecar)
        if bound["report"]["passed"] is not True:
            failures.append(
                old.gate(
                    sidecar,
                    sha256_file(sidecar),
                    "bound_publication",
                    True,
                    bound["report"]["passed"],
                    UPSTREAM_TASK,
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            old.gate(
                path,
                pin,
                "qualified_update_adapter",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                UPSTREAM_TASK,
            )
        )
    return diagnostics(failures)


def measure(root: Path, raw: Path) -> Json:
    """Authenticate current and sealed evidence before the forty-eight-update oracle fixture."""
    progress("preconditions")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    resources = dict(
        methods.base.legacy.resources(raw),
        free_disk_bytes=shutil.disk_usage(raw).free,
        tools={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        child_heartbeat_s=30,
        optional_hardware_required=False,
    )
    authority = methods.authority(root, raw / "authority")
    failures = diagnostics(authority["gate_check_summary"])
    task: Json = next((t for t in authority["tasks"] if t["id"] == TASK), {})
    if task.get("deliverable") != f"results/{NAME}.json":
        failures.append(
            old.gate(
                root / methods.ACTIVE,
                None,
                "task_deliverable",
                f"results/{NAME}.json",
                task.get("deliverable"),
                TASK,
            )
        )
    if (
        not resources["private_disk_backed_scratch"]
        or not all(resources["tools"].values())
        or resources["available_memory_bytes"] < resources["minimum_memory_bytes"]
        or resources["free_disk_bytes"] < 104857600
    ):
        failures.append(old.gate(raw, None, "memory_disk_tools", True, resources, TASK))
    refs = []
    for relative, pin, fields in [
        (
            f"results/{methods.NAME}.json",
            METHODS_PIN,
            ["methods_ready_score", "learning_inputs_ready_score"],
        )
    ]:
        path = root / relative
        try:
            value = read(dict(path=str(path), sha256=pin))
            refs.append(reference(path))
            for field in fields:
                if value.get(field) != 1:
                    failures.append(old.gate(path, pin, field, 1, value.get(field), methods.TASK))
        except (OSError, ValueError) as error:
            failures.append(
                old.gate(
                    path,
                    pin,
                    "methods_learning_inputs",
                    1,
                    None if isinstance(error, FileNotFoundError) else str(error),
                    methods.TASK,
                )
            )
    primary = root / "results" / (upstream.NAME + ".json")
    sidecar = ROOT / "results/raw" / upstream.NAME / "terminal_validation.json"
    terminal = json.loads(sidecar.read_bytes())
    failures += qualification_diagnostics(
        primary, old.UPSTREAM_PIN, Path(terminal["publication"]["sidecar_path"])
    )
    qualified = upstream.consumer(
        primary, Path(terminal["publication"]["sidecar_path"]), replay=False
    )
    if qualified["passed"]:
        sealed = methods.read(
            dict(path=str(ROOT / "results" / (methods.NAME + ".json")), sha256=METHODS_PIN)
        )
        original_request = (
            Path(sealed["primitive_reference"]["path"]).parent / "historical/request-8516.json"
        )
        request = json.loads(original_request.read_bytes())
        request["primary"] = dict(
            path=str(primary), snapshot_path=str(primary), sha256=old.UPSTREAM_PIN
        )
        atomic_json(raw / "upstream_request.json", request)
        receipt = child(
            "upstream_cold_replay",
            [
                sys.executable,
                "-u",
                str(ROOT / methods.CLI),
                "--history-request",
                str(raw / "upstream_request.json"),
                "--history-output",
                str(raw / "upstream_response.json"),
            ],
            raw / "upstream_consumer",
            deadline=180,
        )
        observed = (
            json.loads((raw / "upstream_response.json").read_bytes()) if receipt["passed"] else {}
        )
        qualified.update(
            passed=receipt["passed"] and observed.get("qualified") is True,
            receipts=[receipt],
            sealed_family_receipt=observed,
        )
    if not qualified["passed"]:
        failures.append(
            old.gate(
                primary,
                old.UPSTREAM_PIN,
                "original_consumer",
                True,
                qualified["passed"],
                UPSTREAM_TASK,
            )
        )
    progress("historical_accounting")
    history = historical_accounting()
    atomic_json(raw / "historical_accounting.json", history)
    cfg = fixture.configuration()
    atomic_json(raw / "fixture_configuration.json", cfg)
    argv = [
        sys.executable,
        "-u",
        str(ROOT / CLI),
        "--configuration",
        str(raw / "fixture_configuration.json"),
    ]
    receipts = []
    for name, extras, expected in [
        ("baseline", [], 0),
        ("restart", ["--crash"], 73),
        ("resume", [], 0),
    ]:
        directory = raw / ("baseline" if name == "baseline" else "restart")
        receipts.append(
            child(
                name,
                [*argv, "--fixture-worker", str(directory), *extras],
                raw / "fixtures",
                expected=expected,
                deadline=60,
            )
        )
    rows = read(reference(raw / "baseline/accounting.json"))
    restart = read(reference(raw / "baseline/final.json")) == read(
        reference(raw / "restart/final.json")
    ) and rows == read(reference(raw / "restart/accounting.json"))
    controls = accounting_controls()
    work = dict(
        root=str(root),
        raw=str(raw),
        authority=authority,
        preconditions=resources,
        failures=failures,
        source_refs=refs + history["source_refs"] + [reference(primary)]
        if primary.exists()
        else refs + history["source_refs"],
        code_refs=[reference(ROOT / p) for p in OWNED],
        history_reference=reference(raw / "historical_accounting.json"),
        fixture_rows=rows,
        control_rows=controls,
        restart_passed=restart,
        upstream_consumer=qualified,
        process_receipts=receipts,
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        adversarial_findings=[],
    )
    original = read(dict(path=str(ROOT / "results" / (old.NAME + ".json")), sha256=HISTORICAL_PIN))
    original_logs = [
        Path(r["stdout_path"])
        for r in original["validation_receipts"]
        if r["name"] in {"owned_tests", "historical_union", "adversarial"}
    ]
    original_logs += [Path(original["terminal_validation_sidecar_path"])]
    work["original_failure_references"] = [reference(p) for p in original_logs]
    work["source_refs"] += [
        dict(path=s["snapshot_path"], sha256=s["sha256"])
        for s in authority.get("authority_snapshots", {}).values()
        if s.get("snapshot_path")
    ]
    work["primitive_files"] = [
        reference(p) for p in sorted(raw.rglob("*.json")) if p.name != "measurement.json"
    ]
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", 3, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A valid failure artifact preserves findings and cannot open the learning gate."""
    from carnot.reporting.v733_update_validation import union_receipt

    history = read(work["history_reference"])
    totals = aggregate(history["rows"])
    union = union_receipt(work.get("collected_node_ids", []), receipts)
    required = {
        "owned_tests",
        "coverage_report",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    }
    checks = (
        required <= {r["name"] for r in receipts if r["passed"]}
        and all(r["passed"] for r in receipts + work["process_receipts"])
        and union["passed"]
    )
    rows = [
        dict(
            unit_id=f"fixture:{i}",
            family="constructed_oracle",
            status="completed",
            absolute_metric=r["changed_coordinate_count"],
            raw_numerator=r["changed_coordinate_count"],
            raw_denominator=len(r["before"]["parameters"]),
            **r,
        )
        for i, r in enumerate(work["fixture_rows"])
    ]
    gates = dict(
        diagnostics=not work["failures"],
        arithmetic=history["reconciliation_passed"],
        restart=work["restart_passed"],
        historical_union=union["passed"],
        owned_checks=checks,
        original_consumer=work["upstream_consumer"]["passed"],
        unchanged_adversarial_checks=not work["adversarial_findings"],
    )
    ready = all(gates.values())
    verdict = (
        "disqualified"
        if not checks
        or work["adversarial_findings"]
        or not gates["arithmetic"]
        or not gates["restart"]
        else "blocked"
        if work["failures"]
        else "circular_positive"
    )
    failures = diagnostics(work["failures"])
    failures += [
        old.gate(
            Path(r["stdout_path"]), r["stdout_sha256"], "required_owned_check", True, False, TASK
        )
        for r in receipts
        if not r["passed"]
    ]
    terminal = output.parent / "raw" / output.stem / "terminal_validation.json"
    value: Json = dict(
        experiment_id=8530,
        task_id=TASK,
        milestone="2026.10.733",
        run_date="20261015",
        honest_verdict=f"complete_{verdict}_v733_update_accounting_qualification",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[
            dict(
                scope="imported exposed V726/V732 candidate evidence only",
                original_producer_hashes=history["source_refs"],
            )
        ],
        rows=rows,
        intended_count=48,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            constructed_updates=48, original_updates_replayed=1800, scientific_streams_trained=0
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed oracle fixtures and exposed historical development evidence",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(checks and not work["adversarial_findings"]),
        flagged_adversarial=bool(work["adversarial_findings"]),
        acceptance_gates=gates,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(terminal),
        adversarial_findings=work["adversarial_findings"],
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(name=r["name"], duration_s=r["duration_s"])
            for r in receipts + work["process_receipts"]
        ],
        random_seed=7338530,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["primitive_files"],
        cited_upstream_artifacts=work["source_refs"],
        online_adapter_ready_score=int(ready),
        accounting_schema="carnot.causal_coordinate_accounting.v1",
        metric_reconciliation_rows=history["metric_reconciliation_rows"],
        **totals,
        coefficient_change_count=totals["changed_coordinate_count"],
        label_updates=totals["applied_update_events"],
        accounting_scope="replay of original committed Exp8517 primitive events; no new scientific training",
        fixture_accounting=aggregate(work["fixture_rows"]),
        accounting_controls=work["control_rows"],
        original_failure_references=work["original_failure_references"],
        collected_node_ids=work.get("collected_node_ids", []),
        shard_receipts=[r for r in receipts if r.get("collected_node_ids")],
        historical_union_receipt=union,
        downstream_consumer_receipt=dict(
            scope="exact final bytes",
            receipt_path=str(terminal),
            prepublication_fixture_passed=work["restart_passed"],
        ),
        upstream_consumer_receipt=work["upstream_consumer"],
        primitive_reference=reference(raw / "measurement.json"),
        historical_accounting_reference=work["history_reference"],
        output_path=str(output),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation") for n in [8531, 8532]
        ],
        methodology_note="No model loads. Original state/cost replay separates assignments, stored coordinates and scaled effective coefficients. Forty-eight constructed updates test causal release, conflicting redelivery and real exit73/resume. Exposed history and oracle fixtures cannot establish learning benefit. Original quarantine and stale sidecars remain immutable.",
    )
    value["field_principles"] = {
        k: "Bound to actual invocation and immutable primitive operands; qualification grants no scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        physical_coordinate_writes="Sum of sparse coefficient assignments, including repeated indices and zero gradients; checkpoint serialization excluded.",
        changed_coordinate_count="Sum of unequal serialized parameter values per committed event, independently compared before/after.",
        unique_changed_coordinates="Union of seed/arm/coordinate identities, not a sum of repeated changes.",
        decay_scale_changes="Changes to shared lazy-decay multiplier, separate from physical coefficient assignments.",
        effective_coefficient_changes="Sum of changes to parameter times decay_scale, including logical regularization effects.",
        applied_update_events="Number of unique committed numerical update events reconciled against primitive state and independent cost rows.",
        coefficient_change_count="Compatibility metric corrected to actual changed stored coordinates, retained to audit the old1800 error.",
        label_updates="Original committed update events; retained even when unchanged validators flag its structural equality with applied_update_events.",
        online_adapter_ready_score="Requires complete diagnostics, arithmetic, causal restart fixtures, complete terminal test union, unchanged validators and consumer.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute from independent seals so repairing local summary hashes cannot authorize drift."""
    from tempfile import TemporaryDirectory

    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for receipt in value["validation_receipts"] + work["process_receipts"]:
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            keys = [
                "name",
                "argv",
                "actual_exit",
                "exit_code",
                "passed",
                "timed_out",
                "duration_s",
                "stdout_sha256",
                "stderr_sha256",
            ]
            if any(saved[k] != receipt[k] for k in keys):
                return False
            if any(
                sha256_file(Path(receipt[k + "_path"])) != receipt[k + "_sha256"]
                for k in ["stdout", "stderr"]
            ):
                return False
        for ref in [
            *work["source_refs"],
            *work["code_refs"],
            *work["primitive_files"],
            *work["original_failure_references"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        history = read(work["history_reference"])
        if history != historical_accounting():
            return False
        if work["control_rows"] != accounting_controls():
            return False
        with TemporaryDirectory(prefix="exp8530-replay-", dir="/var/tmp") as directory:
            fixture.worker(fixture.configuration(), Path(directory))
            if (
                json.loads((Path(directory) / "accounting.json").read_bytes())
                != work["fixture_rows"]
            ):
                return False
        expected = build(
            work, value["validation_receipts"], Path(work["raw"]), Path(value["output_path"])
        )
        return bool(value == expected)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def stream_consumer(primary: Path, sidecar: Path) -> Json:
    """Exercise the actual scalar learning gate and reject stale or unqualified bytes."""
    try:
        value = json.loads(primary.read_bytes())
        bound = read_bound_sidecar(primary, sidecar)
        selected = reader_receipt(
            TASK, primary.parent, field="online_adapter_ready_score", expected=1
        )
        passed = (
            value["task_id"] == TASK
            and value["online_adapter_ready_score"] == 1
            and value["required_checks_passed"] is True
            and value["flagged_adversarial"] is False
            and bound["report"]["passed"] is True
            and selected["passed"]
            and replay(primary)
        )
        return dict(
            passed=bool(passed),
            selected=selected,
            primary_sha256=sha256_file(primary),
            consumer="Exp8531 scalar gate plus bound primitive accounting replay",
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, error=str(error))
