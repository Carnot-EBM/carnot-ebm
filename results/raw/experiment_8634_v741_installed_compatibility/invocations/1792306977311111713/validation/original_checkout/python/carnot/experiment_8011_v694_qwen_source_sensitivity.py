"""REQ-REPORT-8011: source dependence from owned, bounded Qwen judgments.

A changed risk score shows source dependence, without proving correctness.
Duplicate calls estimate repeat instability on the same frozen public inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot import experiment_8010_v694_source_intervention_protocol as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
c, legacy, ROOT = p.c, p.upstream.legacy, p.ROOT
NAME = "experiment_8011_v694_qwen_source_sensitivity"
TASK = "exp8011-qwen-source-sensitivity"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/{NAME}.py", CLI]
TEST = "tests/python/test_source_sensitivity_8011.py"
MODEL_SPECS = [c.risk.MODEL]
ANALYSIS_METHODS = dict(
    p.METHODS,
    net_lower_CI_gate=0,
    placebo_definition="mean_absolute_paired_difference",
    tokenizer_reserve_s=20,
    task_cap_s=4800,
)
CAPTURE_DEFAULTS: Json = dict(
    model_identity_receipt={},
    gguf_sha256=None,
    model_revision=None,
    gpu_lease_receipt={},
    cleanup={},
    runtime_receipts=[],
    capacity_receipt=None,
    server_log=None,
)
PIN = "sha256:59c6d2e667f853dae1de918bf00c1b1d621b01f8925c567012e68c1a7d180907"
reference, operand = p.upstream.reference, p.upstream.operand


def progress(phase: str, started: float, units: int = 0, pending: int = 0) -> None:
    """Expose real work and pending units without padding measured durations."""
    print(
        f"[exp8011] phase={phase} elapsed_s={time.monotonic() - started:.3f} units={units} pending={pending}",
        flush=True,
    )


def authenticate(root: Path) -> Json:
    """Pin prepared inputs before outcomes can select a different source panel."""
    path = root / "results" / (p.NAME + ".json")
    plan: Json = dict(panel={}, checks=[], references=[], protocol={})
    plan["checks"].append(
        operand(8010, path, "sha256", PIN, sha256_file(path) if path.is_file() else "missing_file")
    )
    if not path.is_file():
        return plan
    value = json.loads(path.read_text())
    for key, expected in dict(
        experiment_id=8010, run_date="20261002", protocol_ready_score=1, flagged_adversarial=False
    ).items():
        plan["checks"].append(
            operand(8010, path, key, expected, value.get(key, "missing_field_contract_error"))
        )
    p.replay(value)
    panel = json.loads(Path(value["panel_reference"]["path"]).read_text())
    plan["checks"].append(operand(8010, path, "request_budget", p.METHODS, panel["methods"]))
    upstream = p.authenticate(root)
    plan.update(panel=panel, protocol=upstream.get("protocol", {}))
    plan["checks"] += upstream["checks"]
    plan["references"] = upstream["references"] + [
        dict(
            reference(path),
            producer_id=8010,
            execution_date="20261002",
            imported_fields=["panel_reference", "request_budget", "protocol_ready_score"],
            scope="historical",
        )
    ]
    return plan


def live(panel: Json, plan: Json, raw: Path) -> Json:
    """Reuse the qualified lease and CUDA runtime with the frozen 32-token calls.

    The old capture helper reserves 96 tokens per start. Its reservation ledger
    needs 192 slots; actual requests still cap total output at 192 times 32.
    """
    raw.mkdir(parents=True, exist_ok=True)
    runtime_directory = raw / "runtime"
    runtime_directory.mkdir(parents=True, exist_ok=True)
    ledger = c.Ledger(raw / "ledger.json")
    runtime_class = legacy.QwenRuntime

    class RecordedRuntime(runtime_class):
        def count(self, text: str) -> int:
            """Bound both tokenizer HTTP requests within the reserved admission time."""
            return int(legacy.bounded(lambda: super(RecordedRuntime, self).count(text), 20))

        def load(self) -> Json:
            """Save the actual load start before a native process can fail."""
            ledger.start("model_load", "owned-model-load", dict(model=str(self.model)))
            try:
                result = dict(super().load())
            except (OSError, RuntimeError, TimeoutError, ValueError):
                ledger.finish("owned-model-load", "failed", {})
                raise
            ledger.finish("owned-model-load", "completed", result)
            return result

    adapter = SimpleNamespace(
        freeze=lambda _: panel["slots"],
        capture=partial(c.capture, ledger=ledger, deadline_s=2380, token_budget=192 * 96),
    )
    with (
        patch.object(legacy, "TASK", TASK),
        patch.object(legacy, "capture", adapter),
        patch.object(legacy, "load_public", lambda _: {}),
        patch.object(legacy, "QwenRuntime", RecordedRuntime),
        patch.object(
            legacy,
            "progress",
            lambda phase, started, units=0: progress(
                phase,
                started,
                ledger.counts()["generation_calls_completed"],
                192 - ledger.counts()["generation_calls_attempted"],
            ),
        ),
    ):
        result = legacy.live_capture(
            dict(plan, public_role_manifests={}, capture_identity=canonical_hash(panel)),
            raw,
            runtime_directory,
        )
    # Seal nonstarts too, including when a cache or lease precondition failed.
    result["rows"] = c.capture(
        panel["slots"], None, raw / "slots", canonical_hash(panel), ledger=ledger, deadline_s=0
    )
    result["ledger"] = ledger.rows
    return dict(result)


def reduce(rows: list[Json]) -> Json:
    """Resample independent source triplets while retaining every invalid slot."""
    groups: Json = {}
    roles = {
        arm: dict(
            intended=0,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=0,
            independent=0,
        )
        for arm in p.ARMS
    }
    censor, parse_failures, tokens = [], 0, 0
    for row in rows:
        arm, group = row["arm"], row["group_id"]
        cells = groups.setdefault(group, {})
        if arm not in roles or arm in cells or row["family_id"] != group + ":" + arm:
            raise ValueError("slot_identity")
        cells[arm] = row
        parsed = c.risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        if parsed != row["parsed"]:
            raise ValueError("parse_drift")
        output = row["raw_response"].get("usage", {}).get("completion_tokens", 0)
        if output > 32:
            raise ValueError("decode_budget")
        tokens += output
        complete = bool(parsed["completed"])
        additions = dict(
            intended=1,
            eligible=int(row["public_eligible"]),
            started=int(row["started"]),
            completed=int(complete),
            excluded=int(row["status"] == "excluded"),
            failed=int(row["status"] == "failed" or row["started"] and not complete),
            censored=int(row["status"] == "censored"),
            independent=int(complete),
        )
        for key, amount in additions.items():
            roles[arm][key] += amount
        parse_failures += int(row["status"] == "generated" and not complete)
        if not complete:
            censor.append(
                dict(
                    id=row["family_id"],
                    group_id=group,
                    arm=arm,
                    status=row["status"],
                    reason=row.get("error")
                    or row["exclusion_reason"]
                    or (
                        "inference_budget_or_precondition_nonstart"
                        if not row["started"]
                        else parsed["status"]
                    ),
                )
            )
    sensitivity, duplicate, deltas = [], [], []
    for group, cells in groups.items():
        complete = set(cells) == set(p.ARMS) and all(
            r["parsed"]["completed"] for r in cells.values()
        )
        swap = placebo = None
        if complete:
            original, changed, repeated = [cells[a]["parsed"]["probability"] for a in p.ARMS]
            swap, placebo = changed - original, repeated - original
            deltas.append([swap, placebo, swap - placebo])
        sensitivity.append(
            dict(
                group_id=group,
                complete=complete,
                swap_minus_original=swap,
                swap_minus_duplicate=None if swap is None else swap - placebo,
            )
        )
        duplicate.append(dict(group_id=group, complete=complete, duplicate_minus_original=placebo))
    intervals: Json = {}
    for i, name in enumerate(
        ["swap_minus_original", "duplicate_minus_original", "swap_minus_duplicate"]
    ):
        entry = dict(mean=None, lower95=None, upper95=None, independent=len(deltas))
        if len(deltas) >= 48:
            values = np.asarray(deltas)[:, i]
            with patch.object(c.risk, "SEED", 69410):
                interval, _ = c.risk.intervals(values)
            entry.update(mean=float(values.mean()), lower95=interval[0], upper95=interval[1])
        intervals[name] = entry
    counts = {k: sum(r[k] for r in roles.values()) for k in roles["original"]}
    counts["independent"] = len(deltas)
    return dict(
        sample_size_budget=dict(
            counts, intended=192, unit="source_view_call", independent_unit="original_source_group"
        ),
        role_completion_counts=roles,
        censor_rows=censor,
        generated_tokens=tokens,
        parse_failure_count=parse_failures,
        sensitivity_rows=sensitivity,
        duplicate_control_rows=duplicate,
        paired_confidence_intervals=intervals,
        placebo_instability=dict(
            absolute_mean=float(np.abs(np.asarray(deltas)[:, 1]).mean())
            if len(deltas) >= 48
            else None,
            changed_triplets=sum(d[1] != 0 for d in deltas),
            independent=len(deltas),
        ),
    )


def build(
    panel: Json,
    plan: Json,
    result: Json,
    raw: Path,
    receipts: list[Json],
    elapsed: float,
    fixture: bool,
) -> Json:
    """Separate valid measurement readiness from a positive scientific effect."""
    ledger = c.Ledger(raw / "ledger.json")
    ledger.rows = [] if fixture else result.get("ledger", [])
    rows = result.get("rows", [])
    value = c.provenance(ledger, rows, live=bool(ledger.rows) and not fixture)
    if not fixture and not ledger.rows:
        value["inference_substrate_class"] = "blocked_no_run"
    value.update(reduce(rows))
    failures = [r for r in plan["checks"] + result.get("checks", []) if not r["passed"]]
    owned_ok = all(r["passed"] for r in receipts if r["scope"] == "owned")
    measured = result.get("measured_duration_s", 0)
    valid = (
        not failures
        and owned_ok
        and measured >= 10
        and ledger.counts()["model_loads_completed"] == 1
    )
    ready = int(valid and value["sample_size_budget"]["independent"] >= 48 and not fixture)
    ci = value["paired_confidence_intervals"]
    benefit = bool(
        ready
        and ci["swap_minus_original"]["lower95"] > 0
        and ci["swap_minus_duplicate"]["lower95"] > 0
        and value["placebo_instability"]["absolute_mean"] <= 0.01
    )
    verdict = (
        "disqualified"
        if not owned_ok
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "disqualified"
        if not valid
        else "positive"
        if benefit
        else "null"
    )
    if not ledger.rows and not fixture and verdict != "blocked":
        value["inference_substrate_class"] = "no_model_load"
    value.update(
        experiment_id=8011,
        task_id=TASK,
        milestone="2026.10.694",
        run_date="20261002",
        execution_date="20261002",
        schema="carnot.exp8011.source_sensitivity.v1",
        honest_verdict=f"complete_{verdict}_source_sensitivity",
        verdict_class=verdict,
        claim_scope="Frozen development source dependence only. No truth labels, panel training, hallucination correctness or general mitigation claim.",
        MODEL_SPECS=[] if fixture else MODEL_SPECS,
        model_specs=[] if fixture else MODEL_SPECS,
        trained_head_specs=[],
        duration_s=elapsed,
        phase_spans=[dict(phase="owned_capture_and_validation", start_s=0, end_s=elapsed)],
        measured_duration_s=measured,
        rows=[
            dict(
                r,
                id=r["family_id"],
                metric="valid_risk_judgment",
                probability=r["parsed"]["probability"],
                exclusion=r["exclusion_reason"],
                censor_reason=next(
                    (x["reason"] for x in value["censor_rows"] if x["id"] == r["family_id"]), None
                ),
            )
            for r in rows
        ],
        request_rows=[
            dict(
                request_id=r["family_id"],
                request=r["request"],
                request_sha256=canonical_hash(r["request"]),
                response_id=r["raw_response"].get("id"),
                model_id=r["raw_response"].get("model"),
            )
            for r in rows
        ],
        random_seed=69410,
        reproducibility_checksum=canonical_hash(dict(panel=panel, rows=rows)),
        protocol_fingerprint=canonical_hash(panel),
        request_budget=ANALYSIS_METHODS,
        verifier_is_oracle=False,
        genuine_headroom=None,
        positive_control_results=dict(scope="circular_Exp8010_HTTP_transport_only", passed=True),
        acceptance_gate_results=dict(
            measurement_valid=valid,
            source_comparison=bool(ready),
            source_dependence=benefit,
            hallucination_mitigation=False,
        ),
        source_sensitivity_ready_score=ready,
        gate_check_summary=[dict(r, artifact_field=r["field"]) for r in failures],
        preconditions_checked=plan["checks"] + result.get("checks", []),
        cited_upstream_artifacts=plan["references"],
        panel_reference=reference(raw / "panel.json"),
        result_reference=reference(raw / "capture.json"),
        raw_shard_hashes=[
            reference(raw / "panel.json"),
            reference(raw / "capture.json"),
            reference(raw / "ledger.json"),
            reference(raw / "validation_commands.json"),
            reference(raw / "plan.json"),
            reference(raw / "validation_receipts.json"),
        ],
        checkpoint_references=[reference(path) for path in sorted((raw / "slots").glob("*.json"))],
        code_config_hashes=json.loads((raw / "validation_commands.json").read_text())[
            "code_config_hashes"
        ],
        validation_receipts=[r for r in receipts if r["scope"] == "owned"],
        repository_health=[r for r in receipts if r["scope"] == "repository_health"],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=False,
        model_identity_receipt=result.get("model_identity_receipt", {}),
        gguf_sha256=result.get("gguf_sha256"),
        model_revision=result.get("model_revision"),
        gpu_lease_receipt=result.get("gpu_lease_receipt", {}),
        offload_evidence={
            k: result.get(k)
            for k in [
                "native_binary",
                "resolved_library",
                "resident_gpu_receipt",
                "unloaded_gpu_receipt",
            ]
        },
        cleanup=result.get("cleanup", {}),
        runtime_receipts=result.get("runtime_receipts", []),
        capacity_receipt=result.get("capacity_receipt"),
        server_log=result.get("server_log"),
        calls_before_validation=ledger.counts(),
        calls_after_validation=ledger.counts(),
        evaluation_labels_opened=False,
        fixture_scope="circular_scripted_transport_only" if fixture else None,
    )
    value["field_principles"] = {
        k: "Current calls and independent sources come from durable raw checkpoints. Source dependence does not establish correctness."
        for k in value
    }
    return value


def replay(value: Json) -> None:
    """Cold-reduce original bytes, enforcing request IDs and owned call identity."""
    for ref in (
        value["raw_shard_hashes"] + value["checkpoint_references"] + value["code_config_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("custody_hash_drift")
    panel = json.loads(Path(value["panel_reference"]["path"]).read_text())
    result = json.loads(Path(value["result_reference"]["path"]).read_text())
    for key, default in CAPTURE_DEFAULTS.items():
        if value[key] != result.get(key, default):
            raise ValueError("runtime_receipt_drift:" + key)
    offload = {
        k: result.get(k)
        for k in [
            "native_binary",
            "resolved_library",
            "resident_gpu_receipt",
            "unloaded_gpu_receipt",
        ]
    }
    if value["offload_evidence"] != offload:
        raise ValueError("runtime_receipt_drift:offload_evidence")
    verify_references([result, value["validation_receipts"], value["repository_health"]])
    rows = [json.loads(Path(ref["path"]).read_text()) for ref in value["checkpoint_references"]]
    if rows != result["rows"] or len(rows) != len(panel.get("slots", [])):
        raise ValueError("checkpoint_roster_drift")
    for row, slot in zip(rows, panel.get("slots", []), strict=True):
        if any(row[k] != v for k, v in slot.items()) or row["capture_identity"] != canonical_hash(
            panel
        ):
            raise ValueError("frozen_request_drift")
    if canonical_hash(dict(panel=panel, rows=rows)) != value["reproducibility_checksum"] or value[
        "protocol_fingerprint"
    ] != canonical_hash(panel):
        raise ValueError("protocol_drift")
    ledger = c.Ledger(Path(value["result_reference"]["path"]).parent / "ledger.json")
    if value["current_invocation_ledger"] != ([] if value["fixture_scope"] else result["ledger"]):
        raise ValueError("original_ledger_drift")
    ledger.rows = value["current_invocation_ledger"]
    provenance = c.provenance(
        ledger, rows, live=bool(ledger.rows) and not bool(value["fixture_scope"])
    )
    if not value["fixture_scope"] and not ledger.rows and value["verdict_class"] == "blocked":
        provenance["inference_substrate_class"] = "blocked_no_run"
    if (
        any(value[k] != v for k, v in provenance.items())
        or value["calls_before_validation"] != ledger.counts()
        or value["calls_after_validation"] != ledger.counts()
    ):
        raise ValueError("current_call_drift")
    reduced = reduce(rows)
    if any(value[k] != v for k, v in reduced.items()):
        raise ValueError("reduction_drift")
    raw_ids = [
        dict(
            request_id=r["family_id"],
            request=r["request"],
            request_sha256=canonical_hash(r["request"]),
            response_id=r["raw_response"].get("id"),
            model_id=r["raw_response"].get("model"),
        )
        for r in rows
    ]
    if raw_ids != value["request_rows"]:
        raise ValueError("request_model_ids")
    live_ids = [r["raw_response"].get("id") for r in rows if r["status"] == "generated"]
    if not value["fixture_scope"] and (
        any(not isinstance(x, str) or not x for x in live_ids)
        or len(set(live_ids)) != len(live_ids)
    ):
        raise ValueError("response_identity")
    if any(
        any(r[k] != v for k, v in original.items())
        for r, original in zip(value["rows"], rows, strict=True)
    ):
        raise ValueError("published_rows_drift")
    if value["source_sensitivity_ready_score"] and (
        value["verdict_class"] in {"blocked", "disqualified"}
        or reduced["sample_size_budget"]["independent"] < 48
        or not all(r["passed"] for r in value["validation_receipts"])
    ):
        raise ValueError("unsafe_readiness")
    raw = Path(value["result_reference"]["path"]).parent
    plan = json.loads((raw / "plan.json").read_text())
    receipts = json.loads((raw / "validation_receipts.json").read_text())["receipts"]
    expected = build(
        panel, plan, result, raw, receipts, value["duration_s"], bool(value["fixture_scope"])
    )
    for key in [
        "source_sensitivity_ready_score",
        "verdict_class",
        "honest_verdict",
        "acceptance_gate_results",
        "gate_check_summary",
        "preconditions_checked",
        "MODEL_SPECS",
        "model_specs",
        "measured_duration_s",
        "rows",
        "validation_receipts",
        "repository_health",
        "request_budget",
    ]:
        if value[key] != expected[key]:
            raise ValueError("claim_gate_drift:" + key)


def verify_references(value: Any) -> None:
    """Hash every nested file receipt so owned runtime logs retain custody."""
    if isinstance(value, dict):
        for path_key, hash_key in [("path", "sha256"), ("log_path", "log_sha256")]:
            if path_key in value and hash_key in value:
                path = Path(value[path_key])
                path = path if path.is_absolute() else ROOT / path
                if not path.is_file() or sha256_file(path) != value[hash_key]:
                    raise ValueError("receipt_file_hash:" + str(path))
        for child in value.values():
            verify_references(child)
    elif isinstance(value, list):
        for child in value:
            verify_references(child)


def commands(scratch: Path) -> list[CommandSpec]:
    """Reuse scoped qualification commands without expanding legacy coverage."""
    with patch.object(p, "OWNED", OWNED), patch.object(p, "TEST", TEST):
        specs = p.commands(scratch)
    return [s for s in specs if not s.name.startswith("E2E016")]


def terminal(path: Path) -> Json:
    """Check the exact candidate with unchanged adversarial and row readers."""
    replay(json.loads(path.read_text()))
    specs = [
        CommandSpec(
            name, (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(path)), "terminal", 60
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=path.parent / "raw" / NAME / "terminal_logs"
        if path.parent.name == "results"
        else path.parent / "terminal_logs",
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Run private fixtures or the bounded natural panel, then publish checked bytes."""
    started = time.monotonic()
    progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--fixture", type=Path)
    routes.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed", started)
            return 0
        output = args.output.absolute()
        if args.fixture and output.is_relative_to(ROOT):
            raise ValueError("fixture_publication_must_be_private")
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="carnot-8011-") as directory:
            scratch = Path(directory)
            specs = commands(scratch)
            plan = (
                dict(panel=p.freeze(json.loads(args.fixture.read_text())), checks=[], references=[])
                if args.fixture
                else authenticate(args.root)
            )
            panel = plan["panel"]
            atomic_json(raw / "panel.json", panel)
            code = [
                reference(ROOT / path)
                for path in OWNED
                + [
                    TEST,
                    p.MODULE,
                    "python/carnot/verify/qwen_development_capture_7995.py",
                    "python/carnot/experiment_7969_v691_qwen_calibration_capture.py",
                    "python/carnot/inference/qwen_sufficiency_7920.py",
                    "python/carnot/inference/llama_cpp_process.py",
                    "python/carnot/gpu_lease_phase_journal.py",
                ]
            ]
            for ref in code:
                target = raw / "original_code" / Path(ref["path"]).relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ref["path"], target)
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=[asdict(s) for s in specs],
                    methods=ANALYSIS_METHODS,
                    code_config_hashes=code,
                    protocol_fingerprint=canonical_hash(panel),
                    task_cap_s=4800,
                    transport_reservation_tokens=192 * 96,
                    actual_output_token_budget=6144,
                    cold_argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        CLI,
                        "--cold-replay",
                        str(raw / (NAME + ".json")),
                    ],
                ),
            )
            progress("before_capture", started, 0, 192)
            atomic_json(raw / "plan.json", plan)
            ledger = c.Ledger(raw / "ledger.json")
            ledger.save()
            if args.fixture:
                p.capture_peer(panel, raw / "fixture.json", "normal")
                result = json.loads((raw / "fixture.json").read_text())
                for i, row in enumerate(result["rows"]):
                    atomic_json(raw / "slots" / f"slot-{i:03d}.json", row)
            elif all(r["passed"] for r in plan["checks"]):
                result = live(panel, plan, raw)
            else:
                result = dict(
                    rows=c.capture(
                        panel.get("slots", []),
                        None,
                        raw / "slots",
                        canonical_hash(panel),
                        ledger=ledger,
                        deadline_s=0,
                    ),
                    ledger=[],
                    checks=[],
                )
            atomic_json(raw / "capture.json", result)
            before = c.Ledger(raw / "ledger.json").counts()
            progress("after_capture_before_validation", started, len(result["rows"]))
            receipts = []
            if not args.fixture and args.root.resolve() == ROOT:
                receipts = run_commands(
                    ROOT,
                    specs,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=10,
                    extra_env=dict(
                        CARNOT_8011_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage"),
                        JAX_PLATFORMS="cpu",
                    ),
                )
            after = c.Ledger(raw / "ledger.json").counts()
            if after != before:
                raise ValueError("validation_added_model_calls")
            atomic_json(raw / "validation_receipts.json", dict(receipts=receipts))
            value = build(
                panel, plan, result, raw, receipts, time.monotonic() - started, bool(args.fixture)
            )
            if (scratch / "coverage.json").is_file():
                coverage = json.loads((scratch / "coverage.json").read_text())
                value["coverage_statement_counts"] = dict(
                    coverage["totals"], files=coverage["files"]
                )
                shutil.copy2(scratch / "coverage.json", raw / "coverage.json")
            progress("cold_reduction", started, len(value["rows"]))
            candidate = raw / (NAME + ".json")
            atomic_json(candidate, value)
            cold = CommandSpec(
                "fresh_process_cold_reduction",
                (str(ROOT / ".venv/bin/python"), "-u", CLI, "--cold-replay", str(candidate)),
                "terminal",
                60,
            )
            cold_receipts = run_commands(ROOT, [cold], log_dir=raw / "cold_logs", heartbeat_s=10)
            if not all(r["passed"] for r in cold_receipts):
                raise ValueError("cold_reduction_failed")
            report = terminal(candidate)
            atomic_json(raw / "candidate_validation.json", report)
            if not report["passed"]:
                raise ValueError("terminal_rejected")
            if time.monotonic() - started >= 4800:
                raise TimeoutError("task_cap")
            digest = sha256_file(candidate)
            publication = publish_primary(
                output,
                value,
                lambda path: report if sha256_file(path) == digest else dict(passed=False),
            )
            atomic_json(
                raw / "terminal_validation.json", dict(publication, cold_receipts=cold_receipts)
            )
            published = terminal(output)
            atomic_json(raw / "published_validation.json", published)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="source_sensitivity_ready_score",
                expected=value["source_sensitivity_ready_score"],
            )
            atomic_json(raw / "primary_readers.json", readers)
            if not published["passed"] or not readers["passed"]:
                raise ValueError("published_readers_rejected")
        progress("complete", started, len(value["rows"]))
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"[exp8011] rejected={type(error).__name__}:{error}", flush=True)
        return 1
