"""REQ-REPORT-8348 / REQ-VERIFY-8348: bind delayed learning to original bytes.

Mechanical readiness concerns causal state, not a positive utility result.
The evaluator for H2 and retention remains a separate future dependency.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v720_frozen_input_contract as authority
from carnot.reporting import sentence_spline_fit_8334 as original
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from carnot.reporting.v709_execution import child
from carnot.verify import continuous_local_learning_8348 as k
from carnot.verify.cached_sentence_custody_8305 import check_predictor

Json = dict[str, Any]
ROOT = original.ROOT
NAME = "experiment_8348_v720_continuous_local_learning"
TASK = "exp8348-continuous-local-learning"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_continuous_local_learning_8348.py"
OWNED = [
    "python/carnot/verify/continuous_local_learning_8348.py",
    "python/carnot/reporting/continuous_local_learning_8348.py",
    "python/carnot/reporting/continuous_local_execution_8348.py",
    CLI,
]
PINS = dict(
    original.PINS,
    **{
        "results/experiment_8346_v720_frozen_input_contract.json": "sha256:33f8095a3671eae941a8880550a1b1a5f02f73530c96a41e201388ac9ec81de4",
        "results/experiment_8347_v720_local_consumer_qualification.json": "sha256:0297d25cbdc6d3133a5bc9a93a40d16d2e863b017969687d384e595c94ff7a37",
    },
)
MODEL_SPECS: list[Json] = []
progress = k.progress
reference = original.reference


def require(work: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """A missing operand is recorded as missing rather than numeric zero."""
    row = dict(
        upstream=path.stem,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )
    work["gates"].append(row)
    if not row["passed"]:
        work["failures"].append(row)
        raise ValueError(field)


def bind(work: Json, ref: Json, raw: Path, *, parse: bool = True) -> Json:
    """Copy authenticated operands; opaque label bytes are never JSON-decoded here."""
    path = Path(ref["path"])
    require(work, path, "sha256", ref["sha256"], sha256_file(path) if path.is_file() else None)
    dest = raw / "inputs" / (ref["sha256"][7:] + "-" + path.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    dest.chmod(0o400)
    work["refs"].append(dict(reference(dest), source_path=str(path)))
    return dict(json.loads(dest.read_bytes())) if parse else dict(reference(dest))


def authenticate(root: Path, raw: Path, work: Json) -> Json:
    """Passed byte-bound upstream receipts authorize reuse without new fitting."""
    upstream = {}
    for number, field in [(8346, "frozen_heads_ready_score"), (8347, "local_kernel_ready_score")]:
        name = next(p for p in PINS if p.startswith(f"results/experiment_{number}_"))
        path = root / name
        value = bind(work, dict(path=str(path), sha256=PINS[name]), raw)
        validate_primary(value, path)
        terminal = bind(work, reference(Path(value["terminal_validation_sidecar_path"])), raw)
        side = Path(terminal["publication"]["sidecar_path"])
        report = read_bound_sidecar(path, side)
        bind(work, reference(side), raw)
        for f, want, got in [
            ("terminal_passed", True, report["report"]["passed"]),
            ("required_checks_passed", True, value.get("required_checks_passed")),
            ("flagged_adversarial", False, value.get("flagged_adversarial")),
            (field, 1, value.get(field)),
        ]:
            require(work, path, f, want, got)
        upstream[str(number)] = value
    work["authority"] = authority.authority(root, raw / "authority")
    for snapshot in work["authority"]["authority_snapshots"].values():
        work["refs"].append(
            dict(
                path=snapshot["snapshot_path"],
                sha256=snapshot["sha256"],
                source_path=snapshot["source_path"],
            )
        )
    require(
        work,
        root / "research-roadmap.yaml",
        "exact_task_authority",
        True,
        work["authority"]["activated"],
    )
    source = bind(work, dict(path=str(root / original.SOURCE), sha256=PINS[original.SOURCE]), raw)
    protocol = bind(
        work, dict(path=str(root / original.PROTOCOL), sha256=PINS[original.PROTOCOL]), raw
    )
    checkpoint_ref = upstream["8346"]["frozen_policy"]["checkpoint"]
    checkpoint = bind(work, checkpoint_ref, raw)
    head = next(h for h in checkpoint["heads"] if h["arm"] == "spline34")
    slots = bind(work, source["predictor_shards"]["reserved"], raw)["rows"]
    require(
        work,
        Path(source["predictor_shards"]["reserved"]["path"]),
        "intended_slots",
        128,
        len(slots),
    )
    for row, roster in zip(slots, protocol["original_roles"]["evaluation"], strict=True):
        check_predictor(row)
        if any(row[f] != roster[f] for f in ["unit_id", "source_cluster_id"]):
            raise ValueError("frozen_source_order")
    public = [
        dict(
            slot=p["slot"],
            unit_id=p["unit_id"],
            source_cluster_id=p["source_cluster_id"],
            x=None if p["x"] is None else [p["x"][0], *p["x"][12:16]],
        )
        for p in slots
    ]
    labels = bind(work, source["evaluator_shards"]["reserved"], raw, parse=False)
    fit = bind(work, source["predictor_shards"]["fit"], raw)["rows"]
    work.update(
        heads_sha256=checkpoint_ref["sha256"],
        historical=source["historical_model_provenance"],
        protocol_sha256=PINS[original.PROTOCOL],
    )
    return dict(
        head={f: head[f] for f in ["arm", "coefficients", "temperature", "geometry"]},
        slots=public,
        labels=labels,
        fit=fit,
    )


def measure(root: Path, raw: Path) -> Json:
    """Measure natural issue/release work and actual owned worker deaths."""
    from carnot.reporting.continuous_local_execution_8348 import cli

    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        gates=[],
        failures=[],
        refs=[],
        bundle={},
        state={},
        checks={},
        crash_replay_rows=[],
        child_receipts=[],
        phase_spans=[],
        authority={},
    )
    progress("before_authentication")
    try:
        require(
            work,
            raw,
            "private_resources",
            True,
            raw.stat().st_mode & 0o077 == 0 and shutil.disk_usage(raw).free > 1_000_000_000,
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            require(
                work,
                ROOT / ".venv/bin" / tool,
                "executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        work["bundle"] = authenticate(root, raw, work)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if not work["failures"]:
            work["failures"].append(
                dict(
                    upstream="source_authority",
                    path=str(root),
                    hash=None,
                    artifact_field="authenticated_inputs",
                    op="==",
                    expected=True,
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_authentication", len(work["refs"]), 0)
    if work["bundle"] and not work["failures"]:
        bundle = work["bundle"]
        atomic_json(raw / "bundle.json", bundle)
        work["update_budget_control"] = k.control(bundle["head"])
        work["reachable_action_rows"] = k.reachability(bundle)
        progress("before_benchmark_natural_trajectory")
        for label, crash, mutation in [
            ("uninterrupted", 0, 0),
            ("crash32", 32, 0),
            ("crash64", 64, 0),
            ("future", 0, 49),
        ]:
            directory = raw / label
            argv = cli() + [
                "--worker",
                str(raw / "bundle.json"),
                "--worker-dir",
                str(directory),
                "--crash",
                str(crash),
                "--mutate-from",
                str(mutation),
            ]
            receipt = child(
                label,
                argv,
                raw / "workers",
                deadline=180,
                expected=73 if crash else 0,
                heartbeat=20,
            )
            work["child_receipts"].append(receipt)
            if crash and receipt["passed"]:
                resumed = child(
                    label + "_resume",
                    cli() + ["--worker", str(raw / "bundle.json"), "--worker-dir", str(directory)],
                    raw / "workers",
                    deadline=180,
                    heartbeat=20,
                )
                work["child_receipts"].append(resumed)
        progress("after_benchmark_natural_trajectory", 4, 0)
        if all(r["passed"] for r in work["child_receipts"]):
            state = json.loads((raw / "uninterrupted/final.json").read_bytes())
            work["state"] = state
            future = json.loads((raw / "future/final.json").read_bytes())
            for slot in (32, 64):
                resumed = json.loads((raw / f"crash{slot}/final.json").read_bytes())
                work["crash_replay_rows"].append(
                    dict(
                        slot=slot,
                        passed=resumed == state,
                        uninterrupted_hash=canonical_hash(state),
                        resumed_hash=canonical_hash(resumed),
                        crash_exit=73,
                        checkpoint=reference(raw / f"crash{slot}/checkpoint-{slot}.json"),
                    )
                )
            sparse = [u for u in state["updates"] if u["arm"] == "online_sparse"]
            dense = [u for u in state["updates"] if u["arm"] == "online_dense"]
            error = max(
                abs(a - b)
                for s, d in zip(sparse, dense, strict=True)
                for a, b in zip(s["coefficients"], d["coefficients"], strict=True)
            )
            primitive = k.journal(raw / "uninterrupted/events.jsonl")
            kinds = [(v["kind"], v["slot"]) for v in primitive]
            work["checks"] = dict(
                issue_before_release=all(
                    kinds.index(("issue", s)) < kinds.index(("release", s)) for s in range(9, 97)
                ),
                future_label_invariance=state["issued"][: 57 * 5] == future["issued"][: 57 * 5],
                sparse_dense=error <= 1e-10
                and all(
                    a["action"] == b["action"]
                    for a, b in zip(
                        [v for v in state["issued"] if v["arm"] == "online_sparse"],
                        [v for v in state["issued"] if v["arm"] == "online_dense"],
                        strict=True,
                    )
                ),
                exact_restart=all(v["passed"] for v in work["crash_replay_rows"]),
                complete_accounting=len(state["issued"]) == 480
                and len(state["updates"]) == 440
                and len(state["pending"]) == 8
                and len(state["retention"]) == 640,
                hard_invariants=all(
                    u["coefficients"][0] == bundle["head"]["coefficients"][0]
                    and (
                        u["arm"] == "calibration_only"
                        or u["coefficients"][1] == bundle["head"]["coefficients"][1]
                    )
                    and all(-4 <= c <= 4 for c in u["coefficients"][1:])
                    for u in state["updates"]
                ),
            )
            work["dense_sparse_error_max"] = error
            work["later_source_attribution"] = k.attribution(bundle, state)
            work["costs"] = k.journal(raw / "uninterrupted/costs.jsonl")
            for window in (0, 32, 64, 96):
                atomic_json(
                    raw / f"retention-{window}.json",
                    dict(
                        targets_opened=False,
                        rows=[v for v in state["retention"] if v["window"] == window],
                    ),
                )
        else:
            work["checks"] = dict(worker_exits=False)
    work.update(
        duration_s=time.monotonic() - began,
        preconditions_checked=True,
        phase_spans=[
            dict(
                phase="authentication_and_trajectory",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes=[reference(ROOT / p) for p in OWNED]
        + [reference(ROOT / original.PROTOCOL)],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Readiness can pass with zero decision gain; utility is reserved for Exp8351."""
    state = work["state"]
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    owned_ok = all(work["checks"].values()) and all(r["passed"] for r in work["child_receipts"])
    verdict = (
        "disqualified" if not passed or not owned_ok else "blocked" if work["failures"] else "null"
    )
    ready = int(verdict == "null" and bool(state))
    rows = state.get("issued", [])
    completed = sum(v["status"] == "completed" for v in rows)
    seals = [reference(raw / f"retention-{w}.json") for w in (0, 32, 64, 96)] if state else []
    evidence = [reference(raw / "measurement.json")]
    for name in [
        "uninterrupted/events.jsonl",
        "uninterrupted/state.json",
        "uninterrupted/costs.jsonl",
        "crash32/events.jsonl",
        "crash64/events.jsonl",
        "future/events.jsonl",
    ]:
        if (raw / name).is_file():
            evidence.append(reference(raw / name))
    value: Json = dict(
        experiment_id=8348,
        task_id=TASK,
        milestone="2026.10.720",
        run_date="20261009",
        honest_verdict="complete_" + verdict + "_continuous_local_learning",
        verdict_class=verdict,
        gate_check_summary=work["gates"] + work["failures"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        generator_weight_updates=0,
        historical_model_provenance=work.get("historical", []),
        rows=rows,
        intended_count=480,
        completed_count=completed,
        failed_count=0,
        censored_count=480 - len(rows),
        excluded_count=len(rows) - completed,
        independent_count=len({v["source_cluster_id"] for v in rows if v["status"] == "completed"}),
        sample_size_budget=dict(
            stream_sources=96,
            arms=5,
            retention_sources=32,
            windows=[0, 32, 64, 96],
            max_updates_per_arm=88,
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and owned_ok,
        flagged_adversarial=not passed or not owned_ok,
        acceptance_gates=work["checks"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        finding_dispositions=work.get("finding_dispositions", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=k.SEED,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=evidence + seals,
        cited_upstream_artifacts=[
            dict(r, imported_fields="frozen head, source identity or historical provenance")
            for r in work["refs"]
        ],
        heads_sha256=work.get("heads_sha256"),
        trajectory_ready_score=ready,
        issued_rows=rows,
        update_rows=state.get("updates", []),
        retention_prediction_seals=seals,
        retention_shadow_rows=state.get("retention", []),
        retention_targets_opened=0,
        update_budget_control=work.get("update_budget_control", {}),
        reachable_action_rows=work.get("reachable_action_rows", []),
        crash_replay_rows=work["crash_replay_rows"],
        later_source_attribution=work.get("later_source_attribution", []),
        pending_feedback=state.get("pending", []),
        release_rows=state.get("releases", []),
        durable_update_cost_rows=work.get("costs", []),
        dense_sparse_error_max=work.get("dense_sparse_error_max"),
        measurement_reference=reference(raw / "measurement.json"),
        learning_scope="Tier1 continuous online numeric learning with Tier2 durable state; no new semantic constraint types",
        methodology_note="Original fit/tune checkpoint, frozen roster and delay8; labels released only after durable issue. Sparse/dense freeze slope, intercept, temperature and knots with no online decay. Cached observations are exposed development; H1/H2 and retention benefit belong to future evaluators.",
        future_dependencies=["exp8350-static-benefit-audit", "exp8351-learning-retention-audit"],
        authority=work["authority"],
        owned_coverage_reference=work.get("owned_coverage_reference"),
        execution_manifest_reference=work.get("execution_manifest"),
    )
    value["field_principles"] = {
        f: "Bind the "
        + f
        + " claim to authenticated inputs, measured causal primitives and explicit scope."
        for f in value
    }
    value["field_principles"].update(
        trajectory_ready_score="Causality, exact restart, accounting and passed checks only; zero gain is allowed.",
        update_budget_control="Constructed fit-only crossing limits an update-budget null; it does not retune natural learning.",
        later_source_attribution="Isolate each admitted delta on later distinct source features; reserve utility for independent evaluator.",
        retention_prediction_seals="Seal all four feature-only windows before retention target access.",
        reproducibility_checksum="Bind every terminal claim to the canonical artifact bytes.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold recomputation rejects rehashed summaries and changed primitive state."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["measurement_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        raw = Path(ref["path"]).parent
        work = json.loads(Path(ref["path"]).read_bytes())
        for operand in work["refs"] + work["code_config_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(operand["path"])) != operand["sha256"]:
                return False
        if work["state"]:
            if authenticated_projection(work) != work["bundle"]:
                return False
            snapshots = work["authority"]["authority_snapshots"]
            with TemporaryDirectory(prefix="carnot-8348-cold-") as directory:
                scratch = Path(directory)
                authenticated = assess_authorities(
                    Path(snapshots["design"]["snapshot_path"]),
                    Path(snapshots["staged"]["snapshot_path"]),
                    Path(snapshots["active"]["snapshot_path"]),
                    scratch / "authority",
                    milestone="2026.10.720",
                    first_id=8346,
                    count=14,
                )
                if any(
                    authenticated[f] != work["authority"][f]
                    for f in [
                        "activated",
                        "planning_matched",
                        "canonical_tasks_sha256",
                        "contract_rows",
                    ]
                ):
                    return False
                reconstructed = k.run(work["bundle"], scratch)
                if k.journal(scratch / "events.jsonl") != k.journal(
                    raw / "uninterrupted/events.jsonl"
                ):
                    return False
            if reconstructed != work["state"]:
                return False
            if (
                k.control(work["bundle"]["head"]) != work["update_budget_control"]
                or k.reachability(work["bundle"]) != work["reachable_action_rows"]
                or k.attribution(work["bundle"], reconstructed) != work["later_source_attribution"]
            ):
                return False
            for window in (0, 32, 64, 96):
                if json.loads((raw / f"retention-{window}.json").read_bytes()) != dict(
                    targets_opened=False,
                    rows=[v for v in reconstructed["retention"] if v["window"] == window],
                ):
                    return False
            for slot in (32, 64):
                if json.loads((raw / f"crash{slot}/final.json").read_bytes()) != reconstructed:
                    return False
        return build(work, raw, value["validation_receipts"]) == value
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def authenticated_projection(work: Json) -> Json:
    """Rebuild learner inputs from pinned copies, never a rehashed bundle summary."""

    def operand(ref: Json, *, parse: bool = True) -> Json:
        saved = next(
            r
            for r in work["refs"]
            if r["source_path"] == ref["path"] and r["sha256"] == ref["sha256"]
        )
        return (
            dict(json.loads(Path(saved["path"]).read_bytes()))
            if parse
            else reference(Path(saved["path"]))
        )

    primaries = {}
    for name, pin in PINS.items():
        saved = next(r for r in work["refs"] if r["source_path"].endswith(name))
        if saved["sha256"] != pin:
            raise ValueError("upstream_pin")
        primaries[name] = json.loads(Path(saved["path"]).read_bytes())
    receipt = primaries["results/experiment_8346_v720_frozen_input_contract.json"]
    qualified = primaries["results/experiment_8347_v720_local_consumer_qualification.json"]
    if receipt["frozen_heads_ready_score"] != 1 or qualified["local_kernel_ready_score"] != 1:
        raise ValueError("upstream_readiness")
    source = primaries[original.SOURCE]
    head = next(
        h
        for h in operand(receipt["frozen_policy"]["checkpoint"])["heads"]
        if h["arm"] == "spline34"
    )
    slots = operand(source["predictor_shards"]["reserved"])["rows"]
    public = [
        dict(
            slot=p["slot"],
            unit_id=p["unit_id"],
            source_cluster_id=p["source_cluster_id"],
            x=None if p["x"] is None else [p["x"][0], *p["x"][12:16]],
        )
        for p in slots
    ]
    return dict(
        head={f: head[f] for f in ["arm", "coefficients", "temperature", "geometry"]},
        slots=public,
        labels=operand(source["evaluator_shards"]["reserved"], parse=False),
        fit=operand(source["predictor_shards"]["fit"])["rows"],
    )
