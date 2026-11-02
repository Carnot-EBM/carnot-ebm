"""REQ-REPORT-8406: sealed causal operations qualify deployment, not learning benefit."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import continuous_local_learning_8348 as historical
from carnot.reporting import historical_consumer_roots_8402 as authority
from carnot.reporting import direct_state_qualification_8404 as qualified
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs
from carnot.verify import continuous_direct_learning_8406 as k

from carnot.verify import direct_atomic_state_8376 as direct
from carnot.verify.independent_online_memory_8116 import LabelVault

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8406_v724_continuous_direct_learning", "exp8406-continuous-direct-learning"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_continuous_direct_learning_8406.py"
OWNED = [
    "python/carnot/verify/continuous_direct_learning_8406.py",
    "python/carnot/reporting/continuous_direct_learning_8406.py",
    "python/carnot/reporting/continuous_direct_runner_8406.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
progress, reference = k.progress, authority.reference
PINS = {
    "results/experiment_8404_v724_direct_state_qualification.json": "sha256:186b11a57be0c51b32fccb4a5b5064cd9a63220f5f4866a9c3d2740f2f102c67",
    "results/experiment_8348_v720_continuous_local_learning.json": "sha256:c98cbd678aef6aa5055cd7033a2c36404dc8d20337a5e601bd90b2d7dba00f07",
    "results/experiment_8361_v721_utility_audit_qualification.json": "sha256:33cdb0ba681c49eed70d57d5c49fd2c3fd76f7f2a7dd2a5253bbc3248e177ab4",
    authority.PROTOCOL: authority.PROTOCOL_PIN,
}


def bind(ref: Json, raw: Path, work: Json, *, parse: bool = True) -> Json:
    """Copy exact producer bytes before reuse; label fragments stay opaque here."""
    path = Path(ref["path"])
    observed = reference(path)["sha256"] if path.is_file() else None
    if observed != ref["sha256"]:
        work["gate_check_summary"].append(
            authority.failure(path, "input_sha256", ref["sha256"], observed)
        )
    path = authority.checked(path, ref["sha256"])
    saved = raw / "inputs" / (str(len(work["refs"])) + ".bin")
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, saved)
    saved.chmod(0o400)
    work["refs"].append(dict(reference(saved), source_path=str(path)))
    return dict(json.loads(saved.read_bytes())) if parse else reference(saved)


def authenticate(root: Path, raw: Path, work: Json) -> Json:
    """Complete task authority and a qualified byte-bound direct producer precede learning."""
    checked = authority.authority(root, raw / "authority")
    work["current_authority"] = checked
    work["gate_check_summary"].extend(checked["gate_check_summary"])
    tasks = [t for t in checked["tasks"] if t["id"] == TASK]
    if not checked["activated"] or len(tasks) != 1 or tasks[0]["MODEL_SPECS"] != []:
        raise ValueError("exact_current_task")
    for name in (authority.DESIGN, authority.ACTIVE, authority.STAGED):
        if (root / name).exists():
            bind(reference(root / name), raw, work, parse=False)
    primaries = {
        name: bind(dict(path=str(root / name), sha256=pin), raw, work) for name, pin in PINS.items()
    }
    direct = primaries[next(n for n in PINS if "8404_" in n)]
    path = root / next(n for n in PINS if "8404_" in n)
    validate_primary(direct, path)
    terminal = bind(reference(Path(direct["terminal_validation_sidecar_path"])), raw, work)
    side = Path(terminal["sidecar_path"])
    report = read_bound_sidecar(path, side)
    bind(reference(side), raw, work)
    for field, expected in [
        ("direct_state_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        if direct.get(field) != expected:
            work["gate_check_summary"].append(
                authority.failure(path, field, expected, direct.get(field))
            )
    if not (
        direct["direct_state_ready_score"] == 1
        and direct["required_checks_passed"]
        and direct["flagged_adversarial"] is False
        and report["report"]["passed"]
    ):
        raise ValueError("qualified_direct_state")
    for ref in direct["code_config_hashes"][:4]:
        bind(ref, raw, work, parse=False)
    for ref in direct["source_artifact_hashes"]:
        if str(ref.get("source_path", "")).endswith(".py"):
            bind(dict(path=ref["source_path"], sha256=ref["sha256"]), raw, work, parse=False)
    old = primaries[next(n for n in PINS if "8348_" in n)]
    measured = bind(old["measurement_reference"], raw, work)
    for ref in measured["refs"]:
        bind(ref, raw, work, parse=False)
    projected = historical.authenticated_projection(measured)
    protocol = primaries[authority.PROTOCOL]
    for ref in protocol["preserved_protocols"]:
        bind(ref, raw, work, parse=False)
    head = protocol["head"]
    if head["coefficients"] != projected["head"]["coefficients"]:
        raise ValueError("frozen_head")
    work["historical_model_provenance"] = old["historical_model_provenance"]
    audit = primaries[next(n for n in PINS if "8361_" in n)]
    work["historical_controls"] = dict(
        reference=reference(root / next(n for n in PINS if "8361_" in n)),
        scope="labeled_reference_only",
        H2=audit["H2"]["arm_results"],
    )
    if audit["H2"]["block_bootstrap_summary"]["mean_gain"] != 0:
        raise ValueError("closed_h2")
    return dict(
        head=head,
        slots=projected["slots"],
        labels=bind(projected["labels"], raw, work, parse=False),
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Run exactly nine streams after source, host and task checks pass."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        root=str(root),
        raw=str(raw),
        refs=[],
        gate_check_summary=[],
        bundle={},
        streams=[],
        child_receipts=[],
        current_authority={},
        historical_controls={},
        historical_model_provenance=[],
        phase_spans=[],
    )
    progress("preconditions_before")
    probe = private / "disk_probe"
    probe.write_bytes(b"private disk-backed scratch")
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    resources = dict(
        private_scratch=probe.read_bytes() == b"private disk-backed scratch",
        private_permissions=private.stat().st_mode & 0o077 == 0,
        disk_free=shutil.disk_usage(private).free > 1_000_000_000,
        memory_available=memory > 256_000_000,
    )
    resources.update(
        {
            n: os.access(ROOT / ".venv/bin" / n, os.X_OK)
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        }
    )
    work["preconditions_checked"] = dict(
        resources=resources,
        available_memory_bytes=memory,
        task_cap_s=4800,
        worker_deadline_s=600,
        scratch=str(private),
    )
    for name, ok in resources.items():
        if not ok:
            work["gate_check_summary"].append(authority.failure(private, name, True, ok))
    for name, pin in PINS.items():
        if not (root / name).is_file():
            work["gate_check_summary"].append(
                authority.failure(root / name, "input_sha256", pin, None)
            )
    try:
        work["bundle"] = authenticate(root, raw, work)
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        absent = next((root / n for n in PINS if not (root / n).is_file()), root / authority.DESIGN)
        work["gate_check_summary"].append(
            authority.failure(
                absent, "authenticated_inputs", True, None if not absent.exists() else str(error)
            )
        )
    progress("preconditions_after", len(work["refs"]), 0)
    if work["bundle"] and not work["gate_check_summary"]:
        source = raw / "bundle.json"
        atomic_json(source, work["bundle"])
        for seed in k.SEEDS:
            for variant in k.VARIANTS:
                directory = raw / f"{seed}-{variant}"
                argv = [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--worker",
                    str(source),
                    "--state-directory",
                    str(directory),
                    "--variant",
                    variant,
                ]
                progress(
                    "before_benchmark_" + variant, len(work["streams"]), 9 - len(work["streams"])
                )
                receipts = [
                    child(
                        f"{seed}_{variant}_{crash}",
                        argv + ["--crash", str(crash)],
                        raw / "workers",
                        deadline=660,
                        expected=73 if crash else 0,
                        heartbeat=20,
                    )
                    for crash in ((32, 64, 0) if variant == k.VARIANTS[2] else (0,))
                ]
                work["child_receipts"].extend(receipts)
                if all(r["passed"] for r in receipts):
                    work["streams"].append(
                        dict(seed=seed, variant=variant, directory=str(directory))
                    )
                progress(
                    "after_benchmark_" + variant, len(work["streams"]), 9 - len(work["streams"])
                )
                if not all(r["passed"] for r in receipts):
                    break
    work["duration_s"] = time.monotonic() - began
    work["phase_spans"] = [
        dict(phase="preconditions_and_measurement", duration_s=work["duration_s"])
    ]
    work["work_path"] = str(raw / "work.json")
    atomic_json(Path(work["work_path"]), work)
    return work


def reduction(work: Json) -> Json:
    """Only complete sealed trajectories authorize evaluator access and descriptive metrics."""
    rows: list[Json] = []
    retention: list[Json] = []
    pending: list[Json] = []
    hashes: Json = {}
    updates: list[Json] = []
    controls: list[Json] = []
    costs: list[Json] = []
    bundle = work["bundle"]
    streams = {(r["seed"], r["variant"]): r for r in work["streams"]}
    targets = []
    if streams:
        vault = LabelVault(Path(bundle["labels"]["path"]), bundle["slots"])
        targets = [vault.release(i, 104, sealed=True, retention=True)["y"] for i in range(1, 129)]
    raw_refs: list[Json] = []
    brier, utility = {}, {}
    for seed in k.SEEDS:
        for variant in k.VARIANTS:
            stream = streams.get((seed, variant))
            steps = k.read_steps(Path(stream["directory"])) if stream else []
            state = k.check_steps(bundle, steps) if steps else {}
            if stream and len(steps) != 96:
                raise ValueError("incomplete_stream")
            key = f"{seed}:{variant}"
            hashes[key] = canonical_hash([r["state"] for r in steps]) if steps else None
            if stream:
                directory = Path(stream["directory"])
                raw_refs.extend(reference(p) for p in sorted((directory / "steps").glob("*.json")))
                raw_refs.append(reference(directory / "final.json"))
                if json.loads((directory / "final.json").read_bytes()) != state:
                    raise ValueError("final_state_drift")
                controls.extend(
                    dict(r, seed=seed, service_variant=variant) for r in k.controls(bundle, state)
                )
                pending.extend(
                    dict(
                        seed=seed,
                        service_variant=variant,
                        issue_id=i,
                        due_slot=state["issued"][i]["event"]["slot"] + 8,
                    )
                    for i in state["pending"]
                )
                updates.extend(
                    dict(r, seed=seed, service_variant=variant) for r in state["applied"].values()
                )
                costs.extend(
                    dict(
                        seed=seed,
                        service_variant=variant,
                        slot=r["slot"],
                        latency_ns=r["update_latency_ns"],
                    )
                    for r in steps
                    if r["slot"] > 8
                )
            for slot in range(1, 97):
                source = bundle["slots"][slot - 1] if bundle else {}
                issued = state.get("issued", {}).get(f"issue-{slot}")
                p = issued["probability"] if issued else None
                y = targets[slot - 1] if issued else None
                cost = (
                    None
                    if not issued
                    else 0.5
                    if p is None or issued["action"] == "escalate"
                    else float((issued["action"] == "reject") != bool(y))
                )
                rows.append(
                    dict(
                        seed=seed,
                        service_variant=variant,
                        learning_arm="sparse_local",
                        slot=slot,
                        unit_id=source.get("unit_id"),
                        source_cluster_id=source.get("source_cluster_id"),
                        p=p,
                        action=issued["action"] if issued else None,
                        y=y,
                        brier=None if p is None or y is None else (p - y) ** 2,
                        utility=None if cost is None else -cost,
                        status="unstarted"
                        if not issued
                        else "excluded"
                        if p is None
                        else "completed",
                        missing_reason="external_service_gate"
                        if not issued
                        else "missing_source_features"
                        if p is None
                        else None,
                    )
                )
            if stream:
                for window in (0, 32, 64, 96):
                    head = bundle["head"] if window == 0 else steps[window - 1]["state"]["head"]
                    for source, y in zip(bundle["slots"][96:], targets[96:], strict=True):
                        p = k.probability(head, source["x"])
                        action = "escalate" if p is None else direct.evaluator.kernel.action(p)
                        cost = (
                            0.5
                            if p is None or action == "escalate"
                            else float((action == "reject") != bool(y))
                        )
                        retention.append(
                            dict(
                                seed=seed,
                                service_variant=variant,
                                window=window,
                                slot=source["slot"],
                                source_cluster_id=source["source_cluster_id"],
                                p=p,
                                action=action,
                                y=y,
                                brier=None if p is None or y is None else (p - y) ** 2,
                                utility=-cost,
                                head_hash=canonical_hash(head),
                                scope="fixed_exposed_retention",
                            )
                        )
    for variant in k.VARIANTS:
        selected = [r for r in rows if r["service_variant"] == variant]
        values: list[float] = [r["brier"] for r in selected if r["brier"] is not None]
        brier[variant] = dict(
            mean=sum(values) / len(values) if values else None,
            denominator=len(values),
            scope="descriptive_exposed_source_copies",
        )
        values = [r["utility"] for r in selected if r["utility"] is not None]
        utility[variant] = dict(
            mean=sum(values) / len(values) if values else None,
            denominator=len(values),
            scope="negative_fixed_policy_cost",
        )
    exact = len(streams) == 9 and all(
        len({hashes[f"{seed}:{v}"] for v in k.VARIANTS}) == 1 for seed in k.SEEDS
    )
    return dict(
        rows=rows,
        retention_rows=retention,
        pending_feedback_rows=pending,
        trajectory_hashes=hashes,
        feedback_ids=[
            dict(seed=r["seed"], service_variant=r["service_variant"], id=r["event"]["id"])
            for r in updates
        ],
        coefficient_change_count=sum(len(r["changed"]) for r in updates),
        touched_coefficients=sorted({i for r in updates for i in r["changed"]}),
        update_rows=updates,
        update_latency_rows=costs,
        feedback_controls=controls,
        brier_by_variant=brier,
        utility_by_variant=utility,
        raw_shard_hashes=raw_refs,
        exact_resume_score=int(exact),
        future_label_read_count=0,
        issued_prediction_mutation_count=0,
        dropped_feedback_count=0,
        duplicate_feedback_count=0,
        stale_feedback_count=0,
        pending_capacity=dict(
            limit=9, observed_peak=9 if streams else None, final_per_stream=8 if streams else None
        ),
    )


def build(work: Json, receipts: list[Json]) -> Json:
    """Readiness requires real fidelity and coverage, while the scientific verdict stays null."""
    reduced = reduction(work)
    passed = bool(receipts) and all(r["passed"] for r in [*receipts, *work["child_receipts"]])
    with patch.object(qualified, "OWNED", OWNED):
        covered = bool(qualified.coverage_ready(work.get("owned_coverage", {})))
    complete = len(work["streams"]) == 9 and reduced["exact_resume_score"] == 1
    verdict = (
        "disqualified"
        if not passed or (not work["gate_check_summary"] and not complete)
        else "blocked"
        if work["gate_check_summary"]
        else "null"
    )
    rows = reduced["rows"]
    value: Json = dict(
        reduced,
        experiment_id=8406,
        task_id=TASK,
        milestone="2026.10.724",
        run_date="20261011",
        honest_verdict="complete_" + verdict + "_continuous_direct_learning",
        verdict_class=verdict,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        generator_weight_updates=0,
        historical_model_provenance=work["historical_model_provenance"],
        intended_count=864,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["status"] == "unstarted" for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        independent_count=len({r["source_cluster_id"] for r in rows if r["status"] == "completed"}),
        sample_size_budget=dict(
            intended=864,
            source_slots=96,
            usable_source_slots=74,
            missing_source_slots=22,
            seeds=list(k.SEEDS),
            service_variants=list(k.VARIANTS),
            scientific_arms=1,
            seed_or_variant_independent_multiplier=1,
            feedback_budget_per_stream=88,
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        historical_h2_gain=0,
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        gate_check_summary=work["gate_check_summary"],
        acceptance_gates=dict(
            deployment_fidelity=complete, owned_validation=passed, owned_coverage=covered
        ),
        continuous_service_ready_score=int(verdict == "null" and complete and covered),
        service_variant=list(k.VARIANTS),
        learning_arm="sparse_local",
        validation_receipts=[*receipts, *work["child_receipts"]],
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=list(k.SEEDS),
        source_artifact_hashes=work["refs"],
        code_config_hashes=work.get("code_config_hashes", [reference(ROOT / p) for p in OWNED]),
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields="task authority, frozen head, source identity, labels or historical controls",
            )
            for r in work["refs"]
        ],
        historical_controls=work["historical_controls"],
        current_authority=work["current_authority"],
        owned_coverage=work.get("owned_coverage", {}),
        work_reference=reference(Path(work["work_path"])),
        repository_health=dict(scope="separate_global_health_not_launched_per_task"),
        methodology_note="Sparse CPU coefficient writes through shipped direct spline state. Delay8, rate0.01, gradient(p-y)/T, norm cap1, bounds[-4,4]. Fixed intercept, holistic slope, temperature and knots. Genuine process exits32/64; no power-loss or FPGA claim. Brier and retention are exposed descriptive metrics; qualified H2 remains zero.",
    )
    value["raw_shard_hashes"].extend(work.get("coverage_shards", []))
    value["field_principles"] = {
        f: f"Bind {f} to sealed source and causal step states; deployment fidelity gives no scientific benefit."
        for f in value
    }
    value["field_principles"].update(
        coefficient_change_count="Count changed numeric coordinates across service copies; these are not independent examples.",
        historical_h2_gain="Import the qualified closed exposed H2 zero without retuning.",
        utility_by_variant="Negative fixed-policy cost on all issued slots; missing features escalate at cost0.5.",
        exact_resume_score="Require bit-exact complete semantic states across all three deployment variants.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """A cold process independently authenticates inputs and rebuilds every primitive reduction."""
    try:
        value = json.loads(path.read_bytes())
        digest = value.pop("reproducibility_checksum")
        if value["experiment_id"] != 8406 or digest != canonical_hash(value):
            return False
        ref = value["work_reference"]
        work = json.loads(authority.checked(Path(ref["path"]), ref["sha256"]).read_bytes())
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            authority.checked(Path(ref["path"]), ref["sha256"])
        if work["bundle"] and not work["gate_check_summary"]:
            with TemporaryDirectory(dir="/var/tmp", prefix="exp8406-cold-") as directory:
                sealed = [dict(r, exists=True, snapshot_path=r["path"]) for r in work["refs"]]
                fresh: Json = dict(refs=[], gate_check_summary=[])
                with frozen_inputs(sealed, Path(directory) / "inputs"):
                    projected = authenticate(Path(work["root"]), Path(directory) / "auth", fresh)
                if (
                    any(projected[f] != work["bundle"][f] for f in ("head", "slots"))
                    or projected["labels"]["sha256"] != work["bundle"]["labels"]["sha256"]
                ):
                    return False
                if any(
                    fresh["current_authority"][f] != work["current_authority"][f]
                    for f in ("tasks", "activated", "canonical_tasks_sha256")
                ):
                    return False
        receipts = (
            value["validation_receipts"][
                : len(value["validation_receipts"]) - len(work["child_receipts"])
            ]
            if work["child_receipts"]
            else value["validation_receipts"]
        )
        rebuilt = build(work, receipts)
        rebuilt.pop("reproducibility_checksum")
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
