"""REQ-VERIFY-8211: preserve the qualified numerical method on natural bytes.

Small CPU heads change across requests. Historical generator calls remain
historical, and exposed development trajectories supply no generalization credit.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import yaml

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.verify import calibrated_memory_methods_8180 as m
from carnot.verify import delayed_energy_memory_8143 as delayed
from carnot.verify.hard_exit_learning_qualification_8206 import run_check

Json = dict[str, Any]
ROOT = m.ROOT
NAME = "experiment_8211_v709_calibrated_memory_trajectory"
TASK = "exp8211-calibrated-memory-trajectory"
MODULE = "python/carnot/verify/calibrated_memory_trajectory_8211.py"
RUNNER = "python/carnot/reporting/calibrated_trajectory_execution_8211.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_calibrated_memory_trajectory_8211.py"
UPSTREAM = "results/experiment_8206_v709_hard_exit_learning_qualification.json"
UPSTREAM_HASH = "sha256:3ff40ac3a82fc4a9f8e42f626c5d19db10bdfd5546d7aa0e998353a4bce87fd8"
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so CPU fitting cannot look like silent inference."""
    print(f"[exp8211] phase={phase} completed={completed} pending={pending}", flush=True)


def append(path: Path, row: Json) -> None:
    """Persist a causal boundary before the learner can observe the next event."""
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def journal(path: Path) -> list[Json]:
    """Read complete durable records without inventing a missing delivery."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class ReleasedLabels(delayed.DelayedLabels):
    """Retain raw outcomes exactly when their registered release clock arrives."""

    def __init__(self, path: Path, rows: list[Json], output: Path):
        super().__init__(path, rows)
        self.output = output
        self.delivered = {r["label_slot"] for r in journal(output)}

    def __getitem__(self, index: Any) -> Any:
        slot = int(index) + 1
        if self.state.get("phase") != "release" or self.state["cursor"] != slot + 20:
            raise ValueError("unsealed_release")
        if slot in self.delivered:
            raise ValueError("duplicate_delivery")
        target = super().__getitem__(index)
        append(
            self.output,
            dict(label_slot=slot, release_slot=slot + 20, outcome=self.opened[str(slot)]),
        )
        self.delivered.add(slot)
        return target


def run_seed(
    rows: list[Json],
    path: Path,
    seed: int,
    raw: Path,
    *,
    state: Json | None = None,
    crash_slot: int = 0,
) -> Json:
    """Seal issues before feedback and reject checkpoint rollback against journals."""
    raw.mkdir(parents=True, exist_ok=True)
    issued = journal(raw / "issued.jsonl")
    released = journal(raw / "released.jsonl")
    if issued and (
        state is None
        or [r["issued"] for r in issued] != state["issued"]
        or [r["label_slot"] for r in released] != state["consumed"]
        or state["pending"]
        != [
            r["slot"]
            for r in state["issued"]
            if r["slot"] not in state["released"] and r["slot"] not in state["lost"]
        ]
    ):
        raise ValueError("rollback")
    labels = ReleasedLabels(path, rows, raw / "released.jsonl")
    labels.state = state or {}

    def seal(kind: str, current: Json) -> None:
        labels.state = current
        append(
            raw / "issued.jsonl",
            dict(
                seed=seed,
                issued=current["issued"][-1],
                heads_hash=canonical_hash(current["arms"]),
                pending=current["pending"],
            ),
        )
        if current["cursor"] == crash_slot:
            atomic_json(raw / "crash.json", current)
            progress("intentional_hard_exit", current["cursor"], len(current["pending"]))
            os._exit(73)

    final = m.run(rows, labels, seed, state=state, seal=seal)
    atomic_json(raw / "final.json", final)
    return final


def authenticate(root: Path, binder: Any, fixture: bool) -> Json:
    """Authenticate this branch only; optional sibling experiments cannot gate it."""
    binder.upstream = "exp8206-hard-exit-learning-qualification"
    for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
        operand = ROOT / ".venv/bin" / name
        binder.require(operand, "runtime_" + name, True, operand.is_file())
    path = root / UPSTREAM
    value = binder.read(path, None if fixture else UPSTREAM_HASH)
    for field, expected in [
        ("experiment_id", 8206),
        ("calibrated_memory_ready_score", 1),
        ("stream_input_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
        ("numerical_protocol_sha256", m.PROTOCOL_HASH),
    ]:
        binder.require(path, field, expected, value.get(field))
    if not fixture:
        terminal = m.engine.methods.historical.terminal(path, value, binder)
        binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    binder.bind(ROOT / m.PROTOCOL, m.PROTOCOL_HASH)
    binder.bind(ROOT / m.schedule.PROTOCOL, m.schedule.PROTOCOL_HASH)
    exclusion = ROOT / "ops/exclusion_manifest.yaml"
    binder.bind(exclusion)
    binder.require(
        exclusion,
        "experiment8211_not_retired",
        True,
        not any(
            r.get("experiment_id") == 8211
            for r in yaml.safe_load(exclusion.read_text()).get("retired_experiments", [])
        ),
    )
    for previous, digest in [
        (m.UPSTREAM, "sha256:5cb110c748960181176793a2d46b18702a53f2e5e49bfed113b3c46c28c29028"),
        (
            "results/experiment_8171_v706_released_feedback_learning.json",
            "sha256:7123a7bebb9963319855a4e27301bff27699105e83cb974cc6a6556d100f3a5c",
        ),
    ]:
        historical = binder.read(root / previous, digest)
        binder.require(
            path,
            previous + ".input_manifests",
            value["input_manifests"],
            historical["input_manifests"],
        )
    for role in ["stream", "retention"]:
        for ref in [
            value["input_manifests"][role + "_feature_manifest"],
            value["input_manifests"]["evaluator_label_manifests"][role],
        ]:
            binder.bind(Path(ref["path"]), ref["sha256"])
    return dict(value)


def evidence(
    states: list[Json], public: list[Json], releases: list[list[Json]], retained: list[Json]
) -> Json:
    """Derive rows and center influence from actual installed states, without tuning."""
    issued_rows: list[Json] = []
    release_rows: list[Json] = []
    admission_rows: list[Json] = []
    overlap: list[Json] = []
    rows: list[Json] = []
    retention_predictions: list[Json] = []
    operations = buffers = 0
    for state, deliveries in zip(states, releases, strict=True):
        seed, geometry = state["seed"], state["geometry"]
        labels = {r["label_slot"]: r["outcome"] for r in deliveries}
        release_rows.extend(dict(r, seed=seed) for r in deliveries)
        for row, issued in zip(public, state["issued"], strict=True):
            issued = dict(issued, predictions={a: issued["predictions"][a] for a in m.ARMS})
            for arm in m.ARMS:
                p = issued["predictions"][arm]
                issued_rows.append(
                    dict(
                        seed=seed,
                        slot=row["slot"],
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        arm=m.NAMES[arm],
                        probability=p,
                        energy=None
                        if p is None
                        else -math.log(max(1e-300, p) / max(1e-300, 1 - p)),
                        action=None if p is None else m.engine.historical.radial.action(p),
                        prediction_hash=issued["prediction_hash"],
                    )
                )
            if row["slot"] >= 65:
                label = labels.get(
                    row["slot"], dict(y=None, exclusion_reason="feedback_unresolved_tail")
                )
                rows.extend(
                    dict(r, arm=m.NAMES[r["arm"]])
                    for r in m.engine.historical.scored(row, issued, label, "later_stream", seed)
                )
        for event in state["events"]:
            if event["kind"] == "qualified_candidates":
                for arm in [a for a in m.ARMS if a in event["candidates"]]:
                    candidate = event["candidates"][arm]
                    fit = candidate["head"]["fit"]
                    n, k = fit["sample_count"], len(candidate["head"]["centers"])
                    operations += n * k * 9 * (fit["iterations"] + 1)
                    buffers += 8 * (n * (k + 2) + n + k + 2)
            if event["kind"] == "commit_candidate":
                for arm in [a for a in m.ARMS if a in event["candidates"]]:
                    candidate = event["candidates"][arm]
                    centers = candidate["base"]["centers"]
                    for index in range(len(centers) - 4, len(centers)):
                        for earlier in range(index):
                            distance = sum(
                                (a - b) ** 2
                                for a, b in zip(
                                    centers[index]["x"], centers[earlier]["x"], strict=True
                                )
                            )
                            overlap.append(
                                dict(
                                    seed=seed,
                                    arm=m.NAMES[arm],
                                    install_slot=event["slot"],
                                    center_index=index,
                                    earlier_center_index=earlier,
                                    center=centers[index],
                                    earlier_center=centers[earlier],
                                    overlap=math.exp(-distance / (4 * geometry["sigma"] ** 2)),
                                    overlap_kind="continuous_gaussian_inner_product",
                                    diagnostic_only=True,
                                    retention_labels_used=False,
                                )
                            )
            if event["kind"] == "durable_update":
                for arm in m.ARMS:
                    head = event["heads"][arm]
                    later = [
                        p
                        for p in state["issued"][event["slot"] :]
                        if p["predictions"][arm] is not None
                    ]
                    admission_rows.append(
                        dict(
                            seed=seed,
                            arm=m.NAMES[arm],
                            install_slot=event["slot"],
                            head=head,
                            remaining_original_slots=256 - event["slot"],
                            later_available_count=len(later),
                            changed_later_probabilities=sum(
                                abs(p["predictions"][arm] - p["predictions"]["frozen_qwen_offset"])
                                > 1e-12
                                for p in later
                            ),
                            changed_later_decisions=sum(
                                m.engine.historical.radial.action(p["predictions"][arm])
                                != m.engine.historical.radial.action(
                                    p["predictions"]["frozen_qwen_offset"]
                                )
                                for p in later
                            ),
                            update_label_ids=[
                                slot for slot in state["training"] if slot + 20 <= event["slot"]
                            ],
                            admission_label_ids=[
                                slot
                                for slot in state["used_admission"]
                                if slot + 20 <= event["slot"]
                            ],
                        )
                    )
        for row in retained:
            predictions = {
                a: None
                if row["values"] is None
                else m.probability(state["arms"][a], geometry, row["values"])
                for a in m.ARMS
            }
            retention_predictions.append(
                dict(
                    seed=seed,
                    slot=row["slot"],
                    predictions=predictions,
                    prediction_hash=canonical_hash(predictions),
                )
            )
    return dict(
        issued_rows=issued_rows,
        release_rows=release_rows,
        admission_rows=admission_rows,
        support_overlap_rows=overlap,
        rows=rows,
        retention_predictions=retention_predictions,
        hardware_receipt=dict(
            update_operations=operations,
            update_operations_kind="estimated_Gaussian_coordinate_work_from_measured_fit_iterations",
            update_buffer_bytes=buffers,
            storage_bytes=sum(len(json.dumps(s).encode()) for s in states),
            generator_weight_bytes_written=0,
            device_speed_claim=False,
        ),
    )


def copy_child_coverage(raw: Path) -> None:
    """Keep real hard-exit shards in the owned coverage parent until combination."""
    parent = os.environ.get("CARNOT_8211_COVERAGE_PARENT")
    if parent:
        for shard in (raw / "child_coverage").glob(".coverage.*"):
            shutil.copyfile(shard, Path(parent) / (".coverage.8211-" + shard.name[10:]))


def measure(root: Path, raw: Path, *, fixture: bool = False, **kwargs: Any) -> Json:
    """Bind inputs before natural fitting, then seal trajectories before retention."""
    began = time.monotonic_ns()
    began_wall = time.time_ns()
    progress("preconditions_start")
    raw.mkdir(parents=True, exist_ok=True)
    binder = m.engine.methods.Custody(raw)
    for operand in Path(
        os.environ.get("CARNOT_8211_PREFLIGHT", str(raw / "absent_preflight"))
    ).glob("*"):
        if operand.is_file():
            binder.bind(operand)
    work: Json = dict(
        input_ready=0,
        rows=[],
        state_manifest=[],
        input_manifests={},
        issued_rows=[],
        release_rows=[],
        admission_rows=[],
        support_overlap_rows=[],
        restart_state_hashes=[],
        child_exit_rows=[],
        retention_predictions=[],
        hardware_receipt={},
        trajectory_path=None,
        trajectory_sha256=None,
        cited_upstream_artifacts=[],
    )
    try:
        upstream = authenticate(root, binder, fixture)
        manifests = upstream["input_manifests"]
        public = json.loads(Path(manifests["stream_feature_manifest"]["path"]).read_text())["rows"]
        retained = json.loads(Path(manifests["retention_feature_manifest"]["path"]).read_text())[
            "rows"
        ]
        m.engine.historical.public_rows(public, 256)
        m.engine.historical.public_rows(retained, 64)
        path = Path(manifests["evaluator_label_manifests"]["stream"]["path"])
        from carnot.reporting.calibrated_trajectory_execution_8211 import restart_specs

        specs = restart_specs(raw, path, manifests["stream_feature_manifest"])
        atomic_json(raw / "restart_commands.json", dict(commands=specs))
        states, releases = [], []
        progress("preconditions_complete", len(binder.checks), 0)
        for seed in range(101, 121):
            progress("before_natural_benchmark", seed - 101, 121 - seed)
            directory = raw / f"seed-{seed}"
            state = run_seed(public, path, seed, directory)
            states.append(state)
            releases.append(journal(directory / "released.jsonl"))
            work["state_manifest"].append(
                dict(
                    seed=seed,
                    state=dict(
                        path=str(directory / "final.json"),
                        sha256=sha256_file(directory / "final.json"),
                    ),
                )
            )
            progress("after_natural_benchmark", seed - 100, 120 - seed)
        receipts = [run_check(ROOT, spec, raw, raw / "child_logs") for spec in specs]
        baseline = json.loads((raw / "uninterrupted/final.json").read_text())
        resumed = json.loads((raw / "restart/final.json").read_text())
        crash = json.loads((raw / "restart/crash.json").read_text())
        parity = (
            m.stable(baseline) == m.stable(resumed) == m.stable(states[0])
            and journal(raw / "uninterrupted/issued.jsonl") == journal(raw / "restart/issued.jsonl")
            and crash["cursor"] == 90
            and bool(crash["pending"])
        )
        work.update(evidence(states, public, releases, retained))
        work.update(
            input_ready=1,
            input_manifests=manifests,
            child_exit_rows=receipts,
            restart_state_hashes=[
                dict(
                    baseline_sha256=canonical_hash(m.stable(baseline)),
                    resumed_sha256=canonical_hash(m.stable(resumed)),
                    pending_state_sha256=canonical_hash(crash["pending"]),
                    saved_cursor=crash["cursor"],
                    pending_count=len(crash["pending"]),
                    passed=parity and all(r["passed"] for r in receipts),
                )
            ],
            cited_upstream_artifacts=[
                dict(
                    experiment_id=8206,
                    path=str(root / UPSTREAM),
                    sha256=sha256_file(root / UPSTREAM),
                    fields_imported=[
                        "calibrated_memory_ready_score",
                        "stream_input_ready_score",
                        "input_manifests",
                        "numerical_protocol_sha256",
                    ],
                )
            ],
        )
        trajectory = raw / "trajectory.json"
        atomic_json(trajectory, dict(states=states, releases=releases, evidence=deepcopy(work)))
        work.update(trajectory_path=str(trajectory), trajectory_sha256=sha256_file(trajectory))
        progress("trajectory_and_retention_predictions_sealed", len(states), 0)
        vault = m.engine.historical.LabelVault(
            Path(manifests["evaluator_label_manifests"]["retention"]["path"]), retained
        )
        for prediction in work["retention_predictions"]:
            row = retained[prediction["slot"] - 1]
            label = vault.release(row["slot"], 256, sealed=True, retention=True)
            work["rows"].extend(
                dict(r, arm=m.NAMES[r["arm"]])
                for r in m.engine.historical.scored(
                    row, prediction, label, "retention", prediction["seed"]
                )
            )
        copy_child_coverage(raw)
    except m.engine.methods.historical.InputFailure:
        progress("required_external_operand_blocked", len(binder.checks), 0)
    work.update(
        gate_check_summary=binder.checks,
        source_artifact_hashes=binder.refs,
        fixture_mode=fixture,
        preconditions_checked=dict(
            runtime_executable=str(ROOT / ".venv/bin/python"),
            private_writable_storage=True,
            no_model_load=True,
            natural_fitting=bool(work["input_ready"]),
            checks=len(binder.checks),
        ),
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                MODULE,
                RUNNER,
                CLI,
                TEST,
                m.MODULE,
                m.schedule.MODULE,
                m.PROTOCOL,
                m.schedule.PROTOCOL,
            ]
        },
        raw_shard_hashes=[
            dict(path=str(p), sha256=sha256_file(p))
            for p in sorted(raw.rglob("*"))
            if p.is_file() and "inputs" not in p.parts and p.parent != raw / "logs"
        ],
        started_monotonic_ns=began,
        ended_monotonic_ns=time.monotonic_ns(),
        started_wall_ns=began_wall,
        duration_s=(time.monotonic_ns() - began) / 1e9,
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["state_manifest"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Trajectory readiness requires complete causal work, never favorable learning."""
    owned = bool(receipts) and all(r["passed"] and r.get("normal_exit", True) for r in receipts)
    complete = (
        len(work["state_manifest"]) == 20
        and len(work["issued_rows"]) == 256 * 20 * 5
        and len(work["release_rows"]) == 236 * 20
        and all(r["passed"] for r in work["restart_state_hashes"])
    )
    owned = owned and (not work["input_ready"] or complete)
    verdict = "disqualified" if not owned else "blocked" if not work["input_ready"] else "null"
    units = [
        r
        for r in work["rows"]
        if r["condition"] == "later_stream"
        and r["seed"] == 101
        and r["arm"] == "frozen_qwen_offset"
        and r["metric"] == "brier"
    ]
    value = dict(
        work,
        experiment_id=8211,
        task_id=TASK,
        milestone="2026.10.709",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            "causal_trajectory_benefit_reserved_for_8212"
            if verdict == "null"
            else "owned_validation"
            if verdict == "disqualified"
            else next(r["check"] for r in work["gate_check_summary"] if not r["passed"])
        ),
        verdict_class=verdict,
        required_checks_passed=owned,
        learning_trajectory_ready_score=int(owned and bool(work["input_ready"]) and complete),
        continuous_self_learning_task=True,
        no_model_weight_mutation=True,
        verifier_is_oracle=True,
        exposure_scope="historically_exposed_development_replay_chronology",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if work["input_ready"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        intended_count=192,
        completed_count=sum(r["status"] == "completed" for r in units),
        excluded_count=sum(r["status"] == "excluded" for r in units) if units else 192,
        censored_count=sum(r["status"] == "censored" for r in units),
        failed_count=0,
        independent_count=len(
            {r["source_cluster_id"] for r in units if r["status"] == "completed"}
        ),
        random_seed=101,
        reductions=m.engine.historical.reductions(work["rows"]),
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        acceptance_gates=dict(
            qualification="Exp8206 both readiness fields ==1",
            protocol=m.PROTOCOL_HASH,
            chronology="issue fsync, release at origin+20, one-use admission, no tail flush",
            recovery="genuine exit73, complete state and issued-row parity",
            validation="all owned checks and100 percent newly owned statements",
            benefit="reserved for independent Exp8212 audit; readiness does not imply benefit",
        ),
        claim_scope="Natural causal calibrated-memory trajectory only; benefit requires the separate frozen audit.",
        methodology_note="No model loaded. Frozen V707 CPU fits and original five controls use released labels and unchanged action thresholds. Gaussian overlap is diagnostic only. No future-origin permutation, retention tuning or generator weight update.",
        support_overlap_method=dict(
            source="https://arxiv.org/abs/2511.12828",
            scope="Diagnostic motivated by activation overlap; no KAN forgetting theorem is transferred to Gaussian heads",
            formula="exp(-squared_center_distance/(4*sigma^2))",
            binary_gaussian_support="full space; continuous normalized overlap is reported instead of a tuned support threshold",
        ),
    )
    from scripts.experiment_template import normalize_artifact_for_template_write

    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind exact original sources, causal events and current no-model execution; exposed development earns no generalization credit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reexecute sealed primitives before accepting any headline or fresh checksum."""
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value) or value["experiment_id"] != 8211:
            return False
        headline = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        if any(
            value[key] != headline[key]
            for key in [
                "completed_count",
                "learning_trajectory_ready_score",
                "required_checks_passed",
                "verdict_class",
            ]
        ):
            return False
        for receipt in value["validation_receipts"] + value["child_exit_rows"]:
            for prefix in ["stdout", "stderr"]:
                if (
                    prefix + "_path" in receipt
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        for name, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                return False
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        if value["input_ready"]:
            trajectory = Path(value["trajectory_path"])
            if sha256_file(trajectory) != value["trajectory_sha256"]:
                return False
            sealed = json.loads(trajectory.read_text())
            for key in [
                "issued_rows",
                "release_rows",
                "admission_rows",
                "support_overlap_rows",
                "hardware_receipt",
                "retention_predictions",
                "restart_state_hashes",
                "state_manifest",
            ]:
                if value[key] != sealed["evidence"][key]:
                    return False
            manifests = value["input_manifests"]
            public = json.loads(Path(manifests["stream_feature_manifest"]["path"]).read_text())[
                "rows"
            ]
            retained = json.loads(
                Path(manifests["retention_feature_manifest"]["path"]).read_text()
            )["rows"]
            states = []
            for index, (original, deliveries) in enumerate(
                zip(sealed["states"], sealed["releases"], strict=True)
            ):
                directory = Path(value["state_manifest"][index]["state"]["path"]).parent
                if (
                    [r["issued"] for r in journal(directory / "issued.jsonl")] != original["issued"]
                    or journal(directory / "released.jsonl") != deliveries
                    or json.loads((directory / "final.json").read_text()) != original
                ):
                    return False
                labels: list[int | None] = [None] * 256
                for delivery in deliveries:
                    labels[delivery["label_slot"] - 1] = delivery["outcome"]["y"]
                rebuilt = m.run(public, labels, original["seed"])
                if m.stable(rebuilt) != m.stable(original):
                    return False
                states.append(rebuilt)
                progress("cold_replay_states", index + 1, 20 - index - 1)
            reconstructed = evidence(states, public, sealed["releases"], retained)
            for key in reconstructed:
                if key != "rows" and reconstructed[key] != value[key]:
                    return False
            vault = m.engine.historical.LabelVault(
                Path(manifests["evaluator_label_manifests"]["stream"]["path"]), public
            )
            for deliveries in sealed["releases"]:
                for delivery in deliveries:
                    if delivery["outcome"] != vault.release(
                        delivery["label_slot"], delivery["release_slot"], sealed=True
                    ):
                        return False
            vault = m.engine.historical.LabelVault(
                Path(manifests["evaluator_label_manifests"]["retention"]["path"]), retained
            )
            for prediction in reconstructed["retention_predictions"]:
                row = retained[prediction["slot"] - 1]
                label = vault.release(row["slot"], 256, sealed=True, retention=True)
                reconstructed["rows"].extend(
                    dict(r, arm=m.NAMES[r["arm"]])
                    for r in m.engine.historical.scored(
                        row, prediction, label, "retention", prediction["seed"]
                    )
                )
            if reconstructed["rows"] != value["rows"]:
                return False
        rebuilt_value = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        return all(
            value[k] == rebuilt_value[k]
            for k in [
                "reductions",
                "completed_count",
                "excluded_count",
                "censored_count",
                "independent_count",
                "intended_count",
                "verdict_class",
                "honest_verdict",
                "required_checks_passed",
                "learning_trajectory_ready_score",
            ]
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
