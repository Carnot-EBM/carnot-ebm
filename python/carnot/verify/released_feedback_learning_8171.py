"""REQ-VERIFY-8171: replay exposed sources on the qualified release-aware clock.

The existing learner remains the numerical authority. Scoped adapters replace
its old schedule and authenticate the new qualification without changing any
historical implementation or primary artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import admission_horizon_methods_8152 as schedule
from carnot.verify import delayed_energy_memory_8143 as previous
from carnot.verify import native_radial_8105 as native

Json = dict[str, Any]
ROOT = schedule.ROOT
NAME = "experiment_8171_v706_released_feedback_learning"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/released_feedback_learning_8171.py"
RUNNER = "python/carnot/reporting/released_feedback_execution_8171.py"
TEST = "tests/python/test_released_feedback_learning_8171.py"
UPSTREAM = "results/experiment_8165_v706_learning_qualification.json"
MODEL_SPECS: list[Json] = []
engine = schedule.engine
progress = schedule.progress
ORIGINAL_TRAIN = engine.train
ORIGINAL_DESIGN = engine.historical.radial.design
ORIGINAL_RESTART_SPECS = previous.restart_specs


def authenticate(root: Path, raw: Path, binder: Any) -> Json:
    """The qualified schedule is the sole branch gate; other checks bind bytes."""
    binder.upstream = "exp8165-learning-qualification"
    path = root / UPSTREAM
    value = binder.read(path)
    binder.require(
        path, "learning_protocol_ready_score", 1, value.get("learning_protocol_ready_score")
    )
    terminal = engine.methods.historical.terminal(path, value, binder)
    binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    binder.require(path, "protocol_sha256", schedule.PROTOCOL_HASH, value["protocol_sha256"])
    binder.bind(Path(value["protocol_path"]), schedule.PROTOCOL_HASH)
    exclusion = root / "ops/exclusion_manifest.yaml"
    binder.bind(exclusion)
    retired = yaml.safe_load(exclusion.read_text()).get("retired_experiments", [])
    binder.require(
        exclusion,
        "experiment8171_not_retired",
        True,
        not any(r.get("experiment_id") == 8171 for r in retired),
    )
    for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
        operand = ROOT / ".venv/bin" / tool
        binder.require(operand, "runtime_" + tool, True, operand.is_file())
    binder.bind(
        Path(
            os.environ.get(
                "CARNOT_8105_EXTENSION",
                str(ROOT / "target/experiment-8105-load/_rust.cpython-312-x86_64-linux-gnu.so"),
            )
        )
    )
    upstream = schedule.authenticate(root, raw, binder)
    binder.require(
        path,
        "original_slot_mask",
        value["original_slot_mask"],
        upstream["direct_stream_authentication"]["original_slot_mask"],
    )
    for role in ["stream", "retention"]:
        binder.require(
            path,
            role + "_feature_manifest",
            value["input_manifests"][role + "_feature_manifest"],
            upstream[role + "_feature_manifest"],
        )
    return upstream


@contextmanager
def arithmetic(receipt: Json) -> Iterator[None]:
    """Use loaded Rust for update batches and retain independent NumPy parity.

    Predictions keep the qualified implementation. Each update records actual
    feature work and measured host buffer bytes; these are not device speeds.
    """
    os.environ.setdefault(
        "CARNOT_8105_EXTENSION",
        str(ROOT / "target/experiment-8105-load/_rust.cpython-312-x86_64-linux-gnu.so"),
    )
    module, loaded = native.extension()
    receipt.update(loaded, update_operations=0, update_buffer_bytes=0, update_batches=0)

    def design(state: Json, values: Any) -> NDArray[np.float64]:
        x = np.asarray(values, dtype=np.float64)
        encoded = dict(state, coefficients=[0.0] * (len(state["centers"]) + 1))
        result = np.asarray(module.RustRadial8105(json.dumps(encoded)).design(x.tolist()))
        expected = ORIGINAL_DESIGN(state, values)
        if not np.allclose(result, expected, atol=1e-12, rtol=0):
            raise ValueError("loaded_binding_parity")
        receipt["update_batches"] += 1
        receipt["update_operations"] += len(x) * len(state["centers"]) * 9
        receipt["update_buffer_bytes"] += x.nbytes + result.nbytes
        return result

    def train(head: Json, geometry: Json, pool: list[Json], *, reference: bool = False) -> Json:
        with patch.object(engine.historical.radial, "design", design):
            return ORIGINAL_TRAIN(head, geometry, pool, reference=reference)

    with patch.object(engine, "train", train):
        yield


def run_seed(
    rows: list[Json],
    label_path: Path,
    seed: int,
    raw: Path,
    *,
    state: Json | None = None,
    crash_slot: int = 0,
) -> Json:
    """Seal every issue before decoding feedback, so a crash cannot peek ahead."""
    raw.mkdir(parents=True, exist_ok=True)
    labels = previous.DelayedLabels(label_path, rows)
    labels.state = state or {}
    with (raw / "durable.jsonl").open("a") as stream:

        def seal(kind: str, current: Json) -> None:
            labels.state = current
            record = engine.durable_record(kind, current)
            record.update(
                event_id=f"{seed}:issue:{current['cursor']}",
                rng_state=current["rng_state"],
                optimizer={a: h["optimizer_step"] for a, h in current["arms"].items()},
                issued_ref=engine.shard(raw, dict(issued=current["issued"][-1])),
                heads_ref=engine.shard(raw, dict(arms=current["arms"])),
                candidate_ref=engine.shard(raw, dict(candidates=current["candidates"])),
            )
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if current["cursor"] % 32 == 0:
                atomic_json(raw / "checkpoint.json", current)
            if current["cursor"] == crash_slot:
                atomic_json(raw / "crash_checkpoint.json", current)
                progress("intentional_crash", current["cursor"], len(current["pending"]))
                os._exit(73)

        receipt: Json = {}
        with arithmetic(receipt):
            final = schedule.run(rows, labels, seed, state=state, seal=seal)
    atomic_json(raw / "hardware_receipt.json", receipt)
    atomic_json(raw / "opened_labels.json", labels.opened)
    atomic_json(raw / "final_state.json", final)
    return final


def restart_specs(raw: Path) -> list[Json]:
    """Freeze real crash and resume children through this experiment's CLI."""
    specs = ORIGINAL_RESTART_SPECS(raw)
    for spec in specs:
        spec["argv"] = [
            str(ROOT / CLI) if arg == str(ROOT / previous.CLI) else arg for arg in spec["argv"]
        ]
    return specs


def evidence(work: Json) -> Json:
    """Reconstruct transitions from complete states instead of counting admissions.

    Source IDs, labels and hashes make each center choice and later prediction
    inspectable. A safe installation is mechanics, not a statistical benefit.
    """
    events, installations, future, centers, consumption, references = [], [], [], [], [], []
    hardware: Json = dict(
        actual_loaded=False,
        update_operations=0,
        update_buffer_bytes=0,
        update_batches=0,
        durable_bytes=0,
        NPU_FPGA_speed="unmeasured_hypothesis",
    )
    if not work["input_ready"]:
        return dict(
            event_rows=events,
            installation_rows=installations,
            future_prediction_rows=future,
            center_rows=centers,
            feedback_consumption_rows=consumption,
            reference_rows=references,
            hardware_receipt=hardware,
        )
    manifests = work["input_manifests"]
    rows = json.loads(Path(manifests["stream_feature_manifest"]["path"]).read_text())["rows"]
    label_rows = json.loads(
        Path(manifests["evaluator_label_manifests"]["stream"]["path"]).read_text()
    )["rows"]
    targets = [r.get("y") for r in label_rows]
    for entry in work["state_manifest"]:
        seed = entry["seed"]
        path = Path(entry["state"]["path"])
        state = json.loads(path.read_text())
        references.append(dict(seed=seed, **schedule.reference(state, rows, targets)))
        current = json.loads((path.parent / "hardware_receipt.json").read_text())
        hardware.update({k: current[k] for k in ["actual_loaded", "path", "sha256", "module_file"]})
        for key in ["update_operations", "update_buffer_bytes", "update_batches"]:
            hardware[key] += current[key]
        hardware["durable_bytes"] += (path.parent / "durable.jsonl").stat().st_size
        for index, event in enumerate(state["events"]):
            event_id = f"{seed}:event:{index}"
            events.append(dict(seed=seed, event_id=event_id, **event))
            if event["kind"] == "release_feedback":
                slot = event["label_slot"]
                consumption.append(
                    dict(
                        seed=seed,
                        event_id=event_id,
                        issue_slot=slot,
                        release_slot=event["slot"],
                        source_cluster_id=rows[slot - 1]["source_cluster_id"],
                        role="update"
                        if slot in state["training"]
                        else "admission"
                        if slot in state["used_admission"]
                        else "unused_or_missing",
                        label=targets[slot - 1],
                        state_hash=event["state_hash"],
                    )
                )
            if event["kind"] == "commit_candidate":
                for arm, candidate in event["candidates"].items():
                    update_ids = [s for s in state["training"] if s + 20 <= event["slot"]][-64:]
                    for center in candidate["head"]["centers"]:
                        centers.append(
                            dict(
                                seed=seed,
                                arm=arm,
                                commit_slot=event["slot"],
                                center_id=center["source_id"],
                                center_hash=canonical_hash(center),
                                labels_consumed=update_ids,
                                candidate_hash=canonical_hash(candidate),
                                optimizer_step=candidate["head"]["optimizer_step"],
                            )
                        )
            if event["kind"] in ["admit_once", "defer_candidate"]:
                slot = event["slot"]
                installations.append(
                    dict(
                        seed=seed,
                        **event,
                        remaining_original_slots=256 - slot,
                        remaining_resolved_slots=max(0, 236 - slot),
                        remaining_opportunities=sum(s > slot for s in [64, 144]),
                        rejected_arms=[a for a, step in event.get("steps", {}).items() if not step],
                        later_changed_predictions={
                            a: sum(
                                p["slot"] > slot
                                and p["predictions"][a] is not None
                                and abs(p["predictions"][a] - p["predictions"][engine.ARMS[0]])
                                > 1e-12
                                for p in state["issued"]
                            )
                            for a in event.get("steps", {})
                        },
                    )
                )
        for row, issued in zip(rows[64:], state["issued"][64:], strict=True):
            base = issued["predictions"][engine.ARMS[0]]
            for arm in engine.ARMS[1:]:
                probability = issued["predictions"][arm]
                future.append(
                    dict(
                        seed=seed,
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        slot=row["slot"],
                        arm=arm,
                        prediction=probability,
                        frozen_prediction=base,
                        changed=probability is not None and abs(probability - base) > 1e-12,
                        prediction_hash=issued["prediction_hash"],
                    )
                )
        progress("independent_seed_reduction", seed - 100, 120 - seed)
    return dict(
        event_rows=events,
        installation_rows=installations,
        future_prediction_rows=future,
        center_rows=centers,
        feedback_consumption_rows=consumption,
        reference_rows=references,
        hardware_receipt=hardware,
    )


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Reuse the qualified measurement and swap only its schedule/custody adapters."""
    progress("before_released_feedback_measurement")
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (
        patch.object(previous, "authenticate", authenticate),
        patch.object(previous, "run_seed", run_seed),
        patch.object(previous, "restart_specs", restart_specs),
    ):
        work = previous.measure(root, raw, fixture=fixture)
    work.update(evidence(work))
    work["trajectory_manifest"] = work["state_manifest"]
    work.setdefault("retention_prediction_manifest", {})
    work["v705_protocol"] = schedule.protocol()
    work["numerical_protocol"] = work["v705_protocol"]
    work["protocol_sha256"] = schedule.PROTOCOL_HASH
    protocol_path = raw / "protocol.json"
    protocol_path.write_bytes((ROOT / schedule.PROTOCOL).read_bytes())
    work["protocol_path"] = str(protocol_path)
    work["cited_upstream_artifacts"] = [
        dict(experiment_id=n, fields_imported=fields, sha256=sha256_file(root / name))
        for n, name, fields in [
            (8165, UPSTREAM, ["learning_protocol_ready_score", "protocol_sha256"]),
            (
                8111,
                engine.historical.UPSTREAM,
                ["original_slot_mask", "historical_model_provenance"],
            ),
            (8143, schedule.scalar.UPSTREAM, []),
        ]
        if (root / name).is_file()
    ]
    if work["input_ready"] and not fixture:
        upstream = json.loads((root / engine.historical.UPSTREAM).read_text())
        work["historical_model_provenance"] = upstream.get("historical_model_provenance", {})
    work["preconditions_checked"] = dict(
        runtime_executable=str(ROOT / ".venv/bin/python"),
        model_loads=0,
        gate="Exp8165.learning_protocol_ready_score==1",
        protocol_sha256=schedule.PROTOCOL_HASH,
    )
    work["code_config_hashes"].update(
        {
            p: sha256_file(ROOT / p)
            for p in [
                MODULE,
                RUNNER,
                CLI,
                TEST,
                schedule.MODULE,
                schedule.PROTOCOL,
                "crates/carnot-python/src/radial_8105.rs",
            ]
        }
    )
    work["raw_shard_hashes"].append(
        dict(path=str(protocol_path), sha256=sha256_file(protocol_path))
    )
    work["phase_spans"].append(
        dict(
            phase="independent_transition_reconstruction",
            duration_s=time.monotonic() - started - work["duration_s"],
        )
    )
    work["duration_s"] = time.monotonic() - started
    atomic_json(raw / "primitive_rows.json", dict(rows=work["rows"]))
    work["raw_shard_hashes"].append(
        dict(path=str(raw / "primitive_rows.json"), sha256=sha256_file(raw / "primitive_rows.json"))
    )
    atomic_json(raw / "measurement.json", work)
    progress("after_released_feedback_measurement", len(work["state_manifest"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Normal owned validation grants trajectory readiness, never benefit credit."""
    owned = bool(receipts) and all(r["passed"] and r.get("normal_exit", True) for r in receipts)
    value = previous.build(work, raw, [dict(passed=owned)])
    value.pop("reproducibility_checksum")
    value.update(
        experiment_id=8171,
        task_id="exp8171-released-feedback-learning",
        milestone="2026.10.706",
        run_date="20261005",
        claim_scope="Causal releases and later decisions on historical development replay; H2 benefit reserved for Exp8172.",
        exposure_scope="private_circular_fixture"
        if work["fixture_mode"]
        else "historically_exposed_development_replay_chronology",
        validation_receipts=receipts,
        trained_head_specs=[
            dict(
                schedule.protocol()["trained_head_specs"],
                scope="private_fixture"
                if work["fixture_mode"]
                else "historically_exposed_development_replay",
            )
        ],
        intended_count=256,
        acceptance_gates=dict(
            qualification="Exp8165.learning_protocol_ready_score==1",
            conformance="V705 immutable protocol; scalar events, delay20/capacity32 and sealed retention",
            validation="normal owned checks;100 percent added statements",
            benefit="reserved for Exp8172 H2 audit",
        ),
    )
    value["field_principles"].update(
        {
            k: "Causal original-slot evidence; exposed development and repeated seeds add no generalization credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold execution and independent transitions reject a freshly rehashed forgery."""
    try:
        value = json.loads(path.read_text())
        if (
            value["experiment_id"] != 8171
            or value["task_id"] != "exp8171-released-feedback-learning"
        ):
            return False
        if (
            value["v705_protocol"] != schedule.protocol()
            or value["protocol_sha256"] != schedule.PROTOCOL_HASH
        ):
            return False
        rebuilt = evidence(value)
        if rebuilt != {k: value[k] for k in rebuilt}:
            return False
        with (
            arithmetic({}),
            patch.object(engine, "run", schedule.run),
            patch.object(engine, "protocol", lambda: schedule.protocol()["delayed_memory"]),
        ):
            if not previous.replay(path):
                return False
        expected = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        return all(
            expected[k] == value[k]
            for k in [
                "learning_trajectory_ready_score",
                "verdict_class",
                "required_checks_passed",
                "trained_head_specs",
                "trajectory_manifest",
            ]
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
