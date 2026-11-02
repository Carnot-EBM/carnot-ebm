"""REQ-VERIFY-8434: bounded released feedback changes a historical numeric head.

The qualified store owns all durable parameter writes. Evaluator children keep
unused public targets and tune labels outside the learner's update inputs.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.reporting import block_checkpoint_qualification_8433 as qualification
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.verify import block_feedback_state_8433 as state
from carnot.verify import sparse_energy_7996 as sparse

Json = dict[str, Any]
ROOT = qualification.ROOT
NAME, TASK = "experiment_8434_v726_capacity_online_learning", "exp8434-capacity-online-learning"
CLI = f"scripts/experiments/{NAME}.py"
ARMS = ("frozen", "FIFO_updates", "uniform_updates", "shuffled_released_feedback")
SEEDS = (17, 29, 43, 71, 101)
MODEL_SPECS: list[Json] = []
reference, read = qualification.reference, qualification.read
independent_update = qualification.independent_update


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts make each bounded phase visible to the supervising process."""
    print(f"[exp8434] phase={phase} completed={completed} pending={pending}", flush=True)


def configuration(
    head: Json,
    blocks: list[list[Json]],
    seed: int,
    capacity: int,
    arm: str,
    labels: Json,
    tune: Json,
) -> Json:
    """Freeze the update rule before a learner sees any feedback or test target."""
    if arm not in ARMS or seed not in SEEDS or len(blocks) != 5:
        raise ValueError("stream_configuration")
    cfg = state.config(
        capacity,
        "fifo" if arm in ARMS[:2] else "uniform",
        frozen=arm == "frozen",
        shuffled=arm == ARMS[3],
    )
    cfg.update(
        head=deepcopy(head),
        seed=seed,
        research_arm=arm,
        blocks=blocks,
        evaluator_labels=labels,
        retention_labels=tune,
        learning_rate=0.01,
        l2=0.001,
        importance_weighting=False,
    )
    return dict(cfg)


def steps(directory: Path) -> list[Json]:
    """Audit records live apart from the bounded learner's pending feature records."""
    return [json.loads(p.read_bytes()) for p in sorted((directory / "steps").glob("*.json"))]


def predictions(cfg: Json, before: Json, block: int) -> list[Json]:
    """All siblings receive predictions from the same pre-release head version."""
    answers = [dict(a, cluster=q["id"]) for q in cfg["blocks"][block] for a in q["answers"]]
    probabilities = sparse.predict(before["head"], np.asarray([a["x"] for a in answers]))
    return [
        dict(
            id=a["id"],
            cluster=a["cluster"],
            probability=float(p),
            issued_version=before["version"],
            head_hash=canonical_hash(before["head"]),
        )
        for a, p in zip(answers, probabilities, strict=True)
    ]


def released(cfg: Json, sealed: Json) -> Json:
    """The evaluator decodes targets only after independently checking the issue seal."""
    if sealed["phase"] != "sealed":
        raise ValueError("unsealed_release")
    block = sealed["cursor"] // 20
    answers = [dict(a, cluster=q["id"]) for q in cfg["blocks"][block] for a in q["answers"]]
    probs = sparse.predict(sealed["head"], np.asarray([a["x"] for a in answers]))
    selected = set(sealed["selected_cluster_ids"])
    if (
        len(selected) != cfg["capacity"]
        or sealed["pending"] != [a for a in answers if a["cluster"] in selected]
        or sealed["input_hashes"][str(block)] != canonical_hash(cfg["blocks"][block])
        or len(sealed["issued"]) != (block + 1) * 60
        or any(
            sealed["issued"][a["id"]]["prediction_hash"] != canonical_hash([a, float(p)])
            for a, p in zip(answers, probs, strict=True)
        )
    ):
        raise ValueError("prediction_seal")
    vault = {r["unit_id"]: r["y"] for r in read(cfg["evaluator_labels"])["rows"]}
    result = {r["id"]: vault[r["id"]] for r in sealed["pending"]}
    if any(type(y) is not int or y not in (0, 1) for y in result.values()):
        raise ValueError("unavailable_admitted_target")
    return result


def update_trace(sealed: Json, cfg: Json, block: int, labels: Json) -> list[Json]:
    """Reconstruct tentative steps for audit; the store separately commits their effect."""
    targets = [labels[r["id"]] for r in sealed["pending"]]
    rng = state.restore(sealed["feedback_rng"])
    if cfg["shuffled"]:
        rng.shuffle(targets)
    h = deepcopy(sealed["head"])
    trace = []
    for record, target in zip(sealed["pending"], targets, strict=True):
        x = np.asarray(record["x"], dtype=float)
        theta = sparse.parameters(h)
        ids, gradient = sparse.sparse_gradient(h, x, target)
        probability = float(sparse.predict(h, x[None, :])[0])
        changed = h if cfg["frozen"] else sparse.update(h, x, target, 0.01)[0]
        trace.append(
            dict(
                id=record["id"],
                target=target,
                original_target=labels[record["id"]],
                feedback_id=f"feedback:{block}:{record['id']}",
                probability_before_update=probability,
                log_loss=float(
                    -target * np.log(probability) - (1 - target) * np.log1p(-probability)
                ),
                gradient_ids=ids.tolist(),
                data_gradient=gradient.tolist(),
                l2_gradient=(0.002 * theta).tolist(),
                coefficient_delta=(sparse.parameters(changed) - theta).tolist(),
                before_head_hash=canonical_hash(h),
                after_head_hash=canonical_hash(changed),
                tentative_until_block_commit=True,
                parameter_write=not cfg["frozen"],
            )
        )
        h = changed
    return trace


def semantic_step(step: Json) -> Json:
    """Clock observations differ between runs; every scientific operand must match."""
    return {k: v for k, v in step.items() if k not in {"issued_wall_ns", "released_wall_ns"}}


def check_step(cfg: Json, step: Json) -> Json:
    """Replay from pinned inputs rather than trusting self-consistent checkpoint hashes."""
    b, before = step["block"], step["before_state"]
    sealed = state.seal(before, cfg, b, cfg["blocks"][b])
    labels = released(cfg, sealed)
    after = state.release(sealed, cfg, b, labels)
    if (
        step["sealed_state"] != sealed
        or step["released_feedback"] != labels
        or step["predictions"] != predictions(cfg, before, b)
        or step["updates"] != update_trace(sealed, cfg, b, labels)
        or step["after_state"] != after
        or step["checkpoint_hash"] != canonical_hash(after)
        or step["sealed_checkpoint_hash"] != canonical_hash(sealed)
        or step["issued_wall_ns"] >= step["released_wall_ns"]
    ):
        raise ValueError("primitive_stream_drift")
    return dict(after)


def worker(cfg: Json, directory: Path, crash: bool = False) -> Json:
    """Run only released updates; evaluator children retain all unused target labels."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    config_path = directory / "configuration.json"
    atomic_json(config_path, cfg)
    store = state.Store(directory / "store", cfg)
    store.initialize()
    saved = steps(directory)
    current = store.read()
    if saved and saved[-1]["after_state"] != current:
        raise ValueError("restart_audit_state")
    began = time.monotonic()
    for block in range(current["cursor"] // 20, 5):
        if time.monotonic() - began > 240:
            raise TimeoutError("seed_capacity_deadline")
        progress("before_block", block, 5 - block)
        before = store.read()
        issued = predictions(cfg, before, block)
        store.seal(block, cfg["blocks"][block])
        sealed = store.read()
        issue_path = directory / "issues" / f"{block}.json"
        wall = time.time_ns()
        atomic_json(issue_path, dict(state=sealed, predictions=issued, issued_wall_ns=wall))
        release_path = directory / "releases" / f"{block}.json"
        receipt = child(
            f"release_{block}",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--configuration",
                str(config_path),
                "--release",
                str(issue_path),
                "--operand-output",
                str(release_path),
            ],
            directory / "evaluator_processes",
            deadline=30,
        )
        if not receipt["passed"]:
            raise ValueError("released_feedback_child_failed")
        release = read(reference(release_path))
        labels = release["labels"]
        trace = update_trace(sealed, cfg, block, labels)
        store.release(block, labels)
        after = store.read()
        step = dict(
            block=block,
            before_state=before,
            sealed_state=sealed,
            after_state=after,
            predictions=issued,
            issued_wall_ns=wall,
            released_wall_ns=release["released_wall_ns"],
            released_feedback=labels,
            updates=trace,
            checkpoint_hash=canonical_hash(after),
            sealed_checkpoint_hash=canonical_hash(sealed),
        )
        atomic_json(directory / "steps" / f"{block:02}.json", step)
        retention_receipt = child(
            f"retention_{block}",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--configuration",
                str(config_path),
                "--retention",
                str(directory),
                "--operand-output",
                str(directory / "retention" / f"{block}.json"),
            ],
            directory / "evaluator_processes",
            deadline=30,
        )
        if not retention_receipt["passed"]:
            raise ValueError("retention_child_failed")
        progress("after_block", block + 1, 4 - block)
        if crash and block == 2:
            progress("actual_exit73_after_block2", 3, 2)
            os._exit(73)
    atomic_json(directory / "final.json", store.read())
    return dict(store.read())


def retention(cfg: Json, stream: list[Json]) -> Json:
    """A read-only evaluator scores tune questions without returning labels to learning."""
    tune = [r for r in read(cfg["retention_labels"])["rows"] if r.get("role", "tune") == "tune"]
    if len(tune) != 150 or len({r["question_hash"] for r in tune}) != 50:
        raise ValueError("retention_roster")
    rows = []
    baseline = sparse.predict(cfg["head"], np.asarray([r["x"] for r in tune]))
    for step in stream:
        head = step["after_state"]["head"]
        probabilities = sparse.predict(head, np.asarray([r["x"] for r in tune]))
        for r, p, original in zip(tune, probabilities, baseline, strict=True):
            rows.append(
                dict(
                    unit_id=r["unit_id"],
                    question_hash=r["question_hash"],
                    boundary=step["block"],
                    probability=float(p),
                    target=r["y"],
                    brier=float((p - r["y"]) ** 2),
                    baseline_brier=float((original - r["y"]) ** 2),
                    head_hash=canonical_hash(head),
                    read_only=True,
                )
            )
        progress("retention_boundary", step["block"] + 1, 4 - step["block"])
    return dict(rows=rows, labels_fed_to_learner=0)
