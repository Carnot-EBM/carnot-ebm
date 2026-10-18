"""REQ-VERIFY-8433: two durable transactions protect each bounded feedback block.

Only admitted feature records enter the learner. Synthetic evaluator labels
are supplied after sealing; audit files never enter prediction or update APIs.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.direct_atomic_state_8376 import sync_directory
from carnot.verify import sparse_energy_7996 as optimizer

Json = dict[str, Any]
SCHEMA = dict(
    name="v726-block-feedback-v1",
    inputs=9,
    coefficients=109,
    siblings=3,
    block_questions=20,
    transactions_per_block=2,
)
BARRIERS = ("temporary_write", "file_fsync", "atomic_replace", "acknowledgment")


def head() -> Json:
    """A nontrivial fixed head makes a no-op optimizer visible to controls."""
    return dict(
        arm="spline",
        parameters=(0.03 * np.sin(np.arange(109))).tolist(),
        decay_scale=0.9,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
        temperature=1.3,
        parameter_count=109,
    )


def config(
    capacity: int = 8, policy: str = "fifo", *, frozen: bool = False, shuffled: bool = False
) -> Json:
    """Keep initial numerical and random operands independent of saved state."""
    if capacity not in (8, 16) or policy not in ("fifo", "uniform"):
        raise ValueError("configuration")
    return dict(
        capacity=capacity, policy=policy, frozen=frozen, shuffled=shuffled, seed=17, head=head()
    )


def rng_state(seed: int) -> list[Any]:
    """JSON lists preserve the complete generator position across processes."""
    return list(json.loads(json.dumps(random.Random(seed).getstate())))


def restore(value: list[Any]) -> random.Random:
    """Restore all generator words instead of reseeding a partially consumed stream."""
    rng = random.Random()
    rng.setstate((value[0], tuple(value[1]), value[2]))
    return rng


def initial(cfg: Json) -> Json:
    """The new state shape cannot inherit qualification from the smaller old head."""
    h = cfg["head"]
    if len(h["parameters"]) != 109 or len(h["scaler"]["minimum"]) != 9 or h["temperature"] <= 0:
        raise ValueError("head_schema")
    return dict(
        schema=SCHEMA,
        configuration_hash=canonical_hash(cfg),
        version=0,
        phase="updated",
        cursor=0,
        head=deepcopy(h),
        scaler_temperature_hash=canonical_hash([h["scaler"], h["temperature"]]),
        issued={},
        selected_cluster_ids=[],
        pending=[],
        applied_feedback_ids=[],
        release_hashes={},
        acknowledgment_ids={},
        input_hashes={},
        admission_rng=rng_state(cfg["seed"]),
        feedback_rng=rng_state(cfg["seed"] + 1),
        pending_high_water=0,
        parameter_updates=0,
    )


def questions(block: int) -> list[Json]:
    """Public synthetic features contain no labels or historical model outputs."""
    rng = random.Random(843300 + block)
    return [
        dict(
            id=f"q{20 * block + i:03}",
            answers=[
                dict(id=f"p{20 * block + i:03}-{s}", x=[rng.uniform(0.05, 0.95) for _ in range(9)])
                for s in range(3)
            ],
        )
        for i in range(20)
    ]


def feedback(state: Json) -> Json:
    """Evaluator plumbing exposes only the already admitted, sealed answer labels."""
    if state["phase"] != "sealed":
        raise ValueError("unsealed_evaluator")
    return {r["id"]: (int(r["id"][1:4]) + int(r["id"][-1])) % 2 for r in state["pending"]}


def seal(state: Json, cfg: Json, block: int, records: list[Json]) -> Json:
    """Seal every sibling prediction before admitting any target to the updater."""
    if state["phase"] != "updated" or state["cursor"] != block * 20:
        raise ValueError("seal_order")
    if (
        len(records) != 20
        or len({q["id"] for q in records}) != 20
        or any(len(q["answers"]) != 3 or set(q) != {"id", "answers"} for q in records)
    ):
        raise ValueError("block_roster")
    answers = [dict(a, cluster=q["id"]) for q in records for a in q["answers"]]
    if len({a["id"] for a in answers}) != 60 or any(
        set(a) != {"id", "x", "cluster"} or len(a["x"]) != 9 or not np.isfinite(a["x"]).all()
        for a in answers
    ):
        raise ValueError("answer_roster")
    out = deepcopy(state)
    rng = restore(out["admission_rng"])
    chosen = (
        set(range(cfg["capacity"]))
        if cfg["policy"] == "fifo"
        else set(rng.sample(range(20), cfg["capacity"]))
    )
    out["admission_rng"] = list(json.loads(json.dumps(rng.getstate())))
    out["selected_cluster_ids"] = [q["id"] for i, q in enumerate(records) if i in chosen]
    probabilities = optimizer.predict(
        state["head"], np.asarray([a["x"] for a in answers], dtype=float)
    )
    for record, probability in zip(answers, probabilities, strict=True):
        if record["id"] in out["issued"]:
            raise ValueError("prediction_identity")
        out["issued"][record["id"]] = dict(
            prediction_hash=canonical_hash([record, float(probability)]),
            head_hash=canonical_hash(state["head"]),
            block=block,
        )
    out["pending"] = [a for a in answers if a["cluster"] in out["selected_cluster_ids"]]
    out["pending_high_water"] = max(out["pending_high_water"], len(out["pending"]))
    out.update(phase="sealed", version=state["version"] + 1)
    out["input_hashes"][str(block)] = canonical_hash(records)
    acknowledge(out, block, "seal")
    return out


def release(state: Json, cfg: Json, block: int, labels: Json) -> Json:
    """All tentative answer updates become durable together with the next cursor."""
    if state["phase"] != "sealed" or state["cursor"] != block * 20:
        raise ValueError("unsealed_release")
    if set(labels) != {r["id"] for r in state["pending"]} or any(
        type(y) is not int or y not in (0, 1) for y in labels.values()
    ):
        raise ValueError("release_roster")
    out = deepcopy(state)
    rng = restore(out["feedback_rng"])
    targets = [labels[r["id"]] for r in state["pending"]]
    if cfg["shuffled"]:
        rng.shuffle(targets)
    out["feedback_rng"] = list(json.loads(json.dumps(rng.getstate())))
    for record, target in zip(state["pending"], targets, strict=True):
        identity = f"feedback:{block}:{record['id']}"
        if identity in out["applied_feedback_ids"]:
            raise ValueError("duplicate_feedback")
        if not cfg["frozen"]:
            out["head"], _ = optimizer.update(
                out["head"], np.asarray(record["x"], dtype=float), target, 0.01
            )
            out["parameter_updates"] += 1
        out["applied_feedback_ids"].append(identity)
    out["release_hashes"][str(block)] = canonical_hash(labels)
    out.update(pending=[], phase="updated", cursor=(block + 1) * 20, version=state["version"] + 1)
    acknowledge(out, block, "release")
    return out


def acknowledge(state: Json, block: int, transaction: str) -> None:
    """A stable effect seal survives a crash before a caller receives delivery."""
    state["acknowledgment_ids"][f"{transaction}:{block}"] = canonical_hash(
        [
            state["configuration_hash"],
            transaction,
            block,
            state["head"],
            state["issued"],
            state["cursor"],
        ]
    )


def replay_state(state: Json, cfg: Json) -> bool:
    """The synthetic audit reconstructs semantics without trusting saved checksums.

    This evaluator is separate from Store. Its fixtures qualify recovery only;
    a human-label consumer must bring independent released-feedback operands.
    """
    if type(state.get("version")) is not int or not 0 <= state["version"] <= 10:
        return False
    expected = initial(cfg)
    for version in range(state["version"]):
        block = version // 2
        expected = (
            seal(expected, cfg, block, questions(block))
            if version % 2 == 0
            else release(expected, cfg, block, feedback(expected))
        )
    return bool(state == expected)


class Store:
    """One writer publishes immutable complete versions for concurrent readers."""

    def __init__(self, path: Path, configuration: Json):
        self.path, self.configuration = path, deepcopy(configuration)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def initialize(self) -> None:
        """Initialization cannot silently erase a lost durable current pointer."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (self.path / "current.json").exists():
                if any(self.path.glob("version-*.json")):
                    raise FileNotFoundError("missing_durable_pointer")
                self.commit(initial(self.configuration))

    def read(self) -> Json:
        """Read one immutable snapshot so readers never join half of two versions."""
        pointer = json.loads((self.path / "current.json").read_bytes())
        if Path(pointer["file"]).name != pointer["file"]:
            raise ValueError("pointer_path")
        envelope = json.loads((self.path / pointer["file"]).read_bytes())
        state: Json = envelope["state"]
        if envelope["sha256"] != canonical_hash(state) or pointer["sha256"] != envelope["sha256"]:
            raise ValueError("state_checksum")
        if state["schema"] != SCHEMA or state["configuration_hash"] != canonical_hash(
            self.configuration
        ):
            raise ValueError("state_schema")
        if (
            len(state["pending"]) > 3 * self.configuration["capacity"]
            or len({r["cluster"] for r in state["pending"]}) > self.configuration["capacity"]
        ):
            raise ValueError("pending_capacity")
        return state

    def commit(self, state: Json, hook: Callable[[str], None] = lambda phase: None) -> None:
        """Snapshot durability precedes pointer durability and caller acknowledgment."""
        digest = canonical_hash(state)
        target = self.path / ("version-" + digest.split(":")[1] + ".json")
        if not target.exists():
            atomic_json(target, dict(state=state, sha256=digest))
            sync_directory(self.path)
        temporary = self.path / ".current.tmp"
        with temporary.open("wb") as stream:
            stream.write(json.dumps(dict(file=target.name, sha256=digest), sort_keys=True).encode())
            hook("temporary_write")
            stream.flush()
            os.fsync(stream.fileno())
            hook("file_fsync")
        temporary.replace(self.path / "current.json")
        sync_directory(self.path)
        hook("atomic_replace")
        hook("acknowledgment")

    def seal(
        self, block: int, records: list[Json], hook: Callable[[str], None] = lambda phase: None
    ) -> str:
        """Repeated delivery returns its saved effect without another transaction."""
        return self.apply("seal", block, records, hook)

    def release(
        self, block: int, labels: Json, hook: Callable[[str], None] = lambda phase: None
    ) -> str:
        """Repeated committed labels cannot change coefficients or the random position."""
        return self.apply("release", block, labels, hook)

    def apply(self, transaction: str, block: int, operand: Any, hook: Callable[[str], None]) -> str:
        """Lock reading and committing together so a second caller sees saved effects."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.read()
            identity = f"{transaction}:{block}"
            if identity in state["acknowledgment_ids"]:
                ledger = "input_hashes" if transaction == "seal" else "release_hashes"
                if state[ledger][str(block)] != canonical_hash(operand):
                    raise ValueError("duplicate_conflict")
                return str(state["acknowledgment_ids"][identity])
            changed = (
                seal(state, self.configuration, block, operand)
                if transaction == "seal"
                else release(state, self.configuration, block, operand)
            )
            self.commit(changed, hook)
            return str(changed["acknowledgment_ids"][identity])


def drive(path: Path, cfg: Json, crash: str = "") -> Json:
    """Resume the committed phase; tentative updates never become a second application."""
    store = Store(path, cfg)
    store.initialize()
    while store.read()["cursor"] < 100:
        state = store.read()
        block = state["cursor"] // 20
        for transaction in ("seal", "release"):

            def hook(boundary: str) -> None:
                """Real exits force the operating system to close unsaved process state."""
                if crash == f"{transaction}:{block}:{boundary}":
                    print(f"[exp8433] actual_exit73={crash}", flush=True)
                    os._exit(73)

            if transaction == "seal" and state["phase"] == "updated":
                store.seal(block, questions(block), hook)
            if transaction == "release":
                sealed = store.read()
                store.release(block, feedback(sealed), hook)
        print(f"[exp8433] blocks_completed={block + 1} blocks_pending={4 - block}", flush=True)
        if crash == "after_block2" and block == 2:
            print("[exp8433] actual_exit73=after_block2", flush=True)
            os._exit(73)
    return store.read()
