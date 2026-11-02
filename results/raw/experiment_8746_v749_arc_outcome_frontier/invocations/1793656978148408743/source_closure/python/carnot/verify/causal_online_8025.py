"""REQ-SELF-8025: delayed sparse updates preserve the calibrated input geometry.

Only released labels enter the optimizer. A lazy multiplier applies global L2
without converting local coefficient writes into a dense parameter update.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # type: ignore[import-untyped]

from carnot import experiment_8008_v694_conditioned_energy_fit as conditioned
from carnot.experiment_8021_v695_typed_decision_test import action, loss
from carnot.experiment_8019_v695_eligible_targets import shard
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ("frozen_no_write", "uniform", "decision_loss", "brier_loss", "periodic")
CONFIG = dict(
    delay=20,
    slots=256,
    block=16,
    updates_per_block=4,
    maximum_updates=64,
    learning_rate=0.01,
    l2=0.001,
    seeds=list(range(101, 121)),
    primary_seed=17,
    costs=dict(unsupported_accept=5, supported_reject=1, escalate=0.5),
    priority="issued loss; seeded hash ties",
    periodic=[0, 4, 8, 12],
    terminal_flush=False,
    retention_labels_opened=False,
    benefit_gate="descriptive only; no retention or generalization qualification",
)
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts reveal work without adding artificial duration."""
    print(
        f"[exp8025] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def design(head: Json, row: Json) -> Array:
    """Reuse fitted geometry so feedback cannot change scaling or knot locations."""
    return conditioned.design(
        "conditioned_energy", np.asarray([[row["q"], *row["features"]]]), head["geometry"]
    )[0]


def coefficients(head: Json) -> Array:
    """Effective coefficients include the lazy global regularization multiplier."""
    return np.asarray(head["parameters"], dtype=float) * head["decay_scale"]


def probability(head: Json, x: Array) -> float:
    """Keep the imported affine calibration fixed throughout every trajectory."""
    a, b = head["calibration"]
    return float(expit(a + b * float(x @ coefficients(head))))


def update(head: Json, x: Array, y: int) -> Json:
    """Time local arithmetic separately from gradient construction and storage.

    The same calibrated BCE gradient and global ridge decay apply to every
    adaptive arm. Shared intercept and logit coefficients remain explicit.
    """
    ids = np.flatnonzero(x)
    gradient = (probability(head, x) - y) * head["calibration"][1] * x[ids]
    before = coefficients(head)
    began = time.process_time_ns()
    scale = head["decay_scale"] * (1 - 2 * CONFIG["l2"] * CONFIG["learning_rate"])
    for i, g in zip(ids, gradient, strict=True):
        head["parameters"][int(i)] -= CONFIG["learning_rate"] * float(g) / scale
    head["decay_scale"] = scale
    elapsed = time.process_time_ns() - began
    return dict(
        active_basis_indices=ids.tolist(),
        gradient_norm=float(np.linalg.norm(gradient)),
        logical_gradient_norm=float(
            np.linalg.norm(
                (probability(dict(head, parameters=before.tolist(), decay_scale=1.0), x) - y)
                * head["calibration"][1]
                * x
                + 2 * CONFIG["l2"] * before
            )
        ),
        before_coefficients=before.tolist(),
        after_coefficients=coefficients(head).tolist(),
        hot_update_ns=elapsed,
        bytes_written=(len(ids) + 1) * 8,
        logical_coefficients=len(before),
        local_coefficient_writes=len(ids),
        global_decay_writes=1,
        device="cpu",
    )


def select(block: list[Json], arm: str, seed: int) -> list[Json]:
    """Selection spends four identical gradient opportunities per complete block."""
    tie = lambda r: canonical_hash(dict(seed=seed, identity=r["family_id"]))
    if arm == "periodic":
        return [block[i] for i in CONFIG["periodic"]]
    if arm == "uniform":
        return sorted(block, key=tie)[:4]
    metric = "actual_cost" if arm == "decision_loss" else "brier"
    return sorted(block, key=lambda r: (-r[metric], tie(r)))[:4]


class Ledger:
    """SQLite makes feedback identities unique and commits issue before release."""

    def __init__(self, raw: Path):
        raw.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(raw / "ledger.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events("
            "seq INTEGER PRIMARY KEY, kind TEXT, identity TEXT UNIQUE, payload TEXT)"
        )

    def write(self, kind: str, identity: str, row: Json) -> Json:
        """Report serialized payload bytes and full synchronous commit latency."""
        payload = json.dumps(row, sort_keys=True)
        began = time.perf_counter_ns()
        with self.db:
            self.db.execute(
                "INSERT INTO events(kind,identity,payload) VALUES(?,?,?)", (kind, identity, payload)
            )
        return dict(
            durable_transaction_ns=time.perf_counter_ns() - began,
            payload_bytes=len(payload.encode()),
        )

    def close(self) -> None:
        """Close the writer before independent reduction opens the durable database."""
        self.db.close()


def measure(data: Json, raw: Path) -> Json:
    """Persist the whole trajectory before any separate retention evaluation.

    The evaluator owns its target vault. The learner receives only one due
    target after its current prediction has reached synchronous storage.
    """
    atomic_json(raw / "methods.json", dict(config=CONFIG, claim="exposed development replay"))
    atomic_json(
        raw / "inputs.json", dict(head=data["head"], sources=data["sources"], seeds=data["seeds"])
    )
    ledger = Ledger(raw)
    sources = data["sources"]
    vectors = [design(data["head"], r) if r["public_eligible"] else None for r in sources]
    vault: Json = {}

    def release(origin: int) -> int | None:
        if not vault:
            if "labels" in data:
                vault.update(data["labels"])
            else:
                document = json.loads(Path(data["target_reference"]["path"]).read_text())
                vault.update({r["family_id"]: r["eligible_y"] for r in document["rows"]})
        y = vault[sources[origin]["family_id"]]
        if y is not None and (type(y) is not int or y not in (0, 1)):
            raise ValueError("target_contract")
        return y  # type: ignore[no-any-return]

    progress("benchmark_before", 0, len(data["seeds"]) * len(ARMS) * len(sources))
    done = 0
    for seed in data["seeds"]:
        for arm in ARMS:
            head = copy.deepcopy(data["head"])
            issued: list[Json] = []
            block: list[Json] = []
            updates = 0
            checkpoint = shard(raw, "checkpoints", head)
            for slot, row in enumerate(sources):
                x = vectors[slot]
                p = probability(head, x) if x is not None else None
                pred = dict(
                    arm=arm,
                    seed=seed,
                    slot=slot,
                    family_id=row["family_id"],
                    source_cluster_id=row["source_cluster_id"],
                    probability=p,
                    action=action(p),
                    head_hash=canonical_hash(head),
                    checkpoint=checkpoint,
                    numerator=p,
                    denominator=int(p is not None),
                    eligibility=p is not None,
                    exclusion_reason=row.get("exclusion_reason") if p is None else None,
                    failure_reason=None,
                    censor_reason=None,
                )
                identity = f"{arm}/{seed}/{slot}"
                timing = ledger.write("issue", "issue/" + identity, pred)
                ledger.write(
                    "timing", "issue_time/" + identity, dict(timing, arm=arm, seed=seed, slot=slot)
                )
                issued.append(pred)
                origin = slot - CONFIG["delay"]
                if origin >= 0:
                    old = issued[origin]
                    y = release(origin)
                    valid = y is not None and old["probability"] is not None
                    receipt = dict(
                        arm=arm,
                        seed=seed,
                        family_id=old["family_id"],
                        origin_slot=origin,
                        due_slot=slot,
                        y=y,
                        eligibility=valid,
                        issued_probability=old["probability"],
                        issued_action=old["action"],
                        actual_cost=loss(old["action"], y) if valid else None,
                        brier=(old["probability"] - y) ** 2 if valid else None,
                        exclusion_reason=None if valid else "unknown_or_unavailable_target",
                    )
                    timing = ledger.write("release", "feedback/" + identity, receipt)
                    ledger.write(
                        "timing",
                        "release_time/" + identity,
                        dict(timing, arm=arm, seed=seed, slot=slot),
                    )
                    if valid:
                        block.append(receipt)
                    if len(block) == CONFIG["block"]:
                        if arm != "frozen_no_write" and updates < CONFIG["maximum_updates"]:
                            for chosen in select(block, arm, seed):
                                selected_slot = chosen["origin_slot"]
                                before_head = copy.deepcopy(head)
                                began = time.process_time_ns()
                                info = update(head, vectors[selected_slot], chosen["y"])
                                info["update_cpu_ns"] = time.process_time_ns() - began
                                support = np.asarray(info["active_basis_indices"])
                                local = support[support >= 2]
                                overlap = []
                                for past in range(slot + 1):
                                    px = vectors[past]
                                    if px is not None:
                                        pb, pa = probability(before_head, px), probability(head, px)
                                        overlap.append(
                                            dict(
                                                past_slot=past,
                                                shared_basis=int(np.count_nonzero(px[local])),
                                                probability_delta=pa - pb,
                                                action_changed=action(pb) != action(pa),
                                            )
                                        )
                                began = time.perf_counter_ns()
                                checkpoint = shard(raw, "checkpoints", head)
                                info["checkpoint_write_ns"] = time.perf_counter_ns() - began
                                record = dict(
                                    info,
                                    arm=arm,
                                    seed=seed,
                                    slot=slot,
                                    update_index=updates,
                                    family_id=chosen["family_id"],
                                    origin_slot=selected_slot,
                                    due_slot=chosen["due_slot"],
                                    y=chosen["y"],
                                    before_head_hash=canonical_hash(before_head),
                                    after_head_hash=canonical_hash(head),
                                    checkpoint=checkpoint,
                                    selection_block_ids=[r["family_id"] for r in block],
                                    issued_decision_loss=chosen["actual_cost"],
                                    issued_brier_loss=chosen["brier"],
                                    overlap=overlap,
                                    prediction_transition_count=sum(
                                        r["action_changed"] for r in overlap
                                    ),
                                )
                                timing = ledger.write(
                                    "update", f"update/{arm}/{seed}/{updates}", record
                                )
                                ledger.write(
                                    "timing",
                                    f"update_time/{arm}/{seed}/{updates}",
                                    dict(timing, arm=arm, seed=seed, slot=slot),
                                )
                                updates += 1
                        block = []
                done += 1
                if (slot + 1) % 64 == 0 or slot == len(sources) - 1:
                    progress(
                        f"stream_{arm}_seed_{seed}",
                        done,
                        len(data["seeds"]) * len(ARMS) * len(sources) - done,
                    )
    ledger.close()
    progress("benchmark_after", done, 0)
    return reduce(raw)


def reduce(raw: Path) -> Json:
    """Rebuild metrics from SQLite and verify equations using durable checkpoints.

    Reloading public inputs cannot open a label. Only the recorded due feedback
    rows supply outcomes, including exclusions and terminal censoring.
    """
    data = json.loads((raw / "inputs.json").read_text())
    db = sqlite3.connect(f"file:{raw / 'ledger.sqlite'}?mode=ro", uri=True)
    events = [
        (kind, json.loads(payload))
        for kind, payload in db.execute("SELECT kind,payload FROM events ORDER BY seq")
    ]
    db.close()
    issued = [r for k, r in events if k == "issue"]
    releases = [r for k, r in events if k == "release"]
    updates = [r for k, r in events if k == "update"]
    timings = [r for k, r in events if k == "timing"]
    heads: Json = {}
    seen: Json = {}
    eligible_by_arm: Json = {}
    for kind, r in events:
        identity = f"{r['arm']}/{r['seed']}"
        if kind == "issue":
            head = json.loads(checked(r["checkpoint"]).read_text())
            expected_head = heads.setdefault(identity, copy.deepcopy(data["head"]))
            row = data["sources"][r["slot"]]
            p = probability(head, design(head, row)) if row["public_eligible"] else None
            if (
                head != expected_head
                or canonical_hash(head) != r["head_hash"]
                or p != r["probability"]
                or action(p) != r["action"]
            ):
                raise ValueError("issued_state_drift")
            seen[f"{identity}/{r['slot']}"] = r
        elif kind == "release":
            old = seen[f"{identity}/{r['origin_slot']}"]
            if (
                f"{identity}/{r['due_slot']}" not in seen
                or r["due_slot"] != r["origin_slot"] + CONFIG["delay"]
                or old["probability"] != r["issued_probability"]
                or old["action"] != r["issued_action"]
            ):
                raise ValueError("release_order")
            valid = r["y"] is not None and old["probability"] is not None
            if valid != r["eligibility"] or r["actual_cost"] != (
                loss(old["action"], r["y"]) if valid else None
            ):
                raise ValueError("feedback_metric")
            if valid:
                eligible_by_arm.setdefault(identity, []).append(r)
        elif kind == "update":
            eligible = eligible_by_arm[identity]
            block_number = r["update_index"] // 4
            block = eligible[block_number * 16 : (block_number + 1) * 16]
            chosen = select(block, r["arm"], r["seed"])[r["update_index"] % 4]
            head = heads[identity]
            if (
                r["family_id"] != chosen["family_id"]
                or r["y"] != chosen["y"]
                or r["origin_slot"] + 20 > r["slot"]
                or r["before_head_hash"] != canonical_hash(head)
            ):
                raise ValueError("selection_or_future_access")
            audit = update(head, design(head, data["sources"][r["origin_slot"]]), r["y"])
            if (
                audit["after_coefficients"] != r["after_coefficients"]
                or audit["before_coefficients"] != r["before_coefficients"]
                or canonical_hash(head) != r["after_head_hash"]
            ):
                raise ValueError("update_equation")
            if json.loads(checked(r["checkpoint"]).read_text()) != head:
                raise ValueError("checkpoint_drift")
    labels = {(r["arm"], r["seed"], r["origin_slot"]): r for r in releases}
    baseline = {(r["seed"], r["slot"]): r for r in issued if r["arm"] == "frozen_no_write"}
    rows, later, budgets = [], [], {}
    for r in issued:
        released = labels.get((r["arm"], r["seed"], r["slot"]))
        known = released is not None and released["eligibility"]
        cost = released["actual_cost"] if known else None
        reason = (
            None if known else "pending_delay" if released is None else released["exclusion_reason"]
        )
        row = dict(
            r,
            y=released["y"] if released else None,
            actual_cost=cost,
            numerator=cost,
            denominator=int(known),
            eligibility=known,
            exclusion_reason=reason,
            censor_reason="pending_delay" if released is None else None,
        )
        rows.append(row)
        if r["slot"] >= 36:
            base = baseline[(r["seed"], r["slot"])]
            gain = loss(base["action"], row["y"]) - cost if known else None
            later.append(dict(row, gain=gain, changed_action=r["action"] != base["action"]))
    for seed in data["seeds"]:
        for arm in ARMS:
            relevant = [
                r for r in releases if r["arm"] == arm and r["seed"] == seed and r["eligibility"]
            ]
            written = [r for r in updates if r["arm"] == arm and r["seed"] == seed]
            count = min(64, len(relevant) // 16 * 4) if arm != "frozen_no_write" else 0
            if len(written) != count:
                raise ValueError("unequal_budget")
            lr = [r for r in later if r["arm"] == arm and r["seed"] == seed and r["eligibility"]]
            numerator = sum(r["actual_cost"] for r in lr)
            budgets[f"{arm}/{seed}"] = dict(
                updates=count,
                eligible_releases=len(relevant),
                completed_blocks=len(relevant) // 16,
                maximum_updates=64,
                intended=len(data["sources"]),
                eligible=len(relevant),
                started=len(data["sources"]),
                completed=len(data["sources"]),
                excluded=sum(
                    r["arm"] == arm and r["seed"] == seed and not r["eligibility"] for r in releases
                ),
                failed=0,
                censored=min(CONFIG["delay"], len(data["sources"])),
                independent=len(
                    {
                        r["source_cluster_id"]
                        for r in rows
                        if r["arm"] == arm and r["seed"] == seed and r["eligibility"]
                    }
                ),
                later_cost_numerator=numerator,
                later_cost_denominator=len(lr),
                later_cost=numerator / len(lr) if lr else None,
                gain_numerator=sum(r["gain"] for r in lr),
                changed_actions=sum(r["changed_action"] for r in lr),
            )
    primary = [r for r in rows if r["arm"] == "frozen_no_write" and r["seed"] == data["seeds"][0]]
    known = [r for r in primary if r["eligibility"]]
    return dict(
        rows=rows,
        issued_rows=issued,
        released_feedback_rows=releases,
        update_rows=updates,
        per_arm_budget=budgets,
        checkpoint_sequence=[r["checkpoint"] for r in updates],
        overlap_rows=[
            dict(arm=r["arm"], seed=r["seed"], update_index=r["update_index"], overlap=r["overlap"])
            for r in updates
        ],
        later_cost_rows=later,
        hot_update_ns=[r["hot_update_ns"] for r in updates],
        durable_transaction_ns=[r["durable_transaction_ns"] for r in timings],
        bytes_written=sum(r["bytes_written"] for r in updates),
        durable_serialized_bytes=sum(r["payload_bytes"] for r in timings),
        actual_ledger_bytes=(raw / "ledger.sqlite").stat().st_size,
        sample_size_budget=dict(
            intended=len(primary),
            eligible=len(known),
            started=len(primary),
            completed=len(primary),
            excluded=sum(r["denominator"] == 0 and r["censor_reason"] is None for r in primary),
            failed=0,
            censored=sum(r["censor_reason"] is not None for r in primary),
            independent=len({r["source_cluster_id"] for r in known}),
            independent_datasets=1,
            seeds_are_independent=False,
        ),
        decision_benefit=None
        if not any(r["changed_action"] for r in later)
        else "descriptive_changed_decisions; benefit unqualified",
        retained_labels_opened=False,
    )
