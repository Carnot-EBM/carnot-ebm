"""REQ-VERIFY-8461: change the data derivative while keeping durable store semantics."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.verify import block_feedback_state_8433 as legacy
from carnot.verify import sparse_energy_7996 as sparse

__all__ = [
    "legacy",
    "sparse",
    "Store",
    "config",
    "update",
    "RULE",
    "RULE_HASH",
    "bind",
    "release",
    "replay_state",
    "drive",
]

Json = dict[str, Any]
Array: TypeAlias = NDArray[np.float64]
RULE = dict(
    version="brier-spline-v1",
    data="2*(p-y)*p*(1-p)/T",
    rate=0.01,
    l2=0.001,
    bias_regularized=True,
    logical_coefficients=109,
    physical_support_max=37,
)
RULE_HASH = canonical_hash(RULE)


def update(head: Json, x: Array, y: int, learning_rate: float = 0.01) -> tuple[Json, Json]:
    """A shared multiplier decays all weights, including bias, without dense data writes."""
    if (
        head["arm"] != "spline"
        or type(y) is not int
        or y not in (0, 1)
        or not 0 <= learning_rate <= 1
        or head["temperature"] <= 0
        or head["decay_scale"] <= 0
        or x.shape != (9,)
        or not np.isfinite(x).all()
    ):
        raise ValueError("update_contract")
    if not np.isfinite([*head["parameters"], head["temperature"], head["decay_scale"]]).all():
        raise ValueError("finite_state")
    ids, log_gradient = sparse.sparse_gradient(head, x, y)
    p = float(expit(sparse.logits_jacobian(head, x[None, :])[0][0] / head["temperature"]))
    gradient = log_gradient * (2 * p * (1 - p))
    result = deepcopy(head)
    if learning_rate:
        scale = head["decay_scale"] * (1 - 0.002 * learning_rate)
        for i, g in zip(ids, gradient, strict=True):
            result["parameters"][int(i)] -= learning_rate * float(g) / scale
        result["decay_scale"] = scale
    return result, dict(
        coefficient_touches=len(ids) if learning_rate else 0,
        global_decay_writes=int(learning_rate > 0),
        data_gradient=gradient.tolist(),
        gradient_ids=ids.tolist(),
        bias_regularized=True,
    )


def config(
    head: Json,
    capacity: int = 8,
    policy: str = "fifo",
    *,
    frozen: bool = False,
    shuffled: bool = False,
) -> Json:
    """Rule and implementation hashes bind resumes to the declared numerical operation."""
    if "update_rule" in head:
        raise ValueError("update_rule")
    cfg = legacy.config(capacity, policy, frozen=frozen, shuffled=shuffled)
    cfg.update(
        head=deepcopy(head),
        update_rule=deepcopy(RULE),
        update_rule_hash=RULE_HASH,
        update_adapter_hash=sha256_file(Path(__file__)),
    )
    return dict(cfg)


def bind(state: Json, cfg: Json) -> Json:
    """Explicit checkpoint fields let a consumer reject a different objective immediately."""
    return dict(
        state,
        update_rule=cfg["update_rule"],
        update_rule_hash=cfg["update_rule_hash"],
        update_adapter_hash=cfg["update_adapter_hash"],
    )


def release(state: Json, cfg: Json, block: int, labels: Json) -> Json:
    """Only admitted labels can change the head; the second commit saves every effect."""
    if state["phase"] != "sealed" or state["cursor"] != block * 20:
        raise ValueError("unsealed_release")
    if set(labels) != {r["id"] for r in state["pending"]} or any(
        type(y) is not int or y not in (0, 1) for y in labels.values()
    ):
        raise ValueError("release_roster")
    out = deepcopy(state)
    rng = legacy.restore(out["feedback_rng"])
    targets = [labels[r["id"]] for r in state["pending"]]
    if cfg["shuffled"]:
        rng.shuffle(targets)
    out["feedback_rng"] = list(json.loads(json.dumps(rng.getstate())))
    for record, target in zip(state["pending"], targets, strict=True):
        identity = f"feedback:{block}:{record['id']}"
        if identity in out["applied_feedback_ids"]:
            raise ValueError("duplicate_feedback")
        if not cfg["frozen"]:
            out["head"], _ = update(out["head"], np.asarray(record["x"], dtype=float), target)
            out["parameter_updates"] += 1
        out["applied_feedback_ids"].append(identity)
    out["release_hashes"][str(block)] = canonical_hash(labels)
    out.update(pending=[], phase="updated", cursor=(block + 1) * 20, version=state["version"] + 1)
    legacy.acknowledge(out, block, "release")
    return out


class Store(legacy.Store):  # type: ignore[misc]
    """Inherit all physical writes; override only rule checks and numerical release."""

    def commit(self, state: Json, hook: Callable[[str], None] = lambda phase: None) -> None:
        """Initial snapshots carry rule seals; a supplied conflicting seal stays detectable."""
        if "update_rule_hash" not in state:
            state = bind(state, self.configuration)
        super().commit(state, hook)

    def read(self) -> Json:
        """A correct outer checksum cannot authorize a checkpoint from another update rule."""
        state = super().read()
        if (
            state.get("update_rule_hash") != RULE_HASH
            or state.get("update_rule") != RULE
            or state.get("update_adapter_hash") != sha256_file(Path(__file__))
        ):
            raise ValueError("update_rule")
        if not np.isfinite(
            [
                *state["head"]["parameters"],
                state["head"]["temperature"],
                state["head"]["decay_scale"],
            ]
        ).all():
            raise ValueError("finite_state")
        return dict(state)

    def apply(self, transaction: str, block: int, operand: Any, hook: Callable[[str], None]) -> str:
        """Saved IDs return their exact acknowledgment; conflicting operands are rejected."""
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
                legacy.seal(state, self.configuration, block, operand)
                if transaction == "seal"
                else release(state, self.configuration, block, operand)
            )
            self.commit(changed, hook)
            return str(changed["acknowledgment_ids"][identity])


def replay_state(state: Json, cfg: Json) -> bool:
    """Construct the expected state from independent fixtures instead of trusting saved hashes."""
    try:
        if type(state.get("version")) is not int or not 0 <= state["version"] <= 4:
            return False
        if cfg != config(
            cfg["head"],
            cfg["capacity"],
            cfg["policy"],
            frozen=cfg["frozen"],
            shuffled=cfg["shuffled"],
        ):
            return False
        expected = bind(legacy.initial(cfg), cfg)
        for version in range(state["version"]):
            block = version // 2
            expected = (
                legacy.seal(expected, cfg, block, legacy.questions(block))
                if version % 2 == 0
                else release(expected, cfg, block, legacy.feedback(expected))
            )
        return bool(state == expected)
    except (ValueError, KeyError, TypeError):
        return False


def drive(path: Path, cfg: Json, crash: str = "") -> Json:
    """Two constructed blocks exercise future predictions without running a public online panel."""
    store = Store(path, cfg)
    store.initialize()
    while store.read()["cursor"] < 40:
        state = store.read()
        block = state["cursor"] // 20
        for transaction in ("seal", "release"):

            def hook(boundary: str) -> None:
                """A real process exit tests lost delivery and retained operating-system state."""
                if crash == f"{transaction}:{block}:{boundary}":
                    print(f"[exp8461] actual_exit73={crash}", flush=True)
                    os._exit(73)

            if transaction == "seal" and state["phase"] == "updated":
                store.seal(block, legacy.questions(block), hook)
            if transaction == "release":
                store.release(block, legacy.feedback(store.read()), hook)
        print(f"[exp8461] blocks_completed={block + 1} blocks_pending={1 - block}", flush=True)
    return store.read()
