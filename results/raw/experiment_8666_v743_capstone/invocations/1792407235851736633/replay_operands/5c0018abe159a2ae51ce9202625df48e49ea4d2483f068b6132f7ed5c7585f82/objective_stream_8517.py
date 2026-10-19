"""REQ-VERIFY-8517: keep durable block semantics while selecting the registered derivative.

The original store seals predictions before its evaluator releases admitted labels.
Changing an objective must not change admission, retention access or recovery.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import resource
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import capacity_online_learning_8434 as shipped
from carnot.verify import objective_checkpoint_8461 as adapter

__all__ = [
    "shipped",
    "adapter",
    "configuration",
    "binding",
    "worker",
    "check_step",
    "ARMS",
    "SEEDS",
    "CLI",
]

Json = dict[str, Any]
ARMS = ("frozen", "log_loss", "brier", "brier_shuffled_released")
SEEDS = (17, 29, 43, 71, 101)
CLI = "scripts/experiments/experiment_8517_v732_brier_online_learning.py"
_LOG_UPDATE, _TRACE, _WORKER, _CHECK = (
    shipped.sparse.update,
    shipped.update_trace,
    shipped.worker,
    shipped.check_step,
)


def configuration(
    head: Json, blocks: list[list[Json]], seed: int, arm: str, labels: Json, tune: Json
) -> Json:
    """All arms use the same uniform sets and original numerical starting state."""
    if arm not in ARMS or seed not in SEEDS or len(blocks) != 5:
        raise ValueError("stream_configuration")
    cfg = adapter.config(head, 8, "uniform", frozen=arm == "frozen", shuffled=arm == ARMS[3])
    if arm == "log_loss":
        cfg.update(update_rule=dict(adapter.RULE, version="log-loss-spline-v1", data="(p-y)/T"))
        cfg["update_rule_hash"] = canonical_hash(cfg["update_rule"])
    cfg.update(
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


@contextmanager
def binding(cfg: Json, costs: Json | None = None) -> Iterator[None]:
    """Reuse equations and physical writes; measure actual committed updates separately from audit copies."""
    brier = cfg["research_arm"] != "log_loss"
    update = adapter.update if brier else _LOG_UPDATE
    original = adapter.Store if brier else adapter.legacy.Store
    collect = True

    def timed(head: Json, x: Any, y: int, learning_rate: float = 0.01) -> tuple[Json, Json]:
        """Time only numerical update work; an audit reconstruction is never a second effect."""
        start = time.perf_counter_ns()
        changed, info = update(head, x, y, learning_rate)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        if costs is not None and collect:
            costs["updates"].append(
                dict(
                    compute_s=elapsed,
                    coefficient_touches=info["coefficient_touches"],
                    global_decay_writes=info["global_decay_writes"],
                )
            )
        return dict(changed), dict(info)

    def trace(sealed: Json, config: Json, block: int, labels: Json) -> list[Json]:
        """Brier changes the data derivative while retaining the exact regularizer and IDs."""
        nonlocal collect
        collect = False
        try:
            rows = list(_TRACE(sealed, config, block, labels))
        finally:
            collect = True
        if brier:
            for row in rows:
                p = row["probability_before_update"]
                row["data_gradient"] = (
                    np.asarray(row["data_gradient"]) * (2 * p * (1 - p))
                ).tolist()
        return rows

    class MeasuredStore(original):  # type: ignore[misc, valid-type]
        """Store barriers expose durable I/O costs without replacing the write implementation."""

        def commit(self, state: Json, hook: Callable[[str], None] = lambda phase: None) -> None:
            """Keep serialization, file sync, replacement sync and acknowledgment spans explicit."""
            previous = time.perf_counter_ns()
            spans: Json = {}

            def boundary(phase: str) -> None:
                """Each shipped barrier ends one measured span and retains the supplied crash hook."""
                nonlocal previous
                now = time.perf_counter_ns()
                spans[phase + "_s"] = (now - previous) / 1e9
                previous = now
                hook(phase)

            super().commit(state, boundary)
            if costs is not None:
                costs["commits"].append(dict(version=state["version"], **spans))

    namespace = SimpleNamespace(**dict(vars(adapter.legacy)))
    namespace.Store = MeasuredStore
    if brier:
        namespace.initial = lambda config: adapter.bind(adapter.legacy.initial(config), config)
        namespace.release = adapter.release
    with (
        patch.object(shipped, "state", namespace),
        patch.object(shipped, "CLI", CLI),
        patch.object(shipped, "update_trace", trace),
        patch.object(shipped.sparse, "update", timed),
        patch.object(adapter, "update", timed),
    ):
        yield


def worker(cfg: Json, directory: Path, crash: bool = False) -> Json:
    """Persist cost observations alongside science, including a genuine child exit and resumed history."""
    path = directory / "costs.json"
    costs: Json = dict(updates=[], commits=[])
    if path.exists():
        costs = shipped.read(shipped.reference(path))
    with binding(cfg, costs):
        # The existing crash happens after block2; its progress boundary saves costs first.
        progress = shipped.progress

        def boundary(phase: str, completed: int = 0, pending: int = 0) -> None:
            """Costs survive process death without entering the learner's checkpoint or predictions."""
            costs["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            atomic_json(path, costs)
            progress(phase, completed, pending)

        with patch.object(shipped, "progress", boundary):
            result = dict(_WORKER(cfg, directory, crash))
    return result


def check_step(cfg: Json, step: Json) -> Json:
    """Reconstruct the exact committed rule from original features and released labels."""
    with binding(cfg):
        return dict(_CHECK(cfg, step))
