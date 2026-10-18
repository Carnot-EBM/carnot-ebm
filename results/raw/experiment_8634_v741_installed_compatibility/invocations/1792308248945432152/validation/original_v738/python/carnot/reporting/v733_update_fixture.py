"""REQ-REPORT-8530: small oracle blocks exercise the shipped store and original head.

Only constructed labels enter this fixture. Its process death tests durability;
it cannot establish scientific learning benefit on the historical question panel.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import brier_online_learning_8517 as original
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v733_update_metrics import coordinate_row
from carnot.verify import objective_checkpoint_8461 as adapter

Json = dict[str, Any]


def configuration() -> Json:
    """Retain the original V726 head while restricting work to two constructed blocks."""
    return dict(adapter.config(original.source()["heads"][0], 8, "uniform"))


def worker(cfg: Json, directory: Path, crash: bool = False) -> Json:
    """Record effects from actual committed calls and resume after a genuine exit73."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "accounting.json"
    rows = json.loads(path.read_bytes()) if path.exists() else []
    store = adapter.Store(directory / "store", cfg)
    store.initialize()
    update = adapter.update
    for block in range(store.read()["cursor"] // 20, 2):
        before = store.read()
        try:
            adapter.release(before, cfg, block, {})
        except ValueError:
            pass
        else:
            raise ValueError("unsealed_release_accepted")
        store.seal(block, adapter.legacy.questions(block))
        sealed = store.read()
        labels = adapter.legacy.feedback(sealed)
        observed: list[Json] = []

        def measured(head: Json, x: Any, y: int, learning_rate: float = 0.01) -> tuple[Json, Json]:
            """Observe serialized numbers without using the store's parameter update counter."""
            changed, info = update(head, x, y, learning_rate)
            observed.append(
                coordinate_row(
                    head,
                    changed,
                    list(info["gradient_ids"]),
                    event=f"{block}:{len(observed)}",
                    namespace="fixture",
                )
            )
            return dict(changed), dict(info)

        with patch.object(adapter, "update", measured):
            ack = store.release(block, labels)
        committed = store.read()
        if store.release(block, labels) != ack or store.read() != committed:
            raise ValueError("duplicate_release_changed_state")
        conflict = deepcopy(labels)
        conflict[next(iter(conflict))] ^= 1
        try:
            store.release(block, conflict)
        except ValueError:
            pass
        else:
            raise ValueError("conflicting_release_accepted")
        rows.extend(observed)
        atomic_json(path, rows)
        print(f"[exp8530] fixture_completed={block + 1} fixture_pending={1 - block}", flush=True)
        if crash:
            print("[exp8530] actual_exit=73 after_durable_release", flush=True)
            os._exit(73)
    atomic_json(directory / "final.json", store.read())
    return dict(store.read())
