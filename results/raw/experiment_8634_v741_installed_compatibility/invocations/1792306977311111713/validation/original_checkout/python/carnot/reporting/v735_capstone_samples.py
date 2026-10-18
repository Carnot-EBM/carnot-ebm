"""REQ-REPORT-8564: bounded original samples grant no new independent evidence."""

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v734_capstone_samples as previous
from carnot.reporting import hardware_operation_evidence_8563 as cost
from carnot.reporting import arc_panel_replay_8561 as arc
from carnot.reporting import canonical_update_8555 as learning
from carnot.reporting import v735_contract_methods as methods


def samples() -> dict[str, Any]:
    """Reuse one original event and episode; reconcile complete clocks through the actual reader."""
    result = dict(previous.samples())
    path = methods.ROOT / "results/experiment_8555_v735_canonical_update_accounting.json"
    value = learning.read(learning.reference(path))
    event = json.loads(Path(value["event_rows_reference"]["path"]).read_bytes())[0]
    records, metrics, visible, relations = learning.canonical([event])
    result["learning"]["canonical_sample"] = dict(
        source=learning.reference(path),
        event=event["event_id"],
        records=records,
        metrics=metrics,
        visibility=visible,
        relations=relations,
    )
    from carnot.reporting import v733_update_accounting as stream
    from unittest.mock import patch

    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    with patch.multiple(stream, TASK=learning.TASK, replay=learning.replay):
        result["learning"]["actual_stream_consumer"] = stream.stream_consumer(
            path, Path(terminal["publication"]["sidecar_path"])
        )
    result["native_cost"] = cost.costs(methods.ROOT)
    cells = result["native_cost"]["cells"]
    result["native_cost"]["sample_clock_reconciliation"] = [
        dict(
            arm=c["arm"],
            batch=c["batch"],
            repeat=c["repeat"],
            wall_s=c["wall_s"],
            recomputed_wall_s=sum(c["seconds"].values()),
        )
        for c in cells[:2]
    ]
    result["native_consumer"] = dict(
        accepted=result["native_cost"]["qualified"], missing_reason=None
    )
    result["arc"]["actual_panel_consumer"] = arc.panel_consumer(
        methods.ROOT / "results" / (arc.NAME + ".json")
    )
    return result
