"""REQ-REPORT-8963: inference receives graph context without evaluation truth.

Views contain only the frozen encoder operands. Labels live in separate files;
the final-test reader requires a durable prediction seal before opening truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def detector_view(units: list[Json], rows: list[Json]) -> list[Json]:
    """Project allowed inputs without inspecting labels, distances or answer membership."""
    result = []
    for unit, row in zip(units, rows, strict=True):
        answer = row["answer"].strip()
        result.append(
            dict(
                nodes=unit["nodes"],
                edges=unit["edges"],
                query_node=unit["start"],
                proposed_node=answer if answer in unit["nodes"] else None,
                features=row["features"],
            )
        )
    return result


def release_final_test(labels: Path, seal: Path) -> list[int]:
    """Refuse missing, altered or incomplete predictions before exposing held-out labels."""
    saved = json.loads(seal.read_text())
    predictions = saved["rows"]
    if saved["prediction_hash"] != canonical_hash(predictions) or any(
        type(r["p_error"]) not in (int, float)
        or not math.isfinite(r["p_error"])
        or not 0 <= r["p_error"] <= 1
        for r in predictions
    ):
        raise ValueError("prediction_seal")
    truth = json.loads(labels.read_text())["rows"]
    if (
        [r["unit_id"] for r in predictions] != [r["unit_id"] for r in truth]
        or len({r["unit_id"] for r in truth}) != len(truth)
        or any(type(r["label"]) is not int or r["label"] not in (0, 1) for r in truth)
    ):
        raise ValueError("final_test_roster")
    return [r["label"] for r in truth]
