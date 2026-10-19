"""REQ-VERIFY-8582: reconstruct the stream instead of trusting repaired envelope hashes.

Original inputs and mathematical operations determine every later state. A
modified summary cannot become valid by merely recomputing its checksum.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from carnot.reporting import causal_brier_stream_8582 as e

Json = dict[str, Any]


@lru_cache(maxsize=512)
def checked_step(
    configuration_path: str, configuration_sha: str, step_path: str, step_sha: str
) -> Json:
    """Cache only byte-pinned replay work; each fresh process independently reconstructs it."""
    cfg = e.old.read(dict(path=configuration_path, sha256=configuration_sha))
    step = e.old.read(dict(path=step_path, sha256=step_sha))
    return dict(e.stream.check_step(cfg, step))


def replay(path: Path) -> bool:
    """Fresh processes authenticate immutable operands, transitions, costs and receipts."""
    try:
        value = json.loads(path.read_bytes())
        work = e.old.read(value["primitive_reference"])
        for ref in [
            *work["source_refs"],
            *work["code_refs"],
            *work["primitive_files"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if e.old.reference(Path(ref["path"])) != ref:
                return False
        if work["owned_ready"] != e.validated(work):
            return False
        expected_upstream = [
            dict(path=str(e.ROOT / "results" / (name + ".json")), sha256=pin)
            for name, pin in e.PINS.items()
        ]
        if not work["failures"] and any(
            ref not in work["source_refs"] for ref in expected_upstream
        ):
            return False
        if work["source"] != e.source():
            return False
        if [(c["seed"], c["arm"]) for c in work["cells"]] != [
            (s, a) for s in e.stream.SEEDS for a in e.stream.ARMS
        ]:
            return False
        for cell in work["cells"]:
            if not cell.get("configuration"):
                continue
            cfg = e.old.read(cell["configuration"])
            original = work["source"]
            expected = e.stream.configuration(
                next(h for h in original["heads"] if h["seed"] == cell["seed"]),
                original["blocks"],
                cell["seed"],
                cell["arm"],
                original["panel"]["evaluator_custody"],
                original["panel"]["trainer_view"],
            )
            if cfg != expected:
                return False
            with e.stream.binding(cfg):
                previous = e.stream.shipped.state.initial(cfg)
            steps = []
            for ref in cell["steps"]:
                step = e.old.read(ref)
                if step["before_state"] != previous:
                    return False
                previous = checked_step(
                    cell["configuration"]["path"],
                    cell["configuration"]["sha256"],
                    ref["path"],
                    ref["sha256"],
                )
                steps.append(step)
            if e.old.read(cell["final"]) != previous:
                return False
            if e.old.read(cell["retention"]) != e.stream.shipped.retention(cfg, steps):
                return False
            for restart in work["restart_rows"]:
                if cell["seed"] == 17 and restart["arm"] == cell["arm"]:
                    resumed = [e.old.read(ref) for ref in restart["steps"]]
                    equal = (
                        [e.stream.shipped.semantic_step(s) for s in steps]
                        == [e.stream.shipped.semantic_step(s) for s in resumed]
                        and e.old.read(restart["final"]) == previous
                        and restart["actual_exit"] == 73
                    )
                    if not equal or not restart["passed"]:
                        return False
            e.old.progress(
                "cold_replay",
                e.stream.SEEDS.index(cell["seed"]) * 4 + e.stream.ARMS.index(cell["arm"]) + 1,
                0,
            )
        if not e.old.report.receipts_valid(
            [*work["process_receipts"], *work["receipts"]], work["adversarial_findings"]
        ):
            return False
        reduced = e.old.reductions(work)
        raw = Path(work["raw"])
        if (
            e.old.read(e.old.reference(raw / "stream_rows.json")) != dict(rows=reduced["rows"])
            or e.old.read(e.old.reference(raw / "released_feedback.json"))
            != reduced["release_rows"]
            or e.old.read(e.old.reference(raw / "retention_rows.json")) != reduced["retention_rows"]
        ):
            return False
        if e.old.read(e.old.reference(raw / "coefficient_transitions.json")) != e.counters(work):
            return False
        return bool(value == e.build(work, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError):
        return False
