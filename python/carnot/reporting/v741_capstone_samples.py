"""REQ-REPORT-8639: fixed arithmetic samples confer no new generalization credit."""

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v740_capstone_samples as historical
from carnot.reporting import v741_capstone_evidence as e
from carnot.reporting import arc_observer_transfer_8637 as arc

Json = dict[str, Any]


def producer(number: int) -> Json:
    """One original producer prevents summary copies from becoming new evidence."""
    paths = list((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
    if len(paths) != 1:
        raise ValueError("original_producer_identity")
    return dict(e.read(e.reference(paths[0])))


def reduce() -> Json:
    """Reduce original windows, installed clocks and actual ARC observation horizons."""
    old = historical.reduce()
    streaming = producer(8629)
    windows = []
    for row in streaming["window_offset_rows"]:
        ref = row["primitive"]
        if e.methods.sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("streaming_window_hash")
        count = len(Path(ref["path"]).read_bytes().splitlines())
        windows.append(
            dict(
                family_id=row["family_id"],
                chunk_size=row["chunk_size"],
                source=ref,
                observed_windows=count,
                matches_original=count == row["total_window_count"]
                and ref["sha256"] == "sha256:" + row["complete_window_digest"],
                qualified=False,
                semantic_correctness_established=False,
            )
        )
    installed = producer(8633)
    clocks = []
    for ref in installed["raw_shard_hashes"]:
        if not ref["path"].endswith("-sessions.json"):
            continue
        sessions = e.read(ref)
        for arm, session in sessions.items():
            elapsed = (session["ended_monotonic_ns"] - session["started_monotonic_ns"]) / 1e9
            clocks.append(
                dict(
                    source=ref,
                    arm=arm,
                    total_s=elapsed,
                    response_count=len(session["responses"]),
                    matches_original=elapsed > 0 and session["exit_code"] == 0,
                    scope="prototype_session_not_registered120_cell_cost",
                )
            )
    current = producer(8637)
    work = e.read(current["work_reference"])
    horizons: list[Json] = []
    for ref in work["references"]:
        original = e.read(ref)
        unit = original["unit"]
        reduced = arc.replay_stream(original["steps"], original["events"], original["end_reason"])
        horizons.extend(
            dict(r, **{k: unit[k] for k in ["game", "seed", "episode_id", "observer"]})
            for r in reduced
        )
    reported = [
        {k: v for k, v in r.items() if k != "classification"}
        for r in current["horizon_status_rows"]
    ]
    return dict(
        v740=old,
        streaming=windows,
        installed_clocks=clocks,
        arc=dict(
            rows=horizons,
            matches_original=horizons == reported,
            supervisor_benefit_established=False,
        ),
        current_learning=dict(
            observed=None,
            missing_reason="Exp8630 cascade skipped; historical samples are unqualified for current benefit",
        ),
        current_coverage_shards=dict(
            observed=None,
            missing_reason="Exp8628 pre-gated; V740 lost original child shards remain lost",
        ),
        current_complete_cost=dict(
            observed=None,
            missing_reason="Exp8635 pre-gated; prototype clocks cannot measure packaging noninferiority",
        ),
        numerical_inference_this_invocation=False,
        independent_units_added=0,
    )
