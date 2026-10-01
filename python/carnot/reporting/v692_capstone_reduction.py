"""REQ-REPORT-7991-V692: reconstruct observations without importing benefit claims.

Source means keep repeated seeds from inflating the scientific sample size.
Qualified readers check the same stored heads without fitting or model calls.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting import v691_capstone as prior
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]


def operand(
    path: Path, upstream: str, field: str, expected: Any, actual: Any, op: str = "=="
) -> Json:
    """Keep missing bytes distinct from observed zero in every failed gate."""
    return dict(
        upstream_id=upstream,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op=op,
        expected=expected,
        observed=actual,
    )


def references(data: Json) -> list[Json]:
    """Normalize producer-specific reference layouts without discarding their roles."""
    found = []
    for key in (
        "source_artifact_hashes",
        "cited_upstream_artifacts",
        "code_config_hashes",
        "raw_shard_hashes",
    ):
        items = data.get(key, [])
        if isinstance(items, dict):
            items = [
                dict(v, path=k) if isinstance(v, dict) else dict(path=k, sha256=v)
                for k, v in items.items()
            ]
        found.extend(
            dict(r, role=r.get("role", key))
            for r in items
            if isinstance(r, dict) and r.get("path") and r.get("sha256")
        )
    return found


def reduce_rows(rows: list[Json]) -> Json:
    """Average probabilities within sources and score only observed binary labels."""
    cells: Any = defaultdict(list)
    rosters: Any = defaultdict(lambda: defaultdict(set))
    sources, seeds, unknown, causal, restart = set(), set(), 0, 0, {}
    for index, row in enumerate(rows):
        if index % 512 == 0:
            print(f"[exp7991] primitive_rows={index}/{len(rows)}", flush=True)
        if not isinstance(row, dict):
            raise ValueError("malformed_primitive_row")
        p, y = row.get("p", row.get("probability")), row.get("y", row.get("label"))
        if p is not None and (
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
        ):
            raise ValueError("invalid_probability")
        source = row.get("source_cluster_id", row.get("source_group", row.get("family_id")))
        if source is not None:
            sources.add(str(source))
        if row.get("seed") is not None and row["seed"] != "mean":
            seeds.add(row["seed"])
        if "label_read_event" in row:
            causal += 1
            if row["label_read_event"] < row["label_release_event"]:
                raise ValueError("causal_label_read_before_release")
        key = tuple(str(row.get(k)) for k in ("family_id", "arm", "seed", "role"))
        if "restart_identity" in row:
            if key in restart and restart[key] != row["restart_identity"]:
                raise ValueError("restart_identity_changed")
            restart[key] = row["restart_identity"]
        if p is None or row.get("status") not in {"completed", "generated"}:
            continue
        if type(y) is not int or y not in (0, 1):
            unknown += 1
            continue
        arm, role = str(row.get("arm", "default")), str(row.get("role", "evaluation"))
        intervention = str(row.get("intervention", "original"))
        family = str(row.get("family_id", source))
        rosters[role, intervention][arm].add((family, str(source), y))
        action = row.get("decision", row.get("action"))
        cost = {"accept": 5 * y, "reject": 1 - y, "escalate": 0.25}.get(action)
        cells[arm, role, intervention, str(source), family].append((float(p), y, cost, action))
    for arms in rosters.values():
        if len({frozenset(ids) for ids in arms.values()}) > 1:
            raise ValueError("same_information_source_or_unknown_label_drift")
    grouped: Any = defaultdict(lambda: defaultdict(list))
    for (arm, role, intervention, source, _), items in cells.items():
        p = sum(r[0] for r in items) / len(items)
        costs = [r[2] for r in items if r[2] is not None]
        grouped[arm, role, intervention][source].append(
            dict(
                brier=(p - items[0][1]) ** 2,
                cost=sum(costs) / len(costs) if costs else None,
                false_accept=sum(r[3] == "accept" and r[1] == 1 for r in items) / len(items),
                automation=sum(r[3] in {"accept", "reject"} for r in items) / len(items),
            )
        )
    metrics = {}
    for (arm, role, intervention), groups in grouped.items():
        measured: Json = dict(role=role, intervention=intervention, independent_sources=len(groups))
        for field in ("brier", "cost", "false_accept", "automation"):
            means = [
                sum(vals) / len(vals)
                for units in groups.values()
                if (vals := [r[field] for r in units if r[field] is not None])
            ]
            measured[field] = sum(means) / len(means) if means else None
        metrics[
            arm
            if role == "evaluation" and intervention == "original"
            else f"{arm}:{role}:{intervention}"
        ] = measured
    return dict(
        primitive_rows=len(rows),
        independent_sources=len(sources),
        seeds=len(seeds),
        scored_rows=sum(len(v) for v in cells.values()),
        unknown_label_rows=unknown,
        arm_metrics=metrics,
        paired_sources_equal=True,
        causal_label_reads=causal,
        restart_identities=len(restart),
        unknown_label_rule="score only binary observed labels",
        independently_unexposed_sources=0,
        independent_benefit=False,
    )


def audit(number: int, data: Json, root: Path) -> Json:
    """Use qualified readers only where their actual checkpoint prerequisites exist."""
    measured = reduce_rows(data.get("rows", []))
    checks: Json = {}
    if number == 7980 and data.get("public_features"):
        from carnot import experiment_7980_v692_evidence_features as features

        features.replay(data)
        roster = json.loads(Path(data["fresh_roster"]["path"]).read_bytes())["rows"]
        checks.update(
            features_reconstructed=True,
            reserved_sources=len(roster),
            reserved_independence="unavailable" if not roster else "exposure audit required",
            exposed_development_sources=len({r["source_cluster_id"] for r in data["rows"]}),
        )
    if number == 7981 and data.get("raw_response_shards"):
        from carnot.verify import qwen_stream_capture_7981 as capture

        checks["capture_counts"] = capture.reduce(data["rows"])
    if number == 7982 and data.get("heads_seal"):
        from carnot import experiment_7982_v692_multivariate_energy as energy

        checks["head_reconstruction"] = energy.replay(data)
    if number == 7984 and data.get("checkpoints"):
        from carnot import experiment_7984_v692_evidence_ablation as ablation

        checks["ablation_reconstruction"] = ablation.replay(data)
    if number == 7989 and data.get("rows"):
        from carnot.reporting import service_cost_7989 as service

        checks.update(service.reduce(data["rows"]))
        for row in data["rows"]:
            spans = dict(
                zip(
                    service.PHASES,
                    (b - a for a, b in zip(row["ticks_ns"], row["ticks_ns"][1:])),
                    strict=True,
                )
            )
            if (
                spans != row["exclusive_phase_spans"]
                or min(spans.values()) < 0
                or sum(spans.values()) != row["wall_ns"]
            ):
                raise ValueError("service_span_drift")
        checks["exclusive_spans_reconstructed"] = True
        try:
            service.replay(data)
            checks["service_replay"] = dict(passed=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks["service_replay"] = dict(passed=False, error=str(error))
    if number == 7990 and data.get("board_rows"):
        from carnot.reporting import experiment_7990_v692_hardware_evidence as hardware

        checks["board_reconstruction"] = hardware.cold_reduce(root, data)
        checks["hardware_bounds"] = [hardware.bounds(0, 1, None)]
    if number == 7988 and "new_event_rows" in data:
        from carnot.reporting import arc_supervisor_v692_delta as arc

        checks["arc_replay_errors"] = arc.replay(data)
    return dict(**measured, checkpoint_reduction=checks)
