"""REQ-REPORT-8121: admit independent historical inputs without probing boards."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_batch_8121 as h
from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting import radial_hardware_8108 as prior

Json = dict[str, Any]
RESOURCES = [
    *prior.RESOURCES,
    "CLAUDE.md",
    "CODEX.md",
    "research-references.md",
    "python/carnot/reporting/radial_hardware_8108.py",
    "tests/python/test_primary_publication_7928.py",
]
UPSTREAMS = [
    (8108, "v701_radial_hardware_boundary", "hardware_boundary_ready_score"),
    (8119, "v702_batched_service_cost", "host_service_ready_score"),
    (8116, "v702_independent_online_memory", "learning_trajectory_ready_score"),
]


def shard(ref: Json, root: Path, raw: Path, data: Json, label: str) -> Json:
    """Copy hash-bound operands before reduction so old results remain replayable."""
    path = custody.seal(
        root / ref.get("path", "missing_primitive"), ref.get("sha256"), raw, data, label
    )
    return json.loads(path.read_bytes()) if path else {}


def load(root: Path, raw: Path) -> Json:
    """Qualify each branch separately; failed historical primaries stay untouched.

    The numerical readiness gate supplies analytic fixture inputs. Each board
    still needs its own original receipt and transcript hashes. Optional batch
    and update evidence cannot borrow numerical or board readiness.
    """
    data: Json = dict(
        fixture=False,
        systems=[],
        boards=[],
        costs=[],
        updates=[],
        checks=[],
        references=[],
        batches=h.CONFIG["batches"],
        numerical_available=False,
        batch_available=False,
        touches_available=False,
    )
    for resource in RESOURCES:
        custody.seal(root / resource, None, raw, data, "local_resource")
    qualified: dict[int, Json] = {}
    for eid, name, field in UPSTREAMS:
        path = root / f"results/experiment_{eid}_{name}.json"
        value, valid = prior.authenticate(path, eid, field, raw, data)
        for ref in value.get("code_config_hashes", []) if valid else []:
            valid = (
                custody.seal(root / ref["path"], ref["sha256"], raw, data, f"exp{eid}") is not None
                and valid
            )
        qualified[eid] = value if valid else {}
        print(f"[exp8121] input_check exp{eid} qualified={valid}", flush=True)
    numerical = qualified[8108]
    if numerical:
        data["systems"] = shard(
            numerical["replay_input_reference"], root, raw, data, "exp8108"
        ).get("systems", [])
        data["numerical_available"] = len(data["systems"]) == prior.CONFIG["systems"]
        data["checks"].append(
            custody.operand(
                "exp8108",
                root / numerical["replay_input_reference"]["path"],
                "numerical_fixture_count",
                prior.CONFIG["systems"],
                len(data["systems"]),
            )
        )
    originals = {b["board"]: b for b in numerical.get("board_rows", [])}
    for name, contract in prior.CONTRACTS.items():
        board = deepcopy(originals.get(name, dict(board=name, custody_valid=False)))
        begin = len(data["checks"])
        receipt = shard(
            dict(path=board.get("source_path", "missing_" + name), sha256=board.get("source_hash")),
            root,
            raw,
            data,
            name,
        )
        transcript = receipt.get("kv260_terminal_transcript_path") or receipt.get(
            "raw_dispatch_transcript_path"
        )
        digest = receipt.get("kv260_terminal_transcript_sha256") or next(
            (
                r.get("latest_receipt_hash")
                for r in receipt.get("board_rows", [])
                if r.get("board") == name
            ),
            None,
        )
        if transcript and digest:
            shard(
                dict(
                    path=transcript,
                    sha256=digest if digest.startswith("sha256:") else "sha256:" + digest,
                ),
                root,
                raw,
                data,
                name,
            )
        observed = [board.get("processor_class"), board.get("k_max"), board.get("blocker")]
        data["checks"].append(
            custody.operand(
                name,
                root / board.get("source_path", "missing_" + name),
                "recorded_workload_boundary",
                list(contract),
                observed,
            )
        )
        board["custody_valid"] = bool(
            board["custody_valid"] and all(c["passed"] for c in data["checks"][begin:])
        )
        data["boards"].append(board)
        print(
            f"[exp8121] board_check completed={len(data['boards'])} pending={3 - len(data['boards'])}",
            flush=True,
        )
    for eid, destination in [(8119, "batch_available"), (8116, "touches_available")]:
        value = qualified[eid]
        if not value:
            continue
        refs = [
            r
            for r in value.get("raw_shard_hashes", [])
            if Path(r["path"]).name == "primitive_rows.json" and "failed_attempts" not in r["path"]
        ]
        payload = shard(refs[0] if len(refs) == 1 else {}, root, raw, data, f"exp{eid}")
        data[destination] = bool(payload)
        if eid == 8119 and payload:
            data["batches"] = value["measurement_config"]["batches"]
            if value.get("complete_service_ready_score") == 1:
                data["costs"] = payload.get("whole_service_cost_rows", [])
        if eid == 8116 and payload:
            data["updates"] = [
                dict(
                    r,
                    coefficient_touches=len(r.get("touched_coefficients", [])),
                    center_touches=len(r.get("before_weights", [])),
                    durable_writes=1,
                    minimum_coefficient_write_bytes=8 * len(r.get("touched_coefficients", [])),
                )
                for r in payload.get("rows", [])
                if "touched_coefficients" in r
            ]
    for check in data["checks"]:
        check["artifact_field"] = check["field"]
    return data
