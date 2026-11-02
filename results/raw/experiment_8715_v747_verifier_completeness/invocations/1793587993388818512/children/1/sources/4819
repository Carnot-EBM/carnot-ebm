"""REQ-REPORT-8134: keep each original receipt independent of service science."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting import radial_hardware_8108 as prior

Json = dict[str, Any]


def load(root: Path, raw: Path) -> Json:
    """Save authenticated original bytes without repeating availability probes.

    Host timing can qualify while complete service remains blocked. Board
    receipts use their own pins, so a service failure cannot erase their custody.
    """
    data: Json = dict(
        fixture=False,
        boards=[],
        systems=[],
        pairs=[],
        modeled=[],
        checks=[],
        references=[],
        cited=[],
        host_qualified=False,
        complete_service=False,
    )
    for resource in [
        *custody.RESOURCES,
        "ops/north-star.md",
        "openspec/capabilities/verification/spec.md",
        "tests/python/test_primary_publication_7928.py",
    ]:
        custody.seal(root / resource, None, raw, data, "local_resource")
    historical, valid = prior.authenticate(
        root / "results/experiment_8121_v702_hardware_batch_boundary.json",
        8121,
        "hardware_boundary_ready_score",
        raw,
        data,
    )
    originals = {b["board"]: b for b in historical.get("board_rows", [])}
    for name, contract in prior.CONTRACTS.items():
        board = deepcopy(originals.get(name, dict(board=name, custody_valid=False)))
        start = len(data["checks"])
        source = root / board.get("source_path", "missing_" + name)
        saved = custody.seal(source, board.get("source_hash"), raw, data, name)
        receipt = json.loads(saved.read_bytes()) if saved else {}
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
            custody.seal(
                root / transcript,
                digest if digest.startswith("sha256:") else "sha256:" + digest,
                raw,
                data,
                name,
            )
        observed = [board.get("processor_class"), board.get("k_max"), board.get("blocker")]
        data["checks"].append(
            custody.operand(name, source, "recorded_workload_boundary", list(contract), observed)
        )
        board["custody_valid"] = bool(
            board.get("custody_valid") and all(c["passed"] for c in data["checks"][start:])
        )
        # Missing execution dates remain missing; receipt dates are not executions.
        board["current_hardware_execution"] = False
        board.pop("original_receipt", None)
        board.pop("original_transcript", None)
        data["boards"].append(board)
        print(
            f"[exp8134] board_auth completed={len(data['boards'])} pending={3 - len(data['boards'])}",
            flush=True,
        )
    if valid:
        ref = historical["replay_input_reference"]
        saved = custody.seal(root / ref["path"], ref["sha256"], raw, data, "exp8121_numerical")
        if saved:
            data["systems"] = json.loads(saved.read_bytes()).get("systems", [])
    path = root / "results/experiment_8132_v703_service_cost.json"
    service, valid = prior.authenticate(path, 8132, "host_service_ready_score", raw, data)
    if valid:
        start = len(data["checks"])
        for field, destination in [
            ("component_cost_rows", "pairs"),
            ("modeled_acquisition_bounds", "modeled"),
        ]:
            ref = service[field]
            saved = custody.seal(root / ref["path"], ref["sha256"], raw, data, "exp8132_" + field)
            if saved:
                payload = json.loads(saved.read_bytes())
                data[destination] = payload["pairs"] if field == "component_cost_rows" else payload
        data["host_qualified"] = bool(
            data["pairs"] and all(c["passed"] for c in data["checks"][start:])
        )
    data["complete_service"] = bool(valid and service.get("complete_service_ready_score") == 1)
    data["checks"].append(
        custody.operand(
            "exp8132",
            path,
            "complete_service_ready_score",
            1,
            service.get("complete_service_ready_score"),
        )
    )
    for eid, value in [(8121, historical), (8132, service)]:
        data["cited"].append(
            dict(
                experiment_id=eid,
                source_references=[
                    r
                    for r in data["references"]
                    if Path(r.get("original_path", "")).name.startswith(f"experiment_{eid}_")
                ],
                fields_imported="individual custody, numerical fixtures, component rows, acquisition model",
                historical_model_provenance=value.get("cited_upstream_artifacts", []),
                current_model_invocations=0,
            )
        )
    for check in data["checks"]:
        check["artifact_field"] = check["field"]
    return data
