"""REQ-REPORT-8203: terminal readers resolve each external prerequisite separately."""

from __future__ import annotations

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting import hardware_service_inputs_8190 as old
from carnot.reporting import radial_hardware_8108 as reader
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.hardware_service_8190 import REOPEN
from scripts import summarize_artifact

Json = dict[str, Any]
BRANCHES = {
    "history": (8190, "hardware_boundary_ready_score", "v707_hardware_service_boundary"),
    "selective": (8195, "selective_fit_ready_score", "v708_selective_energy_fit"),
    "learning": (8198, "calibrated_memory_ready_score", "v708_calibrated_online_memory"),
    "service": (8201, "complete_service_ready_score", "v708_observed_service_cost"),
}


def load(root: Path, raw: Path) -> Json:
    """Seal inputs before arithmetic; old board transcripts retain independent pins."""
    raw.mkdir(parents=True, exist_ok=True)
    data: Json = dict(
        checks=[],
        references=[],
        cited=[],
        heads=[],
        samples=[],
        training=[],
        boards=[],
        history={},
        trained_head_specs=[],
        branches={},
    )
    for path, field, observed in [
        (Path(sys.executable), "python_runtime_supported", sys.version_info >= (3, 11)),
        (raw, "private_writable_storage", raw.is_dir()),
    ]:
        gate = custody.operand("runtime", path, field, True, observed)
        data["checks"].append(gate)
    probe = raw / ".storage_probe"
    probe.write_bytes(b"writable")
    probe.unlink()
    print("[exp8203] phase=source_summaries_before completed=0 pending=1", flush=True)
    summary = raw / "upstream_summaries.log"
    with summary.open("w") as stream, redirect_stdout(stream):
        summarize_artifact.main(
            [
                str(root / "results" / f"experiment_{eid}_{suffix}.json")
                for eid, _, suffix in BRANCHES.values()
            ]
        )
    data["references"].append(custody.reference(summary))
    print("[exp8203] phase=source_summaries_after completed=1 pending=0", flush=True)
    retired_path = root / "ops/exclusion_manifest.yaml"
    policy = yaml.safe_load(retired_path.read_text()) if retired_path.is_file() else {}
    retired = {
        r.get("experiment_id")
        for key in ("retired", "retired_experiments")
        for r in policy.get(key, [])
    }
    boards = old.load(root, raw / "boards")
    data["boards"] = boards["boards"]
    data["references"] += boards["references"]
    data["checks"] += [c for c in boards["checks"] if c["upstream"] in reader.CONTRACTS]
    for board in data["boards"]:
        source = root / board.get("source_path", "missing_board_receipt")
        source_ref = next(
            (r for r in boards["references"] if r.get("original_path") == str(source)), None
        )
        receipt = json.loads(Path(source_ref["path"]).read_bytes()) if source_ref else {}
        transcript = (
            receipt.get("kv260_terminal_transcript_path")
            or receipt.get("raw_dispatch_transcript_path")
            or str(source)
        )
        transcript_path = root / transcript
        board.update(
            workload=board.get("scope", "historical obligation only"),
            timestamp=board.get("last_actual_execution_date") or receipt.get("run_date"),
            timestamp_absence_reason="original execution date unavailable"
            if not board.get("last_actual_execution_date")
            else None,
            source_transcript=str(transcript_path),
            source_transcript_sha256=custody.sha256_file(transcript_path)
            if transcript_path.is_file()
            else None,
            reopen_condition=REOPEN[board["board"]],
            current_hardware_execution=False,
            current_reachability="not_probed",
        )
    data["boards"] += [
        dict(
            board=name,
            custody_valid=False,
            status="blocked_authenticated_access",
            reopen_condition=REOPEN[name],
        )
        for name in ("NPU", "TSU")
    ]
    for name, (eid, score, suffix) in BRANCHES.items():
        print(
            f"[exp8203] before_branch={name} completed={len(data['branches'])} pending={len(BRANCHES) - len(data['branches'])}",
            flush=True,
        )
        path = root / "results" / f"experiment_{eid}_{suffix}.json"
        value, valid = reader.authenticate(path, eid, score, raw, data)
        gate = custody.operand(name, path, "not_retired", True, eid not in retired)
        data["checks"].append(gate)
        if valid:
            receipt = reader_receipt(value["task_id"], root / "results", field=score)
            data["checks"].append(
                custody.operand(name, path, "terminal_readers_agree", True, receipt["passed"])
            )
            valid = valid and receipt["passed"] and gate["passed"]
        data["branches"][name] = valid
        data["cited"].append(
            dict(
                experiment_id=eid,
                fields_imported=[score, "rows", "trained_head_specs"],
                sha256=custody.sha256_file(path) if path.is_file() else None,
            )
        )
        if valid and name == "history":
            data["history"] = value
            data["research_reuse_frequency"] = dict(
                scope="controlled historical research protocol; not demand",
                counts={
                    c: sum(r["condition"] == c for r in value["workload_rows"])
                    for c in {r["condition"] for r in value["workload_rows"]}
                },
            )
        if valid and name in {"selective", "learning"}:
            try:
                head_ref = dict(
                    path=value["frozen_heads_path"], sha256=value["frozen_heads_sha256"]
                )
                saved = custody.seal(checked(head_ref), head_ref["sha256"], raw, data, name)
                evidence = checked(value["measurement_reference"])
                custody.seal(evidence, value["measurement_reference"]["sha256"], raw, data, name)
                heads = json.loads(saved.read_bytes())["heads"] if saved else []
                rows = json.loads(evidence.read_bytes())["materialized_rows"]
                samples = [r for r in rows if r["x"] is not None]
                for head in heads:
                    ids = set(head["head_fit_source_ids"])
                    head["bounded_samples"] = samples
                    head["excluded_samples"] = [r for r in rows if r["x"] is None]
                    head["bounded_training"] = [r for r in samples if r["source_cluster_id"] in ids]
                data["heads"] += heads
                data["samples"] += samples
                data["training"] += heads[0]["bounded_training"]
                data["trained_head_specs"] += value["trained_head_specs"]
                data["historical_model_provenance"] = value.get("historical_model_provenance", {})
            except (KeyError, ValueError, OSError) as error:
                data["branches"][name] = False
                data["checks"].append(
                    custody.operand(
                        name, path, "frozen_head_contract", "qualified radial head", str(error)
                    )
                )
        if valid and name == "service":
            data["history"] = value
        print(
            f"[exp8203] after_branch={name} completed={len(data['branches'])} pending={len(BRANCHES) - len(data['branches'])}",
            flush=True,
        )
    for c in data["checks"]:
        c["artifact_field"] = c.get("artifact_field", c["field"])
    return data
