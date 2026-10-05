"""REQ-REPORT-8148: authenticate original board receipts and each service separately."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting import radial_hardware_8108 as prior
from carnot.reporting.hardware_workload_8148 import FIELDS

Json = dict[str, Any]
NAMES = {
    8134: "experiment_8134_v703_hardware_service_boundary",
    8145: "experiment_8145_v704_natural_service_cost",
    8146: "experiment_8146_v704_live_service_cost",
}


def load(root: Path, raw: Path) -> Json:
    """Save exact historical bytes without model, board or availability probes.

    The original board pins remain independent even if science receipts fail.
    Retired experiments are terminal external blocks and never cause a retry.
    """
    data: Json = dict(
        fixture=False, boards=[], systems=[], checks=[], references=[], cited=[], branches={}
    )
    for resource in custody.RESOURCES:
        custody.seal(root / resource, None, raw, data, "local_resource")
    history, _ = prior.authenticate(
        root / "results" / (NAMES[8134] + ".json"), 8134, "hardware_boundary_ready_score", raw, data
    )
    originals = {b["board"]: b for b in history.get("board_rows", [])}
    for name, contract in prior.CONTRACTS.items():
        board = deepcopy(originals.get(name, dict(board=name, custody_valid=False)))
        begin = len(data["checks"])
        source = root / board.get("source_path", "missing_" + name)
        saved = custody.seal(source, board.get("source_hash") or "sha256:missing", raw, data, name)
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
        board.update(
            custody_valid=bool(
                saved
                and board.get("custody_valid")
                and all(c["passed"] for c in data["checks"][begin:])
            ),
            original_receipt_run_date=receipt.get("run_date"),
            original_observed_latest_receipt_date=receipt.get("gate_check_summary", {}).get(
                "observed_latest_receipt_date"
            )
            if isinstance(receipt.get("gate_check_summary"), dict)
            else None,
        )
        data["boards"].append(board)
        print(
            f"[exp8148] board_auth completed={len(data['boards'])} pending={3 - len(data['boards'])}",
            flush=True,
        )
    retired: set[int] = set()
    manifest = root / "ops/exclusion_manifest.yaml"
    if manifest.exists():
        import yaml

        policy = yaml.safe_load(manifest.read_text())
        retired = {
            r["experiment_id"]
            for key in ["retired", "retired_experiments"]
            for r in policy.get(key, [])
            if "experiment_id" in r
        }
    for eid in [8145, 8146]:
        branch = f"exp{eid}"
        path = root / "results" / (NAMES[eid] + ".json")
        value, qualified = prior.authenticate(path, eid, FIELDS[branch], raw, data)
        data["checks"].append(
            custody.operand(branch, path, "not_retired", True, eid not in retired)
        )
        qualified = qualified and eid not in retired
        pairs: list[Json] = []
        if qualified:
            start = len(data["checks"])
            refs = value.get("raw_shard_hashes", [])
            ref = value.get("primitive_rows") or next(
                (r for r in refs if r["path"].endswith("primitive_rows.json")), {}
            )
            saved = custody.seal(
                root / ref.get("path", "missing_primitives"),
                ref.get("sha256") or "sha256:missing",
                raw,
                data,
                branch,
            )
            if saved:
                work = json.loads(saved.read_bytes())
                pairs = work.get("pairs", []) + work.get("updates", [])
            qualified = bool(pairs and all(c["passed"] for c in data["checks"][start:]))
        data["branches"][branch] = dict(
            qualified=qualified,
            score=value.get(FIELDS[branch]),
            path=str(path),
            hash=custody.sha256_file(path) if path.exists() else None,
            pairs=pairs,
        )
        data["cited"].append(
            dict(
                experiment_id=eid,
                historical_model_provenance=value.get(
                    "cited_upstream_artifacts", value.get("MODEL_SPECS", [])
                ),
                original_primary_hash=custody.sha256_file(path) if path.exists() else None,
                current_model_invocations=0,
                imported_readiness_field=FIELDS[branch],
            )
        )
        if eid == 8145:
            data["trained_head_specs"] = value.get("trained_head_specs", [])
    # A missing readiness field must still name the original requested operand.
    for check in data["checks"]:
        check["artifact_field"] = check["field"]
    return data
