"""REQ-REPORT-8176: authenticate branches and original board receipts separately."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting import hardware_workload_inputs_8162 as old
from carnot.reporting import radial_hardware_8108 as prior
from carnot.reporting.hardware_workload_8176 import FIELDS

Json = dict[str, Any]
NAMES = {
    8162: "experiment_8162_v705_hardware_workload_boundary",
    8159: old.NAMES[8159],
    8173: "experiment_8173_v706_service_validation",
    8174: "experiment_8174_v706_complete_request_cost",
}


def shard(value: Json, name: str, root: Path, raw: Path, data: Json, branch: str) -> Json:
    """Admit only a pinned primitive shard, keeping absence distinct from zero."""
    ref = value.get(name) or next(
        (r for r in value.get("raw_shard_hashes", []) if Path(r["path"]).name == name + ".json"), {}
    )
    return old.read_ref(root, ref, raw, data, branch) if ref else {}


def load(root: Path, raw: Path) -> Json:
    """Preserve historical inputs and gate each scientific scope on its own field.

    A successful validation repair cannot supply absent acquisition clocks.
    Original board pins are checked even if every cost branch is blocked. No
    access probe, model load, purchase or flash belongs to this reduction.
    """
    data: Json = dict(
        fixture=False,
        checks=[],
        references=[],
        cited=[],
        boards=[],
        branches={},
        state={},
        trained_head_specs=[],
        startup_costs={},
    )
    for resource in [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "research-hardware-wishlist.md",
        "research-references.md",
        "ops/north-star.md",
        "openspec/change-proposals/research-roadmap-v705-preserved-20261005.md",
        "openspec/change-proposals/v706-service-validation-protocol.json",
        "scripts/experiment_template.py",
        "python/carnot/reporting/precision_fallback_8042.py",
        ".venv/bin/python",
        ".venv/bin/pytest",
        ".venv/bin/ruff",
        ".venv/bin/mypy",
    ]:
        custody.seal(root / resource, None, raw, data, "local_resource")
    history, _ = prior.authenticate(
        root / "results" / (NAMES[8162] + ".json"), 8162, "hardware_boundary_ready_score", raw, data
    )
    snapshot = shard(history, "replay_input_reference", root, raw, data, "exp8162")
    data["state"] = snapshot.get("state", {})
    data["cited"].append(
        dict(
            experiment_id=8162,
            fields_imported=["board_rows", "replay_input_reference"],
            sha256=custody.sha256_file(root / "results" / (NAMES[8162] + ".json"))
            if (root / "results" / (NAMES[8162] + ".json")).is_file()
            else None,
        )
    )
    for index, ref in enumerate(snapshot.get("references", [])):
        p = Path(ref["path"])
        data["checks"].append(
            custody.operand(
                "exp8162",
                p,
                "evidence_sha256",
                ref["sha256"],
                custody.sha256_file(p) if p.is_file() else None,
            )
        )
        data["references"].append(ref)
        if index % 128 == 0 or index + 1 == len(snapshot["references"]):
            print(
                f"[exp8176] historical_auth completed={index + 1} pending={len(snapshot['references']) - index - 1}",
                flush=True,
            )
    originals = {b["board"]: b for b in history.get("board_rows", [])}
    for name, contract in prior.CONTRACTS.items():
        begin = len(data["checks"])
        board = deepcopy(originals.get(name, dict(board=name, custody_valid=False)))
        path = root / board.get("source_path", "missing_" + name)
        saved = custody.seal(path, old.PINS[name], raw, data, name)
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
            custody.operand(name, path, "recorded_workload_boundary", list(contract), observed)
        )
        board.update(
            custody_valid=bool(saved and all(c["passed"] for c in data["checks"][begin:])),
            original_receipt_run_date=receipt.get("run_date"),
            source_hash=old.PINS[name],
        )
        data["boards"].append(board)
        print(
            f"[exp8176] board_auth completed={len(data['boards'])} pending={3 - len(data['boards'])}",
            flush=True,
        )
    policy = (
        yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_text())
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else {}
    )
    retired = {
        r.get("experiment_id")
        for key in ["retired", "retired_experiments"]
        for r in policy.get(key, [])
    }
    for eid, field in FIELDS.items():
        branch = f"exp{eid}"
        path = root / "results" / (NAMES[eid] + ".json")
        value, qualified = prior.authenticate(path, eid, field, raw, data)
        data["checks"].append(
            custody.operand(branch, path, "not_retired", True, eid not in retired)
        )
        data["checks"].append(
            custody.operand(
                branch,
                path,
                "admissible_verdict",
                True,
                value.get("verdict_class") in {"positive", "null"},
            )
        )
        begin = len(data["checks"])
        work = shard(value, "primitive_rows", root, raw, data, branch)
        upstream = shard(value, "input_data", root, raw, data, branch)
        pairs = work.get("pairs", []) if eid == 8159 else []
        if eid == 8173:
            composed = next(
                (s for s in upstream.get("sources", []) if s["value"]["experiment_id"] == 8160), {}
            )
            work = composed.get("work", {})
            pairs = work.get("host_groups", [])
        if eid == 8174:
            if not data["state"] and upstream.get("head"):
                head = upstream["head"]
                data["state"] = dict(
                    geometry=upstream["geometry"],
                    centers=head["centers"],
                    coefficients=[head["intercept"], *head["weights"]],
                )
                data["trained_head_specs"] = upstream.get("trained_head_specs", [])
            for request in work.get("requests", []):
                for ref in [*request.get("evidence", []), request.get("store", {})]:
                    old.read_ref(root, ref, raw, data, branch)
                if request["status"] != "completed":
                    continue
                total = request["response_ns"] - request["arrival_ns"]
                arithmetic = request["arithmetic_end_ns"] - request["arithmetic_start_ns"]
                acquisition = request["acquisition_end_ns"] - request["arrival_ns"]
                queue = request["batch_start_ns"] - request["acquisition_end_ns"]
                persistence = request["commit_end_ns"] - request["commit_start_ns"]
                components = dict(
                    arithmetic_ns=arithmetic,
                    acquisition_ns=acquisition,
                    queue_ns=queue,
                    persistence_ns=persistence,
                    residual_host_and_readout_ns=total
                    - arithmetic
                    - acquisition
                    - queue
                    - persistence,
                )
                store = root / request["store"]["path"]
                arm = dict(
                    arm=request["arm"],
                    duration_ns=total,
                    components=components,
                    requests=[request],
                    durable_state_bytes=store.stat().st_size if store.is_file() else None,
                )
                pairs.append(
                    dict(
                        unit_id=request["unit_id"],
                        condition=request["condition"],
                        status=request["status"],
                        arms=[arm],
                    )
                )
            data["startup_costs"] = value.get("startup_costs", {})
        if eid == 8159 and upstream.get("head"):
            head = upstream["head"]
            data["state"] = dict(
                geometry=upstream["geometry"],
                centers=head["centers"],
                coefficients=[head["intercept"], *head["weights"]],
            )
            data["trained_head_specs"] = upstream.get("trained_head_specs", [])
        if eid == 8159:
            for pair in pairs:
                for arm in pair["arms"]:
                    store = root / arm["store_path"]
                    custody.seal(store, arm["store_sha256"], raw, data, branch)
                    arm["durable_state_bytes"] = store.stat().st_size if store.is_file() else None
        qualified = bool(
            qualified
            and eid not in retired
            and value.get("verdict_class") in {"positive", "null"}
            and all(c["passed"] for c in data["checks"][begin:])
        )
        data["branches"][branch] = dict(
            qualified=qualified,
            score=value.get(field),
            pairs=pairs,
            captures=work.get("captures", []),
            startup_amortized_ns=(
                work.get("startup_ns", 0)
                + sum(r.get("acquisition_ns", 0) for r in work.get("warmups", []))
            )
            / 32,
            path=str(path),
            hash=custody.sha256_file(path) if path.is_file() else None,
        )
        data["cited"].append(
            dict(
                experiment_id=eid,
                fields_imported=[field, "primitive_rows", "input_data"],
                sha256=data["branches"][branch]["hash"],
                historical_model_provenance=value.get(
                    "historical_model_provenance", value.get("MODEL_SPECS", [])
                ),
                historical_model_invocation_counts=value.get("model_invocation_counts", {}),
                current_model_invocations=0,
            )
        )
        print(
            f"[exp8176] branch_auth branch={branch} qualified={qualified} completed={len(pairs)} pending=0",
            flush=True,
        )
    for check in data["checks"]:
        check["artifact_field"] = check["field"]
    return data
