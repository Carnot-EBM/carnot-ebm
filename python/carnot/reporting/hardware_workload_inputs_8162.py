"""REQ-REPORT-8162: authenticate boards independently of V705 cost readiness."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting import radial_hardware_8108 as prior

Json = dict[str, Any]
NAMES = {
    8148: "experiment_8148_v704_hardware_workload_boundary",
    8159: "experiment_8159_v705_durable_batch_service",
    8160: "experiment_8160_v705_shared_acquisition_cost",
}
FIELDS = {8159: "host_batch_ready_score", 8160: "acquisition_composition_ready_score"}
PINS = {
    "KV260": "sha256:acdfd841c75649279515fab6b91115d07587094b8703cc72a04febae123236d2",
    "PolarFire": "sha256:341ca079f8ca42ed26dcda1d57edba21cede9c0ca965265fed05121feec2b588",
    "GateMate": "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66",
}


def read_ref(root: Path, ref: Json, raw: Path, data: Json, branch: str) -> Json:
    """Save pinned primitive bytes; an absent shard grants no cost evidence."""
    saved = custody.seal(
        root / ref.get("path", "missing_shard"),
        ref.get("sha256", "sha256:missing"),
        raw,
        data,
        branch,
    )
    return json.loads(saved.read_bytes()) if saved else {}


def load(root: Path, raw: Path) -> Json:
    """Preserve failed primaries and authenticate each branch without any probe.

    Board pins bind to the original receipts, so replacing one receipt or the
    custody summary cannot create fresh fabric credit. Cost shards remain usable
    only when their own terminal report and exact readiness operand pass.
    """
    data: Json = dict(
        fixture=False,
        boards=[],
        branches={},
        checks=[],
        references=[],
        cited=[],
        state={},
        trained_head_specs=[],
    )
    for resource in [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/exclusion_manifest.yaml",
        "research-hardware-wishlist.md",
        "research-references.md",
        "ops/north-star.md",
        "openspec/change-proposals/research-roadmap-v704-preserved-20261005.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/precision_fallback_8042.py",
        ".venv/bin/python",
        ".venv/bin/pytest",
        ".venv/bin/ruff",
        ".venv/bin/mypy",
    ]:
        custody.seal(root / resource, None, raw, data, "local_resource")
    history, _ = prior.authenticate(
        root / "results" / (NAMES[8148] + ".json"), 8148, "hardware_boundary_ready_score", raw, data
    )
    originals = {b["board"]: b for b in history.get("board_rows", [])}
    for name, contract in prior.CONTRACTS.items():
        begin = len(data["checks"])
        board = deepcopy(originals.get(name, dict(board=name, custody_valid=False)))
        path = root / board.get("source_path", "missing_" + name)
        saved = custody.seal(path, PINS[name], raw, data, name)
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
            source_hash=PINS[name],
        )
        data["boards"].append(board)
        print(
            f"[exp8162] board_auth completed={len(data['boards'])} pending={3 - len(data['boards'])}",
            flush=True,
        )
    manifest = root / "ops/exclusion_manifest.yaml"
    policy = yaml.safe_load(manifest.read_text()) if manifest.exists() else {}
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
        work = read_ref(root, value.get("primitive_rows", {}), raw, data, branch)
        upstream = read_ref(root, value.get("input_data", {}), raw, data, branch)
        pairs = work.get("pairs" if eid == 8159 else "host_groups", [])
        completed_stores = 0
        total_stores = sum(len(p["arms"]) for p in pairs)
        for pair in pairs:
            for arm in pair["arms"]:
                store = root / arm.get("store_path", "missing_store")
                pinned = custody.seal(
                    store, arm.get("store_sha256", "sha256:missing"), raw, data, branch
                )
                arm["durable_state_bytes"] = store.stat().st_size if pinned else None
                completed_stores += 1
                if completed_stores % 128 == 0 or completed_stores == total_stores:
                    print(
                        f"[exp8162] store_auth completed={completed_stores} pending={total_stores - completed_stores}",
                        flush=True,
                    )
        qualified = bool(
            qualified
            and eid not in retired
            and value.get("verdict_class") in {"positive", "null"}
            and pairs
            and all(c["passed"] for c in data["checks"][begin:])
        )
        data["branches"][branch] = dict(
            qualified=qualified,
            score=value.get(field),
            pairs=pairs,
            captures=work.get("captures", []),
            path=str(path),
            hash=custody.sha256_file(path) if path.exists() else None,
            startup_amortized_ns=(
                work.get("startup_ns", 0)
                + sum(r.get("acquisition_ns", 0) for r in work.get("warmups", []))
            )
            / 32,
        )
        data["cited"].append(
            dict(
                experiment_id=eid,
                primary_hash=data["branches"][branch]["hash"],
                historical_model_provenance=upstream.get(
                    "historical_model_provenance", value.get("MODEL_SPECS", [])
                ),
                current_model_invocations=0,
            )
        )
        if eid == 8159 and upstream.get("head"):
            head = upstream["head"]
            data["state"] = dict(
                geometry=upstream["geometry"],
                centers=head["centers"],
                coefficients=[head["intercept"], *head["weights"]],
            )
            data["trained_head_specs"] = upstream.get("trained_head_specs", [])
        print(
            f"[exp8162] cost_auth branch={branch} qualified={qualified} completed={len(pairs)} pending=0",
            flush=True,
        )
    for check in data["checks"]:
        check["artifact_field"] = check["field"]
    return data
