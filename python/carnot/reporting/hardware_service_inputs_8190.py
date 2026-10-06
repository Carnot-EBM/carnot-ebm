"""REQ-REPORT-8190: admit only byte-bound reuse evidence and original board pins."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting import hardware_workload_inputs_8176 as previous
from carnot.reporting import radial_hardware_8108 as prior
from carnot.verify.exact_request_8188 import components

Json = dict[str, Any]
NAMES = dict(previous.NAMES, **{})
NAMES[8176] = "experiment_8176_v706_hardware_workload_boundary"
NAMES[8188] = "experiment_8188_v707_exact_request_service"
PINS = previous.old.PINS
RESOURCES = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "ops/exclusion_manifest.yaml",
    "scripts/experiment_template.py",
    "python/carnot/reporting/precision_fallback_8042.py",
    "openspec/change-proposals/v706-service-validation-protocol.json",
    "research-hardware-wishlist.md",
    "research-references.md",
    "ops/north-star.md",
    ".venv/bin/python",
    ".venv/bin/pytest",
    ".venv/bin/ruff",
    ".venv/bin/mypy",
]
CLOCKS = [
    ("key_hash_start_ns", "key_hash_end_ns", "key_hash_ns"),
    ("cache_read_start_ns", "cache_read_end_ns", "cache_read_ns"),
    ("generation_start_ns", "generation_end_ns", "generation_ns"),
    ("parsing_start_ns", "parsing_end_ns", "parse_ns"),
    ("feature_start_ns", "feature_end_ns", "features_ns"),
    ("cache_write_start_ns", "cache_write_end_ns", "cache_write_ns"),
    (
        "boundary_crossing_start_ns",
        "boundary_crossing_end_ns",
        "native_crossing_including_arithmetic_ns",
    ),
    ("serialization_start_ns", "serialization_end_ns", "decision_serialization_ns"),
    ("commit_start_ns", "commit_end_ns", "decision_write_ack_ns"),
]


def load(root: Path, raw: Path) -> Json:
    """Authenticate each scope without repairing an unavailable historical branch.

    The original board pins are independent of software costs. Cached service
    primitives supply the new estimand; full requests remain historical context.
    Sealed bytes retain their source paths for later independent cold replay.
    """
    data: Json = dict(
        fixture=False,
        checks=[],
        references=[],
        cited=[],
        boards=[],
        requests=[],
        state={},
        trained_head_specs=[],
        service_score=None,
        service_qualified=False,
        startup_ns=0,
        source={},
        historical_context={},
    )
    for resource in RESOURCES:
        custody.seal(root / resource, None, raw, data, "local_resource")
    history, _ = prior.authenticate(
        root / "results" / (NAMES[8176] + ".json"), 8176, "hardware_boundary_ready_score", raw, data
    )
    originals = {b["board"]: b for b in history.get("board_rows", [])}
    history_path = root / "results" / (NAMES[8176] + ".json")
    data["cited"].append(
        dict(
            experiment_id=8176,
            fields_imported=["board_rows"],
            sha256=custody.sha256_file(history_path) if history_path.is_file() else None,
        )
    )
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
            f"[exp8190] board_auth completed={len(data['boards'])} pending={3 - len(data['boards'])}",
            flush=True,
        )
    policy = (
        yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_text())
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else {}
    )
    retired = {
        r.get("experiment_id")
        for k in ("retired", "retired_experiments")
        for r in policy.get(k, [])
    }
    for eid, field in (
        (8188, "cached_service_ready_score"),
        (8174, "complete_service_ready_score"),
    ):
        path = root / "results" / (NAMES[eid] + ".json")
        value, qualified = prior.authenticate(path, eid, field, raw, data)
        begin = len(data["checks"])
        for key, wanted, observed in [
            ("not_retired", True, eid not in retired),
            ("admissible_verdict", True, value.get("verdict_class") in {"positive", "null"}),
        ]:
            data["checks"].append(custody.operand(f"exp{eid}", path, key, wanted, observed))
        work = previous.shard(value, "primitive_rows", root, raw, data, f"exp{eid}")
        if eid == 8188:
            upstream = previous.shard(value, "input_data", root, raw, data, "exp8188")
            head = upstream.get("head", {})
            if head:
                data["state"] = dict(
                    geometry=upstream["geometry"],
                    centers=head["centers"],
                    coefficients=[head["intercept"], *head["weights"]],
                )
            data["trained_head_specs"] = upstream.get("trained_head_specs", [])
            data["source"] = dict(
                path=str(path), hash=custody.sha256_file(path) if path.is_file() else None
            )
            data["service_score"] = value.get(field)
            data["startup_ns"] = work.get("startup_ns", 0)
            data["service_qualified"] = qualified and all(
                c["passed"] for c in data["checks"][begin:]
            )
            requests = work.get("requests", [])
            for index, r in enumerate(requests):
                start = len(data["checks"])
                for ref in [*r.get("evidence", []), r.get("store", {})]:
                    custody.seal(
                        root / ref.get("path", "missing_evidence"),
                        ref.get("sha256", "sha256:missing"),
                        raw,
                        data,
                        r["unit_id"],
                    )
                try:
                    costs = components(r)
                except KeyError:
                    costs = {}
                data["requests"].append(
                    dict(
                        request=r,
                        components=costs,
                        qualified=all(c["passed"] for c in data["checks"][start:]),
                    )
                )
                if index % 32 == 0 or index + 1 == len(requests):
                    print(
                        f"[exp8190] request_auth completed={index + 1} pending={len(requests) - index - 1}",
                        flush=True,
                    )
        else:
            data["historical_context"] = dict(
                qualified=qualified,
                component_cost_rows=value.get("component_cost_rows", []),
                startup_costs=value.get("startup_costs", {}),
                requests=work.get("requests", []),
                scope="historical complete requests; outside reuse estimand",
            )
        data["cited"].append(
            dict(
                experiment_id=eid,
                fields_imported=[field, "primitive_rows", "input_data"]
                if eid == 8188
                else [field, "primitive_rows", "component_cost_rows"],
                sha256=custody.sha256_file(path) if path.is_file() else None,
            )
        )
    for check in data["checks"]:
        check["artifact_field"] = check.get("artifact_field", check["field"])
    return data
