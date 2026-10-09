"""REQ-VERIFY-8340: versioned readers and CUDA health require separate evidence.

The previous failed consumer remains historical evidence. New identity hashes
can permit a leased diagnostic, but a calendar change cannot repair a device.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import v719_contract_replay as versioned
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference, checked
from carnot.reporting.roadmap_contract import parse_design as parse_design
from carnot.reporting.v717_contract_methods import PIN as PROTOCOL_PIN
from carnot.reporting.v718_contract_replay import authenticate
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import runtime_change_boundary_8307 as old

__all__ = ["old", "parse_design", "build", "replay"]

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8340_v719_runtime_reader_qualification",
    "exp8340-runtime-reader-qualification",
    "2026.10.719",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_runtime_reader_8340.py"
OWNED = [
    "python/carnot/verify/runtime_reader_8340.py",
    "python/carnot/verify/runtime_reader_report_8340.py",
    "python/carnot/verify/runtime_reader_execution_8340.py",
    CLI,
]
UPSTREAM = "results/experiment_8307_v717_runtime_change_boundary.json"
MODEL_SPECS: list[Json] = []
design = versioned.design


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual progress so waiting children remain visible to the conductor."""
    print(f"[exp8340] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """Exclude only the checksum itself to give every other field byte custody."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def authority(root: Path, raw: Path, milestone: str = MILESTONE) -> Json:
    """Reuse full-task comparison directly, independent of any Exp8332 score."""
    try:
        return dict(versioned.authority(root, raw, milestone))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            activated=False,
            refs=[],
            gate_check_summary=[
                old.gate(root / "research-roadmap.yaml", "execution_authority", True, str(error))
            ],
        )


def changes(data: Json) -> list[Json]:
    """Compare causal identity bytes, ignoring a library's filename and clocks."""
    rows = []
    for field in ("driver", "kernel", "devices", "masks", "libraries", "nodes", "permitted_uuid"):
        left, right = data["previous"].get(field), data["current"].get(field)
        available = left is not None and right is not None
        if field == "libraries":
            available &= bool(left and right) and all(
                r.get("sha256") for r in (left or []) + (right or [])
            )
            left, right = (
                sorted(r.get("sha256") or "" for r in left or []),
                sorted(r.get("sha256") or "" for r in right or []),
            )
        rows.append(
            dict(
                unit_id=field,
                field=field,
                previous=left,
                observed=right,
                available=bool(available),
                changed=bool(available and old.normalized(left) != old.normalized(right)),
                status="completed" if available else "censored",
            )
        )
    return rows


def reduce(data: Json, passed: bool) -> Json:
    """Reader readiness and authenticated change never substitute for copy parity."""
    rows = changes(data)
    changed = any(r["changed"] for r in rows)
    value = old.reduce(data, passed)
    ready = bool(changed and value["cuda_context_ready_score"])
    gates = [g for g in data["checks"] if not g.get("passed", False)]
    gates += data["diagnostic"].get("checks", [])
    gates += [
        old.gate(Path(r["path"]), "library.sha256", "readable SHA-256", None)
        for r in data["current"].get("libraries", [])
        if r.get("sha256") is None
    ]
    gates += [
        old.gate(Path(r["stdout_path"]), "inventory.normal_exit", True, r["passed"])
        for r in data["observation_receipts"]
        if not r["passed"]
    ]
    if not passed:
        gates.append(
            old.gate(
                Path(data.get("boundary_path", ROOT / UPSTREAM)),
                "required_checks_passed",
                True,
                False,
            )
        )
    if not gates and not ready:
        gates.append(
            old.gate(
                Path(data.get("boundary_path", ROOT / UPSTREAM)),
                "authenticated_causal_delta" if not changed else "leased_context_copy_ready",
                True,
                changed if not changed else ready,
            )
        )
    verdict = "disqualified" if not passed else "blocked" if gates else "null"
    for g in gates:
        g["upstream_id"] = {
            "research-roadmap.yaml": "V719_authority",
            "v717-local-learning-protocol.json": "V717_protocol",
            Path(UPSTREAM).name: "exp8307-runtime-change-boundary",
        }.get(Path(g.get("path", g.get("artifact_path", ""))).name, TASK)
    value.update(
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            "owned_checks"
            if not passed
            else "external_operands"
            if any(not c.get("passed", False) for c in data["checks"])
            else "cuda_environment_unchanged"
            if not changed
            else "cuda_context"
        ),
        verdict_class=verdict,
        gate_check_summary=gates,
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["available"] for r in rows),
        censored_count=sum(not r["available"] for r in rows),
        runtime_reader_ready_score=int(
            passed
            and data.get("authority", {}).get("activated", False)
            and all(c.get("passed", False) for c in data["checks"])
        ),
        runtime_changed_score=int(changed),
        cuda_context_ready_score=int(ready and passed and not gates),
        change_evidence=rows,
        device_identity=data["current"].get("devices", []),
        lease_binding=data["diagnostic"].get("lease_receipt", {}),
        runtime_library_hashes=data["current"].get("libraries", []),
        sample_size_budget=dict(intended_operands=len(rows), independent_scientific_samples=0),
        acceptance_gates=dict(
            reader=passed, authenticated_change=changed, cuda_context_ready=ready
        ),
    )
    return value


def measure(root: Path, raw: Path) -> Json:
    """Authenticate failed source evidence before observing fresh read-only identities."""
    progress("authority_and_source_before")
    auth = authority(root, raw / "authority")
    data: Json = dict(
        previous={},
        current={},
        historical={},
        checks=[],
        refs=[],
        diagnostic=dict(rows=[], checks=[], lease_receipt={}, cleanup_receipt={}),
        receipt_rows=[],
        fixture=False,
        observation_receipts=[],
        authority=auth,
        boundary_path=str(raw / "measurement.json"),
    )
    data["checks"].append(
        old.gate(root / "research-roadmap.yaml", "full_V719_authority", True, auth["activated"])
    )
    data["refs"].extend(
        snapshot(root / p, raw / "authority_inputs", Path(p).stem)
        for p in ["research-roadmap.yaml", "openspec/change-proposals/research-roadmap-vNEXT.md"]
    )
    source = authenticate(root / UPSTREAM, raw / "source", data["refs"], data["checks"])
    if source:
        try:
            terminal = json.loads(Path(source["terminal_validation_sidecar_path"]).read_bytes())
            if not terminal["normal_process_exit"] or not terminal["report"]["passed"]:
                raise ValueError("source_terminal_checks")
            operand = dict(
                path=source["runtime_binding_path"], sha256=source["runtime_binding_sha256"]
            )
            require_reference(operand)
            original = json.loads(checked(operand).read_bytes())
            data["baseline_reference"] = snapshot(checked(operand), raw / "source", "baseline")
            data["refs"].append(data["baseline_reference"])
            data["baseline_source_reference"] = next(
                r for r in data["refs"] if r["path"] == str(root / UPSTREAM)
            )
            data.update(previous=original, historical=source)
            data["current"], data["observation_receipts"] = old.inventory(
                data["previous"], raw / "inventory"
            )
            failure = next(
                r
                for r in source["validation_receipts"]
                if r["name"] == "runtime_and_publication_consumers"
            )
            for stream in ("stdout", "stderr"):
                operand = dict(path=failure[stream + "_path"], sha256=failure[stream + "_sha256"])
                require_reference(operand)
                data["refs"].append(
                    snapshot(Path(operand["path"]), raw / "source", "original_failure_" + stream)
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            data["checks"].append(
                old.gate(root / UPSTREAM, "runtime_identity_custody", True, str(error))
            )
    protocol = root / "openspec/change-proposals/v717-local-learning-protocol.json"
    data["refs"].append(snapshot(protocol, raw / "protocol", "frozen_science"))
    data["checks"].append(
        old.gate(
            protocol,
            "frozen_protocol_sha256",
            PROTOCOL_PIN,
            sha256_file(protocol) if protocol.is_file() else None,
        )
    )
    if all(c.get("passed", False) for c in data["checks"]) and any(
        r["changed"] for r in changes(data)
    ):
        progress("changed_leased_probe_before")
        data["diagnostic"] = old.probe_changed(data, raw / "changed_cuda")
        progress("changed_leased_probe_after")
    binding = raw / "current_binding.json"
    atomic_json(binding, data["current"])
    data["current_reference"] = reference(binding)
    data["observed_wall_ns"] = time.time_ns()
    progress("authority_and_source_after")
    return data


from carnot.verify.runtime_reader_report_8340 import build as build, replay as replay  # noqa: E402
