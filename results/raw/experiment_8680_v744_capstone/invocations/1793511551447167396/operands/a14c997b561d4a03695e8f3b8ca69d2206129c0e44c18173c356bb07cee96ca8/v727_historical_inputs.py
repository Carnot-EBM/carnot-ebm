"""REQ-REPORT-8443: original producer authority prevents historical relabeling."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v726_capstone_evidence as cap
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v721_capstone_evidence import frozen_inputs
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = cap.ROOT
DESIGN = "openspec/change-proposals/research-roadmap-v726-preserved-20261012.md"
DESIGN_PIN = "sha256:fa83ab5bd372b7ca9953ccc82953e51d0f126b3563bd907b7cde59c6d30298ab"
ACTIVE = "results/raw/experiment_8429_v726_contract_methods/invocations/1791757329212215566/authority/active-7717c59393378fee17f418312586baaf489259d2bb9db7b3f8562c7d3145a7ca.bin"
ACTIVE_PIN = "sha256:7717c59393378fee17f418312586baaf489259d2bb9db7b3f8562c7d3145a7ca"
PINS = {**cap.PINS, 8407: "033cdbbe98be7790889354e50646f49c24a9e270cbb5c3c83b8d708c540c293f"}
FIELDS = {
    8430: "feature_panel_ready_score",
    8431: "head_ready_score",
    8433: "block_store_ready_score",
    8434: "learning_stream_ready_score",
    8407: "python_cost_ready_score",
}


def custody(root: Path, raw: Path) -> list[Json]:
    """Copy bounded original bytes before a reader can resolve mutable aliases."""
    refs = []
    for name, pin, alias in [
        (DESIGN, DESIGN_PIN, cap.methods.DESIGN),
        (ACTIVE, ACTIVE_PIN, cap.methods.ACTIVE),
    ]:
        path = root / name
        if sha256_file(path) != pin:
            raise ValueError("historical_authority_pin")
        ref = cap.freeze(path, raw)
        ref["source_path"] = str(ROOT / alias)
        refs.append(ref)
    return refs


def qualify(number: int, primary: Json, authority: list[Json], scratch: Path) -> Json:
    """Shipped replay checks primitives; reported readiness alone cannot qualify inputs."""
    field = FIELDS[number]
    observed: Any = None
    failures = []
    try:
        if primary.get("sha256") != "sha256:" + PINS[number]:
            raise ValueError("original_producer_pin")
        if number != 8407:
            for ref, pin in zip(authority, [DESIGN_PIN, ACTIVE_PIN], strict=True):
                if sha256_file(Path(ref["snapshot_path"])) != pin:
                    raise ValueError("historical_authority_pin")
        value = cap.load(primary)
        observed = value.get(field)
        with frozen_inputs(authority, scratch):
            if number == 8407:
                module = importlib.import_module("carnot.reporting.python_transaction_cost_8407")
                terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
                work = module.authority.checked(
                    Path(value["work_reference"]["path"]), value["work_reference"]["sha256"]
                )
                operands = json.loads(work.read_bytes())
                sealed = [dict(r, exists=True, snapshot_path=r["path"]) for r in operands["refs"]]
                with frozen_inputs(sealed, scratch / "service-inputs"):
                    actual = module.authenticate(
                        Path(operands["root"]),
                        scratch / "service-auth",
                        dict(refs=[], gate_check_summary=[]),
                    )
                checked = (
                    actual["head"] == operands["bundle"]["head"]
                    and actual["slots"] == operands["bundle"]["slots"]
                )
                checked &= actual["labels"]["sha256"] == operands["bundle"]["labels"]["sha256"]
                checked &= len(operands["results"]) == 162
                for ref in [
                    *value["source_artifact_hashes"],
                    *value["code_config_hashes"],
                    *operands["results"],
                ]:
                    module.authority.checked(Path(ref["path"]), ref["sha256"])
                checked &= terminal["primary_sha256"] == primary["sha256"]
                side = read_bound_sidecar(Path(primary["path"]), Path(terminal["sidecar_path"]))
                checked &= side["report"]["passed"] is True
                checked &= value["required_checks_passed"] and not value["flagged_adversarial"]
            else:
                tasks = yaml.safe_load(Path(authority[1]["snapshot_path"]).read_bytes())["tasks"]
                task = next(t for t in tasks if t["id"].startswith(f"exp{number}-"))
                checked = bool(cap.outcome(task, primary, scratch)["eligible"])
        if not checked or observed != 1:
            raise ValueError("original_producer_replay_and_readiness")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        checked = False
        failures = [
            dict(
                check="historical_producer_qualification",
                upstream_id=f"exp{number}",
                artifact_path=primary["path"],
                artifact_hash=primary.get("sha256"),
                artifact_field=field,
                op="==",
                expected=1,
                observed=observed,
                reason=str(error),
                passed=False,
            )
        ]
    return dict(
        experiment_id=number,
        field=field,
        observed=observed,
        qualified=bool(checked),
        primary_sha256=primary.get("sha256"),
        qualification_scope="original complete workload authority and all operand hashes; cost statistics not imported"
        if number == 8407
        else "original producer primitive replay",
        gate_check_summary=failures,
    )


def worker(request: Path, output: Path) -> int:
    """A fresh bounded process owns replay allocations and keeps scratch on disk."""
    args = json.loads(request.read_bytes())
    with TemporaryDirectory(prefix="exp8443-history-", dir="/var/tmp") as directory:
        result = qualify(args["number"], args["primary"], args["authority"], Path(directory))
    atomic_json(output, result)
    return 0
