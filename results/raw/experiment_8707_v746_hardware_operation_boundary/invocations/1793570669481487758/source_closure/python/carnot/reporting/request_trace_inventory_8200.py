"""REQ-REPORT-8200: copy task-authorized evidence before opening call rows.

Task authorities choose the paths. A filesystem glob could silently select a
different attempt and would not preserve the intended research conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from carnot.verify import request_trace_8200 as e

Json = dict[str, Any]


def operand(check: str, path: Path, expected: Any, observed: Any, op: str = "==") -> Json:
    """Blocked artifacts name the actual operand instead of a generic deficit."""
    return dict(
        check=check,
        upstream=path.stem,
        path=str(path),
        hash=e.sha256_file(path) if path.is_file() else None,
        artifact_field=check,
        op=op,
        expected=expected,
        observed=observed,
        passed=observed == expected if op == "==" else observed >= expected,
    )


def copy_bytes(path: Path, raw: Path) -> Json:
    """Content-addressed copies preserve historical bytes without changing inputs."""
    ref = e.reference(path)
    frozen = raw / "inputs" / (ref["sha256"].split(":")[1] + path.suffix)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        shutil.copyfile(path, frozen)
        frozen.chmod(0o444)
    return dict(ref, frozen_path=str(frozen))


def authorities(root: Path, raw: Path) -> Json:
    """Only completed milestone task contracts can name inventory inputs."""
    checks, refs, tasks = [], [], []
    source = root / "research-complete.yaml"
    checks.append(operand("task_authority_exists", source, True, source.is_file()))
    if source.is_file():
        ref = copy_bytes(source, raw)
        refs.append(ref)
        data = yaml.safe_load(Path(ref["frozen_path"]).read_text())
        for milestone in data["milestones"]:
            version = int(milestone["id"].rsplit(".", 1)[-1])
            if 701 <= version <= 707:
                tasks.extend(dict(t, milestone=version) for t in milestone["tasks"])
        for path in sorted(
            (root / "openspec/change-proposals").glob("research-roadmap-v70[1-7]-preserved-*.md")
        ):
            refs.append(copy_bytes(path, raw))
    e.seal(raw / "task_authorities.json", dict(tasks=tasks, references=refs, checks=checks))
    return dict(tasks=tasks, refs=refs, checks=checks)


def inventory(root: Path, raw: Path, authority: Json) -> Json:
    """Keep raw rows intact; unavailable fields remain unavailable in the census."""
    records, inventory_rows, refs, checks, citations = (
        [],
        [],
        list(authority["refs"]),
        list(authority["checks"]),
        [],
    )
    mapping: Json = {}
    total = len(authority["tasks"])
    for index, task in enumerate(authority["tasks"]):
        path = root / task["deliverable"]
        checks.append(operand("authorized_primary_exists", path, True, path.is_file()))
        if path.is_file():
            ref = copy_bytes(path, raw)
            refs.append(ref)
            value = json.loads(Path(ref["frozen_path"]).read_text())
            citations.append(
                dict(
                    experiment_id=value.get("experiment_id"),
                    sha256=ref["sha256"],
                    fields_imported=["call_ledger", "model_invocation_counts", "raw_shard_hashes"],
                )
            )
            ledger = value.get("call_ledger", [])
            for position, row in enumerate(ledger):
                original = dict(row)
                original["provenance"] = dict(
                    task_id=task["id"],
                    milestone=task["milestone"],
                    primary=ref,
                    ledger_index=position,
                )
                records.append(original)
                inventory_rows.append(
                    dict(
                        inventory_index=len(records) - 1,
                        task_id=task["id"],
                        original_call_id=row.get("call_id"),
                        unit_id=row.get("unit_id"),
                        issued_at=row.get("issued_at"),
                        ledger_index=position,
                        primary_sha256=ref["sha256"],
                        frozen_path=ref["frozen_path"],
                    )
                )
            if value.get("experiment_id") == 8188:
                for method in value.get("source_artifact_hashes", []):
                    if "measurement_code" in method["path"]:
                        source = Path(method["path"])
                        observed = e.sha256_file(source) if source.is_file() else None
                        checks.append(
                            operand("immutable_method_sha256", source, method["sha256"], observed)
                        )
                        if checks[-1]["passed"]:
                            refs.append(copy_bytes(source, raw))
                for shard in value["raw_shard_hashes"]:
                    if Path(shard["path"]).name == "input_data.json":
                        source = Path(shard["path"])
                        checks.append(
                            operand(
                                "pinned_input_sha256",
                                source,
                                shard["sha256"],
                                e.sha256_file(source) if source.is_file() else None,
                            )
                        )
                        if checks[-1]["passed"]:
                            frozen = copy_bytes(source, raw)
                            refs.append(frozen)
                            data = json.loads(Path(frozen["frozen_path"]).read_text())
                            runtime, protocol = data["runtime_freeze"], data["protocol"]
                            mapping = dict(
                                model_gguf=protocol["gguf_sha256"],
                                model_revision=protocol["model_revision"],
                                tokenizer=runtime["tokenizer_sha256"],
                                runtime=protocol["runtime_sha256"],
                                chat_template_sha256=protocol["chat_template_sha256"],
                                quantization="Q4_K_M",
                                runtime_version="b9606-9b4dae81f",
                            )
                            binary = Path(runtime["runtime"]["path"])
                            checks.append(
                                operand(
                                    "pinned_runtime_sha256",
                                    binary,
                                    runtime["runtime"]["sha256"],
                                    e.sha256_file(binary) if binary.is_file() else None,
                                )
                            )
                            if binary.is_file():
                                refs.append(copy_bytes(binary, raw))
        e.progress("inventory", index + 1, total - index - 1)
    e.seal(
        raw / "inventory.json",
        dict(
            records=records,
            replay_identity=mapping,
            inventory_rows=inventory_rows,
            references=refs,
            checks=checks,
        ),
    )
    return dict(
        records=records,
        replay_identity=mapping,
        inventory_rows=inventory_rows,
        refs=refs,
        checks=checks,
        citations=citations,
    )
