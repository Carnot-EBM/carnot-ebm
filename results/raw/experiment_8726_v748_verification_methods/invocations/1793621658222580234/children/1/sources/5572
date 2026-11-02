"""REQ-REPORT-8565: reuse original qualified bytes rather than rerunning controllers.

Each branch carries its own hash checks. Imported historical failures stay in
the historical provenance and cannot disqualify a separate qualified family.
"""

from __future__ import annotations

import json
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting import v736_contract_methods as e
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def checked(ref: Json) -> bool:
    """A missing source is unknown, while a changed source invalidates receipt reuse."""
    if "config" in ref:
        return bool(canonical_hash(ref["config"]) == ref["sha256"])
    path = Path(ref["path"])
    return path.is_file() and sha256_file(path) == ref["sha256"]


def family(ref: Json, number: int) -> Json:
    """Authenticate original source, test, dependency and primitive manifests independently."""
    failures = []
    references = [ref]
    value: Json = {}
    try:
        value = e.read(ref)
        references += (
            value.get("code_config_hashes", [])
            + value.get("source_artifact_hashes", [])
            + value.get("raw_shard_hashes", [])
        )
        logs = list(value.get("validation_receipts", []))
        if value.get("primitive_reference"):
            primitive = e.read(value["primitive_reference"])
            references.append(value["primitive_reference"])
            references += primitive.get("sources", [])
            logs += primitive.get("receipts", [])
        references += [
            dict(path=row[stream + "_path"], sha256=row[stream + "_sha256"])
            for row in logs
            for stream in ["stdout", "stderr"]
            if stream + "_path" in row
        ]
        for source in references:
            if not checked(source):
                failures.append(
                    e.gate(
                        Path(str(source.get("path", ref["path"]))),
                        "original_dependency_sha256",
                        source["sha256"],
                        sha256_file(Path(source["path"]))
                        if "path" in source and Path(source["path"]).is_file()
                        else None,
                        f"exp{number}",
                    )
                )
        if not value.get("required_checks_passed"):
            failures.append(
                e.gate(
                    Path(ref["path"]),
                    "required_checks_passed",
                    True,
                    value.get("required_checks_passed"),
                    f"exp{number}",
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            e.gate(Path(ref["path"]), "original_family_receipt", True, str(error), f"exp{number}")
        )
    return dict(
        experiment_id=number,
        qualified=not failures,
        reference=ref,
        references=references,
        gate_check_summary=failures,
        historical_model_provenance=value.get("historical_model_provenance", []),
        reuse_mode="hash_authenticated_original_receipt",
    )


def historical(p: Json) -> Json:
    """Head, exact node union and native inputs remain independent of optional CUDA."""
    rows = []
    for i, n in enumerate([8431, 8529, 8552, 8554, 8558]):
        e.progress(f"family_{n}_before", i, 5 - i)
        rows.append(family(p["original_producers"][str(n)], n))
        e.progress(f"family_{n}_after", i + 1, 4 - i)
    by = {r["experiment_id"]: r for r in rows}
    failures = [g for row in rows for g in row["gate_check_summary"]]
    head_ok = by[8431]["qualified"]
    union_ok = by[8554]["qualified"]
    try:
        head = e.read(p["original_producers"]["8431"])
        ref = dict(path=head["frozen_heads_path"], sha256=head["frozen_heads_sha256"])
        if not checked(ref):
            raise ValueError("original_head_scaler_temperature_hash")
        by[8431]["references"].append(ref)
        receipt = e.read(p["original_producers"]["8554"])
        primitive = e.read(receipt["primitive_reference"])
        from carnot.reporting.historical_learning_8554 import exact_union

        union = exact_union(primitive["nodes"], primitive["receipts"])
        union_ok = bool(
            union_ok
            and len(primitive["nodes"]) == 158
            and union == receipt["exact_union_receipt"]
            and union["passed"]
        )
        if not union_ok:
            failures.append(
                e.gate(
                    Path(receipt["primitive_reference"]["path"]),
                    "exact_158_node_original_union",
                    True,
                    False,
                    "exp8554",
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        head_ok = False
        union_ok = False
        failures.append(
            e.gate(
                Path(p["original_producers"]["8431"]["path"]),
                "original_head_and_node_receipt",
                True,
                str(error),
                "exp8431/exp8554",
            )
        )
    source_ok = by[8529]["qualified"] and by[8552]["qualified"]
    return dict(
        families=rows,
        gate_check_summary=failures,
        learning_inputs_ready_score=int(head_ok and union_ok),
        native_inputs_ready_score=int(head_ok and by[8558]["qualified"]),
        canary_inputs_ready_score=int(head_ok and source_ok),
        references=[r for row in rows for r in row["references"]],
        historical_model_provenance=[
            dict(experiment_id=r["experiment_id"], provenance=r["historical_model_provenance"])
            for r in rows
        ],
    )


def runtime(root: Path, raw: Path | None = None) -> Json:
    """Read only supplied evidence in a supervised child; no model or device is opened."""
    path = root / "ops/cuda-runtime-repair-receipt.json"
    if not path.is_file():
        return dict(
            ready=False,
            gate_check_summary=[
                e.gate(path, "supplied_runtime_repair_receipt", True, None, "exp8410")
            ],
            duration_s=0.0,
        )
    with TemporaryDirectory(prefix="exp8565-runtime-", dir="/var/tmp") as directory:
        output = Path(directory) / "reader.json"
        code = "from pathlib import Path; import sys; from carnot.reporting.v735_contract_methods import runtime; from carnot.reporting.current_work_receipt import atomic_json; atomic_json(Path(sys.argv[2]),runtime(Path(sys.argv[1])))"
        receipt = child(
            "runtime_reader",
            [sys.executable, "-u", "-c", code, str(root), str(output)],
            raw or Path(directory) / "logs",
            deadline=60,
        )
        result = (
            dict(json.loads(output.read_bytes()))
            if receipt["passed"]
            else dict(
                ready=False,
                failures=[
                    e.gate(path, "bounded_exp8410_reader", 0, receipt["actual_exit"], "exp8410")
                ],
            )
        )
    result["reader_receipt"] = receipt
    return result
