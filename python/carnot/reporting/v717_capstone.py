"""REQ-VERIFY-8317: reuse qualified bounded execution and atomic publication."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v717_capstone_evidence as e
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v710_contract_replay import require_reference

Json = dict[str, Any]
_manifest = qualified.manifest


def reconcile_manifest(root: Path, output: Path, value: Json) -> None:
    """Append authenticated exact repeats while preserving every historical manifest byte."""
    path = root / "ops/exclusion_manifest.yaml"
    if not path.is_file():
        return
    original = path.read_text()
    existing = yaml.safe_load(original).get("retired_extras", [])
    ids = {row["id"] for row in existing if "id" in row}
    additions = []
    for row in value["retirements"]:
        for prior, evidence in zip(row["same_verdict_entries"], row["prior_evidence"]):
            identity = (
                "v717_exact_repeat_"
                + canonical_hash([row["task_id"], prior["experiment_id"], prior["verdict"]])[7:23]
            )
            authenticated = evidence["authenticated"] and any(
                e.read(ref).get("task_id") == prior["experiment_id"]
                and e.read(ref).get("honest_verdict") == prior["verdict"]
                for ref in evidence["references"]
            )
            if identity not in ids and authenticated:
                ref = evidence["references"][0]
                require_reference(ref)
                additions.append(
                    dict(
                        id=identity,
                        experiment_scope=row["task_id"] + ": " + row["scope"],
                        reason="Exact repeated terminal verdict: " + prior["verdict"],
                        retired_milestone=e.MILESTONE,
                        retired_by_artifact=str(output),
                        retire_if_same_verdict=True,
                        prior_path=ref["path"],
                        prior_sha256=ref["sha256"],
                        producer_path=str(output),
                        producer_sha256=sha256_file(output),
                        reopening_condition=row["reopening_condition"],
                    )
                )
                ids.add(identity)
    if additions:
        appended = original + "\n" + yaml.safe_dump(additions, sort_keys=False)
        if len(yaml.safe_load(appended)["retired_extras"]) != len(existing) + len(additions):
            raise ValueError("retirement_append_schema")
        path.write_text(appended)


def manifest(private: Path) -> list[Json]:
    """Freeze owned-only coverage and both private authority and audit E2Es."""
    probe = private / "write_probe"
    probe.write_bytes(b"private scratch")
    if probe.read_bytes() != b"private scratch":
        raise ValueError("private_scratch_not_writable")
    with patch.object(qualified, "m", e):
        plan = list(_manifest(private))
    plan[0]["deadline"] = 360
    plan.append(
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        )
    )
    return plan


def write_note(output: Path) -> None:
    """Each disposition needs a falsifiable reopening condition and a bounded claim."""
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    destination = Path(work["root"]) / "docs/research-notes/v717-outcomes.md"
    lines = [
        "# V717 outcomes — 2026-10-08",
        "",
        f"Verdict: {value['honest_verdict']}.",
        f"Fourteen dispositions; executed {value['actual_executed_task_count']}, pre-gate {value['pre_gate_count']}, absent {value['missing_output_count']}.",
        f"Audit readiness {value['capstone_execution_ready_score']}; science readiness {value['science_ready_score']}; local mechanics readiness {value['local_mechanics_ready_score']}.",
        "H1 has 128 intended slots; H2 has 88 later slots, stream96 and retention32 at windows0/32/64/96. Alpha .025 each; intervals are descriptive exposed development only.",
        "Spline energy is a logistic re-expression of the same basis. Local sentence or update decision changes are unmeasured. Both generalization scores remain zero.",
        "V713 source deletion remains parked/unmeasured. A runtime canary is not H1 evidence. Historical Qwen calls count once and grant no current-call credit.",
        "",
        "| Task | Disposition | Next evidence condition |",
        "|---|---|---|",
    ]
    for row, retirement in zip(value["rows"], value["retirements"]):
        lines.append(
            f"| {row['task_id']} | {row['disposition']}; {row['honest_verdict'] or 'producer verdict unavailable'} | {retirement['reopening_condition']} |"
        )
    lines += [
        "",
        *[f"- PRD gap {g['gap']}: {g['next_evidence_condition']}" for g in value["three_prd_gaps"]],
        "",
        "PolarFire graduation preserves board-local Linux CPU dispatch and output parity, not FPGA fabric acceleration. KV260 and GateMate retain unmet board obligations.",
        "CUDA error101 root cause remains unproved. Require authenticated changed substrate before another probe. Scientific hypotheses do not retire for unavailable CUDA.",
        "Permanent recommendations apply only to exact repeated prior_failures scopes. Unchanged GateMate probes and missing-evidence capstone retries retire; physical and science obligations stay open.",
        f"Publication paper_ready={value['paper_ready']}; unmet gates={value['unmet_gates']}. No external publication occurred.",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    reconcile_manifest(Path(work["root"]), output, value)


def main(argv: list[str] | None = None) -> int:
    """The qualified runner owns private children and recovery; this adapter owns scope."""
    args = sys.argv[1:] if argv is None else argv
    with patch.object(qualified, "m", e), patch.object(qualified, "manifest", manifest):
        result = qualified.main(args)
    if result == 0 and "--cold-replay" not in args:
        output = (
            Path(args[args.index("--output") + 1])
            if "--output" in args
            else e.ROOT / "results" / (e.NAME + ".json")
        )
        write_note(output)
    return int(result)
