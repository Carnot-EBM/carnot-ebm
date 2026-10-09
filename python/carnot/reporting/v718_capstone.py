"""REQ-VERIFY-8331: reuse qualified children, finding consumption and publication.

The adapter owns scope and final replay. The existing publisher owns atomic
replacement, and historical rows without ids use its shipped compatibility fix.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v717_capstone as legacy
from carnot.reporting import v717_contract_runner as runner
from carnot.reporting import v718_capstone_evidence as e
from carnot.reporting import v718_replay_runner as qualified
from carnot.reporting.v709_execution import child
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
_preflight = runner.preflight
_controls = runner.controls


def controls(value: Json, raw: Path) -> list[Json]:
    """Reject changes to both self accounting and frozen validation dispositions."""
    receipts = list(_controls(value, raw))
    for field in ["self_row", "validation_disposition"]:
        if field == "validation_disposition" and not value["validation_receipts"]:
            continue
        changed = deepcopy(value)
        if field == "self_row":
            changed["rows"][-1]["verdict_class"] = "positive"
        else:
            changed["validation_receipts"][0]["passed"] = not changed["validation_receipts"][0][
                "passed"
            ]
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        path = raw / (field + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + field,
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(path),
                ],
                raw / "cold",
                expected=1,
                deadline=60,
                heartbeat=20,
            )
        )
    return receipts


def preflight(plan: list[Json]) -> list[Json]:
    """Check tools and scratch capacity before aggregation starts, without a model load."""
    failures = list(_preflight(plan))
    free = shutil.disk_usage("/tmp").free
    e.progress("resources_checked")
    if free < 1_000_000_000:
        failures.append(
            e.failure(Path("/tmp"), "private_scratch_capacity_bytes", 1_000_000_000, free)
        )
    return failures


def manifest(private: Path) -> list[Json]:
    """Freeze coverage on this adapter and run both private consumer E2Es."""
    probe = private / "write_probe"
    probe.write_bytes(b"private scratch")
    if probe.read_bytes() != b"private scratch":
        raise ValueError("private_scratch_not_writable")
    with patch.object(runner, "m", e):
        plan = list(legacy._manifest(private))
    plan[0]["deadline"] = 360
    plan[1]["argv"][-1:-1] = [
        "tests/python/test_v718_contract_replay_8318.py::test_finding_policy",
        "tests/python/test_v717_capstone_8317.py::test_append_only_retirement",
    ]
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
    """Give every branch one falsifiable condition without implying continuation."""
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["work_reference"]["snapshot_path"]).read_bytes())
    root = Path(work["root"])
    path = root / "docs/research-notes/v718-outcomes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V718 outcomes — 2026-10-09",
        "",
        f"Verdict: {value['honest_verdict']}.",
        f"Executed {value['actual_executed_task_count']}; pre-gate {value['pre_gate_count']}; absent {value['missing_output_count']}.",
        "Local learning has not earned continuation: H1/H2 remain unmeasured. A qualified static head, sealed natural trajectory and independently replayable audit are the next evidence condition. Missing runtime does not retire a scientific hypothesis.",
        "H1: 128 intended; H2: 88 later, stream96, retention32 at windows0/32/64/96. All arm statistics are unavailable. Exposed cached development gives zero generalization credit.",
        "Constructed capacity invariants and feedback-loss controls remain outside H1/H2; no natural benefit or regret guarantee follows.",
        "V717 original eight producers, two pre-gates and four absences remain historical; exact-zero informational findings, V716 authority failure and reduction_drift are preserved. Current replays do not rewrite historical verdicts.",
        "No current model calls. Imported Qwen observations retain historical provenance. A runtime canary cannot enter H1 or reopen V713 capture.",
        "",
        "| Task | Disposition | Next evidence condition |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {r['task_id']} | {r['disposition']}; {r['honest_verdict'] or 'producer verdict absent'} | {c} |"
        for r, c in zip(value["rows"], value["next_evidence_conditions"], strict=True)
    )
    lines.extend(
        [
            "",
            *[
                f"- PRD gap {g['gap']}: {g['next_evidence_condition']}"
                for g in value["three_prd_gaps"]
            ],
            "",
            "ARC requires overlapping arm support and new authenticated outcomes. KV260 supports quadratic Ising overlays at k<=5; spline updates and persistence remain unsupported. GateMate requires dated physical change before IDCODE0x20000001, n16 flash and sample/hash smoke. PolarFire retains authenticated board-local Linux CPU scope only.",
            f"paper_ready={value['paper_ready']}; unmet_gates={value['unmet_gates']}. No external publication.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    with patch.object(legacy, "e", e):
        legacy.reconcile_manifest(root, output, value)


def main(argv: list[str] | None = None) -> int:
    """Run unconditionally with the requested date and cold-check published bytes."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--date" in args:
        index = args.index("--date")
        if index + 1 >= len(args) or args[index + 1] != "20261009":
            e.progress("date_rejected")
            return 2
        args[index + 1] = "20261008"
    with (
        patch.object(runner, "m", e),
        patch.object(runner, "manifest", manifest),
        patch.object(runner, "preflight", preflight),
        patch.object(runner, "controls", controls),
        patch.object(runner, "publish", qualified.publish),
        patch.object(qualified, "e", e),
    ):
        result = runner.main(args)
    if result == 0 and "--cold-replay" not in args:
        output = (
            Path(args[args.index("--output") + 1])
            if "--output" in args
            else e.ROOT / "results" / (e.NAME + ".json")
        )
        receipt = child(
            "published_cold_replay",
            [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--cold-replay",
                str(output),
            ],
            output.parent / "raw" / output.stem / "post_publication",
            deadline=60,
            heartbeat=20,
        )
        if not receipt["passed"]:
            raise ValueError("published_reduction_drift")
        write_note(output)
    return int(result)
