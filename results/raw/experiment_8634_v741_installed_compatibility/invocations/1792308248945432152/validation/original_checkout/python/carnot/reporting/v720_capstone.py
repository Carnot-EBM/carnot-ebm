"""REQ-VERIFY-8359: qualify scoped execution before one atomic terminal primary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v717_contract_runner as runner
from carnot.reporting import v717_capstone as retirement
from carnot.reporting import v718_capstone as controls
from carnot.reporting import v720_replay_execution as terminal
from carnot.reporting import v720_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
_preflight = runner.preflight


def manifest(private: Path) -> list[Json]:
    """Freeze new-code coverage and unchanged terminal/retirement consumers."""
    with patch.object(terminal, "e", e):
        plan = list(terminal.manifest(private))
        checks = terminal.preflight(private)
    atomic_json(private / "preconditions.json", checks)
    consumers = next(p for p in plan if p["name"] == "unchanged_consumers")
    consumers["argv"][-1:-1] = [
        "tests/python/test_v718_contract_replay_8318.py::test_finding_policy",
        "tests/python/test_v717_capstone_8317.py::test_append_only_retirement",
    ]
    for spec in plan:
        if spec["name"] in {"private_E2E018", "private_E2E021", "unchanged_consumers"}:
            spec["argv"][-1] = (
                "--basetemp=/tmp/exp8359-" + canonical_hash(str(private))[7:23] + "-" + spec["name"]
            )
    return plan


def preflight(plan: list[Json]) -> list[Json]:
    """Executables must exist before owned children can authenticate evidence."""
    with patch.object(runner, "m", e):
        return list(_preflight(plan))


def publish(value: Json, output: Path, raw: Path) -> None:
    """The existing typed consumer and atomic publisher check the exact candidate."""
    attempts: list[Json] = []

    def validate(path: Path) -> Json:
        with patch.object(terminal, "e", e):
            report = dict(terminal.validate(path, raw / "terminal" / str(len(attempts))))
        attempts.append(report)
        return report

    publication = publish_primary(output, value, validate)
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            attempts=attempts,
            checks=attempts[-1]["checks"],
            normal_process_completion=True,
        ),
    )
    e.progress("published", 1, 0)


def write_note(output: Path) -> None:
    """Every branch has a falsifiable condition; a diagnostic is never a repaired verdict."""
    value = json.loads(output.read_bytes())
    work = e.read(value["work_reference"])
    destination = Path(work["root"]) / "docs/research-notes/v720-outcomes.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V720 outcomes — 2026-10-09",
        "",
        f"Verdict: {value['honest_verdict']}.",
        f"Executed {value['actual_executed_task_count']}; pre-gate {value['pre_gate_count']}; absent {value['missing_output_count']}.",
        "H1 uses all128 intended slots. H2 uses88 later slots, stream96 and retention32 at windows0/32/64/96. Alpha .025 each. Sealed-row diagnostics are recomputed independently; failed audit coverage keeps science unqualified. Missing science is not a null.",
        "Cached observations are exposed development. Both generalization scores remain zero. Spline energy is the sigmoid expression of the same coefficients. No generator weights change.",
        "Capacity, table fidelity, refresh and CPU timings are separate engineering results. Constructed success cannot prove natural decision gain, whole-service acceleration or NFR-01.",
        "V717/V718/V719 failed historical dispositions remain failures. Exact repeated failed scopes retire; unmeasured hypotheses stay open. Legacy retirement entries remain intact.",
        "Current model loads/generation calls:0. Imported Qwen work is historical. A canary cannot enter H1/H2 or reopen parked V713 capture.",
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
            "ARC requires authenticated cross-game support. KV260 supports quadratic Ising k<=5 only. PolarFire remains board-local Linux CPU, without fabric evidence. GateMate requires recovered history, dated physical change, IDCODE0x20000001, n16 flash, then device sample/hash smoke.",
            f"paper_ready={value['paper_ready']}; unmet_gates={value['unmet_gates']}. No external publication or activation.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n")
    with patch.object(retirement, "e", e):
        retirement.reconcile_manifest(Path(work["root"]), output, value)


def main(argv: list[str] | None = None) -> int:
    """Run unconditionally; worker and replay modes cannot launch new science."""
    args = list(sys.argv[1:] if argv is None else argv)
    e.progress("start_no_model_load_current_model_calls_0")
    if "--worker-request" in args:
        return e.worker(
            Path(args[args.index("--worker-request") + 1]),
            Path(args[args.index("--worker-output") + 1]),
        )
    if "--date" in args:
        index = args.index("--date")
        if index + 1 >= len(args) or args[index + 1] != "20261009":
            return 2
        args[index + 1] = "20261008"
    with (
        patch.object(runner, "m", e),
        patch.object(runner, "manifest", manifest),
        patch.object(runner, "preflight", preflight),
        patch.object(runner, "publish", publish),
        patch.object(runner, "controls", controls.controls),
        patch.object(controls, "e", e),
    ):
        result = runner.main(args)
    if result == 0 and "--cold-replay" not in args:
        output = (
            Path(args[args.index("--output") + 1])
            if "--output" in args
            else e.ROOT / "results" / (e.NAME + ".json")
        )
        write_note(output)
    return int(result)
