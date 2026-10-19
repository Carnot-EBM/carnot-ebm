"""REQ-VERIFY-8345: qualify bounded execution and immutable mixed-history checks."""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v718_capstone as adapter
from carnot.reporting import v718_capstone_evidence as historical
from carnot.reporting import v718_contract_replay as old_contract
from carnot.reporting import v718_replay_history as old_history
from carnot.reporting import v719_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v710_contract_replay import require_reference

Json = dict[str, Any]
_manifest, _preflight = adapter.manifest, adapter.preflight


def manifest(private: Path) -> list[Json]:
    """Freeze new-code coverage and original consumer assertions before measurement."""
    with patch.object(adapter, "e", e):
        plan = list(_manifest(private))
    plan[0]["deadline"] = 600
    plan.insert(
        2,
        dict(
            name="v718_mixed_history",
            argv=[
                str(e.ROOT / ".venv/bin/coverage"),
                "run",
                "--rcfile=" + str(private / "coverage.ini"),
                str(e.ROOT / e.CLI),
                "--historical-check",
                str(private / "mixed_history.json"),
            ],
            expected=0,
            deadline=180,
            scope="owned",
        ),
    )
    return plan


def preflight(plan: list[Json]) -> list[Json]:
    """Check private storage and executable availability without loading a model."""
    with patch.object(adapter, "e", e):
        return list(_preflight(plan))


def historical_check(output: Path) -> int:
    """Run every unchanged mixed-history assertion under byte-bound V718 authority.

    The original test's child isolation is authenticated by its file hash. This
    worker records its own memory, so process isolation cannot conceal growth.
    """
    e.progress("historical_before")
    before = e.memory()
    primary = json.loads((e.ROOT / "results/experiment_8331_v718_capstone.json").read_bytes())
    refs = primary["source_artifact_hashes"][:3]
    for ref in refs:
        require_reference(ref)
    source = e.ROOT / "tests/python/test_v718_capstone_8331.py"
    saved = primary["code_config_hashes"]
    original_hash = next(r["sha256"] for r in saved if r["path"] == str(source))
    result: Json = dict(
        source_sha256=sha256_file(source),
        frozen_source_sha256=original_hash,
        isolated_child="CARNOT_8331_MIXED_CHILD" in source.read_text(),
    )
    os.environ["CARNOT_8331_MIXED_CHILD"] = "1"
    try:
        with ExitStack() as stack:
            for module in [historical, old_contract]:
                for name, ref in zip(["DESIGN", "STAGED", "ACTIVE"], refs, strict=True):
                    stack.enter_context(
                        patch.object(module, name, ref.get("snapshot_path", ref["path"]))
                    )
            old_design = old_history.design
            stack.enter_context(
                patch.object(
                    old_history,
                    "design",
                    lambda root, milestone: (
                        Path(refs[0]["snapshot_path"])
                        if milestone == "2026.10.718"
                        else old_design(root, milestone)
                    ),
                )
            )
            runpy.run_path(str(source))["test_actual_mixed_history"](output.parent / "history_work")
        result["assertions_passed"] = True
    except (AssertionError, OSError, ValueError, KeyError, TypeError) as error:
        result.update(assertions_passed=False, error=repr(error))
    after = e.memory()
    result.update(
        before=before,
        after=after,
        growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
        memory_passed=after["peak_rss_mb"] - before["peak_rss_mb"] <= 500,
    )
    result["passed"] = (
        result["assertions_passed"] and result["memory_passed"] and result["isolated_child"]
    )
    atomic_json(output, result)
    print("memory_receipt=" + json.dumps(result, sort_keys=True), flush=True)
    e.progress("historical_after", 1, 0)
    return int(not result["passed"])


def write_note(output: Path) -> None:
    """Record every disposition and a falsifiable condition without publication claims."""
    value = json.loads(output.read_bytes())
    work = e.read(value["work_reference"])
    destination = Path(work["root"]) / "docs/research-notes/v719-outcomes.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V719 outcomes — 2026-10-09",
        "",
        f"Verdict: {value['honest_verdict']}.",
        f"Executed {value['actual_executed_task_count']}; pre-gate {value['pre_gate_count']}; absent {value['missing_output_count']}.",
        "H1:128 intended. H2:88 later, stream96, retention32 at windows0/32/64/96. Missing sealed science is blocked. Alpha .025 each; exposed development gives zero generalization credit.",
        f"Static head readiness={value['static_ready_score']}; reserved sources with sealed predictions={value['H1']['sealed_predictions']['completed_source_count']}; evaluator access={value['H1']['sealed_predictions']['evaluator_access_count']}. Prediction seals do not supply evaluator observations or scientific controls.",
        "Constructed finite capacity stays outside H1/H2. No natural benefit or regret guarantee follows. Static readiness, durable local correctness and later decision improvement are separate.",
        "Historical V718 failures (coverage, executor branches, +596MB teardown) are execution evidence. gate_check_summary[15].hash drift does not establish an H1/H2 null.",
        "No current model calls; imported Qwen calls remain historical. A canary cannot enter cached H1/H2 or reopen V713 capture.",
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
            "ARC needs overlapping support. KV260 is quadratic Ising k<=5; spline updates and persistence remain unsupported. GateMate needs dated physical change, IDCODE0x20000001, n16 flash and sample/hash smoke. PolarFire remains board-local Linux CPU only.",
            f"paper_ready={value['paper_ready']}; unmet_gates={value['unmet_gates']}. No external publication.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n")
    with patch.object(adapter.legacy, "e", e):
        adapter.legacy.reconcile_manifest(Path(work["root"]), output, value)


def main(argv: list[str] | None = None) -> int:
    """The qualified runner owns checks, recovery and atomic terminal publication."""
    args = list(sys.argv[1:] if argv is None else argv)
    e.progress("start")
    if "--worker-request" in args:
        return e.worker(
            Path(args[args.index("--worker-request") + 1]),
            Path(args[args.index("--worker-output") + 1]),
        )
    if "--historical-check" in args:
        return historical_check(Path(args[args.index("--historical-check") + 1]))
    with (
        patch.object(adapter, "e", e),
        patch.object(adapter, "manifest", manifest),
        patch.object(adapter, "preflight", preflight),
        patch.object(adapter, "write_note", write_note),
    ):
        return int(adapter.main(args))
