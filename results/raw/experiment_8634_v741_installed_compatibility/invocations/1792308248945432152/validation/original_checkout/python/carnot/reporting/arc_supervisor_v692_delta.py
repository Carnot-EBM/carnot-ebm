"""Audit new outcomes while preserving failed history. Spec: REQ-REPORT-7988.

The eligible predecessor supplies the content frontier. A repaired historical
check does not change the failed primary or create new live evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import yaml

from carnot.reporting import arc_supervisor_v690_delta as runner
from carnot.reporting.current_work_receipt import sha256_file

ROOT = runner.ROOT
PRIOR = ROOT / "results/experiment_7962_v690_arc_supervisor_delta.json"
INVENTORY = ROOT / "results/raw/experiment_7962_v690_arc_supervisor_delta/receipt_inventory.json"
HISTORY = ROOT / "results/experiment_7975_v691_arc_supervisor_delta.json"
HISTORY_INVENTORY = (
    ROOT / "results/raw/experiment_7975_v691_arc_supervisor_delta/receipt_inventory.json"
)
REGISTRY = runner.REGISTRY
OUTPUT = ROOT / "results/experiment_7988_v692_arc_supervisor_delta.json"
PINNED = {
    str(PRIOR): "sha256:4999f7c1398c5998d8194aa6315767654e77f02e0aba138d4e826e07e3b943f6",
    str(INVENTORY): "sha256:9eed81ea6f70d410198013c58e26d3cc1e30cdc5072e550d3ba86e41264d46c1",
    str(HISTORY): "sha256:a28af99cf2de2a4ba93348d7c2fb3ce18879a46a468fb02961f407180fc48a4f",
    str(
        HISTORY_INVENTORY
    ): "sha256:66e6fcf9b58f85a9e4eac69345aca9de8c621adde9a39787239f5619687a8b59",
    str(REGISTRY): "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
}
EXPERIMENT_ID = 7988
PRIOR_ID = 7962
MILESTONE = "2026.10.692"
RUN_DATE = "20261001"
MODULE = "python/carnot/reporting/arc_supervisor_v692_delta.py"
CLI = "scripts/experiments/experiment_7988_v692_arc_supervisor_delta.py"
TEST = "tests/python/test_arc_supervisor_delta_7988.py"
ADDED = [MODULE, CLI, runner.MODULE, runner.CLI]
INCLUDE = ",".join("*/" + path for path in ADDED)
CONSUMERS = ["tests/python/test_arc_supervisor_delta_7975.py", runner.TEST, *runner.CONSUMERS]
COVERAGE_TESTS = ["tests/python/test_arc_supervisor_delta_7975.py", runner.TEST]
APPLICABLE_E2E = ["E2E-017"]
TERMINAL_CHECKS = [
    dict(
        script=CLI,
        flag="--cold-replay",
        cwd="private_scratch",
        unset_environment=["PYTHONPATH"],
        expected_exit=0,
        deadline_s=60,
    )
]
previous = runner.previous


def inputs() -> dict[str, Any]:
    """Authenticate fallback authority and keep original failures as history."""
    checked = runner.inputs(sys.modules[__name__])
    for path in (HISTORY, HISTORY_INVENTORY):
        actual = sha256_file(path) if path.is_file() else "missing"
        checked["checks"].append(
            previous.operand(
                path, "sha256", PINNED[str(path)], actual, actual if path.is_file() else None
            )
        )
    history = (
        json.loads(HISTORY.read_text())
        if HISTORY.is_file() and sha256_file(HISTORY) == PINNED[str(HISTORY)]
        else {}
    )
    for path, document, fields in (
        (
            PRIOR,
            checked["prior"],
            {
                "experiment_id": PRIOR_ID,
                "task_id": "exp7962-arc-supervisor-delta",
                "run_date": RUN_DATE,
                "honest_verdict": "complete_null_no_new_supervisor_outcomes",
            },
        ),
        (
            HISTORY,
            history,
            {
                "experiment_id": 7975,
                "task_id": "exp7975-arc-supervisor-delta",
                "run_date": RUN_DATE,
                "execution_date": RUN_DATE,
                "verdict_class": "disqualified",
                "arc_evidence_ready_score": 0,
            },
        ),
    ):
        checked["checks"].extend(
            previous.operand(
                path, field, expected, document.get(field, "missing"), PINNED[str(path)]
            )
            for field, expected in fields.items()
        )
    exclusion = ROOT / "ops/exclusion_manifest.yaml"
    document = yaml.safe_load(exclusion.read_text()) if exclusion.is_file() else {}
    retired = any(
        row.get("experiment_id") in {PRIOR_ID, EXPERIMENT_ID}
        or any(
            str(identity).split("-")[0] in {"exp7962", "exp7988"}
            for identity in row.get("experiment_ids", [])
        )
        for entries in document.values()
        if isinstance(entries, list)
        for row in entries
        if isinstance(row, dict)
    )
    checked["checks"].append(
        previous.operand(
            exclusion,
            "retired",
            False,
            retired if exclusion.is_file() else "missing",
            sha256_file(exclusion) if exclusion.is_file() else None,
        )
    )
    checked["prior"].setdefault("historical_required_failures", []).append(
        dict(
            upstream_id="exp7975-arc-supervisor-delta",
            path=str(HISTORY),
            sha256=PINNED[str(HISTORY)],
            verdict_class=history.get("verdict_class"),
            arc_evidence_ready_score=history.get("arc_evidence_ready_score"),
            gate_check_summary=history.get("gate_check_summary", []),
        )
    )
    checked["additional_source_hashes"] = {
        str(p): PINNED[str(p)] for p in (HISTORY, HISTORY_INVENTORY)
    }
    checked["failures"] = [r for r in checked["checks"] if r["expected"] != r["observed"]]
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze the reused private routes and run only the applicable end-to-end check."""
    return [
        s
        for s in runner.commands(private, sys.modules[__name__])
        if not s["name"].startswith("e2e_016")
    ]


def artifact_fields(value: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    """Retain receipt-level stagnations once and make empty evidence explicit."""
    games: dict[str, dict[str, Any]] = {}
    seen: set[tuple[Any, ...]] = set()
    for row in value["new_event_rows"]:
        cell = games.setdefault(str(row["game"]), {}).setdefault(
            row["arm"], dict(fired=0, helped=0, actions_to_levelup=[], stagnations_unredirected=0)
        )
        cell["fired"] += 1
        cell["helped"] += int(row["resolved_by_levelup"] is True)
        cell["actions_to_levelup"].append(row["actions_to_levelup"])
        identity = (row["game"], row["arm"], row["receipt_id"], row["invocation_id"], row["seed"])
        if identity not in seen:
            cell["stagnations_unredirected"] += row.get("stagnations_unredirected") or 0
            seen.add(identity)
    return dict(
        frontier=dict(
            upstream_id="exp7962-arc-supervisor-delta",
            path=str(INVENTORY),
            sha256=PINNED[str(INVENTORY)],
            role="last_eligible_content_frontier",
            producer_invocation_date=RUN_DATE,
            preserved_disqualified_upstream="exp7975-arc-supervisor-delta",
            receipt_inventory=value.get("receipt_inventory", {}),
        ),
        per_game_arm_statistics=games,
        empty_ledger_control=previous.reduce([]),
        sample_interpretation="directional_observational"
        if len(value["new_event_rows"]) >= 8 and len(games) >= 3
        else "descriptive",
        code_config_hashes=value.get("source_artifact_hashes", {}),
        raw_shard_hashes=value.get("scan_source_hashes", {}),
        defaults_changed=False,
        causal_benefit_claimed=False,
        historical_input_receipts=[
            dict(
                path=str(p),
                sha256=PINNED[str(p)],
                role="preserved_disqualified_history",
                producer_invocation_date=RUN_DATE,
                verdict_class="disqualified",
                arc_evidence_ready_score=0,
            )
            for p in (HISTORY, HISTORY_INVENTORY)
        ],
        future_controlled_generalization_evidence=[
            "Frozen curated arm IDs, source hashes and unchanged model weights",
            "Randomized arm and control assignment before outcomes, with matched action/time budgets",
            "At least eight authenticated firings across three distinct held-out hidden games",
            "Exact run, receipt, seed and invocation IDs with own-attempt live entrypoint provenance",
            "Content-hashed fired/helped/action-count, stagnation, failure and censoring outcome receipts",
            "Preregistered paired game-level effect and uncertainty; empty ledger control stays at zero",
        ],
    )


def replay(value: dict[str, Any]) -> list[str]:
    """Reject invented supplemental counts while accepting older private fixtures."""
    expected = artifact_fields(value, {})
    return runner.replay(value) + [
        key
        for key in ("per_game_arm_statistics", "empty_ledger_control", "sample_interpretation")
        if key in value and value[key] != expected[key]
    ]


def execute(output: Path, private: Path) -> int:
    """Use the qualified publication path with only the current declared scope."""
    return runner.execute(output, private, sys.modules[__name__])


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Cold replay from private scratch tests the exact bytes without PYTHONPATH."""
    report = previous.terminal(candidate, private, durable)
    launcher = "import os, runpy, sys; os.chdir(sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    row = previous.run(
        dict(
            name="cold_primary_replay",
            argv=[
                "env",
                "-u",
                "PYTHONPATH",
                str(ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                launcher,
                str(private),
                str(ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ],
            deadline_s=60,
            expected_exit=0,
            expected_text=None,
            classification="required",
        ),
        private,
        durable,
    )
    report["reports"].append(row)
    report["passed"] = report["passed"] and row["passed"]
    return report
