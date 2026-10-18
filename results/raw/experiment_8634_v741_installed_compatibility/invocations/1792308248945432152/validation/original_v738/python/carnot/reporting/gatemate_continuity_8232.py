"""REQ-VERIFY-8232: preserve a physical blocker without retrying unchanged hardware.

The existing receipt parser is used only in dry-run mode. A accepted receipt
records a physical change; it cannot prove that JTAG, flash or device execution
works. Historical byte copies keep those distinctions independently readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from scripts.experiments import experiment_7146_v627_gatemate_changed_state as legacy

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8232_v711_gatemate_continuity"
CLI = "scripts/experiments/" + NAME + ".py"
UPSTREAM = "results/experiment_8216_v709_hardware_workload_obligations.json"
PIN = "sha256:5b31ca2cca90b76d47f7c1b9575264e352a3990ba303992c8083f97c0fff823f"
HISTORY_PIN = "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66"
REOPEN = "docs/research-notes/v711-gatemate-reopen.md"
CONFIG: Json = dict(seed=7118232, cutoff_date="20260823", run_date="20261007")
DOCS = [
    "ops/exclusion_manifest.yaml",
    "ops/hardware-bringup-prep.md",
    "docs/jtag-wiring-gatemate-dirtyjtag.md",
    "research-hardware-wishlist.md",
    "ops/known-issues.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    REOPEN,
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so an unchanged board never causes silent waiting."""
    print(f"[exp8232] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(path: Path, raw: Path, data: Json) -> Json:
    """Keep immutable input bytes while retaining the operator's original location."""
    ref = copy_bytes(path, raw)
    frozen = dict(ref, path=ref["frozen_path"], original_path=ref["path"])
    data["references"].append(frozen)
    return frozen


def load(root: Path, raw: Path) -> Json:
    """Authenticate one named board lineage and scan only established receipt files.

    Missing documents or historical bytes block their operands. The audit still
    retains one GateMate obligation, rather than silently replacing it with a
    sibling board or an invented failed device measurement.
    """
    data: Json = dict(
        board={},
        history_ready=False,
        checks=[],
        references=[],
        cited=[],
        receipt_rows=[],
        physical_change={},
        fixture=False,
    )
    path = root / UPSTREAM
    progress("authenticate_before", 0, 1)
    try:
        source = freeze(path, raw, data)
        value = json.loads(Path(source["path"]).read_bytes())
        for field, expected, observed in [
            ("sha256", PIN, source["sha256"]),
            ("experiment_id", 8216, value["experiment_id"]),
            ("required_checks_passed", True, value["required_checks_passed"]),
            ("flagged_adversarial", False, value["flagged_adversarial"]),
            ("fixture_mode", False, bool(value.get("fixture_mode"))),
        ]:
            data["checks"].append(operand(field, path, expected, observed))
        side = path.parent / "raw" / path.stem / "validators" / (PIN.split(":")[1] + ".json")
        bound = read_bound_sidecar(path, side)
        freeze(side, raw, data)
        data["checks"].append(operand("report.passed", side, True, bound["report"]["passed"]))
        boards = [b for b in value["board_rows"] if b["board"] == "GateMate"]
        if len(boards) != 1:
            raise ValueError("gatemate_row_count")
        data["board"] = boards[0]
        data["checks"].append(
            operand("blocked_idcode", path, "0xffffffff", boards[0]["blocked_idcode"])
        )
        for field, pin in [
            ("source_path", "source_hash"),
            ("source_transcript", "source_transcript_sha256"),
        ]:
            history = root / boards[0][field]
            observed = sha256_file(history) if history.is_file() else None
            data["checks"].append(operand(field + ".sha256", history, boards[0][pin], observed))
            data["checks"].append(operand("historical_lineage_pin", history, HISTORY_PIN, observed))
            if history.is_file():
                freeze(history, raw, data)
        data["cited"].append(
            dict(
                experiment_id=8216,
                path=str(path),
                sha256=source["sha256"],
                fields_imported=["board_rows[GateMate]", "required_checks_passed"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["checks"].append(
            operand("gatemate_history", path, "authenticated board lineage", str(error))
        )
    data["history_ready"] = all(c["passed"] for c in data["checks"])
    for i, relative in enumerate(DOCS):
        document = root / relative
        data["checks"].append(operand("named_input.exists", document, True, document.is_file()))
        if document.is_file():
            freeze(document, raw, data)
        progress("authenticate_document", i + 1, len(DOCS) - i - 1)
    followup = root / "ops/operator-followup.md"
    if followup.is_file():
        freeze(followup, raw, data)
    data["receipt_rows"], data["physical_change"] = legacy.audit_receipts(
        root,
        cutoff_date=CONFIG["cutoff_date"],
        run_date=CONFIG["run_date"],
        dry_run=True,
    )
    change = data["physical_change"]
    data["checks"].append(
        operand("physical_change_evidence.exists", followup, True, change["exists"])
    )
    data["receipt_search_paths"] = [str(root / p) for p in legacy.RECEIPT_PATHS]
    progress("authenticate_after", 1, 0)
    return data


def reduce(data: Json) -> Json:
    """Close owned documentation work while leaving device acceptance pending.

    The excluded row is the board obligation, not a synthetic failed probe.
    Its numerator records whether a change receipt exists; missing history keeps
    that measurement unavailable. Audit readiness earns no hardware benefit.
    """
    historical = bool(data["history_ready"])
    change = data["physical_change"]
    changed = bool(change.get("exists"))
    documents = all(
        c["passed"] for c in data["checks"] if c["artifact_field"] == "named_input.exists"
    )
    ready = int(historical and documents)
    operand_name = (
        "gatemate_history"
        if not historical
        else "named_input"
        if not documents
        else "gatemate_device_preflight"
        if changed
        else "gatemate_physical_change"
    )
    row = dict(
        data["board"],
        board="GateMate",
        unit_id="gatemate_obligation",
        source_cluster_id="gatemate_physical_setup",
        arm="documentation_audit",
        condition="changed_receipt" if changed else "unchanged_or_missing_receipt",
        status="excluded",
        metric="physical_change_observed",
        numerator=int(changed) if historical else None,
        denominator=1,
        missing_status=None if historical else "historical_evidence_unavailable",
        exclusion_reason=operand_name,
        blocked_idcode="0xffffffff" if historical else None,
        current_hardware_execution=False,
        current_reachability="not_probed",
        started=False,
        completed=False,
        failed=False,
        censored=False,
        excluded=True,
        terminal_criterion_met=False,
        actual_substrate="none",
        physical_change_recorded=changed,
        execution_ready_score=0,
        flash_success=False,
        on_device_success=False,
        device_hash_parity=None,
        required_next_evidence=[
            "dated operator physical-change receipt",
            "GM1Ax IDCODE 0x20000001",
            "authenticated n16 bitstream flash",
            "device tile/hash smoke parity",
        ],
        reopen_contract_path=REOPEN,
    )
    return dict(
        honest_verdict="complete_blocked_" + operand_name,
        verdict_class="blocked",
        gatemate_obligation_ready_score=ready,
        gatemate_obligation=row,
        physical_change_evidence=change,
        physical_change_receipt_rows=data["receipt_rows"],
        reopen_contract_path=REOPEN,
        rows=[row],
        board_rows=[row],
        intended_count=1,
        completed_count=0,
        failed_count=0,
        censored_count=0,
        excluded_count=1,
        independent_count=0,
        completed_audit_count=1,
        current_device_execution_count=0,
        current_jtag_retry_count=0,
        execution_ready_score=0,
        scientific_benefit_score=0,
        verifier_is_oracle=False,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            authenticated_history=historical,
            documented_reopen=documents,
            physical_change_recorded=changed,
            idcode_preflight=False,
            n16_flash=False,
            n16_device_hash_smoke=False,
        ),
    )
