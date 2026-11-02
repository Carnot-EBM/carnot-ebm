"""REQ-VERIFY-8429: private children qualify immutable no-model terminal bytes."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v725_contract_runner as shipped
from carnot.reporting import v726_contract_methods as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
_SHIPPED_RUN, _SHIPPED_MANIFEST, _MEASURE = shipped.run, shipped.manifest, e.measure


def manifest(private: Path) -> list[Json]:
    """Serial coverage and E2E use shipped commands, preserving child database shards."""
    with patch.object(shipped, "e", e):
        return list(_SHIPPED_MANIFEST(private))


def compatibility(value: Json, raw: Path) -> list[Json]:
    """Private null controls expose consumer limitations before any scientific fitting."""
    rows = []
    for name, delta in [("zero", 0.0), ("nonzero", 0.125)]:
        private = raw / "compatibility" / name
        output = private / "results" / "experiment_8432_v726_criterion_decision_audit.json"
        operands: Json = dict(
            baseline_losses=[0.0] * 8 if name == "zero" else [1.0] + [0.0] * 7,
            candidate_losses=[0.0] * 8,
            independent_control_before=[0.2, 0.2],
            independent_control_after=[0.0, 0.0],
            scope="synthetic private decision costs only",
        )
        atomic_json(private / "loss_operands.json", operands)
        assert sum(operands["baseline_losses"]) / 8 - sum(operands["candidate_losses"]) / 8 == delta
        v = deepcopy(value)
        v.update(
            experiment_id=8432,
            task_id="exp8432-criterion-decision-audit",
            honest_verdict="complete_null_private_compatibility",
            verdict_class="null",
            decision_loss_delta=delta,
            methods_ready_score=0,
            contract_ready_score=0,
            current_contract_ready_score=0,
            control_delta=sum(operands["independent_control_before"]) / 2
            - sum(operands["independent_control_after"]) / 2,
            exposure_scope="synthetic private representative control, no scientific observations",
        )
        v["field_principles"].update(
            decision_loss_delta="Honest private null/nonzero control; unchanged validator findings are retained.",
            control_delta="Nonzero negative control keeps a legitimate null distinct from a stub.",
        )
        v["reproducibility_checksum"] = e.checksum(v)
        publication = publish_primary(
            output,
            v,
            lambda _: dict(
                passed=True, scope="private compatibility audit; scientific qualification unclaimed"
            ),
        )
        consumer = reader_receipt(
            v["task_id"],
            output.parent,
            field="required_checks_passed",
            expected=v["required_checks_passed"],
        )
        checks = []
        for label, argv in [
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(output)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(output)]),
        ]:
            checks.append(
                child(
                    name + "_" + label, [sys.executable, "-u", *argv], private / "logs", deadline=60
                )
            )
        reports = json.loads(Path(checks[0]["stdout_path"]).read_bytes())["reports"]
        findings = [flag for report in reports for flag in report.get("flags", [])]
        rows.append(
            dict(
                control=name,
                decision_loss_delta=delta,
                observations_scope="private synthetic compatibility only",
                publication=publication,
                consumer=consumer,
                checks=checks,
                findings=findings,
                unsupported_informational_findings=[
                    f for f in findings if f["severity"] in {0, "info"}
                ],
                changes_to_metrics_validators_or_authority_readiness=False,
                references=[
                    e.reference(output),
                    e.reference(private / "loss_operands.json"),
                    *[
                        e.reference(Path(c[s + "_path"]))
                        for c in checks
                        for s in ["stdout", "stderr"]
                    ],
                ],
            )
        )
        e.progress("compatibility_after_" + name, len(rows), 2 - len(rows))
    return rows


def measure(root: Path, raw: Path) -> Json:
    """Seal compatibility findings inside measurement before the validation reduction."""
    work = _MEASURE(root, raw)
    private_receipt = child(
        "compatibility_input",
        [sys.executable, "-u", "-c", "print('private consumer input only',flush=True)"],
        raw / "compatibility",
    )
    work["compatibility_audit"] = compatibility(
        e.build(work, [private_receipt], raw, raw / (e.NAME + ".json")), raw
    )
    history = (
        e.ROOT
        / "results/raw/experiment_8429_v726_contract_methods/compatibility_history/observed_findings.json"
    )
    work["compatibility_audit"].append(
        dict(
            control="preserved_original_attempt",
            observations_scope="historical private validation finding",
            findings=e.read(e.reference(history)),
            references=[e.reference(history)],
            resolved_by="independent primitive loss operands and distinct negative control; metrics and validators unchanged",
        )
    )
    atomic_json(raw / "measurement.json", work)
    return work


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Adapters keep the existing terminal validators and subprocess behavior unchanged."""
    with patch.object(shipped, "e", e), patch.object(e, "measure", measure):
        result = _SHIPPED_RUN(root, output, private, control=control)
    e.progress("fourteen_slots_terminal", 14, 0)
    return int(result)


def main(argv: list[str] | None = None) -> int:
    """The shipped CLI enforces the date, private paths and process deadline."""
    with (
        patch.object(shipped, "e", e),
        patch.object(shipped, "run", run),
        patch.object(shipped, "manifest", manifest),
    ):
        return int(shipped.main(argv))
