"""REQ-REPORT-9166-V781: bind immutable protocols before collecting any predictions."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from carnot.reporting import v780_contract_evidence as prior
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.v749_contract_methods import encode

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_9166_v781_contract", "exp9166-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v781_contract_9166.py"
CONFIG = "openspec/change-proposals/v781-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN, ACTIVE = prior.DESIGN, prior.ACTIVE
ARCHIVE = "openspec/change-proposals/research-roadmap-v780-preserved-20261116.md"
QUEUE = "openspec/change-proposals/research-roadmap-v780-queue-preserved-20261116.yaml"
OWNED = [
    "python/carnot/reporting/v781_contract_evidence.py",
    "python/carnot/reporting/v781_contract.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flush boundaries so the shared child supervisor can expose slow operations."""
    print(f"[exp9166] phase={phase}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Retain strict operand types so missing evidence cannot masquerade as zero."""
    row = dict(prior.check(path, field, expected, observed, name))
    if row["upstream"] == prior.TASK:
        row["upstream"] = TASK
    return row


def authority(root: Path) -> Json:
    """Compare whole ordered objects; a matching title cannot hide changed instructions."""
    tasks, rows, checks = [], [], []
    try:
        text = (root / DESIGN).read_text()
        queue = yaml.safe_load((root / ACTIVE).read_bytes())
        machine = json.loads(
            text.split("## Full machine-readable contract", 1)[1]
            .split("```json", 1)[1]
            .split("```", 1)[0]
        )
        tasks = queue["tasks"]
        table = []
        for line in (
            text.split("## Task contract", 1)[1]
            .split("## Full machine-readable", 1)[0]
            .splitlines()
        ):
            cells = [s.strip().strip("`") for s in line.strip().strip("|").split("|")]
            if len(cells) == 5 and cells[0].isdigit():
                table.append(cells)
        for name in [DESIGN, ACTIVE]:
            checks.append(
                check(
                    root / name,
                    "sha256",
                    FROZEN["pins"][name],
                    ref(root / name)["sha256"],
                    "contract_bytes",
                )
            )
        checks.append(
            check(
                root / ACTIVE,
                "milestone",
                "2026.11.781",
                queue.get("milestone", "missing_field"),
                "queue_milestone",
            )
        )
        checks.append(
            check(
                root / ACTIVE,
                "task_count",
                [14] * 3,
                list(map(len, [tasks, machine, table])),
                "task_count",
            )
        )
        for i, task in enumerate(tasks):
            gates = (
                "; ".join(
                    f"{g['upstream']}.{g['artifact_field']} {g['op']} {g['value']}"
                    for g in task.get("gated_on", [])
                )
                or "Unconditional"
            )
            matched = (
                i < len(machine)
                and i < len(table)
                and task in (machine[i], dict(machine[i], prompt=encode(machine[i]["prompt"])))
                and table[i]
                == [str(i + 1), task["id"], str(task["phase"]), task["deliverable"], gates]
                and task["id"].startswith(f"exp{9166 + i}-")
            )
            rows.append(
                dict(
                    unit_id=task["id"],
                    qid=None,
                    group_id=task["id"],
                    sibling=None,
                    arm="full_task_contract",
                    condition="exact_authority",
                    seed=FROZEN["seed"],
                    metric="ordered_full_object_match",
                    absolute_metric=int(matched),
                    numerator=int(matched),
                    denominator=1,
                    status="completed" if matched else "failed",
                    exclusion_reason=None,
                    censor_reason=None,
                )
            )
        checks.append(
            check(
                root / ACTIVE,
                "all_complete_objects",
                True,
                len(rows) == 14 and all(r["numerator"] for r in rows),
                "full_contract",
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        checks.append(
            check(
                root / ACTIVE,
                "full_task_objects",
                "14 exact ordered objects",
                "missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field"
                if isinstance(error, KeyError)
                else str(error),
                "authority",
            )
        )
    return dict(ready=all(c["passed"] for c in checks), tasks=tasks, rows=rows, checks=checks)


def brier_moments(probabilities: list[float], labels: list[int]) -> Json:
    """Check the population identity directly; zero variance needs no invented correlation."""
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=float)
    if (
        p.shape != y.shape
        or p.ndim != 1
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or np.any((p < 0) | (p > 1))
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("moment_support")
    if not len(p):
        return dict(status="unavailable", reason="empty_support")
    sp, sy = float(p.std()), float(y.std())
    covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
    bias = float((p.mean() - y.mean()) ** 2)
    variance = (sp - sy) ** 2
    deficit = 2 * (sp * sy - covariance)
    direct = float(np.mean((p - y) ** 2))
    return dict(
        status="completed",
        brier=direct,
        squared_mean_bias=bias,
        variance_mismatch=variance,
        covariance_deficit=deficit,
        covariance=covariance,
        correlation=covariance / (sp * sy) if sp * sy else None,
        signed_residual=direct - bias - variance - deficit,
    )


def inspect(root: Path) -> Json:
    """Authenticate original producers first, then reuse their cohort and qualified methods."""
    progress("authenticate_inputs_before")
    contract = authority(root)
    checks, references, values = list(contract["checks"]), [], {}
    for name, pin in FROZEN["pins"].items():
        path = root / name
        gate = check(
            path, "sha256", pin, ref(path)["sha256"] or "missing_file", "immutable_v781_operand"
        )
        checks.append(gate)
        references.append(
            dict(
                path=str(path),
                producer_path=str(path),
                sha256=pin,
                imported_fields=["original_authority", "readiness", "rows"],
            )
        )
        if gate["passed"] and name.startswith("results/"):
            values[Path(name).name.split("_")[1]] = json.loads(path.read_bytes())
    empty = dict(
        contract=contract,
        checks=checks,
        source_references=references,
        source_ready=False,
        static_ready=False,
        online_ready=False,
        protocol_references={},
        source={},
        static_protocol={},
        online_protocol={},
        previous_outcome_rows=[],
        archive_lag={},
        retired_qualification={},
        method_receipt_replays={},
        execution_rosters=[],
        partition_union_equals_original=False,
        diagnostic_protocol={},
        retired_schedule={},
        validation_obligations={},
    )
    for number, fields in [
        (
            9152,
            dict(
                contract_ready_score=1,
                source_ready_score=1,
                static_methods_ready_score=1,
                online_methods_ready_score=1,
            ),
        ),
        (9153, dict(schedule_ready_score=0, completed_count=32, censored_count=642)),
        (9165, dict(reconciliation_ready_score=1)),
    ]:
        name = next(p for p in FROZEN["pins"] if p.startswith(f"results/experiment_{number}_"))
        for field, expected in dict(fields, required_checks_passed=True).items():
            checks.append(
                check(
                    root / name,
                    field,
                    expected,
                    values.get(str(number), {}).get(
                        field, "missing_field" if str(number) in values else "missing_file"
                    ),
                    "v780_gate",
                )
            )
    for reference in values.get("9152", {}).get("protocol_references", {}).values():
        path = Path(reference["path"])
        checks.append(
            check(
                path,
                "sha256",
                reference["sha256"],
                ref(path)["sha256"] or "missing_file",
                "exp9152_original_protocol_binding",
            )
        )
        references.append(
            dict(
                reference,
                producer_path=reference.get("producer_path", reference["path"]),
                imported_fields=["original_protocol"],
            )
        )
    affected = [
        r
        for number in ["9152", "9153", "9165"]
        for r in values.get(number, {}).get("code_config_hashes", [])
        if Path(r["path"]).name.startswith("test_")
    ] + [ref(ROOT / TEST)]
    for reference in affected:
        path = Path(reference["path"])
        checks.append(
            check(
                path,
                "sha256",
                reference["sha256"],
                ref(path)["sha256"] or "missing_file",
                "affected_assertion_bytes",
            )
        )
    if not all(r["passed"] for r in checks):
        return empty
    frozen = deepcopy(prior.FROZEN)
    frozen["pins"][ARCHIVE] = frozen["pins"].pop(prior.DESIGN)
    frozen["pins"][QUEUE] = frozen["pins"].pop(prior.ACTIVE)
    older = deepcopy(prior.prior.FROZEN)
    older["pins"][ARCHIVE] = older["pins"].pop(prior.DESIGN)
    with (
        patch.multiple(prior, DESIGN=ARCHIVE, ACTIVE=QUEUE, FROZEN=frozen),
        patch.object(prior.prior, "FROZEN", older),
    ):
        result = prior.inspect(root)
    result["checks"].extend(checks)
    result["source_references"].extend(references)
    result["contract"] = contract
    result["previous_outcome_rows"] = values["9165"]["task_outcome_rows"]
    result["archive_lag"] = values["9165"]["archive_lag"]
    result["historical_model_provenance"] = values["9165"]["historical_model_provenance"]
    schedule = values["9153"]
    plan = schedule["schedule_reference"]
    assertions = (
        plan["original_assertion_hashes"]
        + schedule["current_revision_reference"]["v779_addition_assertion_hashes"]
    )
    node_bindings = [
        dict(
            unit_id=node,
            assertion_reference=next(
                r for r in assertions if r["path"].endswith(node.split("::")[0])
            ),
            current_qualification_credit=0,
        )
        for node in plan["original_node_manifest"]
    ]
    result["validation_obligations"] = dict(
        original_node_ids=plan["historical_original_node_manifest"],
        v779_addition_ids=plan["v779_additions"],
        assertion_references=plan["original_assertion_hashes"],
        v779_addition_assertion_references=schedule["current_revision_reference"][
            "v779_addition_assertion_hashes"
        ],
        historical_rows=deepcopy(schedule["rows"]),
        historical_counts=dict(completed=32, censored=642),
        current_qualification_credit=0,
        node_bindings=node_bindings,
        affected_v780_node_assertions=values["9165"]["validation_scope_manifest"][
            "original_test_node_hashes"
        ],
        affected_v780_references=values["9152"]["code_config_hashes"]
        + values["9165"]["code_config_hashes"],
        affected_current_paths=[*OWNED, TEST, CONFIG],
        affected_node_assertions=list(
            {
                "tests/python/" + Path(reference["path"]).name + "::" + node.name: dict(
                    node_id="tests/python/" + Path(reference["path"]).name + "::" + node.name,
                    sha256=canonical_hash(ast.dump(node, include_attributes=False)),
                    producer_reference=reference,
                    current_qualification_credit=0,
                )
                for reference in affected
                for node in ast.parse(Path(reference["path"]).read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            }.values()
        ),
        current_assertion_reference=ref(ROOT / TEST),
        elapsed_and_watchdog_distinct=True,
        complete_current_execution_ready=False,
    )
    result["diagnostic_protocol"]["brier_moments"] = dict(
        weighting="equal qid; three original siblings per qid",
        moments="population",
        formula="Brier=(mean(p)-mean(y))**2+(sd(p)-sd(y))**2+2*(sd(p)*sd(y)-cov(p,y))",
        components=["squared_mean_bias", "variance_mismatch", "covariance_deficit"],
        identity_tolerance=1e-12,
        residual="direct Brier minus components; signed, unclamped",
        zero_variance="direct covariance; correlation unavailable",
        empty_support="unavailable",
        supports=[
            "static TEST100 all siblings",
            "future groups2-5 in ten streams/40 qids/120 siblings",
            "each retention TEST100 panel separately",
        ],
        arms="all five static and all six online; identical supports",
        resampling="original paired qid or whole-stream units; retention remains100 independent qids",
        selection="none",
        primary_gate_rescue=False,
    )
    progress("authenticate_inputs_after")
    return dict(result)
