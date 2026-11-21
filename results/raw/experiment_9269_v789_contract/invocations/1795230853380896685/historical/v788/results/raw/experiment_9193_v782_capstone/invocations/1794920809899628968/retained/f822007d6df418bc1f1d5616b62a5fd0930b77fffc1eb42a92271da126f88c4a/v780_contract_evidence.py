"""REQ-REPORT-9152-V780: smaller execution partitions keep original scientific authority."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v779_contract_evidence as prior
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v749_contract_methods import encode

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_9152_v780_contract", "exp9152-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v780_contract_9152.py"
CONFIG = "openspec/change-proposals/v780-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN, ACTIVE = prior.DESIGN, prior.ACTIVE
ARCHIVE = "openspec/change-proposals/research-roadmap-v779-preserved-20261116.md"
QUEUE = "openspec/change-proposals/research-roadmap-v779-queue-preserved-20261116.yaml"
OWNED = [
    "python/carnot/reporting/v780_contract_evidence.py",
    "python/carnot/reporting/v780_contract.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flushed boundaries and the reused child supervisor expose slow operations."""
    print(f"[exp9152] phase={phase}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Keep strict gate types and identify the producer even for absent operands."""
    row = dict(prior.check(path, field, expected, observed, name))
    if row["upstream"] == prior.TASK:
        row["upstream"] = TASK
    return row


def authority(root: Path) -> Json:
    """Whole-object equality prevents a matching title from hiding a changed prompt."""
    tasks, rows, checks = [], [], []
    try:
        text = (root / DESIGN).read_text()
        queue = yaml.safe_load((root / ACTIVE).read_bytes())
        machine = parse_design(
            "## Exact task contract\n<!-- V780_TASK_CONTRACT_START -->"
            + text.split("<!-- V780_TASK_CONTRACT_START -->", 1)[1],
            milestone="2026.11.780",
        )[1]
        tasks = queue["tasks"]
        table = []
        for line in (
            text.split("## Task contract\n", 1)[1]
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
                "2026.11.780",
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
                and task["id"].startswith(f"exp{9152 + i}-")
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


def inspect(root: Path) -> Json:
    """Revalidate original human custody and qualified receipts, never mutable replacements."""
    progress("authenticate_inputs_before")
    contract = authority(root)
    result: Json = dict(
        contract=contract,
        checks=list(contract["checks"]),
        source_references=[],
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
    )
    values: Json = {}

    def load(path: Path, pin: str, fields: list[str]) -> Json:
        """Decode a producer only after its exact bytes authenticate."""
        observed = ref(path)["sha256"] or "missing_file"
        gate = check(path, "sha256", pin, observed, "immutable_v780_operand")
        result["checks"].append(gate)
        result["source_references"].append(
            dict(path=str(path), producer_path=str(path), sha256=pin, imported_fields=fields)
        )
        return json.loads(path.read_bytes()) if gate["passed"] else {}

    for name, pin in FROZEN["pins"].items():
        if name.startswith("results/"):
            values[Path(name).name.split("_")[1]] = load(
                root / name, pin, ["original_authority", "readiness", "rows"]
            )
        elif name not in [DESIGN, ACTIVE]:
            result["checks"].append(
                check(
                    root / name,
                    "sha256",
                    pin,
                    ref(root / name)["sha256"] or "missing_file",
                    "preserved_v779_bytes",
                )
            )
            result["source_references"].append(
                dict(
                    ref(root / name),
                    producer_path=str(root / name),
                    imported_fields=["preserved_authority"],
                )
            )
    for n, fields in [
        (
            9138,
            dict(
                contract_ready_score=1,
                source_ready_score=1,
                static_methods_ready_score=1,
                online_methods_ready_score=1,
            ),
        ),
        (9139, dict(schedule_ready_score=1)),
        (9140, dict(forward_validation_ready_score=0)),
        (9141, dict(contrast_protocol_ready_score=1)),
        (9151, dict(reconciliation_ready_score=1)),
    ]:
        for field, expected in dict(fields, required_checks_passed=n != 9140).items():
            path = root / next(
                p for p in FROZEN["pins"] if p.startswith(f"results/experiment_{n}_")
            )
            result["checks"].append(
                check(
                    path, field, expected, values[str(n)].get(field, "missing_field"), "v779_gate"
                )
            )
    if not all(r["passed"] for r in result["checks"]):
        result.update(source_ready=False, static_ready=False, online_ready=False)
        return result
    early = result
    frozen = deepcopy(prior.FROZEN)
    frozen["pins"][ARCHIVE] = frozen["pins"].pop(DESIGN)
    original_authority = prior.authority
    with patch.multiple(prior, DESIGN=ARCHIVE, ACTIVE=QUEUE, FROZEN=frozen):
        inherited_contract = original_authority(root)
    with patch.multiple(prior, FROZEN=frozen, authority=lambda _: inherited_contract):
        result = prior.inspect(root)
    result["checks"].extend(early["checks"])
    result["source_references"].extend(early["source_references"])
    result.update(
        {
            k: early[k]
            for k in [
                "contract",
                "execution_rosters",
                "partition_union_equals_original",
                "diagnostic_protocol",
                "retired_schedule",
            ]
        }
    )
    contrast = values["9141"]
    masks = {}
    for key in ["diagnostic_protocol_reference", "evaluator_mask_reference"]:
        reference = contrast[key]
        masks[key] = load(
            Path(reference["path"]), reference["sha256"], ["evaluator_only_original_protocol"]
        )
    if not all(r["passed"] for r in result["checks"]):
        result.update(source_ready=False, static_ready=False, online_ready=False)
        return result
    refs = result["protocol_references"]
    view = json.loads(Path(refs["inference_view_reference"]["path"]).read_bytes())["rows"]
    qids = sorted({r["qid"] for r in view})
    originals = [
        json.loads(Path(refs[k]["path"]).read_bytes())["qids"]
        for k in ["shard_a_roster_reference", "shard_b_roster_reference"]
    ]
    rosters = [
        dict(
            shard=letter,
            lexical_start=i * 100,
            lexical_stop=(i + 1) * 100,
            qids=qids[i * 100 : (i + 1) * 100],
            rows=[r for r in view if r["qid"] in qids[i * 100 : (i + 1) * 100]],
        )
        for i, letter in enumerate("abc")
    ]
    union_ok = (
        sum(len(r["qids"]) for r in rosters) == len(set(qids)) == 300
        and set(qids) == set(originals[0]) | set(originals[1])
        and len({r["unit_id"] for roster in rosters for r in roster["rows"]}) == len(view) == 900
    )
    result["checks"].append(
        check(
            Path(refs["inference_view_reference"]["path"]),
            "disjoint_union_original_population",
            True,
            union_ok,
            "execution_partition",
        )
    )
    result.update(execution_rosters=rosters, partition_union_equals_original=union_ok)
    result["diagnostic_protocol"] = dict(
        inherited_protocol_reference=contrast["diagnostic_protocol_reference"],
        evaluator_mask_reference=contrast["evaluator_mask_reference"],
        inherited=masks["diagnostic_protocol_reference"],
        scale_ranking=dict(
            unit="fixed TEST100 qids/all siblings, every arm, every stream/released-group update",
            timing="before and after each update; rejected updates retain unchanged state",
            raw_logits="s before sigmoid",
            RMS="sqrt(mean(s**2))",
            AUROC="P(s_error>s_correct)+0.5*P(tie); all error/correct pairs",
            one_class_auroc="unavailable",
            Brier="mean((sigmoid(s)-y)**2)",
            counts="error and correct sibling counts; independent support in qids",
            selection="none; descriptive only, no new acceptance bound",
        ),
    )
    forward = values["9140"]
    original_ids = forward["schedule_reference"]["original_node_manifest"]
    all_ids = {r["unit_id"] for r in forward["rows"]}
    new_ids = sorted(all_ids - set(original_ids))
    completed_ids = [r["unit_id"] for r in forward["rows"] if r["status"] == "completed"]
    result["checks"].append(
        check(
            root / "results/experiment_9140_v779_forward_qualification.json",
            "original_and_new_node_scope",
            [621, 53, 674, 23, 0],
            [
                len(original_ids),
                len(new_ids),
                len(all_ids),
                len(completed_ids),
                len(set(completed_ids) & set(original_ids)),
            ],
            "retired_scope",
        )
    )
    result["retired_schedule"] = dict(
        original_nodes=len(original_ids),
        original_node_ids=original_ids,
        v779_additions=len(new_ids),
        new_node_ids=new_ids,
        completed_node_ids=completed_ids,
        completed=forward["completed_count"],
        censored=forward["censored_count"],
        completed_scope="23 newly owned checks only; zero original-node completion credit",
        sum_of_watchdog_ceilings_s=27880.142389,
        original_budget_equation=forward["schedule_reference"]["budget_equation"],
        complete_runtime_s=None,
        unknown_slow_child_cost_s=None,
        original_schedule_reference=ref(
            root / "results/experiment_9139_v779_validation_schedule.json"
        ),
        original_forward_reference=ref(
            root / "results/experiment_9140_v779_forward_qualification.json"
        ),
        original_obligations=deepcopy(forward["rows"]),
        remaining_obligations=deepcopy(forward["remaining_obligations"]),
        retirement_scope="sum-of-ceilings forecast only; original tests, assertions, watchdogs, primaries and quarantines retained",
        new_authority="measured process durations distinct from watchdog ceilings; unknown cost stays unknown",
    )
    result["previous_outcome_rows"] = values["9151"]["task_outcome_rows"]
    result["archive_lag"] = values["9151"]["archive_lag"]
    result["historical_model_provenance"] = values["9151"]["historical_model_provenance"]
    result["source_ready"] = bool(result["source_ready"] and union_ok)
    progress("authenticate_inputs_after")
    return result
