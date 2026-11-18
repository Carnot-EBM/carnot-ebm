"""REQ-REPORT-9138-V779: reuse original authority without rerunning method development."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import logit_methods_runner_9126 as static_runner
from carnot.reporting import online_methods_runner_9132 as online_runner
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v778_capstone_evidence import check as original_check

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9138_v779_contract", "exp9138-contract"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v779_contract_9138.py"
CONFIG = "openspec/change-proposals/v779-contract-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
DESIGN, ACTIVE = "openspec/change-proposals/research-roadmap-vNEXT.md", "research-roadmap.yaml"
ARCHIVE = "openspec/change-proposals/research-roadmap-v778-preserved-20261116.md"
OWNED = [
    "python/carnot/reporting/v779_contract_evidence.py",
    "python/carnot/reporting/v779_contract.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flush each boundary while the existing subprocess supervisor supplies heartbeats."""
    print(f"[exp9138] phase={phase}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Original strict operand types keep missing evidence distinct from zero."""
    row = dict(original_check(path, field, expected, observed, name))
    producer = re.match(r"experiment_(\d+)_", path.name)
    row["upstream"] = "exp" + producer[1] if producer else TASK
    return row


def design_objects(text: str) -> tuple[list[Json], list[Json]]:
    """Adapt the new visible table while keeping the existing machine-contract parser."""
    machine = parse_design(
        "## Exact task contract\n"
        + "<!-- V779_TASK_CONTRACT_START -->"
        + text.split("<!-- V779_TASK_CONTRACT_START -->", 1)[1],
        milestone="2026.11.779",
    )[1]
    table = []
    for line in (
        text.split("## Task contract and dependency graph", 1)[1]
        .split("```mermaid", 1)[0]
        .splitlines()
    ):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            identity, title = cells[1].split(" — ", 1)
            identity = identity.strip("`")
            table.append(
                dict(
                    order=int(cells[0]),
                    id=identity,
                    title=f"Exp{identity[3:7]}: {title}",
                    phase=int(cells[2]),
                    deliverable=cells[3].strip("`"),
                    gates=cells[4],
                )
            )
    return table, machine


def authority(root: Path) -> Json:
    """Full object equality includes prompts and metadata, beyond visible task titles."""
    tasks, rows, checks = [], [], []
    try:
        queue = yaml.safe_load((root / ACTIVE).read_bytes())
        table, machine = design_objects((root / DESIGN).read_text())
        tasks = queue["tasks"]
        checks = [
            check(
                root / DESIGN,
                "sha256",
                FROZEN["pins"][DESIGN],
                ref(root / DESIGN)["sha256"],
                "design_bytes",
            ),
            check(root / ACTIVE, "milestone", "2026.11.779", queue["milestone"], "queue_milestone"),
            check(
                root / ACTIVE,
                "task_count",
                [14] * 3,
                list(map(len, [tasks, table, machine])),
                "task_count",
            ),
        ]
        for i, task in enumerate(tasks):
            matched = (
                i < len(machine)
                and i < len(table)
                and task in (machine[i], dict(machine[i], prompt=encode(machine[i]["prompt"])))
                and all(table[i][k] == task[k] for k in TABLE_FIELDS)
                and table[i]["order"] == i + 1
                and task["id"].startswith(f"exp{9138 + i}-")
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
    """Pins precede every dependent field; retained original producers resolve historical paths."""
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
    )
    values: Json = {}
    for name, pin in FROZEN["pins"].items():
        path = root / name
        result["checks"].append(
            check(path, "sha256", pin, ref(path)["sha256"] or "missing_file", "immutable_bytes")
        )
        if result["checks"][-1]["passed"]:
            result["source_references"].append(
                dict(ref(path), imported_fields=["original_authority"])
            )
            if name.startswith("results/"):
                values[Path(name).name.split("_")[1]] = json.loads(path.read_bytes())
    if len(values) != 5:
        return result
    source, retired, static, online, capstone = [
        values[str(n)] for n in [9124, 9125, 9126, 9132, 9137]
    ]
    saved_path = Path(capstone["work_reference"]["path"])
    result["checks"].append(
        check(
            saved_path,
            "sha256",
            capstone["work_reference"]["sha256"],
            ref(saved_path)["sha256"] or "missing_file",
            "capstone_work",
        )
    )
    if not result["checks"][-1]["passed"]:
        return result
    saved = json.loads(saved_path.read_bytes())

    def bind(reference: Json) -> Path:
        """Resolve only exact retained bytes; never replace an old hash with checkout bytes."""
        original = reference["path"]
        retained = saved["operands"].get(original, reference)
        path = Path(retained["path"])
        observed = ref(path)["sha256"] or "missing_file"
        result["checks"].append(
            check(path, "sha256", reference["sha256"], observed, "original_producer_binding")
        )
        result["source_references"].append(
            dict(
                path=str(path),
                sha256=reference["sha256"],
                producer_path=original,
                imported_fields=["immutable_original_bytes"],
            )
        )
        return path

    for artifact in [source, retired, static, online]:
        for reference in [
            *artifact["source_references"],
            *artifact["code_config_hashes"],
            *artifact["current_work_receipts"],
        ]:
            if "path" in reference:
                bind(reference)
    bound: Json = {}
    for owner, keys in [
        (
            source,
            [
                "source_manifest_reference",
                "evaluator_label_reference",
                "inference_view_reference",
                "shard_a_roster_reference",
                "shard_b_roster_reference",
            ],
        ),
        (static, ["static_protocol_reference", "work_reference", "raw_sample_roster_reference"]),
        (
            online,
            [
                "online_protocol_reference",
                "work_reference",
                "initial_roster_reference",
                "donor_map_reference",
            ],
        ),
    ]:
        for key in keys:
            path = bind(owner[key])
            bound[(owner["experiment_id"], key)] = path
            if key != "work_reference":
                result["protocol_references"][key] = dict(
                    ref(path), producer_path=owner[key]["path"]
                )
    queue_ref = saved["snapshots"][ACTIVE]
    queue_path = bind(queue_ref)
    authority_ref = saved["operands"][
        str(ROOT / "results/experiment_9097_v776_human_authority.json")
    ]
    human_path = bind(authority_ref)
    if not all(
        c["passed"]
        for c in result["checks"]
        if c["check"]
        not in {"full_contract", "queue_milestone", "task_count", "design_bytes", "authority"}
    ):
        return result
    for artifact, fields in [
        (source, dict(contract_ready_score=1, source_ready_score=1, required_checks_passed=True)),
        (static, dict(methods_ready_score=1, required_checks_passed=True)),
        (online, dict(online_methods_ready_score=1, required_checks_passed=True)),
        (
            retired,
            dict(
                receipt_ready_score=0,
                bounded_scope_retired=True,
                original_quarantine_preserved=True,
            ),
        ),
        (capstone, dict(reconciliation_ready_score=1, required_checks_passed=True)),
    ]:
        path = next(
            root / name
            for name in FROZEN["pins"]
            if name.startswith(f"results/experiment_{artifact['experiment_id']}_")
        )
        for field, expected in fields.items():
            result["checks"].append(
                check(path, field, expected, artifact.get(field, "missing_field"), "upstream_gate")
            )
    table, machine = parse_design((root / ARCHIVE).read_text(), milestone="2026.11.778")
    queue = yaml.safe_load(queue_path.read_bytes())
    old_tasks = source["full_task_objects"]
    old_match = len(old_tasks) == len(queue["tasks"]) == len(machine) == len(table) == 14 and all(
        q in (t, dict(t, prompt=encode(t["prompt"])))
        and t in (m, dict(m, prompt=encode(m["prompt"])))
        and all(t[k] == v[k] for k in TABLE_FIELDS)
        for t, q, m, v in zip(old_tasks, queue["tasks"], machine, table, strict=False)
    )
    result["checks"].append(
        check(root / ARCHIVE, "full_task_objects", True, old_match, "preserved_v778_contract")
    )
    labels = json.loads(bound[(9124, "evaluator_label_reference")].read_bytes())["rows"]
    view = json.loads(bound[(9124, "inference_view_reference")].read_bytes())["rows"]
    human = json.loads(human_path.read_bytes())
    originals = {r["unit_id"]: r for r in human["rows"]}
    qids = sorted({r["qid"] for r in view})
    roles = {
        role: len({r["qid"] for r in labels if r["role"] == role})
        for role in ["TRAIN", "CAL", "TEST"]
    }
    shards = [
        json.loads(bound[(9124, k)].read_bytes())["qids"]
        for k in ["shard_a_roster_reference", "shard_b_roster_reference"]
    ]
    source_ok = (
        len(view) == len(labels) == len(originals) == 900
        and len(qids) == 300
        and roles == dict(TRAIN=150, CAL=50, TEST=100)
        and shards == [qids[:150], qids[150:]]
        and Counter(r["qid"] for r in view) == Counter({q: 3 for q in qids})
        and all(
            set(r) == {"unit_id", "qid", "sibling", "question", "answer_text"}
            and all(
                "sha256:" + hashlib.sha256(r[k].encode()).hexdigest()
                == originals[r["unit_id"]][f"{k if k == 'question' else 'answer'}_sha256"]
                for k in ["question", "answer_text"]
            )
            for r in view
        )
        and all(
            all(
                r[k] == originals[r["unit_id"]][k]
                for k in ["qid", "role", "annotator1", "annotator2"]
            )
            for r in labels
        )
    )
    result["checks"].append(
        check(human_path, "original_human_roles_text_labels", True, source_ok, "source_closure")
    )
    result["source"] = dict(
        answer_count=len(view), qid_count=len(qids), roles=roles, shards=list(map(len, shards))
    )
    result["static_protocol"] = json.loads(bound[(9126, "static_protocol_reference")].read_bytes())
    result["online_protocol"] = json.loads(bound[(9132, "online_protocol_reference")].read_bytes())
    progress("original_method_receipts_before")
    for number, artifact, runner, label in [
        (9126, static, static_runner, "static"),
        (9132, online, online_runner, "online"),
    ]:
        method_work = json.loads(bound[(number, "work_reference")].read_bytes())
        validation = dict(
            checks=artifact["validation_receipts"],
            owned_statement_coverage=artifact["owned_statement_coverage"],
        )
        method_inputs = {r["path"]: bind(r) for r in method_work["scope"]["code_test_hashes"]}
        # A historical command keeps its original identity while reading its retained bytes.
        with patch.object(
            runner.d,
            "ref",
            lambda p: dict(ref(p), sha256=ref(method_inputs.get(str(p), p))["sha256"]),
        ):
            receipt_ok = bool(runner.receipts_pass(method_work, validation))
        primitive_ok = artifact["rows"] == (
            method_work["controls"] if number == 9126 else method_work["result"]["rows"]
        )
        if number == 9126:
            primitive_ok = (
                primitive_ok
                and all(r["passed"] for r in method_work["controls"])
                and artifact["five_arm_specification"] == result["static_protocol"]["arms"]
                and len(result["static_protocol"]["arms"]) == 5
            )
        else:
            primitive_ok = (
                primitive_ok
                and method_work["result"]["exact_resume"]
                and all(r["passed"] for r in method_work["result"]["causal_controls"])
                and artifact["seven_bound_specification"]
                == result["online_protocol"]["benefit_bounds"]
                and len(result["online_protocol"]["arms"]) == 6
                and len(result["online_protocol"]["benefit_bounds"]["bounds"]) == 7
            )
        result["method_receipt_replays"][label] = receipt_ok and primitive_ok
        result["checks"].append(
            check(
                bound[(number, "work_reference")],
                "original_receipts_and_primitives",
                True,
                receipt_ok and primitive_ok,
                "method_receipt_replay",
            )
        )
    progress("original_method_receipts_after")
    gates_ok = all(c["passed"] for c in result["checks"] if c["check"] == "upstream_gate")
    result.update(
        source_ready=bool(source_ok and gates_ok and old_match),
        static_ready=bool(result["method_receipt_replays"]["static"] and gates_ok),
        online_ready=bool(result["method_receipt_replays"]["online"] and gates_ok),
    )
    result["retired_qualification"] = {
        k: retired[k]
        for k in [
            "honest_verdict",
            "receipt_ready_score",
            "bounded_scope_retired",
            "original_quarantine_preserved",
            "intended_count",
            "attempted_count",
            "completed_count",
            "failed_count",
            "censored_count",
            "sample_size_budget",
            "required_command_union",
            "acceptance_gates",
        ]
    }
    result["retired_qualification"]["original_reference"] = ref(
        root / "results/experiment_9125_v778_receipt_qualification.json"
    )
    result["retired_qualification"]["original_quarantine_references"] = [
        r
        for r in result["source_references"]
        if any(
            p in r.get("producer_path", "")
            for p in ["1a8053cdf2fb", "18dfabaaed325", "428d1696eef", "05c1d05bebcb"]
        )
    ]
    result["previous_outcome_rows"] = capstone["task_outcome_rows"]
    archive_path = bind(saved["snapshots"]["research-complete.yaml"])
    end = yaml.safe_load(archive_path.read_bytes())["milestones"][-1]["id"]
    result["archive_lag"] = dict(
        observed_archive_end=end,
        completed_scheduling_frontier="2026.11.778",
        archive_lagged=end != "2026.11.778",
        missing_experiment_evidence=[
            r["experiment_id"]
            for r in capstone["task_outcome_rows"]
            if r["primary_status"] == "missing_file"
        ],
    )
    result["historical_model_provenance"] = capstone["historical_model_provenance"]
    progress("authenticate_inputs_after")
    return result
