"""REQ-REPORT-9095-V775: immutable evidence keeps blocked science distinct from accounting."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import service_cost_9093 as costs
from carnot.reporting import susceptibility_9094 as susceptibility
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v768_capstone import edge as edge

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
TASK_PIN = "sha256:d58d1e9fd18c20e320da27687eec491027773fc6f40ab085b3a03f30b89b5790"
PINS = {
    "research-roadmap.yaml": "sha256:10caaae6c06ef27417685d999ca8ff6293e275409eb08a75746b32d99502df37",
    DESIGN: "sha256:ce767f80004cb4b9aef48f4b49c46ce024f06e8bea251a4cf9aa4d33c631ab1b",
    "ops/exclusion_manifest.yaml": "sha256:081d0cb3f30a97c2afae61715d9c84de5254acae75eef050e31a3a221bdfc8a0",
    "research-complete.yaml": "sha256:366e30ae4351c694ead5145f4d56d0d04b2d95ce6980ef6c2205a9919a10f199",
    "ops/conductor-log.md": "sha256:a50fe4531d5e9092b990a365cff65f0cb1fe261212ecc6c90b35c15f65b56aba",
    "results/experiment_9081_v774_capstone.json": "sha256:f6097c48c395164b77ef62a87f3274e492badc99bc9fd1fde0f0f277ae37f704",
    "results/experiment_9082_v775_contract.json": "sha256:d23c560c4553a7ee755d8bf5d76c71aa5286465b44d8b2587a02ff72d8284182",
    "results/experiment_9083_v775_prospective_custody.json": "sha256:ee870a237979ed37a6ff2b1b5adae4a2021a463659c62bc3df712e5997a62232",
    "results/experiment_9084_v775_human_methods.json": "sha256:4249a4e9b12e62f5638c161b546049946f30cf2f82a64be9525707e4b672f4df",
    "results/experiment_9085_collect_a.json": "sha256:21d52f0766ab49863f59b3791086c636ef5154283bb15061737978f6d95fe374",
    "results/experiment_9086_collect_b.json": "sha256:8d8a1da380c417c403c829a92e1064a8bf26b7ef714afd6f43a6a1de080435a1",
    "results/experiment_9088_human_embeddings.json": "sha256:198fd340c2c7c8b0be4da1f024ee53a329c7526bd6fd22f7062aa1ce48f50258",
    "results/experiment_9093_v775_service_cost.json": "sha256:a02f9f1b3cdd5264ddbddabbd91fc8af33d106fa7e145db65650cde5760d9271",
    "results/experiment_9094_v775_susceptibility.json": "sha256:c16cca543256c4dd908010776367dd0dffb9de323e85391f9c5074583702e3a5",
}
SUFFIXES = (
    "contract",
    "prospective-custody",
    "human-methods",
    "collect-a",
    "collect-b",
    "sealed-admission",
    "human-embeddings",
    "static-seal",
    "online-seal",
    "graph-decision",
    "human-decision",
    "service-cost",
    "susceptibility",
    "capstone",
)
TASKS = [f"exp{9082 + i}-{suffix}" for i, suffix in enumerate(SUFFIXES)]


def project(value: Json) -> Json:
    """Keep large numerical arrays in authenticated case files instead of duplicating them."""
    return {
        k: v
        for k, v in value.items()
        if k
        not in {
            "binary_energy_rows",
            "stability_rows",
            "derivative_rows",
            "susceptibility_rows",
            "finite_perturbation_rows",
        }
    }


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Keep exact absent operands and numeric zeros inspectable in blocked results."""
    return costs.gate(
        str(path), field, expected, observed, name, "exp9095-capstone", reference(path)["sha256"]
    )


def authenticate(root: Path, raw: Path) -> Json:
    """Authenticate fixed source bytes before allowing any dependent row reduction."""
    checks, snapshots, values, operands = [], {}, {}, {}
    for name, pin in PINS.items():
        path = root / name
        checks.append(
            check(path, "sha256", pin, reference(path)["sha256"] or "missing_file", "source_bytes")
        )
        if checks[-1]["passed"]:
            snapshots[name] = retain(path, raw, pin)
    for n in range(9082, 9095):
        expected = sorted(Path(p).name for p in PINS if p.startswith(f"results/experiment_{n}_"))
        observed_names = sorted(p.name for p in (root / "results").glob(f"experiment_{n}_*.json"))
        checks.append(
            check(
                root / "results",
                f"experiment_{n}_inventory",
                expected,
                observed_names,
                "primary_inventory",
            )
        )
    tasks = []
    if "research-roadmap.yaml" in snapshots and DESIGN in snapshots:
        tasks = yaml.safe_load(Path(snapshots["research-roadmap.yaml"]["path"]).read_bytes())[
            "tasks"
        ]
        table, machine = parse_design(
            Path(snapshots[DESIGN]["path"]).read_text(), milestone="2026.11.775"
        )
        keys = [t["title"][:50].strip() for t in tasks]
        matched = len(tasks) == len(table) == len(machine) == len(set(keys)) == 14
        matched &= canonical_hash(tasks) == TASK_PIN
        for i, task in enumerate(tasks):
            matched &= task in (machine[i], dict(machine[i], prompt=encode(machine[i]["prompt"])))
            matched &= all(table[i][k] == task[k] for k in TABLE_FIELDS)
            matched &= task["id"] == TASKS[i] and table[i]["order"] == i + 1
        checks.append(check(root / DESIGN, "full_task_objects", True, matched, "task_authority"))
    for name, ref in sorted(snapshots.items()):
        if not name.startswith("results/experiment_") or "9081_" in name:
            continue
        value = project(json.loads(Path(ref["path"]).read_bytes()))
        n = int(name.split("experiment_")[1][:4])
        task = TASKS[n - 9082]
        scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
        identity = (
            (value.get("experiment") == n and value.get("title") == tasks[n - 9082]["title"])
            if scheduler and tasks
            else value.get("experiment_id") == n and value.get("task_id") == task
        )
        checks.append(check(root / name, "producer_identity", True, identity, "producer_identity"))
        entry: Json = dict(value=value, reference=ref, selected_name=name, current_code_drift=[])
        values[task] = entry
        bindings = value.get("code_config_hashes", []) + value.get("raw_shard_hashes", [])
        bindings += [r for r in value.get("current_work_receipts", []) if "path" in r]
        for binding in bindings:
            path = Path(binding["path"])
            observed = reference(path)["sha256"] or "missing_file"
            row = check(path, "sha256", binding["sha256"], observed, "producer_binding")
            if row["passed"]:
                operands[str(path)] = retain(path, raw, binding["sha256"])
            elif binding in value.get("code_config_hashes", []):
                entry["current_code_drift"].append(row)
            else:
                checks.append(row)
    return dict(checks=checks, snapshots=snapshots, values=values, operands=operands, tasks=tasks)


def primitive(work: Json, ref: Json) -> Json:
    """Read only retained, byte-authenticated primitives rather than reported totals."""
    saved = work["operands"][ref["path"]]
    require(saved["sha256"] == ref["sha256"], "primitive_identity")
    return json.loads(Path(saved["path"]).read_bytes())  # type: ignore[no-any-return]


def decision(branch: str, value: Json) -> Json:
    """A measured null closes its frozen comparison; a missing result is never a null."""
    cls = value.get("verdict_class")
    action = (
        "retain_inventory_require_complete_service"
        if branch == "cost" and cls == "null"
        else "close_unchanged_scope"
        if cls == "null"
        else "retain_qualified_result"
        if cls in {"positive", "circular_positive"}
        else "blocked_require_changed_prerequisite"
    )
    reopen = {
        "custody": "Enforce a complete historical-attempt exclusion inventory and source-time access recorder before a new request; no new seed or renamed gate.",
        "detector": "Authenticated disjoint A/B collections, qualified admission and immutable paired static predictions under the original seven bounds.",
        "online": "Qualified admission and predict-before-feedback paired seals with exact recovery under the original ten bounds.",
        "human": "Original author source and historical producer hashes must agree, then frozen Qwen embeddings; LPHB remains exposed development.",
        "cost": "Matched scientific per-request clocks, quality, recovery and measured power integrals; inventory null closes no scientific comparison.",
        "susceptibility": "A new physical implementation with measured force response, parity, timing and energy; CPU cancellation alone is insufficient.",
        "ARC": "New uncensored hidden-game paired episodes across held families; unchanged receipts are context only.",
    }
    return dict(
        branch=branch,
        decision=action,
        prior_verdict=value.get("honest_verdict"),
        reopening_prerequisite=reopen[branch],
        independent_generalization_score=0,
    )


def outcomes(work: Json) -> list[Json]:
    """Account for every slot without promoting scheduler OK lines to scientific results."""
    log_ref = work["snapshots"].get("ops/conductor-log.md")
    log = Path(log_ref["path"]).read_text().splitlines() if log_ref else []
    rows = []
    for i, task in enumerate(TASKS[:-1]):
        obj = work["tasks"][i] if work["tasks"] else {}
        entry = work["values"].get(task, {})
        value = entry.get("value", {})
        events = [line for line in log if obj and obj["title"][:50].strip() in line]
        scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
        disposition = (
            "authenticated_scheduler_skip"
            if scheduler
            else "producer_primary"
            if entry
            else "unstarted_upstream_retired"
            if any("upstream retired" in e for e in events)
            else "missing_primary"
        )
        rows.append(
            dict(
                task_id=task,
                experiment_id=9082 + i,
                task=obj,
                primary_path=obj.get("deliverable"),
                primary_status="present"
                if entry.get("selected_name") == obj.get("deliverable")
                else "missing_file",
                selected_path=entry.get("selected_name"),
                reference=entry.get("reference"),
                disposition=disposition,
                scheduler_events=events,
                honest_verdict=None if scheduler else value.get("honest_verdict"),
                scheduler_honest_verdict=value.get("honest_verdict") if scheduler else None,
                scheduler_gate_summary=value.get("gate_check_summary") if scheduler else None,
                scheduler_gates_evaluated=value.get("gates_evaluated", []) if scheduler else [],
                verdict_class=None if scheduler else value.get("verdict_class"),
                qualified=bool(
                    value.get("required_checks_passed") and not entry.get("current_code_drift")
                ),
                current_code_drift=entry.get("current_code_drift", []),
            )
        )
    return rows


def reduce(work: Json) -> Json:
    """Recompute available primitives while missing graph predictions remain unmeasured."""
    require(canonical_hash(work["tasks"]) == TASK_PIN, "task_authority")
    values = {t: e["value"] for t, e in work["values"].items()}
    contract = values[TASKS[0]]
    human = values[TASKS[2]]
    human_rows = primitive(work, human["primitive_reference"])["reduction"]["rows"]
    require(human_rows == human["rows"], "human_primitive_rows")
    require(len({r["unit_id"] for r in human_rows}) == len(human_rows), "human_unit_identity")
    human_summary = dict(
        question_count=len({r["qid"] for r in human_rows}),
        annotator1_observed_count=sum(r["annotator1"] is not None for r in human_rows),
        annotator2_observed_count=sum(r["annotator2"] is not None for r in human_rows),
        annotator2_missing_count=sum(r["annotator2"] is None for r in human_rows),
        disagreement_count=sum(r["disagreement"] for r in human_rows),
        role_counts=dict(Counter(r["role"] for r in human_rows)),
        claim_scope="LPHB exposed development inventory; no trained-head result or independent generalization.",
        status="blocked_historical_producer_hash",
        rows=human_rows,
    )
    for key in (
        "question_count",
        "annotator1_observed_count",
        "annotator2_observed_count",
        "annotator2_missing_count",
        "disagreement_count",
        "role_counts",
    ):
        require(human_summary[key] == human[key], "human_aggregate:" + key)
    cost = values[TASKS[11]]
    cost_rows = primitive(work, cost["raw_shard_hashes"][0])["result"]["operation_cost_rows"]
    require(cost_rows == cost["operation_cost_rows"], "cost_primitive_rows")
    cost_summary = costs.summarize(cost_rows)
    require(all(cost_summary[k] == cost[k] for k in cost_summary), "cost_aggregate")
    diagnostic = values[TASKS[12]]
    cases = [primitive(work, r) for r in diagnostic["raw_shard_hashes"][1:]]
    summary = susceptibility.summarize(cases)
    require(summary == diagnostic["summary"], "susceptibility_aggregate")
    bounds: Json = {}
    for prefix, count in [("seven", 7), ("ten", 10)]:
        policy = contract["frozen_science"][prefix + "_bound_contract"]
        require(len(policy["bounds"]) == count, "original_bound_count")
        bounds[prefix + "_bound_rows"] = [
            dict(
                bound_id=f"{prefix}:{i + 1}",
                **bound,
                value=None,
                numerator=None,
                denominator=0,
                status="unmeasured",
                observed="missing_file",
                passed=False,
                verifier_is_oracle=True,
                claim_scope="Exact graph labels; any graph success is circular_positive.",
            )
            for i, bound in enumerate(policy["bounds"])
        ]
    return dict(
        **bounds,
        human_summary=human_summary,
        cost_summary=dict(cost_summary, rows=cost_rows),
        susceptibility_summary=summary,
        arc_summary=dict(
            contract["arc_summary"],
            new_gameplay_executed=False,
            claim_scope="Unchanged ARC receipt context only; no level solve or new generalization.",
        ),
        continuation_decisions=[
            decision(branch, values.get(task, {}))
            for branch, task in [
                ("custody", TASKS[1]),
                ("detector", TASKS[9]),
                ("online", TASKS[8]),
                ("human", TASKS[10]),
                ("cost", TASKS[11]),
                ("susceptibility", TASKS[12]),
                ("ARC", "absent"),
            ]
        ],
        retirement_recommendations=[
            dict(
                recommendation="retire_prospective_chain",
                task_id="exp9083-prospective-custody",
                prior_task_id="exp9069-source-custody",
                unchanged_verdict=False,
                repeated_blocker_family="custody_source_authority",
                prior_verdict="complete_blocked_a_access_coverage",
                prior_capstone_verdict="complete_blocked_declared_gate",
                current_verdict=values[TASKS[1]]["honest_verdict"],
                prior_reference=work["snapshots"]["results/experiment_9081_v774_capstone.json"],
                rationale="The custody/source-authority prerequisite failed again; changed wording and fresh seeds cannot establish missing original attempt topology.",
                reopening_prerequisite=decision("custody", {})["reopening_prerequisite"],
                manifest_mutated=False,
                retire_if_same_verdict=True,
            )
        ],
    )
