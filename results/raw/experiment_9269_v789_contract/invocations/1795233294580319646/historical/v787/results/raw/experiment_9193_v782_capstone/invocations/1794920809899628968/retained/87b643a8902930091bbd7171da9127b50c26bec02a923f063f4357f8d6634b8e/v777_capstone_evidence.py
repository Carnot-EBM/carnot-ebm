"""REQ-REPORT-9123-V777: authenticate evidence before any dependent reduction."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v776_capstone_evidence as history
from carnot.reporting import v775_capstone_evidence as inherited_history
from carnot.reporting import v777_contract as contract
from carnot.reporting import complete_cost_9122 as cost_reader
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v768_capstone import edge as edge

Json = dict[str, Any]
ROOT = contract.ROOT
FROZEN = ROOT / "openspec/change-proposals/v777-capstone-inputs.json"
PROTOCOL = json.loads(FROZEN.read_bytes())
PINS, TASK_PIN = PROTOCOL["pins"], PROTOCOL["task_hash"]
SUFFIXES = (
    "contract",
    "forward-evidence",
    "masked-runtime",
    "energy-methods",
    "features-a",
    "features-b",
    "logit-capture",
    "static-heads",
    "online-streams",
    "static-audit",
    "learning-audit",
    "energy-audit",
    "complete-cost",
    "capstone",
)
TASKS = [f"exp{9110 + i}-{s}" for i, s in enumerate(SUFFIXES)]
project = history.project


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Missing bytes and fields must remain distinguishable from measured zero."""
    return dict(history.check(path, field, expected, observed, name), upstream=TASKS[-1])


def authenticate(root: Path, raw: Path) -> Json:
    """Use authenticated historical snapshots instead of changed checkout code."""
    checks, snapshots, values, operands, resolutions = [], {}, {}, {}, []
    for name, pin in PINS.items():
        path = root / name
        row = check(
            path, "sha256", pin, reference(path)["sha256"] or "missing_file", "source_bytes"
        )
        checks.append(row)
        if row["passed"]:
            snapshots[name] = retain(path, raw, pin)
    for name, ref in snapshots.items():
        if name.startswith("results/"):
            n = int(Path(name).name.split("_")[1])
            identity = TASKS[n - 9110] if 9110 <= n < 9123 else f"prior:{n}"
            value = json.loads(Path(ref["path"]).read_bytes())
            scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
            valid = value.get("experiment") == n if scheduler else value.get("experiment_id") == n
            valid = valid and (scheduler or n == 6980 or value.get("task_id") == identity)
            checks.append(check(root / name, "producer_identity", True, valid, "producer_identity"))
            values[identity] = dict(
                value=value, reference=ref, selected_name=name, current_code_drift=[]
            )
    source = values.get(TASKS[0], {}).get("value", {})
    for identity, fields in [
        (
            TASKS[0],
            {
                "contract_ready_score": 1,
                "source_ready_score": 1,
                "online_methods_ready_score": 1,
                "required_checks_passed": True,
            },
        ),
        (TASKS[12], {"cost_inventory_ready_score": 1, "required_checks_passed": True}),
    ]:
        entry = values.get(identity, {})
        for field, expected in fields.items():
            observed = entry.get("value", {}).get(
                field, "missing_field" if entry else "missing_file"
            )
            checks.append(
                check(
                    root / entry.get("selected_name", "results"),
                    field,
                    expected,
                    observed,
                    "upstream_custody_gate",
                )
            )
    catalog = {r["sha256"]: r for r in source.get("source_references", [])}
    bindings = [
        r
        for entry in values.values()
        for key, item in entry["value"].items()
        for r in (
            item
            if key
            in {
                "source_references",
                "raw_shard_hashes",
                "current_work_receipts",
                "code_config_hashes",
            }
            else [item]
            if key.endswith("_reference") and isinstance(item, dict)
            else []
        )
        if r.get("path") and r.get("sha256")
    ]
    for index, bound in enumerate(bindings):
        original = Path(bound["path"])
        if original.suffix in {".gguf", ".so"} or str(original) in operands:
            continue
        candidate = catalog.get(bound["sha256"], bound)
        path = Path(candidate["path"])
        path = root / path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        row = check(
            path,
            "sha256",
            bound["sha256"],
            reference(path)["sha256"] or "missing_file",
            "producer_binding",
        )
        checks.append(row)
        if row["passed"]:
            operands[str(original)] = retain(path, raw, bound["sha256"])
            if path != original:
                resolutions.append(
                    dict(
                        original_path=str(original),
                        snapshot=operands[str(original)],
                        scope="historical provenance only",
                    )
                )
        if index % 100 == 0:
            print(
                f"[exp9123] phase=authenticated_bindings completed={index} pending={len(bindings) - index}",
                flush=True,
            )
    tasks = source.get("full_task_objects", [])
    for task in tasks[:-1]:
        name = task["deliverable"]
        if name not in PINS:
            checks.append(
                check(
                    root / name,
                    "declared_primary",
                    "missing_file",
                    reference(root / name)["sha256"] or "missing_file",
                    "frozen_primary_absence",
                )
            )
    declared_gates = [
        dict(edge(g, values), downstream=t["id"]) for t in tasks for g in t["gated_on"]
    ]
    if tasks:
        checks.append(check(root, "task_hash", TASK_PIN, canonical_hash(tasks), "task_authority"))
        active = snapshots.get("research-roadmap.yaml")
        design = snapshots.get(contract.DESIGN)
        queue = yaml.safe_load(Path(active["path"]).read_bytes()) if active else {}
        table, machine = (
            parse_design(Path(design["path"]).read_text(), milestone="2026.11.777")
            if design
            else ([], [])
        )
        checks.append(
            check(
                root,
                "full_task_objects",
                True,
                len(queue.get("tasks", [])) == len(tasks)
                and all(
                    q in (t, dict(t, prompt=encode(t["prompt"])))
                    for t, q in zip(tasks, queue.get("tasks", []), strict=False)
                ),
                "active_authority",
            )
        )
        matched = len(tasks) == len(machine) == len(table) == 14 and all(
            t in (m, dict(m, prompt=encode(m["prompt"])))
            and all(t[k] == visible[k] for k in TABLE_FIELDS)
            and visible["order"] == i + 1
            for i, (t, m, visible) in enumerate(zip(tasks, machine, table, strict=False))
        )
        checks.append(check(root, "visible_contract", True, matched, "design_authority"))
    return dict(
        checks=checks,
        snapshots=snapshots,
        values=values,
        operands=operands,
        tasks=tasks,
        historical_code_resolutions=resolutions,
        declared_gate_preconditions=declared_gates,
    )


def primitive(work: Json, ref: Json) -> Json:
    """Require the original byte identity before a retained object is consumed."""
    saved = work["operands"][ref["path"]]
    require(saved["sha256"] == ref["sha256"], "primitive_identity")
    return json.loads(Path(saved["path"]).read_bytes())  # type: ignore[no-any-return]


@contextmanager
def historical_binding(work: Json) -> Iterator[None]:
    """Old work envelopes resolve only through the current authenticated catalog."""
    catalog = {r["sha256"]: r for r in work["operands"].values()}

    def read(envelope: Json, ref: Json) -> Json:
        saved = envelope["operands"][ref["path"]]
        require(saved["sha256"] == ref["sha256"], "historical_identity")
        return json.loads(Path(catalog[ref["sha256"]]["path"]).read_bytes())  # type: ignore[no-any-return]

    with (
        patch.object(history, "primitive", read),
        patch.object(inherited_history, "primitive", read),
    ):
        yield


def outcomes(work: Json) -> list[Json]:
    """Preserve exact scheduler alternatives without substituting declared primaries."""
    with patch.multiple(history, TASKS=TASKS, SUFFIXES=SUFFIXES):
        rows = history.outcomes(work)
    for i, row in enumerate(rows):
        row["experiment_id"] = 9110 + i
        if not work["tasks"]:
            row["primary_path"] = (
                f"results/experiment_{9110 + i}_v777_{SUFFIXES[i].replace('-', '_')}.json"
            )
    return rows


def retirement(work: Json, prior: Json) -> list[Json]:
    """Apply declared same-verdict rules only to authenticated original scopes."""
    previous = {r["task_id"]: r for r in work["values"][TASKS[0]]["value"]["previous_outcome_rows"]}
    previous["exp6980-spilled-energy-requalification"] = work["values"]["prior:6980"]["value"]
    previous["exp8435-capacity-benefit-audit"] = prior["original_exp8435_null"]
    current = {r["task_id"]: r for r in outcomes(work)}
    current[TASKS[-1]] = dict(accounting_honest_verdict="complete_blocked_declared_gate")
    rows = []
    for task in work["tasks"]:
        for p in task["prior_failures"]:
            old, new = previous.get(p["experiment_id"], {}), current[task["id"]]
            before = (
                old.get("scheduler_honest_verdict")
                if p["verdict"].startswith("blocked_")
                else old.get("accounting_honest_verdict")
                or old.get("honest_verdict")
                or old.get("scheduler_honest_verdict")
            )
            after = (
                new.get("accounting_honest_verdict")
                or new.get("honest_verdict")
                or new.get("scheduler_honest_verdict")
            )
            same = bool(p["retire_if_same_verdict"] and before == p["verdict"] == after)
            rows.append(
                dict(
                    task_id=task["id"],
                    prior_task_id=p["experiment_id"],
                    declared_prior_verdict=p["verdict"],
                    actual_prior_verdict=before,
                    current_verdict=after,
                    prior_verdict_authenticated=before == p["verdict"],
                    retire_if_same_verdict=p["retire_if_same_verdict"],
                    condition_met=same,
                    changed_mechanism_context=p["addressed_by"],
                    recommendation="retire_unchanged_scope"
                    if same
                    else "require_new_qualified_prerequisite",
                    manifest_mutated=False,
                )
            )
    return rows


def bounds() -> Json:
    """Missing studies retain all predeclared bounds without inventing intervals."""
    definitions = {
        "static": [
            (a, "Brier_improvement", "lower>0")
            for a in [
                "prevalence",
                "text_length_logistic",
                "projected27_linear",
                "full_embedding_regularized_linear",
                "donor_projected27_spline",
            ]
        ],
        "online": [
            (a, "future_Brier_improvement", "lower>0")
            for a in [
                "frozen_spline",
                "bias_only_spline",
                "adaptive_linear",
                "donor_adaptive_spline",
            ]
        ]
        + [
            ("initial_state", "retained_Brier_increase", "upper<=0.005"),
            ("initial_state", "retained_error_increase", "upper<=0"),
        ],
        "energy": [
            (a, "Brier_improvement", "lower>0")
            for a in ["probability_length_logistic", "donor_combined_logistic"]
        ],
    }
    return {
        kind + "_bound_rows": [
            dict(
                unit_id=f"{kind}:{i + 1}",
                bound_id=f"{kind}:{i + 1}",
                group_id=None,
                qid=None,
                arm=kind,
                condition=control,
                control=control,
                seed={"static": 775092, "online": 776105, "energy": 777121}[kind],
                metric=metric,
                criterion=criterion,
                absolute_metric=None,
                value=None,
                numerator=None,
                denominator=0,
                status="censored",
                censor_reason="missing_qualified_scientific_predictions",
                exclusion_reason=None,
                passed=False,
                observed="missing_file",
                independent_generalization_score=0,
            )
            for i, (control, metric, criterion) in enumerate(items)
        ]
        for kind, items in definitions.items()
    }


def reduce(work: Json) -> Json:
    """Recompute historical primitives and retain every currently unmeasured branch."""
    require(canonical_hash(work["tasks"]) == TASK_PIN, "task_authority")
    source = work["values"][TASKS[0]]["value"]
    catalog = {r["sha256"]: r for r in work["operands"].values()}
    capstone_ref = next(
        r
        for r in source["authority_snapshots"]
        if r["source"]["path"].endswith("results/experiment_9109_v776_capstone.json")
    )
    capstone = json.loads(Path(catalog[capstone_ref["sha256"]]["path"]).read_bytes())
    saved = json.loads(Path(catalog[capstone["work_reference"]["sha256"]]["path"]).read_bytes())
    with historical_binding(work):
        prior = history.reduce(saved)
    require(prior == source["preserved_results"], "historical_aggregates")
    # These fixed inputs contain no current scientific predictions to reduce.
    for identity in TASKS[4:12]:
        require(identity not in work["values"], "unexpected_science_requires_new_protocol")
    cost = work["values"][TASKS[12]]["value"]
    cost_work = primitive(work, cost["work_reference"])
    rebuilt = cost_reader.build(
        dict(
            cost_work,
            references=[catalog[r["sha256"]] for r in cost_work["references"]],
            work_reference=cost["work_reference"],
        ),
        dict(passed=cost["required_checks_passed"], receipts=[], coverage={}),
        cost["run_date"],
    )
    require(rebuilt["rows"] == cost["rows"], "cost_primitive_rows")
    rows = [r for r in rebuilt["rows"] if r["scope"] == "science"]
    scientific_counts = dict(
        intended=len(rows),
        attempted=sum(r["status"] != "unstarted" for r in rows),
        completed=sum(r["status"] == "completed" for r in rows),
        failed=sum(r["status"] == "failed" for r in rows),
        censored=sum(r["status"] == "censored" for r in rows),
        unstarted=sum(r["status"] == "unstarted" for r in rows),
    )
    require(scientific_counts["completed"] == 0, "unavailable_cost_measurements")
    decisions = [
        dict(
            branch="forward_evidence",
            decision="stop_until_owned_controls_pass",
            next_prerequisite="Exp9111 operation-specific receipts and all unchanged positive/adversarial controls must pass; preserve its disqualified primary.",
        ),
        dict(
            branch="masked_runtime",
            decision="retain_only_if_original_controls_pass",
            next_prerequisite="Qualified forward receipts, original embedding/logit tolerances, ragged masking, serial/batch parity, state/index controls, actual GPU0 lease/offload and feasibility; otherwise retire this repair scope.",
        ),
        dict(
            branch="static",
            decision="blocked",
            next_prerequisite="Two complete authenticated 450-row shards; original six heads, five Brier lower bounds>0 and defined accepted risk no worse than projected linear.",
        ),
        dict(
            branch="online",
            decision="blocked",
            next_prerequisite="Independent online100 initialization; sealed delayed feedback, exact recovery and all six original stream bounds, including retained Brier<=0.005 and error<=0.",
        ),
        dict(
            branch="energy",
            decision="blocked",
            next_prerequisite="Qualified full-vocabulary logit capture and frozen formulas/gauge controls; both combined Brier lower bounds>0 and defined accepted risk no worse than probability+length.",
        ),
        dict(
            branch="independent_human_evidence",
            decision="defer_until_descriptive_signal",
            next_prerequisite="A descriptive static, online or energy signal with its matched controls must precede genuinely new independent human confirmation. Exposed LPHB cannot close either verifier gap.",
        ),
        dict(
            branch="complete_cost",
            decision="retain_inventory_only",
            next_prerequisite="Original first20 TEST qids/all60 siblings, three fresh processes, matching scientific probabilities/decisions, complete request/update/recovery clocks and measured power integrals.",
        ),
    ]
    return dict(
        prior,
        **bounds(),
        cost_summary=dict(
            rows=rows,
            counts=scientific_counts,
            measured_latency_s=None,
            energy_joules=None,
            complete_service_ratios=[],
            amdahl_ceiling=None,
            historical_inventory=prior["cost_summary"],
            scientific_request_counts=dict(
                intended=len(rows) * 3,
                attempted=0,
                completed=0,
                failed=0,
                censored=0,
                unstarted=len(rows) * 3,
            ),
            scope="Current unavailable scientific services; historical administrative clocks stay separate.",
        ),
        closed_scope_rows=[
            r
            for r in prior["continuation_decisions"]
            if r["branch"] in {"graph_collection", "physical_hardware"}
        ],
        continuation_decisions=decisions,
        retirement_recommendations=retirement(work, prior),
        hardware_operation_rows=cost["hardware_operation_rows"],
        hardware_limits=dict(
            CPU="State, storage, reductions and small fits only.",
            GPU0="Frozen batched features only after lease, memory, offload and parity qualification.",
            GPU1="Reserved; no dual-GPU assumption.",
            KV260="k_max<=5; established Ising scope only; no board operation.",
            PolarFire="CPU susceptibility only; no measured physical speed or joules.",
            GateMate="Physical/JTAG block unchanged; no repeated probe.",
        ),
        hardware_wishlist=dict(
            NPU="Supported operations, measured service bottleneck and complete parity/cost first.",
            TSU="Qualified sampling operation, parity and measured physical energy first.",
            larger_FPGA="Measured fixed-basis update bottleneck and validated implementation first.",
        ),
        historical_code_resolutions=work["historical_code_resolutions"],
        original_exp6980_null=dict(
            reference=work["values"]["prior:6980"]["reference"],
            honest_verdict=work["values"]["prior:6980"]["value"]["honest_verdict"],
            scope="Retired exact-oracle span comparison; whole-answer human scoring does not reverse it.",
        ),
    )
