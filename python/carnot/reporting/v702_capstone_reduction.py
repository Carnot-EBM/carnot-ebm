"""REQ-VERIFY-8122: preserve independent conclusions without promoting exposed data.

The qualified V701 source-level equations remain useful for private H1/H2
controls. V702 moves the H1 audit one slot later and leaves its independently
initialized learning audit in slot seven. This adapter keeps that distinction.
"""

from copy import deepcopy
from typing import Any

from carnot.reporting import v701_capstone_reduction as previous
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def reduce(data: Json) -> Json:
    """Finish every scheduled disposition while keeping invalid measurements excluded."""
    adapted = deepcopy(data)
    tasks = data["tasks"]
    adapted["primaries"][tasks[4]["id"]] = data["primaries"].get(tasks[5]["id"], {})
    adapted["dispositions"][4]["eligible"] = data["dispositions"][5]["eligible"]
    value = previous.reduce(adapted)
    rows = deepcopy(data["dispositions"]) + [value["rows"][-1]]
    for row in rows:
        row.update(
            source_cluster_id=row["unit_id"],
            condition="terminal_accounting",
            metric="authenticated_task",
            raw_numerator=row["numerator"],
            raw_denominator=row["denominator"],
        )
    state = value["verdict_class"]
    first = next((g["check"] for g in data["failures"] if not g.get("passed")), "v702_capstone")
    verdict = "complete_" + state + "_" + (first if state == "blocked" else "v702_capstone")
    oracle = bool(data.get("verifier_is_oracle", False))
    if oracle and state == "positive":
        state, verdict = "circular_positive", "complete_circular_positive_v702_capstone"
    rows[-1].update(honest_verdict=verdict, verdict_class=state, issued_state=verdict)
    retirements = []
    for task, row in zip(tasks, rows, strict=True):
        for prior in task["prior_failures"]:
            evidence = data.get("prior_evidence", {}).get(prior["experiment_id"], {})
            same = prior["verdict"] == row["honest_verdict"]
            retirements.append(
                dict(
                    prior,
                    task_id=task["id"],
                    same_verdict=same,
                    observed_verdict=row["honest_verdict"],
                    scope=task["id"],
                    prior_path=evidence.get("path"),
                    prior_sha256=evidence.get("sha256"),
                    prior_artifact_verdict=evidence.get("honest_verdict"),
                    prior_verdict_matches_artifact=evidence.get("honest_verdict")
                    == prior["verdict"],
                    input_path=row.get("path"),
                    input_sha256=row.get("sha256"),
                    task_config_sha256=canonical_hash(task),
                    changed_mechanism_evidence=prior["addressed_by"],
                    retire_exact_configuration=bool(
                        same
                        and row["eligible"]
                        and row["verdict_class"] == "null"
                        and prior["retire_if_same_verdict"]
                        and evidence.get("sha256")
                        and evidence.get("honest_verdict") == prior["verdict"]
                        and row.get("sha256")
                    ),
                    environmental_block_retires_method_family=False,
                    automatic_reopen=False,
                    reopen_condition="Changed mechanism or separately acquired independent human-labeled corpus.",
                )
            )
    service = data["primaries"].get(tasks[9]["id"], {})
    complete = int(
        data["dispositions"][9]["eligible"] and service.get("complete_service_ready_score") == 1
    )
    value.update(
        honest_verdict=verdict,
        verdict_class=state,
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        retirement_candidates=retirements,
        verifier_is_oracle=int(oracle),
        eligible_count=sum(r["eligible"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        h1_decision_benefit_score=value["h1_development_signal_score"],
        h2_learning_benefit_score=value["h2_development_signal_score"],
        complete_service_evidence_score=complete,
        whole_service_deployment=dict(
            ready=bool(complete),
            host_ready=bool(
                data["dispositions"][9]["eligible"] and service.get("host_service_ready_score") == 1
            ),
            acquisition_ready=bool(
                data["dispositions"][8]["eligible"]
                and data["primaries"].get(tasks[8]["id"], {}).get("acquisition_cost_ready_score")
                == 1
            ),
            independent_deployment_claim=False,
        ),
        exposure_description="Previously exposed RAGTruth development; no independent generalization.",
    )
    value["scope_reduction_compliance"].update(
        historical_111_scopes=["GRPO", "puzzle", "HardNet", "generic external-text"],
        exclusion_manifest_reference=data.get("exclusion_manifest_reference"),
        automatic_reopen=False,
        radial_next_evidence="Retire unchanged qualified null radial configurations; blocked fitting remains untested.",
        center_growth_next_evidence="Retire unchanged qualified null center-growth configurations; disqualified training earns no benefit.",
        promising_finite_next_evidence="Acquire a separately collected independent human-labeled corpus before generalization.",
    )
    return value
