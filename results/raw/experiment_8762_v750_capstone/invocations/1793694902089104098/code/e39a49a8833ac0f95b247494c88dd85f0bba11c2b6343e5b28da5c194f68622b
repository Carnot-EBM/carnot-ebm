"""REQ-VERIFY-8737: enumerate truth independently; register but never score variants."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.verification_methods_8726 import SCHEDULES

Json = dict[str, Any]
NEW = ("omitted_shared_variable_join", "missing_required_premise_as_false")
VARIANTS = ("correct", "reject_all", "accept_all", "reversed_arguments", *NEW)


def truth(facts: list[list[Any]], query: list[list[str]] | None) -> str:
    """Enumerate consistent assignments so a missing premise stays unknown.

    Records explicitly state positive or negative facts. No closed-world rule or
    verifier implementation supplies labels. A conjunction needs every premise;
    an existential claim needs one consistent binding across all its premises.
    """
    if not query:
        return "unknown"
    entities = sorted(
        {str(x) for f in facts for x in (f[0], f[2])}
        | {x for q in query for x in (q[0], q[2]) if not x.startswith("?")}
    )
    variables = sorted({x for q in query for x in q if x.startswith("?")})
    outcomes = []
    for values in product(entities, repeat=len(variables)):
        binding = dict(zip(variables, values))
        atoms = []
        for q in query:
            grounded = [binding.get(x, x) for x in q]
            matching = [f[3] for f in facts if f[:3] == grounded]
            atoms.append(
                "true" if True in matching else "false" if False in matching else "unknown"
            )
        outcomes.append(
            "false" if "false" in atoms else "unknown" if "unknown" in atoms else "true"
        )
    return "true" if "true" in outcomes else "unknown" if "unknown" in outcomes else "false"


def freeze() -> Json:
    """Seal IDs and all probe inputs before either new defect can be executed."""
    worlds, rows = [], []
    for family in ["role_binding", "temporal_order", "two_premise_composition"]:
        for index in range(20):
            a, b, c, d = [f"{family}_{index:02d}_{name}" for name in "abcd"]
            relation = "before" if family == "temporal_order" else "owns"
            facts = [
                [a, relation, b, True],
                [b, "links", c, True],
                [a, relation, c, False],
                [d, "links", c, True],
                [c, "links", d, True],
            ]
            query = (
                [[a, relation, b]]
                if family == "role_binding"
                else [[a, relation, "?join"], ["?join", "links", c]]
            )
            world_id = f"v749/{family}/{index:02d}"
            partition = "development" if index < 10 else "evaluation"
            rename = {x: f"entity_{i}" for i, x in enumerate([d, c, b, a])}
            renamed = [[rename[s], r, rename[o], t] for s, r, o, t in facts]
            renamed_query = [[rename.get(s, s), r, rename.get(o, o)] for s, r, o in query]
            probes = []
            for name, records, predicates in [
                ("truth", facts, query),
                ("equivalent", list(reversed(facts)), list(reversed(query))),
                (
                    "invalid",
                    facts,
                    [[a, relation, c]]
                    if len(query) == 1
                    else [[a, relation, "?join"], ["?join", "links", d]],
                ),
                ("unknown", facts, None),
                ("deleted_records", [], query),
                ("entity_bijection", renamed, renamed_query),
                ("required_premise", facts[1:], query),
                ("required_role", [[s, r, o, False] for s, r, o, _ in facts], query),
            ]:
                probes.append(
                    dict(
                        transform=name,
                        facts=deepcopy(records),
                        query=deepcopy(predicates),
                        expected=truth(records, predicates),
                        supporting_premises=deepcopy(records),
                    )
                )
            answer = (
                f"{a} {relation} {b}"
                if len(query) == 1
                else f"{a} connects through one shared entity to {c}"
            )
            worlds.append(
                dict(
                    world_id=world_id,
                    family=family,
                    partition=partition,
                    facts=facts,
                    query=query,
                    proposed_answer=answer,
                    answer_spans=[dict(start=0, end=len(answer), text=answer)],
                    supporting_premises=facts[:2],
                    probes=probes,
                    valid_alternatives=["equivalent", "entity_bijection"],
                )
            )
            for variant in VARIANTS:
                for arm, schedule in SCHEDULES.items():
                    rows.append(
                        dict(
                            unit_id=world_id + "/" + variant + "/" + arm,
                            world_id=world_id,
                            partition=partition,
                            variant=variant,
                            arm=arm,
                            ordered_transforms=list(schedule),
                            unique_probe_ids=[
                                world_id + "/" + variant + "/" + arm + "/" + t for t in schedule
                            ],
                            status="frozen_unexecuted",
                            actual=None,
                            latency_s=None,
                            predicate_work=None,
                            timed_repetitions=10,
                            independent_cluster=world_id,
                        )
                    )
    return dict(
        schema="carnot.v749.verifier-transfer.v1",
        worlds=worlds,
        rows=rows,
        variants=list(VARIANTS),
        variant_definitions=dict(
            correct="preserve true/false/unknown with consistent bindings",
            reject_all="constant false",
            accept_all="constant true",
            reversed_arguments="swap subject and object in every query",
            omitted_shared_variable_join="satisfy premises separately without equality of shared entity",
            missing_required_premise_as_false="replace unknown due to absent required evidence with false",
        ),
        schedules={k: list(s) for k, s in SCHEDULES.items()},
        schedules_selected_by="Exp8726 only; no new-defect outcomes inspected",
        new_families_select_probes=False,
        early_exit=False,
        timed_repetitions=10,
        unique_probe_invocations=3600,
        world_variant_arm_rows=720,
        independent_worlds=60,
        evaluation_defective_units_per_arm=150,
        evaluation_new_family_units_per_arm=60,
        split_rule="first10 IDs per family development, last10 evaluation; immutable before results",
        truth_source="independent exhaustive signed-fact assignment enumerator",
        analysis=dict(
            resamples=10000,
            cluster="world_id",
            paired=True,
            primary="new-family detection gain lower95 > 0 and no extra correct-verifier false rejection",
            equal_calls_not_equal_time=True,
            repeats_increase_sample_size=False,
        ),
        scientific_claim="constructed oracle-dependent benefits remain circular; independent generalization unmeasured",
        roster_sha256=canonical_hash(worlds),
    )
