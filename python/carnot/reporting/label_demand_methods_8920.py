"""REQ-REPORT-8920: acquire only labels needed by unresolved frozen policies.

Induction on candidate ordinal proves equality: active accepting policies see
one identical label, rejected outcomes never enter their statistics, and the
unchanged exact look rule therefore makes identical stopping decisions. This
is a local proof with constructed truth, not a natural-language benefit claim.
"""

from collections.abc import Callable, Sequence
from itertools import product
import random
from typing import Any

from carnot.reporting import finite_budget_methods_8906 as frozen
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting import shared_terminal_methods_8893 as shared

__all__ = ["frozen"]

Json = dict[str, Any]
ROOT, SEED = frozen.ROOT, 763300
NAME, TASK = "experiment_8920_v763_label_demand_methods", "exp8920-label-demand-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/label_demand_methods_8920.py",
    "python/carnot/reporting/label_demand_runner_8920.py",
    CLI,
]
TESTS = ["tests/python/test_label_demand_methods_8920.py"]
PROOF = __doc__


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose progress without extending the fixed runtime budget."""
    print(f"[exp8920] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Fix acquisition and the prospective contrast before any fresh labels open."""
    return dict(
        methods=frozen.protocol(),
        risk_ceiling=0.05,
        alpha_ledger=frozen.alpha_ledger(),
        selection_labels_per_group=240,
        acquisition="buy iff an unresolved policy accepts; share once per ordinal",
        cutoff_none="immediate refusal at ordinal zero in both references",
        primary_contrast="paired eager minus demand all-policy shared-union labels",
        interval=dict(
            method="one-sided bounded Hoeffding",
            confidence=0.99,
            independent_pairs=40,
            savings_range=[0, 3840],
            lower="mean(savings)-3840*sqrt(log(100)/(2*40))",
        ),
        safety="exact accepted-label/look/stop/served-set equality; zero inclusion violations",
        readiness_requires_savings=False,
        fresh_seeds=list(range(SEED, SEED + 40)),
        proof=PROOF,
        scientific_benefit_claim=False,
    )


class LabelLedger:
    """Separate paid independent draws from evaluator-only census access.

    Source IDs may repeat; the draw key is the billable identity. Reads require
    a successful purchase, and census truth stays unavailable until decisions seal.
    """

    def __init__(self, provider: Callable[[int], int]) -> None:
        self._provider = provider
        self._labels: dict[str, int] = {}
        self.events: list[Json] = []
        self.paid = 0
        self.census_reads = 0
        self.sealed = False

    def buy(self, key: str, source: int, stage: str) -> int:
        """Charge each authorized draw once, even for repeated source items."""
        if stage not in ["selection", "certification"]:
            raise ValueError("label_stage")
        if key in self._labels:
            raise ValueError("duplicate_draw")
        label = self._provider(source)
        if type(label) is not int or label not in [0, 1]:
            raise ValueError("binary_label")
        self._labels[key] = label
        self.paid += 1
        self.events.append(dict(draw=key, source=source, stage=stage, label=label))
        return self.read(key)

    def read(self, key: str) -> int:
        """An unrequested outcome cannot silently become a correct pseudo-label."""
        if key not in self._labels:
            raise ValueError("unrequested_label")
        return self._labels[key]

    def census(self, sources: list[int]) -> list[int]:
        """Evaluator truth is a separate cost and cannot affect service decisions."""
        if not self.sealed:
            raise ValueError("premature_census")
        self.census_reads += len(sources)
        return [self._provider(i) for i in sources]


def acquire(
    masks: Sequence[Sequence[bool] | None],
    ledger: LabelLedger,
    *,
    eager: bool = False,
    predictions: Sequence[Sequence[int]] | None = None,
    looks: list[int] = frozen.LOOKS,
    denominators: list[int] = frozen.DENOMINATORS,
    risk_denominator: int = 20,
) -> Json:
    """Consume immutable source-visible masks in order using the original exact rule.

    Binary truth is shared while each policy retains its source-visible prediction.
    Every binary truth path is represented, including hidden errors.
    The certificate engine sees only accepted labels, never unknown outcomes.
    """
    if not masks or any(
        m is not None and (len(m) != looks[-1] or any(type(x) is not bool for x in m))
        for m in masks
    ):
        raise ValueError("acceptance_mask")
    masks = tuple(None if mask is None else tuple(mask) for mask in masks)
    if predictions is None:
        predictions = [[0] * looks[-1] for _ in masks]
    if len(predictions) != len(masks) or any(
        len(p) != looks[-1] or any(type(x) is not int or x not in [0, 1] for x in p)
        for p in predictions
    ):
        raise ValueError("source_prediction")
    predictions = tuple(tuple(p) for p in predictions)
    states: list[Json] = [
        dict(
            sequence=[],
            looks=[],
            stop=0 if m is None else None,
            certificate_issued=False,
            reason="cutoff_none" if m is None else "pending",
        )
        for m in masks
    ]
    rows, work = [], 0
    for ordinal in range(1, looks[-1] + 1):
        active = [i for i, s in enumerate(states) if s["stop"] is None]
        accepting = [
            i
            for i, mask in enumerate(masks)
            if i in active and mask is not None and mask[ordinal - 1]
        ]
        work += len(active)
        paid = bool(active) if eager else bool(accepting)
        label = ledger.buy(f"candidate/{ordinal}", ordinal - 1, "certification") if paid else None
        rows.append(
            dict(
                ordinal=ordinal,
                active=active,
                accepting=accepting,
                label=label,
                paid=paid,
                status="labeled" if paid else "unlabeled",
            )
        )
        for i in accepting:
            states[i]["sequence"].append([ordinal, label])
        if ordinal not in looks:
            continue
        index = looks.index(ordinal)
        for i in active:
            s = states[i]
            n, errors = (
                len(s["sequence"]),
                sum(x[1] != predictions[i][x[0] - 1] for x in s["sequence"]),
            )
            stats = [(x["accepted"], x["errors"]) for x in s["looks"]] + [(n, errors)]
            path = frozen.decide(
                stats + [(n, errors)] * (len(looks) - len(stats)),
                refusal=True,
                looks=looks,
                denominators=denominators,
                risk_denominator=risk_denominator,
            )
            event = path["looks"][index]
            s["looks"].append(event)
            if event["reason"] != "pending":
                s.update(
                    stop=ordinal,
                    certificate_issued=event["certificate_issued"],
                    reason=event["reason"],
                )
    ledger.sealed = True
    return dict(
        policies=states,
        candidate_rows=rows,
        paid_labels=ledger.paid,
        cheap_policy_evaluations=work,
        label_events=ledger.events,
    )


def draw_manifest() -> Json:
    """Seal independent stage/group streams using fresh seeds without any provider."""
    reps = []
    for seed in range(SEED, SEED + 40):
        groups = {}
        for group in shared.base.GROUPS:
            streams = {}
            for stage, count in [("selection", 240), ("certification", 960)]:
                rng = random.Random(f"{seed}/{group}/{stage}")
                streams[stage] = [rng.randrange(1024) for _ in range(count)]
            groups[group] = streams
        reps.append(dict(replicate=seed - SEED, seed=seed, groups=groups))
    return dict(
        replicates=reps,
        labels_opened=False,
        population_per_group=1024,
        replacement=True,
        paired_arms=["eager", "demand"],
        protocol_hash=canonical_hash(protocol()),
    )


def qualify() -> Json:
    """Enumerate masks and hidden truth, then qualify full-budget accounting controls."""
    rows, work = [], 0
    small: Json = dict(looks=[1, 2, 3, 4], denominators=[2] * 4, risk_denominator=2)
    for code in range(256):
        masks = [[bool((code >> (2 * j + i)) & 1) for j in range(4)] for i in range(2)]
        for bits in product([0, 1], repeat=4):
            demand = acquire(masks, LabelLedger(lambda i: bits[i]), **small)
            eager = acquire(masks, LabelLedger(lambda i: bits[i]), eager=True, **small)
            work += demand["cheap_policy_evaluations"] + eager["cheap_policy_evaluations"]
            predictions = [[0, 1, 0, 1], [1, 0, 1, 0]]
            different = acquire(
                masks, LabelLedger(lambda i: bits[i]), predictions=predictions, **small
            )
            reference = acquire(
                masks, LabelLedger(lambda i: bits[i]), predictions=predictions, eager=True, **small
            )
            work += different["cheap_policy_evaluations"] + reference["cheap_policy_evaluations"]
            equal = (
                demand["policies"] == eager["policies"]
                and different["policies"] == reference["policies"]
            )
            rows.append(
                dict(
                    frozen.row(f"mask/{code}/truth/{''.join(map(str, bits))}", equal),
                    seed=SEED,
                    cluster_id="finite-proof",
                    masks=masks,
                    binary_truth=list(bits),
                    demand_labels=demand["paid_labels"],
                    eager_labels=eager["paid_labels"],
                    decision_hash=canonical_hash(demand["policies"]),
                )
            )
        if (code + 1) % 32 == 0:
            progress("exhaustive_masks", 16 * (code + 1), 16 * (255 - code))
    controls, retained = {}, {}
    cases: list[tuple[str, Sequence[Sequence[bool] | None], bool]] = [
        ("sparse", [[i % 4 == 0 for i in range(960)]] * 7, False),
        ("all_accept", [[True] * 960 for _ in range(7)], False),
        ("none", [None] * 7, True),
        ("hidden_errors", [[i % 4 == 0 for i in range(960)]] * 7, True),
        ("all_error", [[True] * 960] * 7, True),
    ]
    for name, control_masks, hidden in cases:
        provider = lambda i: int(hidden and (name == "all_error" or i % 4 != 0))
        demand = acquire(control_masks, LabelLedger(provider))
        eager = acquire(control_masks, LabelLedger(provider), eager=True)
        equal = demand["policies"] == eager["policies"]
        controls[name + "_exact"] = equal
        retained[name] = dict(demand=demand, eager=eager)
        work += demand["cheap_policy_evaluations"] + eager["cheap_policy_evaluations"]
    controls["sparse_saves"] = (
        retained["sparse"]["demand"]["paid_labels"] < retained["sparse"]["eager"]["paid_labels"]
    )
    controls["all_accept_no_savings"] = (
        retained["all_accept"]["demand"]["paid_labels"]
        == retained["all_accept"]["eager"]["paid_labels"]
    )
    ledger = LabelLedger(lambda i: 1)
    denied, premature = False, False
    try:
        ledger.read("unpaid")
    except ValueError:
        denied = True
    try:
        ledger.census([0])
    except ValueError:
        premature = True
    for i in range(240):
        ledger.buy(f"selection/{i}", 0, "selection")
    ledger.sealed = True
    ledger.census([0, 0])
    access = dict(
        unrequested_read_refused=denied,
        premature_census_refused=premature,
        repeated_source_draws_paid=ledger.paid == 240,
        census_separate=ledger.census_reads == 2 and ledger.paid == 240,
        unknown_rejections_retained=all(
            x["label"] is None for x in retained["none"]["demand"]["candidate_rows"]
        ),
    )
    control_rows = [
        dict(frozen.row("control/" + k, v), seed=SEED) for k, v in (controls | access).items()
    ]
    return dict(
        rows=rows + control_rows,
        proof_rows=rows,
        label_access_controls=access,
        positive_negative_controls=controls,
        retained_control_operands=retained,
        alpha_ledger=frozen.alpha_ledger(),
        proof_obligations=dict(
            proof=PROOF, passed=all(x["metric"] == 1 for x in rows + control_rows)
        ),
        decision_equality_violations=sum(x["metric"] != 1 for x in rows),
        pathwise_inclusion_violations=sum(
            not controls[k] for k in controls if k.endswith("_exact")
        ),
        cheap_policy_evaluations=work,
        scientific_benefit_claim=False,
        numerical_training_calls=0,
    )
