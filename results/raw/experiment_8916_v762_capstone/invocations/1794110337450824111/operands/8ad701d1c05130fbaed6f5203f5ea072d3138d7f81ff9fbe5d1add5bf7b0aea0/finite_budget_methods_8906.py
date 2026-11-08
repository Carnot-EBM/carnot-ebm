"""REQ-REPORT-8906: refuse only when every remaining registered test is impossible.

These finite constructions qualify a stopping rule. They establish neither
empirical label savings nor independent natural-language generalization.
"""

from functools import lru_cache, cache
from itertools import product
import math
import random
from typing import Any

from carnot.reporting import shared_terminal_methods_8893 as frozen
from carnot.reporting import sequential_certificate_methods_8880 as streams
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT, LOOKS, SEED = frozen.ROOT, frozen.LOOKS, 7628906
NAME, TASK = "experiment_8906_v762_finite_budget_methods", "exp8906-finite-budget-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/finite_budget_methods_8906.py",
    "python/carnot/reporting/finite_budget_runner_8906.py",
    CLI,
]
TESTS = ["tests/python/test_finite_budget_methods_8906.py"]
DENOMINATORS = [16800, 16800, 16800, 5600]
PROOF = (
    "At fixed p, couple Bernoulli sums by adding one independent trial. "
    "P(Bin(n+1,p)<=e)<=P(Bin(n,p)<=e), while increasing e increases this CDF. "
    "Inverting the CDF gives an upper bound decreasing in n and increasing in e. "
    "At future draw L after t, every attainable state satisfies n_future<=n+L-t "
    "and e_future>=e. Thus (n+L-t,e) minimizes the exact upper bound. "
    "Rejecting every optimistic future test proves every attainable future test fails. "
    "An already issued certificate is permanent. Until safe refusal, both sequential "
    "arms visit identical prefixes; after it neither can pass. Their decisions are "
    "therefore equal on every path. Fixed final certificates remain included. "
    "Statistical error separately uses the unchanged finite family union bound; "
    "stationary iid group draws independent of selection remain required. "
    "No theorem from the motivating sequential feasibility papers is imported."
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show completed work without moving the execution deadline."""
    print(f"[exp8906] phase={phase} completed={completed} pending={pending}", flush=True)


def alpha_ledger() -> list[Json]:
    """The shared final event is paid once across all three arms."""
    return [
        dict(stage="early", events=84, alpha_denominator=16800),
        dict(stage="shared_final", events=28, alpha_denominator=5600),
    ]


def protocol() -> Json:
    """Freeze future evaluation before opening any new calibration labels."""
    return dict(
        frozen.protocol(),
        seed=SEED,
        arms=["fixed_final", "early_acceptance", "early_acceptance_refusal"],
        alpha_denominators=DENOMINATORS,
        family_alpha_ledger=alpha_ledger(),
        numerical_convention="exact integer binomial CDF at risk1/20; no float tolerance; invalid bound output aborts all arms",
        refusal_rule="all remaining optimistic tests fail; every additional draw accepted with zero new errors",
        proof_obligations=PROOF,
        fresh_calibration_seed=SEED,
        scientific_benefit_claim=False,
        calibration_labels_opened=False,
        census_scope="retain original4096 worlds; evaluator-only truth",
        shared_cost="240 selection requests per group plus maximum policy stop; request ordinals shared once, replacement repeats paid",
        small_proof=dict(
            looks=[2, 4, 6, 8], alpha_denominators=[12, 12, 12, 4], risk_denominator=2
        ),
        randomized_states=4096,
    )


@lru_cache(maxsize=None, typed=True)
def passes(n: int, errors: int, alpha_denominator: int, risk_denominator: int = 20) -> bool:
    """Integer CDF comparison avoids a rounded floating bound changing a decision."""
    if (
        any(type(x) is not int for x in [n, errors, alpha_denominator, risk_denominator])
        or not 0 <= errors <= n
        or alpha_denominator <= 1
        or risk_denominator <= 1
    ):
        raise ValueError("binomial_counts_or_alpha")
    if not n or errors * risk_denominator >= n:
        return False
    term = (risk_denominator - 1) ** n
    numerator = term
    for k in range(errors):
        term = term * (n - k) // ((k + 1) * (risk_denominator - 1))
        numerator += term
    return bool(numerator * alpha_denominator <= risk_denominator**n)


@cache
def bound(n: int, errors: int, denominator: int) -> float:
    """Keep the original exact upper bound for audit, rejecting unusable numerics."""
    value = frozen.base.old.upper(n, errors, 1 / denominator)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("nonfinite_or_invalid_exact_bound")
    return float(value)


def decide(
    statistics: list[tuple[int, int]],
    *,
    refusal: bool = False,
    fixed: bool = False,
    looks: list[int] = LOOKS,
    denominators: list[int] = DENOMINATORS,
    risk_denominator: int = 20,
) -> Json:
    """Only attainable accepted counts and errors justify a permanent refusal."""
    if len(statistics) != len(looks) or len(denominators) != len(looks) or not looks:
        raise ValueError("sufficient_statistic_schedule")
    previous = (0, 0, 0)
    for look, (n, errors) in zip(looks, statistics):
        last_n, last_errors, last_look = previous
        if (
            type(n) is not int
            or type(errors) is not int
            or type(look) is not int
            or not 0 <= errors <= n <= look
            or look <= last_look
            or not 0 <= errors - last_errors <= n - last_n <= look - last_look
        ):
            raise ValueError("sufficient_statistic_transition")
        previous = n, errors, look
    events = []
    for i in [len(looks) - 1] if fixed else range(len(looks)):
        look, (n, errors), denominator = looks[i], statistics[i], denominators[i]
        issued = passes(n, errors, denominator, risk_denominator)
        remaining = [
            dict(
                candidate_draws=later,
                accepted=n + later - look,
                errors=errors,
                alpha=1 / d,
                risk_upper=bound(n + later - look, errors, d),
                certificate_possible=passes(n + later - look, errors, d, risk_denominator),
            )
            for later, d in zip(looks[i + 1 :], denominators[i + 1 :])
        ]
        impossible = not issued and not any(x["certificate_possible"] for x in remaining)
        reason = (
            "certificate"
            if issued
            else "impossible_future"
            if refusal and impossible and remaining
            else "final_refusal"
            if i == len(looks) - 1
            else "pending"
        )
        events.append(
            dict(
                candidate_draws=look,
                accepted=n,
                errors=errors,
                alpha=1 / denominator,
                risk_upper=bound(n, errors, denominator),
                certificate_issued=issued,
                remaining_tests=remaining,
                reason=reason,
            )
        )
        if issued or (refusal and impossible):
            break
    return dict(events[-1], looks=events)


def costs(stops: list[Json]) -> Json:
    """The last unresolved policy determines the group's shared label prefix."""
    groups = frozen.base.GROUPS
    union = {g: max(x["candidate_draws"] for x in stops if x["group"] == g) for g in groups}
    standalone = {
        str(i): sum(240 + x["candidate_draws"] for x in stops if x["policy"] == i) for i in range(7)
    }
    return dict(
        shared_union_labels=960 + sum(union.values()),
        group_prefixes=union,
        standalone_labels=standalone,
        replacement_repeats_paid=True,
    )


def row(unit: str, passed: bool, **fields: Any) -> Json:
    """Proof cases are deterministic checks, never independent scientific samples."""
    return dict(
        unit_id=unit,
        cluster_id="finite-proof",
        arm="proof",
        seed=SEED,
        metric=int(passed),
        denominator=1,
        status="completed",
        missing_reason=None,
        **fields,
    )


def qualify() -> Json:
    """Exhaustive paths and seeded states can falsify the local monotonicity proof."""
    paths, monotonic, randomized = [], [], []
    small: Json = dict(looks=[2, 4, 6, 8], denominators=[12, 12, 12, 4], risk_denominator=2)
    for alphabet in [2, 3]:
        total = alphabet**8
        for index, path in enumerate(product(range(alphabet), repeat=8)):
            stats = []
            for look in small["looks"]:
                prefix = path[:look]
                stats.append(
                    (
                        look if alphabet == 2 else sum(x > 0 for x in prefix),
                        sum(x == (1 if alphabet == 2 else 2) for x in prefix),
                    )
                )
            early, refusal, fixed = (
                decide(stats, **small),
                decide(stats, refusal=True, **small),
                decide(stats, fixed=True, **small),
            )
            passed = (
                early["certificate_issued"] == refusal["certificate_issued"]
                and (not fixed["certificate_issued"] or early["certificate_issued"])
                and refusal["candidate_draws"] <= early["candidate_draws"]
            )
            paths.append(
                row(
                    f"path/{alphabet}/{index}",
                    passed,
                    path=list(path),
                    statistics=stats,
                    early_certificate=early["certificate_issued"],
                    refusal_certificate=refusal["certificate_issued"],
                    fixed_certificate=fixed["certificate_issued"],
                )
            )
            if (index + 1) % 1024 == 0:
                progress(f"exhaustive_{alphabet}", index + 1, total - index - 1)
    for d in [16800, 5600]:
        for n in range(33):
            for e in range(n + 1):
                upper = bound(n, e, d)
                passed = bound(n + 1, e, d) <= upper and (e == n or bound(n, e + 1, d) >= upper)
                monotonic.append(
                    row(
                        f"monotonic/{d}/{n}/{e}",
                        passed,
                        accepted=n,
                        errors=e,
                        alpha_denominator=d,
                        risk_upper=upper,
                    )
                )
        # Only adjacent states around the integer threshold can change a decision.
        for n in range(961):
            last = -1
            while last + 1 <= n and passes(n, last + 1, d):
                last += 1
            checks = sorted({0, n, max(0, last), min(n, last + 1)})
            agrees = all(passes(n, e, d) == (bound(n, e, d) <= 0.05 and n > 0) for e in checks)
            monotonic.append(
                row(
                    f"rounding/{d}/{n}",
                    agrees,
                    accepted=n,
                    largest_passing_error=last,
                    checked_errors=checks,
                )
            )
            if n % 240 == 0:
                progress("exact_thresholds", n, 960 - n)
    rng = random.Random(SEED)
    for index in range(4096):
        stats, n, e, previous = [], 0, 0, 0
        for look in LOOKS:
            added = rng.randrange(look - previous + 1)
            n += added
            e += rng.randrange(min(added, [0, 1, 4, added][index % 4]) + 1)
            stats.append((n, e))
            previous = look
        early, refusal, fixed = (
            decide(stats),
            decide(stats, refusal=True),
            decide(stats, fixed=True),
        )
        randomized.append(
            row(
                f"random/{index}",
                early["certificate_issued"] == refusal["certificate_issued"]
                and (not fixed["certificate_issued"] or early["certificate_issued"]),
                statistics=stats,
                early=early,
                refusal=refusal,
                fixed=fixed,
            )
        )
        if (index + 1) % 512 == 0:
            progress("randomized_states", index + 1, 4096 - index - 1)
    unsafe_stats = [(120, 7), (240, 7), (480, 7), (960, 7)]
    future = decide(unsafe_stats, refusal=True)
    unsafe = dict(
        rule="refuse if current empirical error exceeds .05",
        statistics=unsafe_stats,
        lost_certificate=future["certificate_issued"],
        unsafe_stops_at=120,
        safe_stops_at=future["candidate_draws"],
    )
    cost_rows = []
    for last in [120, 960]:
        stops = [
            dict(policy=i, group=g, candidate_draws=240 if i < 6 else last)
            for g in frozen.base.GROUPS
            for i in range(7)
        ]
        ledger = costs(stops)
        paid = {
            (g, stage, j)
            for g in frozen.base.GROUPS
            for stage, count in [("selection", 240), ("candidate", max(240, last))]
            for j in range(count)
        }
        cost_rows.append(
            row(
                f"cost/{last}",
                ledger["shared_union_labels"] == len(paid),
                stops=stops,
                ledger=ledger,
                unique_request_count=len(paid),
            )
        )
    control = row("unsafe_control_detected", unsafe["lost_certificate"], **unsafe)
    rows = paths + monotonic + randomized + cost_rows + [control]
    return dict(
        rows=rows,
        exhaustive_path_rows=paths,
        bound_monotonicity_rows=monotonic,
        randomized_state_rows=randomized,
        cost_identity_rows=cost_rows,
        unsafe_control=unsafe,
        proof_obligations=dict(proof=PROOF, passed=all(x["metric"] == 1 for x in rows)),
        family_alpha_ledger=alpha_ledger(),
        numerical_training_calls=0,
        scientific_benefit_claim=False,
        proof_receipt_hash=canonical_hash(rows),
    )


def calibration_manifest() -> Json:
    """Freeze forty new independent draw streams without opening any new labels."""
    builder = namespace(streams, "finite8906_draws")
    builder.SEED = SEED
    return dict(builder.draws(1024, replicates=40))
