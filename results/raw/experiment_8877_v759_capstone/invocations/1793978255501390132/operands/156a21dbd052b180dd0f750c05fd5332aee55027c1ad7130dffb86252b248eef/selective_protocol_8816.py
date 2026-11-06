"""REQ-REPORT-8816: freeze selective decisions before untouched evaluation labels open."""

from collections import defaultdict
import random
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
SEEDS = [7558818, 7558819, 7558820, 7558821, 7558822]
MENU = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ARMS = [
    "local_record",
    "evidence_set",
    "joint_symbolic",
    "fixed_evidence",
    "agreement_only",
    "learned_energy",
]


def protocol() -> Json:
    """A fixed risk question cannot retroactively rescue an earlier conjunctive gate."""
    return dict(
        threshold_menu=MENU,
        seeds=SEEDS,
        arms=ARMS,
        calibration_selection_rule="per arm and seed: maximize useful coverage among safe thresholds; ties lowest threshold; no qualifying threshold => no_candidate before evaluation",
        comparator_selection="Exp8818 strongest of local_record/evidence_set/joint_symbolic by calibration correct decisive coverage at safe threshold; ties listed order; freeze before evaluation",
        estimands=dict(
            exp8818=dict(
                correct_decisive_coverage_gain_lower=0.05,
                semantic_false_accept_noninferiority_upper=0.01,
                comparator="strongest prespecified budgeted comparator",
                denominators=dict(
                    correct_decisive_coverage="all scheduled units",
                    semantic_false_accept_rate="truth != support",
                ),
            ),
            exp8819=dict(
                endpoints=[
                    "unconditional_brier",
                    "wrong_decision_rate",
                    "decisive_coverage",
                    "insufficient_to_decision_rate",
                ],
                risk_upper=0.05,
                coverage_lower=0.30,
                useful_coverage_gain_lower=0.05,
                comparator="fixed_evidence at its own frozen safe threshold",
                denominators=dict(
                    brier="all scheduled candidates",
                    wrong_decision_rate="non-escalations",
                    decisive_coverage="all scheduled candidates",
                    insufficient_to_decision_rate="insufficient-truth candidates",
                ),
            ),
        ),
        multiplicity_families=dict(
            exp8818=dict(endpoints=2, confidence=0.975),
            exp8819=dict(endpoints=4, confidence=0.9875),
        ),
        resampling=dict(
            unit="whole world",
            draws=2000,
            seed=7558816,
            nesting="average seeds within each world before paired world resampling",
            interpretation="empirical constructed-population intervals; no distribution-free certification",
        ),
        budget=dict(
            paid_primitive="Python line dispatch inside unchanged solvers, including checks and aggregation",
            limit_per_unit=200000,
            equal_paid_checks=True,
            exhaustion="escalate and retain scheduled denominator",
            exact_oracle="separate reference costs; excluded from paid comparators",
        ),
        matched_coverage="calibration freeze hash-ordered thinning per arm/seed to smallest safe-arm coverage; replay same hash cutoff on evaluation; report achieved coverage and risk",
        label_custody=dict(
            evaluation="private until complete prediction rows seal",
            canary="disjoint private labels; no fitting or threshold selection",
            fit="release only fit labels",
            calibration="release only calibration labels after fit parameters seal",
        ),
        evaluation_selection_forbidden=True,
        insufficient_target="escalate for every candidate, including insufficient",
        v754_outcomes="unchanged nulls; zero false-accept headroom is not useful coverage",
    )


def action(candidate: str, truth: str) -> str:
    """A correct rejection is not a useful decision when the evidence is insufficient."""
    return "escalate" if truth == "insufficient" else "accept" if candidate == truth else "reject"


def interval(values: list[float], confidence: float = 0.9875) -> list[Any]:
    """Resample independent world means; seeds and nuisance views are never new worlds."""
    if not values:
        return [None, None]
    rng = random.Random(7558816)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    tail = int(2000 * (1 - confidence) / 2)
    return [draws[tail], draws[-tail - 1]]


def metrics(rows: list[Json], threshold: float) -> Json:
    """Keep insufficient decisions and all abstentions in their named denominators."""
    wrong = decisive = correct = unknown_decisions = unknown = false_accept = unsupported = 0
    brier = 0.0
    for row in rows:
        probabilities = row["probabilities"]
        index = max(range(3), key=lambda i: probabilities[i])
        decision = (
            ["accept", "reject", "escalate"][index]
            if probabilities[index] >= threshold
            else "escalate"
        )
        target = action(row["candidate"], row["truth"])
        brier += sum(
            (v - int(i == ["accept", "reject", "escalate"].index(target))) ** 2
            for i, v in enumerate(probabilities)
        )
        decisive += decision != "escalate"
        correct += decision != "escalate" and decision == target
        wrong += decision != "escalate" and decision != target
        unknown += row["truth"] == "insufficient"
        unknown_decisions += row["truth"] == "insufficient" and decision != "escalate"
        unsupported += row["truth"] != "support"
        false_accept += (
            row["truth"] != "support" and row["candidate"] == "support" and decision == "accept"
        )
    n = len(rows)
    return dict(
        denominator=n,
        non_escalations=decisive,
        insufficient_denominator=unknown,
        false_accept_denominator=unsupported,
        wrong_decisions=wrong,
        brier=brier / n if n else None,
        wrong_decision_rate=wrong / decisive if decisive else None,
        decisive_coverage=decisive / n if n else 0.0,
        correct_decisive_coverage=correct / n if n else 0.0,
        insufficient_to_decision_rate=unknown_decisions / unknown if unknown else None,
        semantic_false_accept_rate=false_accept / unsupported if unsupported else None,
    )


def select(rows: list[Json], arm: str, seed: int) -> Json:
    """Retain the full calibration menu and freeze one eligible threshold without evaluation."""
    if any(r["split"] != "calibration" for r in rows):
        raise ValueError("calibration_only")
    local = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
    worlds: dict[str, list[Json]] = defaultdict(list)
    for row in local:
        worlds[row["cluster_id"]].append(row)
    menu = []
    for threshold in MENU:
        observed = metrics(local, threshold)
        summaries = [metrics(group, threshold) for group in worlds.values()]
        coverage = interval([x["decisive_coverage"] for x in summaries])
        # Resample the ratio of wrong decisions to non-escalations as whole worlds.
        rng = random.Random(7558816)
        risk_draws = []
        for _ in range(2000 if summaries else 0):
            sample = rng.choices(summaries, k=len(summaries))
            n = sum(x["non_escalations"] for x in sample)
            risk_draws.append(sum(x["wrong_decisions"] for x in sample) / n if n else 1.0)
        risk = sorted(risk_draws)
        upper = risk[-13] if risk else None
        eligible = upper is not None and upper <= 0.05 and coverage[0] >= 0.30
        menu.append(
            dict(
                threshold=threshold,
                metrics=observed,
                risk_upper=upper,
                coverage_lower=coverage[0],
                eligible=eligible,
            )
        )
    safe = [x for x in menu if x["eligible"]]
    best = (
        max(safe, key=lambda x: (x["metrics"]["correct_decisive_coverage"], -x["threshold"]))
        if safe
        else None
    )
    return dict(
        arm=arm,
        seed=seed,
        status="selected" if best else "no_candidate",
        threshold=best["threshold"] if best else None,
        calibration_rows=menu,
        calibration_input_hash=canonical_hash(rows),
    )
