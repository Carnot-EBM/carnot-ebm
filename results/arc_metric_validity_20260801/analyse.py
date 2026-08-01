#!/usr/bin/env python3
"""Estimate whether held-out `change_fidelity` predicts plannability, and what does instead.

Reads `scored.json` (produced by `run.py`; no generated code is executed here -- every engine was
already run in its own killable subprocess) and writes `analysis.json`.

THE ORDER OF OPERATIONS IS PART OF THE METHOD. `min_reachable_two_sided_p` is computed from N and
k and recorded BEFORE the observed statistic, so a reader sees the floor the design allows before
seeing what it produced. When plannability is rare that floor can sit above 0.05, in which case
the honest report is "this corpus cannot answer the question" and no p-value is offered as
evidence in either direction -- which is a decision rule fixed in `preregistration.json`, not one
chosen after seeing the numbers.

CLUSTERING IS NOT A FOOTNOTE HERE. Engines induced for the same game share that game's window,
root, difficulty and goal structure, so a pooled AUC over ~160 engines from ~22 games has an
effective n nearer 22. Both the pooled and the WITHIN-GAME stratified estimates are reported, and
the within-game one is the decision-relevant estimate: the decision this metric would be used for
is "given several candidates induced for THE SAME game, pick one".
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PRIMARY_ROOT = "window_root"
RNG_SEED = 20260801
MC_DRAWS = 200000


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a positive scores above a negative) + 0.5 P(tie). Mann-Whitney U normalised.

    Ties are worth half deliberately: an engine that scores EXACTLY the same fidelity as another
    carries no information about which of them plans, and rounding that to a win in either
    direction would invent discrimination the metric does not have. Six of this corpus's
    highest-fidelity engines sit at exactly 1.0, so the tie rule is load-bearing, not cosmetic.
    """
    if not pos or not neg:
        return None
    # Vectorised, but the SAME quantity as the O(n*m) double loop it replaces: for each positive,
    # `searchsorted(left)` counts negatives strictly below it and `right - left` counts exact ties,
    # which are then weighted 0.5. Done this way because the cluster bootstrap calls this tens of
    # thousands of times per predictor and the naive form made the analysis pass the slow step.
    # `estimator_selftest` checks the result against scipy's independently-written U statistic on
    # every run, so this optimisation cannot silently change the number.
    ns = np.sort(np.asarray(neg, dtype=float))
    ps = np.asarray(pos, dtype=float)
    lo = np.searchsorted(ns, ps, side="left")
    hi = np.searchsorted(ns, ps, side="right")
    wins = float(lo.sum()) + 0.5 * float((hi - lo).sum())
    return wins / (len(pos) * len(neg))


def _ranksum(labels: list[int], values: list[float]) -> float:
    order = np.argsort(np.argsort(np.asarray(values, dtype=float)))
    # average ranks for ties
    vals = np.asarray(values, dtype=float)
    ranks = np.empty(len(vals), dtype=float)
    for v in np.unique(vals):
        m = vals == v
        ranks[m] = np.mean(order[m]) + 1.0
    return float(sum(r for r, lab in zip(ranks, labels, strict=False) if lab == 1))


def perm_p_two_sided(labels: list[int], values: list[float], seed: int = RNG_SEED) -> dict:
    """Exact when the label arrangements can be enumerated, Monte-Carlo otherwise.

    The statistic is the rank-sum of the positives, and the two-sided p is
    P(|T - E[T]| >= |t_obs - E[T]|) over all (or many) relabellings.
    """
    n, k = len(labels), sum(labels)
    if k == 0 or k == n:
        return {"p": None, "method": "degenerate_single_class", "n": n, "k": k}
    obs = _ranksum(labels, values)
    idx = list(range(n))
    total = math.comb(n, k)
    vals = np.asarray(values, dtype=float)
    order = np.argsort(np.argsort(vals))
    ranks = np.empty(n, dtype=float)
    for v in np.unique(vals):
        m = vals == v
        ranks[m] = np.mean(order[m]) + 1.0
    mean_t = k * float(np.mean(ranks))
    dev = abs(obs - mean_t)

    if total <= 500000:
        hits = 0
        for combo in itertools.combinations(idx, k):
            t = float(ranks[list(combo)].sum())
            if abs(t - mean_t) >= dev - 1e-12:
                hits += 1
        return {
            "p": hits / total,
            "method": "exact_enumeration",
            "n": n,
            "k": k,
            "n_arrangements": total,
        }
    rng = random.Random(seed)
    hits = 0
    for _ in range(MC_DRAWS):
        combo = rng.sample(idx, k)
        t = float(ranks[combo].sum())
        if abs(t - mean_t) >= dev - 1e-12:
            hits += 1
    return {
        "p": (hits + 1) / (MC_DRAWS + 1),
        "method": f"monte_carlo_{MC_DRAWS}",
        "n": n,
        "k": k,
        "n_arrangements": total,
    }


def min_reachable_two_sided_p(n: int, k: int) -> float | None:
    """The FLOOR this design can reach: the most extreme label arrangement possible.

    With k positives among n units there are C(n,k) arrangements; the single most extreme one
    carries mass 1/C(n,k) in each tail. Reporting this before the observed p is what stops a
    'p = 0.11, trending' reading of a design that could never have produced anything smaller.
    """
    if n <= 0 or k <= 0 or k >= n:
        return None
    return min(1.0, 2.0 / math.comb(n, k))


def estimator_selftest(seed: int = 12345, trials: int = 40) -> dict:
    """Check this file's OWN statistics against an independent implementation, every run.

    WHY THIS IS NOT OPTIONAL HERE. The headline of this artifact may be a NULL, and a null is
    exactly the result a silently-wrong estimator produces most convincingly: a permutation test
    with an off-by-one in its tail, or an AUC with its classes swapped, returns a comfortable
    "no association" and looks like a clean measurement. So the two load-bearing statistics are
    checked against `scipy.stats.mannwhitneyu(..., method="exact")` -- a separately-written
    implementation of the same quantity -- on randomly-generated small samples of the same shape
    as the real corpus, and the check runs on every invocation rather than once by hand.

    `auc` is checked via the identity U = AUC * n_pos * n_neg (scipy returns U for the first
    sample), which also catches a class swap: a flipped AUC gives 1 - AUC and fails.

    `min_reachable_two_sided_p` is checked at the EXTREME: when the positives take the k highest
    values, the exact two-sided p must equal the stated floor exactly. That is the one arrangement
    where "the smallest p this design can reach" is directly observable.
    """
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(seed)
    n_p_match = n_u_match = 0
    worst_p = worst_u = 0.0
    for _ in range(trials):
        n = int(rng.integers(8, 18))
        k = int(rng.integers(2, 5))
        vals = [round(float(v), 3) for v in rng.random(n)]
        labs = [0] * n
        for i in rng.choice(n, size=k, replace=False):
            labs[int(i)] = 1
        pos = [v for v, lb in zip(vals, labs, strict=True) if lb]
        neg = [v for v, lb in zip(vals, labs, strict=True) if not lb]
        mine = perm_p_two_sided(labs, vals)
        ref = mannwhitneyu(pos, neg, alternative="two-sided", method="exact")
        dp = abs(float(mine["p"]) - float(ref.pvalue))
        worst_p = max(worst_p, dp)
        n_p_match += dp < 1e-9
        du = abs(float(auc(pos, neg)) * len(pos) * len(neg) - float(ref.statistic))
        worst_u = max(worst_u, du)
        n_u_match += du < 1e-9

    # The extremal arrangement: positives take the k largest values, so the observed two-sided p
    # must land exactly on the design's floor.
    n, k = 14, 3
    vals = [float(i) for i in range(n)]
    labs = [0] * (n - k) + [1] * k
    extreme = perm_p_two_sided(labs, vals)
    floor = min_reachable_two_sided_p(n, k)
    return {
        "why": (
            "a wrong estimator returns a convincing null. Both load-bearing statistics are "
            "checked against scipy's independently-written exact Mann-Whitney on every run."
        ),
        "trials": trials,
        "perm_p_matches_scipy_exact": f"{n_p_match}/{trials}",
        "max_abs_p_difference": worst_p,
        "auc_matches_scipy_U_statistic": f"{n_u_match}/{trials}",
        "max_abs_U_difference": worst_u,
        "auc_check_also_catches_a_class_swap": (
            "a swapped AUC would give 1 - AUC and fail the U identity"
        ),
        "extremal_case": {
            "n": n,
            "k": k,
            "observed_two_sided_p": extreme["p"],
            "min_reachable_two_sided_p": floor,
            "equal": abs(float(extreme["p"]) - float(floor)) < 1e-12,
        },
        "all_pass": bool(
            n_p_match == trials
            and n_u_match == trials
            and abs(float(extreme["p"]) - float(floor)) < 1e-12
        ),
    }


def power_positive_control(
    rows: list[dict], effect_aucs: tuple[float, ...] = (0.65, 0.75, 0.85), seed: int = RNG_SEED
) -> dict:
    """Could this design have DETECTED an association of a given size, at this n and clustering?

    THIS IS THE CONTROL A NULL LIVES OR DIES BY, and it is deliberately a control of the
    INSTRUMENT rather than of the world. Pointing at a real rival predictor that happens to
    separate the classes is weaker evidence: if no rival happened to work, the null would be
    uninterpretable, and whether a rival works is a fact about ARC engines, not about whether this
    test can see. So a SYNTHETIC predictor with a KNOWN association to the real `plan_found`
    labels is injected -- same units, same game clustering, same class imbalance, same estimator,
    same cluster bootstrap -- and the fraction of replications whose game-clustered CI excludes
    0.5 is this design's empirical POWER at that effect size.

    Read it as: "an association of AUC x would have been detected p% of the time here." A null
    reported alongside high power at a decision-relevant effect size is evidence of absence. The
    same null alongside low power is only absence of evidence, and says so.

    The synthetic predictor is `label + N(0, sigma)`, with sigma solved from the closed form for
    two unit-variance-separated normals: AUC = Phi(d / sqrt(2)) where d = 1/sigma, hence
    sigma = 1 / (sqrt(2) * Phi^-1(AUC)). Realised AUC is reported next to the target so the
    calibration is checkable rather than asserted.
    """
    from scipy.stats import norm

    labels = [1 if r["plan_found"] else 0 for r in rows]
    games = [r["game"] for r in rows]
    if len(set(labels)) < 2:
        return {"ran": False, "why": "single class -- no labels to inject signal against"}
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "why": (
            "a null is only informative if the design could have seen a real effect. This injects "
            "a KNOWN effect into the real labels, clustering and class balance, and reports how "
            "often the same estimator detects it."
        ),
        "n_units": len(rows),
        "n_positive": sum(labels),
        "n_games": len(set(games)),
        "replications_per_effect": 200,
        "detection_rule": "game-clustered bootstrap 95% CI excludes 0.5",
        "by_effect": {},
    }
    for target in effect_aucs:
        sigma = 1.0 / (math.sqrt(2.0) * float(norm.ppf(target)))
        detected, realised = 0, []
        for _ in range(200):
            vals = [float(lb) + rng.normal(0.0, sigma) for lb in labels]
            pos = [v for v, lb in zip(vals, labels, strict=True) if lb]
            neg = [v for v, lb in zip(vals, labels, strict=True) if not lb]
            realised.append(auc(pos, neg))
            rr = [
                {"game": g, "plan_found": bool(lb), "v": v}
                for g, lb, v in zip(games, labels, vals, strict=True)
            ]
            ci = cluster_bootstrap_auc_ci(rr, "v", draws=2000, seed=int(rng.integers(1, 10**9)))
            if ci is not None and (min(ci) > 0.5 or max(ci) < 0.5):
                detected += 1
        out["by_effect"][f"auc_{target}"] = {
            "target_auc": target,
            "mean_realised_auc": round(float(np.mean(realised)), 4),
            "detection_rate": round(detected / 200, 3),
        }
    powers = {k: v["detection_rate"] for k, v in out["by_effect"].items()}
    out["power_at_auc_0.75"] = powers.get("auc_0.75")
    out["passed"] = bool((powers.get("auc_0.75") or 0.0) >= 0.8)
    out["passed_means"] = (
        "the design detects a moderate (AUC 0.75) association at least 80% of the time, so a "
        "measured null here is evidence of absence rather than absence of evidence."
    )
    return out


def family_max_statistic_p(rows: list[dict], keys: list[str], seed: int = RNG_SEED) -> dict:
    """FWER-adjusted p for the BEST predictor, because the best was picked from a family.

    WHY THIS IS MANDATORY AND NOT A REFINEMENT. This run scores a dozen-plus candidate predictors
    against one outcome and then reports the one furthest from chance. Under a global null where
    NONE of them is associated, the MAXIMUM of a dozen sample AUCs is far from 0.5 by construction
    -- so quoting that winner's own p-value would be reporting a selection effect as a discovery,
    and the actionable half of this run ("what should the roster be rebuilt around") is exactly
    where that error would do damage.

    The max-statistic permutation handles it exactly and without a Bonferroni approximation: under
    the null, relabel and recompute EVERY predictor, take the maximum |AUC - 0.5| across the whole
    family, and compare the observed maximum to that distribution. It automatically accounts for
    the predictors being correlated with each other (several here are different views of the same
    state-graph shape), where Bonferroni would be needlessly conservative.

    Labels are permuted WITHIN GAME, matching the stratified estimate: a between-game shuffle
    would test a null nobody is interested in (that games differ in difficulty) and would be
    anticonservative for the within-game claim.
    """
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)
    usable = {
        g: rs
        for g, rs in by_game.items()
        if any(r["plan_found"] for r in rs) and any(not r["plan_found"] for r in rs)
    }
    if not usable:
        return {"p": None, "why": "no game has both classes"}

    # Only predictors defined on EVERY row of the usable games can enter the family; a predictor
    # with missing values would otherwise change the effective sample between permutations.
    fam = [k for k in keys if all(r.get(k) is not None for rs in usable.values() for r in rs)]
    if not fam:
        return {"p": None, "why": "no predictor is defined on every usable row"}

    vals = {k: {g: [float(r[k]) for r in rs] for g, rs in usable.items()} for k in fam}
    labs = {g: [bool(r["plan_found"]) for r in rs] for g, rs in usable.items()}

    def signed_for(key: str, assign: dict[str, list[bool]]) -> float:
        """The SIGNED within-game weighted AUC. Signed, not absolute, because direction is the
        whole point here: a predictor that lands FAR BELOW 0.5 is strongly associated with the
        outcome and associated the WRONG WAY -- higher score, less plannable. Collapsing that to a
        distance and then reporting `0.5 + distance` would print it as a strong POSITIVE predictor,
        which is the opposite of what it is. (That is not hypothetical: the first version of this
        function did exactly that and reported `change_fidelity` at 0.7857 when its actual
        within-game AUC was 0.2143.)"""
        num = den = 0.0
        for g, lb in assign.items():
            vv = vals[key][g]
            pos = [v for v, x in zip(vv, lb, strict=True) if x]
            neg = [v for v, x in zip(vv, lb, strict=True) if not x]
            a = auc(pos, neg)
            if a is None:
                continue
            w = len(pos) * len(neg)
            num += a * w
            den += w
        return num / den if den else 0.5

    def stat_for(key: str, assign: dict[str, list[bool]]) -> float:
        """Distance from chance. The two-sided test statistic; direction is carried separately."""
        return abs(signed_for(key, assign) - 0.5)

    observed = {k: stat_for(k, labs) for k in fam}
    observed_signed = {k: signed_for(k, labs) for k in fam}
    obs_max_key = max(observed, key=lambda k: observed[k])
    obs_max = observed[obs_max_key]

    rng = random.Random(seed)
    draws = 5000
    hits = 0
    per_key_hits = dict.fromkeys(fam, 0)
    for _ in range(draws):
        assign = {}
        for g, lb in labs.items():
            perm = list(lb)
            rng.shuffle(perm)
            assign[g] = perm
        stats = {k: stat_for(k, assign) for k in fam}
        if max(stats.values()) >= obs_max - 1e-12:
            hits += 1
        for k in fam:
            if stats[k] >= observed[k] - 1e-12:
                per_key_hits[k] += 1
    return {
        "why": (
            "the best predictor was SELECTED from a family, so its own p-value is a selection "
            "effect. This is the exact family-wise adjustment, handling the predictors' mutual "
            "correlation without a Bonferroni approximation."
        ),
        "family": sorted(fam),
        "family_size": len(fam),
        "permutation_scheme": "labels shuffled WITHIN game, matching the stratified estimate",
        "draws": draws,
        "strongest_association_in_family": obs_max_key,
        "its_signed_within_game_auc": round(observed_signed[obs_max_key], 4),
        "its_direction": (
            "higher score -> MORE plannable"
            if observed_signed[obs_max_key] > 0.5
            else "higher score -> LESS plannable (associated the WRONG WAY)"
        ),
        "signed_within_game_auc_per_predictor": {
            k: round(observed_signed[k], 4) for k in sorted(fam)
        },
        "fwer_adjusted_p_for_strongest": (hits + 1) / (draws + 1),
        "unadjusted_p_per_predictor": {k: (per_key_hits[k] + 1) / (draws + 1) for k in sorted(fam)},
        "reading": (
            "fwer_adjusted_p_for_strongest is the probability that a family of this size and "
            "correlation structure would produce a winner this far from chance when NOTHING is "
            "associated. Compare THAT to 0.05, never the winner's own unadjusted p."
        ),
    }


def point_biserial(labels: list[int], values: list[float]) -> dict:
    x = np.asarray(labels, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(set(labels)) < 2 or float(np.std(y)) == 0.0:
        return {"r": None, "why": "no variance in one side"}
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    out: dict[str, Any] = {"r": round(r, 4), "n": n}
    if n > 3 and abs(r) < 1.0:
        z = 0.5 * math.log((1 + r) / (1 - r))
        se = 1.0 / math.sqrt(n - 3)
        lo, hi = z - 1.96 * se, z + 1.96 * se
        out["fisher_ci95"] = [round(math.tanh(lo), 4), round(math.tanh(hi), 4)]
        out["fisher_ci95_caveat"] = (
            "assumes independent units; engines cluster by game, so this interval is too narrow"
        )
    return out


def logistic_slope(rows: list[dict], value_key: str, seed: int = RNG_SEED) -> dict:
    """Unpenalised logistic slope of plan_found on the predictor, with a CLUSTER-bootstrap CI.

    Reported because a slope is the form a reader expects, but it is the WEAKEST of the three
    estimates here and is labelled so. With few positives an unpenalised fit is prone to quasi-
    complete separation, where the maximum-likelihood coefficient diverges and the optimiser
    stops wherever its tolerance runs out -- a large slope then means "the classes are separable",
    not "the effect is large". `separation_suspected` fires on that; when it does, read the AUC.

    The CI resamples GAMES, not engines, for the same reason every other interval here does.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:  # pragma: no cover - sklearn is present in this venv
        return {"slope": None, "why": "sklearn unavailable"}

    def fit(rs: list[dict]) -> float | None:
        y = np.asarray([1 if r["plan_found"] else 0 for r in rs])
        if len(set(y.tolist())) < 2:
            return None
        x = np.asarray([[float(r[value_key])] for r in rs])
        if float(np.std(x)) == 0.0:
            return None
        try:
            # C=inf, not penalty=None: sklearn 1.8 deprecated the latter. Same model --
            # an UNPENALISED fit, which is what a reported slope has to be for its magnitude to
            # mean anything. A default-regularised fit would shrink the coefficient toward zero
            # and quietly bias this run's headline toward "no association".
            m = LogisticRegression(C=float("inf"), max_iter=5000).fit(x, y)
            return float(m.coef_[0][0])
        except Exception:  # noqa: BLE001
            return None

    slope = fit(rows)
    if slope is None:
        return {"slope": None, "why": "degenerate (single class or no predictor variance)"}
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)
    games = sorted(by_game)
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        pool: list[dict] = []
        for g in (rng.choice(games) for _ in games):
            pool.extend(by_game[g])
        s = fit(pool)
        if s is not None:
            draws.append(s)
    draws.sort()
    ci = (
        [round(draws[int(0.025 * len(draws))], 4), round(draws[int(0.975 * len(draws)) - 1], 4)]
        if len(draws) >= 100
        else None
    )
    return {
        "slope": round(slope, 4),
        "odds_ratio_per_unit_fidelity": round(float(math.exp(slope)), 4)
        if abs(slope) < 50
        else None,
        "cluster_bootstrap_ci95_game_resample": ci,
        "n_bootstrap_fits": len(draws),
        "separation_suspected": bool(abs(slope) > 10),
        "read_the_auc_instead_if_separation_suspected": True,
    }


def bootstrap_auc_ci(
    pos: list[float], neg: list[float], draws: int = 20000, seed: int = RNG_SEED
) -> list[float] | None:
    if not pos or not neg:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        p = [rng.choice(pos) for _ in pos]
        q = [rng.choice(neg) for _ in neg]
        a = auc(p, q)
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]


def cluster_bootstrap_auc_ci(
    rows: list[dict], value_key: str, draws: int = 20000, seed: int = RNG_SEED
) -> list[float] | None:
    """Resample GAMES with replacement, not engines. The honest interval.

    Engines within a game are not independent draws -- they share the window, the root and the
    goal. Resampling engines pretends there are ~160 independent units when there are ~22.
    """
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)
    games = sorted(by_game)
    if len(games) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        pick = [rng.choice(games) for _ in games]
        pool: list[dict] = []
        for g in pick:
            pool.extend(by_game[g])
        pos = [r[value_key] for r in pool if r["plan_found"]]
        neg = [r[value_key] for r in pool if not r["plan_found"]]
        a = auc(pos, neg)
        if a is not None:
            vals.append(a)
    if len(vals) < 100:
        return None
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]


def within_game_auc(rows: list[dict], value_key: str) -> dict:
    """The decision-relevant estimate: among candidates for THE SAME game, does higher score
    mean more plannable? Only games containing both classes can contribute."""
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)
    contributions, per_game = [], {}
    for g, rs in sorted(by_game.items()):
        pos = [r[value_key] for r in rs if r["plan_found"]]
        neg = [r[value_key] for r in rs if not r["plan_found"]]
        if not pos or not neg:
            continue
        a = auc(pos, neg)
        w = len(pos) * len(neg)
        per_game[g] = {"auc": round(a, 4), "n_pos": len(pos), "n_neg": len(neg), "n_pairs": w}
        contributions.append((a, w))
    if not contributions:
        return {
            "auc": None,
            "n_informative_games": 0,
            "why": (
                "no game contains BOTH a plannable and an unplannable engine, so no within-game "
                "comparison exists. Any pooled association is then entirely BETWEEN games -- it "
                "measures which GAMES are plannable, not which ENGINES."
            ),
            "per_game": per_game,
        }
    num = sum(a * w for a, w in contributions)
    den = sum(w for _, w in contributions)
    return {
        "auc": round(num / den, 4),
        "n_informative_games": len(contributions),
        "n_concordant_pairs_total": den,
        "per_game": per_game,
    }


def stratified_perm_p(rows: list[dict], value_key: str, seed: int = RNG_SEED) -> dict:
    """Permute plan_found labels WITHIN game only. Tests the within-game claim without borrowing
    any power from between-game differences in difficulty."""
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)
    usable = {
        g: rs
        for g, rs in by_game.items()
        if any(r["plan_found"] for r in rs) and any(not r["plan_found"] for r in rs)
    }
    if not usable:
        return {"p": None, "why": "no game has both classes"}

    def stat(assign: dict[str, list[bool]]) -> float:
        num = den = 0.0
        for g, rs in usable.items():
            labs = assign[g]
            pos = [r[value_key] for r, lb in zip(rs, labs, strict=False) if lb]
            neg = [r[value_key] for r, lb in zip(rs, labs, strict=False) if not lb]
            a = auc(pos, neg)
            if a is None:
                continue
            w = len(pos) * len(neg)
            num += a * w
            den += w
        return num / den if den else 0.5

    obs_assign = {g: [bool(r["plan_found"]) for r in rs] for g, rs in usable.items()}
    obs = stat(obs_assign)
    rng = random.Random(seed)
    # 20000 rather than more: this runs once per predictor across ~31 predictor/subset
    # combinations, and 20000 draws already resolve p to about +/-0.7% -- finer than any decision
    # here turns on, and far coarser than the multiplicity correction that dominates.
    draws = 20000
    hits = 0
    for _ in range(draws):
        assign = {}
        for game_key, labs in obs_assign.items():
            perm = list(labs)
            rng.shuffle(perm)
            assign[game_key] = perm
        if abs(stat(assign) - 0.5) >= abs(obs - 0.5) - 1e-12:
            hits += 1
    n_arr = 1
    for labs in obs_assign.values():
        n_arr *= math.comb(len(labs), sum(labs))
    return {
        "p": (hits + 1) / (draws + 1),
        "observed_within_game_auc": round(obs, 4),
        "method": f"within_game_label_permutation_{draws}",
        "n_arrangements": n_arr,
        "min_reachable_two_sided_p": round(min(1.0, 2.0 / n_arr), 6) if n_arr > 1 else None,
    }


# ---------------------------------------------------------------------------------------------
# row extraction
# ---------------------------------------------------------------------------------------------
def flatten(engines: list[dict], root: str) -> list[dict]:
    """One analysis row per engine, with every predictor pulled to the top level.

    An engine is ANALYSABLE only if it ran AND its held-out slice can grade change at all
    (`n_changing > 0`) AND the plan outcome is determined. Each exclusion is counted and named:
    `change_fidelity` returns 0.0 when there is nothing to grade, and 0.0 is also the value
    meaning 'got every change wrong', so admitting those would enter a NOT-MEASURED as a measured
    worst score.
    """
    rows, excluded = [], []
    for e in engines:
        cell = e.get("cell")
        base = {
            "cell": cell,
            "corpus": e.get("corpus"),
            "game": e.get("game"),
            "arm": e.get("arm"),
            "status": e.get("status"),
        }
        if e.get("status") != "ok":
            excluded.append({**base, "reason": f"engine_status:{e.get('status')}"})
            continue
        h = e.get("heldout") or {}
        ins = e.get("in_sample") or {}
        p = ((e.get("plan") or {}).get(root)) or {}
        g = ((e.get("goal_gate") or {}).get(root)) or {}
        sg = ((e.get("state_graph") or {}).get(root)) or {}
        if "plan_found" not in p:
            excluded.append({**base, "reason": "plan_outcome_undetermined"})
            continue
        if int(h.get("n_changing") or 0) <= 0:
            excluded.append(
                {
                    **base,
                    "reason": "heldout_has_no_changing_transition_so_change_fidelity_unmeasured",
                    "plan_found": bool(p.get("plan_found")),
                }
            )
            continue
        rows.append(
            {
                **base,
                "plan_found": bool(p.get("plan_found")),
                "plan_length": p.get("plan_length"),
                "plan_termination": (p.get("diagnostics") or {}).get("termination_reason"),
                "plan_nodes_expanded": (p.get("diagnostics") or {}).get("nodes_expanded"),
                "change_fidelity": float(h.get("change_fidelity")),
                "heldout_accuracy": float(h.get("accuracy")),
                "heldout_cell_recall": float(h.get("cell_recall")),
                "heldout_n_changing": int(h.get("n_changing")),
                "heldout_invented_changed_cells": int(h.get("invented_changed_cells") or 0),
                "in_sample_accuracy": float(ins.get("accuracy") or 0.0),
                "in_sample_change_fidelity": float(ins.get("change_fidelity") or 0.0),
                "goal_satisfiable": bool(g.get("satisfiable")),
                "goal_kind": g.get("kind"),
                "goal_first_true_depth": g.get("first_true_depth"),
                "goal_reachable_grids_evaluated": g.get("reachable_grids_evaluated"),
                "shipped_gate_trust_energy": e.get("shipped_gate_trust_energy"),
                "shipped_gate_heldout_accuracy": e.get("shipped_gate_heldout_accuracy"),
                "shipped_gate_passes": e.get("shipped_gate_passes"),
                "engine_changes_anything_at_root": bool(sg.get("engine_changes_anything_at_root")),
                "n_distinct_successors_at_root": sg.get("n_distinct_successors_at_root"),
                "n_distinct_changing_successors_at_root": sg.get(
                    "n_distinct_changing_successors_at_root"
                ),
                "probe_distinct_states": sg.get("probe_distinct_states"),
                "probe_mean_new_states_per_expansion": sg.get(
                    "probe_mean_new_states_per_expansion"
                ),
                "probe_depth_reached": sg.get("probe_depth_reached"),
            }
        )
    return rows, excluded  # type: ignore[return-value]


# PREDICTORS THAT ARE THE OUTCOME WEARING A DIFFERENT NAME. These are reported -- suppressing
# them would hide a real structural fact -- but they are barred from being called a "rival
# predictor" or from serving as the positive control, because they do not predict plannability,
# they RE-COMPUTE it.
#
# `_goal_satisfiability_check` is a BFS from the root over `_probe_candidates` (which delegates
# to `_model_candidates`), deduplicated, bounded by the SAME `max_nodes` and -- since 2026-07-31,
# deliberately, via the shared `plan_max_depth_default()` resolver -- the SAME `max_depth`, and
# it stops at the first state where `is_level_complete` is True. `plan_in_model` is the same BFS
# that additionally carries the path. The resolver's own docstring states the sharing is a
# soundness requirement rather than tidiness. So `goal_satisfiable` scoring AUC ~1.0 against
# `plan_found` is arithmetic, not a finding, and treating it as a discovered predictor would be
# exactly the circularity the Circularity / Oracle-Distinctness Discipline exists to catch.
# (Their small residual disagreements are real but incidental: the gate keys `seen` on
# `to_ascii` while the planner uses `_state_key`, and the gate has a `goal_predicate_true_at_root`
# rejection the planner does not, since the planner only tests successors.)
DEFINITIONALLY_COUPLED_TO_OUTCOME = {
    "goal_satisfiable",
    "goal_first_true_depth",
    "goal_reachable_grids_evaluated",
}

RIVALS = [
    ("change_fidelity", "THE PRIMARY: held-out change fidelity (what the A/B moved)"),
    (
        "heldout_accuracy",
        "held-out exact full-grid match -- the floored metric cf was adopted to escape",
    ),
    ("heldout_cell_recall", "held-out changed-cell recall"),
    ("in_sample_accuracy", "in-sample exact accuracy (free, no held-out split needed)"),
    ("in_sample_change_fidelity", "in-sample change fidelity"),
    ("goal_first_true_depth", "how deep the goal predicate first becomes true (None if never)"),
    ("goal_reachable_grids_evaluated", "how much distinct state the goal gate could reach"),
    ("shipped_gate_trust_energy", "the shipped trust gate's own energy"),
    ("shipped_gate_heldout_accuracy", "the shipped gate's internal held-out accuracy"),
    ("n_distinct_successors_at_root", "branching at the root"),
    ("n_distinct_changing_successors_at_root", "branching at the root, changing states only"),
    ("probe_distinct_states", "distinct states a bounded forward probe reaches"),
    ("probe_mean_new_states_per_expansion", "path (~1) vs tree (>>1) vs inert (0)"),
    ("probe_depth_reached", "how deep the bounded probe got"),
    ("engine_changes_anything_at_root", "does ANY action change the root state"),
    ("goal_satisfiable", "the shipped goal gate's verdict"),
    ("shipped_gate_passes", "does the engine clear the shipped trust gate"),
]


def score_predictor(rows: list[dict], key: str) -> dict:
    usable = [r for r in rows if r.get(key) is not None]
    n_missing = len(rows) - len(usable)
    vals = [float(r[key]) for r in usable]
    labs = [1 if r["plan_found"] else 0 for r in usable]
    pos = [v for v, lb in zip(vals, labs, strict=False) if lb]
    neg = [v for v, lb in zip(vals, labs, strict=False) if not lb]
    out: dict[str, Any] = {
        "n_scored": len(usable),
        "n_missing_value": n_missing,
        "n_plannable": len(pos),
        "n_unplannable": len(neg),
        "auc_pooled": None if auc(pos, neg) is None else round(auc(pos, neg), 4),
        "mean_when_plannable": round(float(np.mean(pos)), 4) if pos else None,
        "mean_when_unplannable": round(float(np.mean(neg)), 4) if neg else None,
    }
    if pos and neg:
        out["min_reachable_two_sided_p"] = round(
            min_reachable_two_sided_p(len(usable), len(pos)) or 1.0, 8
        )
        out["perm_p_pooled"] = perm_p_two_sided(labs, vals)
        out["bootstrap_ci95_engine_resample"] = bootstrap_auc_ci(pos, neg)
        rr = [
            {"game": r["game"], "plan_found": r["plan_found"], "v": float(r[key])} for r in usable
        ]
        out["cluster_bootstrap_ci95_game_resample"] = cluster_bootstrap_auc_ci(rr, "v")
        out["within_game"] = within_game_auc(
            [
                {"game": r["game"], "plan_found": r["plan_found"], "v": float(r[key])}
                for r in usable
            ],
            "v",
        )
        out["within_game_perm_p"] = stratified_perm_p(
            [
                {"game": r["game"], "plan_found": r["plan_found"], "v": float(r[key])}
                for r in usable
            ],
            "v",
        )
        out["point_biserial"] = point_biserial(labs, vals)
        out["logistic"] = logistic_slope(rr, "v")
    return out


def main() -> int:  # noqa: C901, PLR0915
    scored = json.loads((HERE / "scored.json").read_text())
    engines = scored["engines"]
    rows, excluded = flatten(engines, PRIMARY_ROOT)

    # ---- who is plannable at all -----------------------------------------------------------
    n_pos = sum(1 for r in rows if r["plan_found"])
    n = len(rows)
    floor = min_reachable_two_sided_p(n, n_pos)

    analysis: dict[str, Any] = {
        "estimator_selftest": estimator_selftest(),
        "what_this_is": (
            "Does held-out change_fidelity predict whether plan_in_model finds a plan? Estimated "
            "over every frozen induced engine that can be scored on both sides, at the CURRENT "
            "shipped search defaults."
        ),
        "primary_root": PRIMARY_ROOT,
        "corpus": {
            "n_engines_in_scored_json": len(engines),
            "n_analysable": n,
            "n_excluded": len(excluded),
            "exclusion_census": _census(excluded, "reason"),
            "n_games": len(sorted({r["game"] for r in rows})),
            "by_corpus": _census(rows, "corpus"),
        },
        # STATED BEFORE THE OBSERVED STATISTIC, per preregistration.json.
        "power_floor_stated_before_the_association": {
            "n_analysable": n,
            "n_plannable": n_pos,
            # NOT rounded. With ~150 units and ~30 positives the floor is astronomically small
            # (order 1e-30), and `round(x, 10)` prints it as 0.0 -- which reads as "the design has
            # no floor at all" instead of "the floor is negligible". Two different statements, and
            # the rounded one is the misleading one.
            "min_reachable_two_sided_p": floor,
            "reading": (
                "the smallest two-sided p this design could produce even if every plannable "
                "engine outscored every unplannable one. A p above this is not 'trending'; a "
                "floor above 0.05 means the corpus cannot establish the association at all."
            ),
        },
    }

    if n_pos == 0 or n_pos == n:
        analysis["verdict"] = {
            "association": "UNESTABLISHED -- single class",
            "detail": f"{n_pos} of {n} analysable engines are plannable; no contrast exists.",
        }

    # ---- the primary + every rival, same machinery ------------------------------------------
    analysis["predictors"] = {}
    for key, why in RIVALS:
        entry = {"what_it_is": why, **score_predictor(rows, key)}
        if key in DEFINITIONALLY_COUPLED_TO_OUTCOME:
            entry["definitionally_coupled_to_outcome"] = True
            entry["why_this_is_not_a_rival_predictor"] = (
                "the shipped goal gate is the SAME bounded BFS from the SAME root over the SAME "
                "candidate generator under the SAME max_nodes and (since the shared "
                "plan_max_depth_default resolver) the SAME max_depth as plan_in_model, stopping "
                "at the first state where is_level_complete is True. It does not predict "
                "plannability, it recomputes it. Reported for completeness; barred from the "
                "rival ranking and from the positive control."
            )
            if key == "goal_first_true_depth":
                entry["and_it_is_undefined_exactly_when_the_outcome_is_negative"] = (
                    "goal-predicate reachability depth was named as a candidate predictor, and "
                    "it cannot be one: the gate only reports first_true_depth when it REACHED "
                    "the goal, so the field is null on precisely the engines that fail to plan. "
                    "Its n_scored below is therefore almost all positives, and any AUC computed "
                    "on the survivors is conditioned on the outcome. A usable version would have "
                    "to be a depth-to-goal ESTIMATE defined for unreachable goals too."
                )
        else:
            entry["definitionally_coupled_to_outcome"] = False
        analysis["predictors"][key] = entry
    analysis["definitionally_coupled_predictors"] = sorted(DEFINITIONALLY_COUPLED_TO_OUTCOME)

    # ---- prespecified falsification checks --------------------------------------------------
    no_tn36 = [r for r in rows if r["game"] != "tn36"]
    analysis["driver_check_tn36_removed"] = {
        "why": (
            "all six perfect-fidelity engines in the earlier 40-candidate join were tn36 progress-"
            "BAR TICKERS -- they model the status indicator exactly and the playfield not at all. "
            "If the association only exists with tn36 in, it is that artifact."
        ),
        "n_analysable_without_tn36": len(no_tn36),
        "n_plannable_without_tn36": sum(1 for r in no_tn36 if r["plan_found"]),
        "change_fidelity": score_predictor(no_tn36, "change_fidelity"),
    }

    perfect = sorted(
        [r for r in rows if r["change_fidelity"] >= 0.999],
        key=lambda r: (r["game"], r["cell"]),
    )
    analysis["degeneracy_audit_of_the_top_of_the_metric"] = {
        "why": (
            "a metric is disqualified if it ranks a non-model at the top. These are every engine "
            "the primary scores at >= 0.999."
        ),
        "n_at_or_above_0999": len(perfect),
        "n_of_those_plannable": sum(1 for r in perfect if r["plan_found"]),
        "games": _census(perfect, "game"),
        "rows": [
            {
                k: r[k]
                for k in (
                    "cell",
                    "game",
                    "corpus",
                    "plan_found",
                    "change_fidelity",
                    "heldout_accuracy",
                    "heldout_n_changing",
                    "engine_changes_anything_at_root",
                    "n_distinct_changing_successors_at_root",
                    "probe_mean_new_states_per_expansion",
                    "goal_kind",
                )
            }
            for r in perfect
        ],
    }

    # ---- THE OUTCOME HAS A DEGENERATE SUBCLASS, FOUND WHILE SCORING --------------------------
    # `plan_in_model` tests `is_level_complete` on SUCCESSORS ONLY -- never on the start grid.
    # So an engine whose induced goal predicate is TRUE AT THE ROOT (the most degenerate predicate
    # expressible, e.g. `lambda g: True`) yields a length-1 "plan" to the very first state the
    # search generates. The shipped goal gate catches exactly this and returns
    # `goal_predicate_true_at_root` -- and the LIVE pipeline runs that gate BEFORE the planner, so
    # those engines never reach a planner in production.
    #
    # That makes them false positives of the OUTCOME variable, not of the exposure. Left in, they
    # add plannable rows whose plannability says nothing about the world model at all. So the
    # primary is re-estimated with the live pipeline's own ordering applied: a root-true goal is a
    # veto, and the engine counts as unplannable.
    #
    # THIS IS POST-HOC. It was found by reading the scored rows, not anticipated in
    # `preregistration.json`, and is reported as a labelled sensitivity beside the preregistered
    # primary rather than replacing it.
    degen = [r for r in rows if r.get("goal_kind") == "goal_predicate_true_at_root"]
    gated_rows = [
        {
            **r,
            "plan_found": (r["plan_found"] and r.get("goal_kind") != "goal_predicate_true_at_root"),
        }
        for r in rows
    ]
    analysis["degenerate_goal_predicate_audit"] = {
        "why": (
            "plan_in_model never evaluates is_level_complete on the start grid, so a goal "
            "predicate that is true at the root produces a length-1 plan to the first state the "
            "search touches. The shipped goal gate rejects exactly that case, and the live "
            "pipeline runs the gate BEFORE the planner -- so these engines are false positives of "
            "the outcome variable, and would never be planned with in production."
        ),
        "post_hoc": True,
        "n_with_root_true_goal": len(degen),
        "n_of_those_counted_plannable_by_the_raw_planner": sum(1 for r in degen if r["plan_found"]),
        "their_plan_lengths": sorted(
            {r.get("plan_length") for r in degen if r["plan_found"] and r.get("plan_length")}
        ),
        "n_plannable_before_gating": sum(1 for r in rows if r["plan_found"]),
        "n_plannable_after_gating": sum(1 for r in gated_rows if r["plan_found"]),
        "primary_re_estimated_with_the_gate_applied_first": score_predictor(
            gated_rows, "change_fidelity"
        ),
        "best_rival_re_estimated": {
            k: score_predictor(gated_rows, k)
            for k, _w in RIVALS
            if k not in DEFINITIONALLY_COUPLED_TO_OUTCOME
        },
        "min_reachable_two_sided_p_after_gating": min_reachable_two_sided_p(
            len(gated_rows), sum(1 for r in gated_rows if r["plan_found"])
        ),
    }

    # ---- MULTIPLICITY: the best rival was SELECTED from a family --------------------------
    analysis["family_multiplicity_check"] = family_max_statistic_p(
        rows, [k for k, _w in RIVALS if k not in DEFINITIONALLY_COUPLED_TO_OUTCOME]
    )

    # ---- CAN THIS DESIGN SEE AN EFFECT AT ALL? (the control a null lives or dies by) --------
    analysis["power_positive_control"] = power_positive_control(rows)

    # ---- THE INERTNESS FLOOR, and what survives it -------------------------------------------
    # An engine that changes nothing at the root CANNOT plan, as a matter of construction rather
    # than of measurement: `plan_in_model` seeds `seen` with the root and skips any successor
    # whose state key is already there, so an inert engine generates no new node and the frontier
    # drains immediately. Any predictor that is really just an inertness detector will therefore
    # post an AUC above chance for free. The question that matters is whether ANY predictor still
    # separates plannable from unplannable AMONG THE ENGINES THAT ARE NOT INERT -- that is the
    # population a selection rule would actually be choosing within.
    live = [r for r in rows if r["engine_changes_anything_at_root"]]
    analysis["inertness_floor"] = {
        "why": (
            "an inert engine cannot plan by construction, so an 'inertness detector' scores above "
            "chance without predicting anything. Everything below is restricted to engines that "
            "DO change the root state, which is the population any selection rule operates in."
        ),
        "n_inert_at_root": len(rows) - len(live),
        "n_inert_that_are_plannable": sum(
            1 for r in rows if not r["engine_changes_anything_at_root"] and r["plan_found"]
        ),
        "inert_plannable_should_be_zero_by_construction": True,
        "n_live": len(live),
        "n_live_plannable": sum(1 for r in live if r["plan_found"]),
        "min_reachable_two_sided_p_within_live": min_reachable_two_sided_p(
            len(live), sum(1 for r in live if r["plan_found"])
        ),
        "predictors_within_live_engines_only": {
            key: score_predictor(live, key)
            for key, _why in RIVALS
            if key not in DEFINITIONALLY_COUPLED_TO_OUTCOME
        },
    }

    # ---- does this run reproduce the A/B's own numbers? --------------------------------------
    analysis["objperc_change_fidelity_reproduction_check"] = _objperc_repro(engines)

    # ---- root substitution + depth checks ---------------------------------------------------
    analysis["root_substitution_check"] = _root_check(engines)
    analysis["depth_40_to_80_check"] = _depth_check(engines)

    analysis["rows"] = rows
    analysis["excluded_rows"] = excluded
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2))

    cf = analysis["predictors"]["change_fidelity"]
    print(f"analysable={n} plannable={n_pos} games={analysis['corpus']['n_games']}")
    print(f"min reachable two-sided p = {floor}")
    print(
        f"change_fidelity AUC pooled = {cf.get('auc_pooled')}  "
        f"perm_p={(cf.get('perm_p_pooled') or {}).get('p')}"
    )
    print(f"  cluster CI (games) = {cf.get('cluster_bootstrap_ci95_game_resample')}")
    _wg = cf.get("within_game") or {}
    print(f"  within-game AUC = {_wg.get('auc')} over {_wg.get('n_informative_games')} games")
    il = analysis["inertness_floor"]
    print(
        f"\ninertness floor: {il['n_inert_at_root']} inert "
        f"(plannable {il['n_inert_that_are_plannable']}), "
        f"{il['n_live']} live (plannable {il['n_live_plannable']})"
    )
    print("live-only AUCs:")
    for kk, vv in sorted(
        il["predictors_within_live_engines_only"].items(),
        key=lambda kv: -abs((kv[1].get("auc_pooled") or 0.5) - 0.5),
    )[:6]:
        wgl = vv.get("within_game") or {}
        print(f"  {kk:<45} pooled={vv.get('auc_pooled')} within_game={wgl.get('auc')}")
    pc = analysis["power_positive_control"]
    if pc.get("by_effect"):
        print(
            "power (detection rate by injected effect): "
            + ", ".join(f"{k}={v['detection_rate']}" for k, v in pc["by_effect"].items())
        )
    fm = analysis["family_multiplicity_check"]
    print(
        f"family multiplicity: strongest={fm.get('strongest_association_in_family')} "
        f"signed_within_game_auc={fm.get('its_signed_within_game_auc')} "
        f"({fm.get('its_direction')}) FWER_p={fm.get('fwer_adjusted_p_for_strongest')}"
    )
    dg = analysis["degenerate_goal_predicate_audit"]
    print(
        f"degenerate goals: {dg['n_with_root_true_goal']} root-true, "
        f"plannable {dg['n_plannable_before_gating']} -> {dg['n_plannable_after_gating']} "
        f"after applying the gate first; cf AUC then "
        f"{dg['primary_re_estimated_with_the_gate_applied_first'].get('auc_pooled')}"
    )
    print("\nrival ranking by pooled AUC:")
    ranked = sorted(
        (
            (k, v.get("auc_pooled"))
            for k, v in analysis["predictors"].items()
            if v.get("auc_pooled") is not None and k not in DEFINITIONALLY_COUPLED_TO_OUTCOME
        ),
        key=lambda kv: -abs(kv[1] - 0.5),
    )
    for k, a in ranked:
        w = analysis["predictors"][k].get("within_game") or {}
        print(f"  {k:<45} pooled={a:<8} within_game={w.get('auc')}")
    print("\n(definitionally coupled, NOT rivals -- the goal gate is the same search:)")
    for k in sorted(DEFINITIONALLY_COUPLED_TO_OUTCOME):
        v = analysis["predictors"].get(k) or {}
        print(f"  {k:<45} pooled={v.get('auc_pooled')}")
    return 0


def _census(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return dict(sorted(out.items()))


def _objperc_repro(engines: list[dict]) -> dict:
    """Is the exposure variable here the SAME number the object-perception A/B measured?

    This run rebuilds each game's window from scratch by calling `build_progress_window` +
    `_split_prefix_heldout` again, months of repo commits after the A/B ran. If that rebuild is
    not deterministic -- or if some upstream default has moved, as `_induce_transitions_k` did on
    the best-of-N side and broke it loudly -- then the `change_fidelity` on the left of this
    association is not the `change_fidelity` the A/B moved, and the whole question is being asked
    about a different quantity. A silent version of that drift is the dangerous one, so it is
    checked rather than assumed: every objperc cell's held-out change_fidelity is compared to the
    value frozen in the A/B's own `rescore.json` (itself already verified against `run_ab.py`'s
    as-run record with 0 mismatches on 116 cells).
    """
    p = (
        Path(__file__).resolve().parents[1]
        / "arc_object_perception_ab_change_fidelity_20260801"
        / "rescore.json"
    )
    if not p.exists():
        return {"checked": False, "why": "frozen rescore.json not found"}
    frozen = {
        c["cell"]: (c.get("full") or {}).get("change_fidelity")
        for c in json.loads(p.read_text()).get("cells", [])
        if c.get("status") == "ok"
    }
    checked, mismatch = 0, []
    for e in engines:
        if e.get("corpus") != "objperc" or e.get("status") != "ok":
            continue
        want = frozen.get(e.get("cell"))
        got = (e.get("heldout") or {}).get("change_fidelity")
        if want is None or got is None:
            continue
        checked += 1
        if abs(float(want) - float(got)) > 1e-6:
            mismatch.append({"cell": e["cell"], "frozen": want, "here": got})
    return {
        "checked": True,
        "why": (
            "the exposure variable must be the same quantity the A/B moved. A rebuilt window that "
            "silently differs would make this association a study of a different metric."
        ),
        "n_cells_compared": checked,
        "n_mismatch": len(mismatch),
        "all_reproduce": len(mismatch) == 0 and checked > 0,
        "mismatches": mismatch[:20],
    }


def _root_check(engines: list[dict]) -> dict:
    both = [
        e
        for e in engines
        if e.get("status") == "ok"
        and "real_root" in (e.get("plan") or {})
        and "window_root" in (e.get("plan") or {})
    ]
    agree = sum(
        1
        for e in both
        if bool(e["plan"]["real_root"].get("plan_found"))
        == bool(e["plan"]["window_root"].get("plan_found"))
    )
    disagree = [
        {
            "cell": e["cell"],
            "real_root": bool(e["plan"]["real_root"].get("plan_found")),
            "window_root": bool(e["plan"]["window_root"].get("plan_found")),
        }
        for e in both
        if bool(e["plan"]["real_root"].get("plan_found"))
        != bool(e["plan"]["window_root"].get("plan_found"))
    ]
    return {
        "why": (
            "the object-perception corpus recorded no E3AgentPolicy.root_grid, so its plan root is "
            "reconstructed as the window's first grid. The best-of-N corpus has BOTH, so the "
            "substitution is checkable there rather than assumed everywhere."
        ),
        "n_engines_with_both_roots": len(both),
        "n_agree_on_plan_found": agree,
        "agreement_rate": round(agree / len(both), 4) if both else None,
        "disagreements": disagree,
    }


def _depth_check(engines: list[dict]) -> dict:
    rows = [
        e
        for e in engines
        if e.get("status") == "ok" and e.get("frozen_plan_found_at_depth40") is not None
    ]
    root = "real_root"
    now = [
        (
            e["cell"],
            bool(e.get("frozen_plan_found_at_depth40")),
            bool(((e.get("plan") or {}).get(root) or {}).get("plan_found")),
        )
        for e in rows
        if root in (e.get("plan") or {})
    ]
    gained = [c for c, was, is_ in now if is_ and not was]
    lost = [c for c, was, is_ in now if was and not is_]
    return {
        "why": (
            "every best-of-N record carries goal_max_depth 40; the shipped default is now 80. "
            "Re-deriving at 80 independently reproduces (or refutes) plan_max_depth_default's "
            "stated 2 -> 6 conversion on this corpus."
        ),
        "n_compared": len(now),
        "n_plannable_at_frozen_depth40": sum(1 for _, was, _i in now if was),
        "n_plannable_now_at_depth80": sum(1 for _c, _w, is_ in now if is_),
        "gained_at_depth80": sorted(gained),
        "lost_at_depth80": sorted(lost),
    }


if __name__ == "__main__":
    raise SystemExit(main())
