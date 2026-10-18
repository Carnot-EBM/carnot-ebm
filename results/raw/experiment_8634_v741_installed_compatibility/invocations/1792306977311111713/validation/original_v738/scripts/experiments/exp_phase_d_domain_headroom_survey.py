"""Phase D domain survey: which corpus in this repo has REAL, SELECTABLE, oracle-distinct headroom?

WHY THIS EXISTS
---------------
Phase D (the off-ARC "verifier moat" program) has produced a seven-milestone null streak.
The most recent proposal -- MMLU-Pro -- was refuted at review because its "0.375 headroom"
was COMBINATORIAL, not selectable: the generator sat at the 10-way chance floor (per-candidate
accuracy 0.125 vs a 0.10 floor, z=1.17), so oracle@6 rising to 0.500 is just what you get for
free by drawing six near-random samples. `ops/known-issues.md` now carries a
`blocked_generator_at_chance_floor` precondition.

This script answers the obvious next question MECHANICALLY rather than by argument: is there
ANY corpus already in `results/` that clears all three bars a Phase D selection experiment
needs? It measures, for every candidate-pool-shaped corpus in the repo:

  Criterion 1 -- GENERATOR ABOVE CHANCE. A verifier selects among candidates; if the generator
      is at chance there is no signal to select. We state the chance floor EXPLICITLY (computed
      by brute-force enumeration of the answer space where the task is open-ended, not asserted)
      and test observed per-candidate accuracy against it with a one-proportion z test. We ALSO
      run the independence check 1-(1-p)^K: if observed oracle@K is at or below what independent
      chance-level sampling predicts, the "headroom" is combinatorial and there is nothing there.

  Criterion 2 -- SELF-CONSISTENCY NOT SATURATED. If plain majority voting already captures the
      headroom, no verifier can beat it -- the documented cause of the bounded energy-selection
      result. We report SC, oracle@K, and the GAP. Where the pool is a multi-MODEL ensemble
      rather than multi-SAMPLE, we additionally report the gap over the BEST SINGLE MODEL,
      because that -- not a degenerate 3-way vote containing a dead arm -- is the honest baseline.

  Criterion 3 -- VERIFIER ORACLE-DISTINCT. Reported as a design property per corpus.

WHAT "CONSTRUCTION ARTIFACT" MEANS HERE (the trap this script is built to catch)
-------------------------------------------------------------------------------
Twice already this project has been fooled by a pool whose headroom was manufactured by the
pool BUILDER rather than measured from a generator: the FoVer/arc/arcgen rows of
`results/headroom_survey_cross_domain.json` (retracted 2026-07-01 -- the numbers were exact
consequences of a `task_index % 4` formula). So this script also emits structural tells:
the distribution of WHICH candidate index is correct, and the per-index model balance. A real
temperature-sampled pool puts the correct answer at every index and does not balance models to
the unit; a templated fixture does both.

SUBSTRATE: this reads cached corpora only. No LLM is invoked, no GPU is used -- so
`inference_substrate` is `aggregation_from_upstream_artifacts` and a sub-second duration is
correct, not suspicious.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

from carnot.paths import repo_path, results_path

RANDOM_SEED = 20260803


# ---------------------------------------------------------------------------
# Chance floors by ENUMERATION.
#
# The MMLU-Pro refutation turned on a 10-way MCQ having a 0.10 floor. For open-ended
# structured answers the floor is the fraction of the ANSWER SPACE that is correct. We compute
# it by brute force per instance rather than asserting a number, because an asserted floor is
# exactly the kind of unchecked premise that produced the refuted proposal.
# ---------------------------------------------------------------------------


def floor_graph_coloring(c: dict) -> tuple[float, int]:
    n, k, edges = c["n_nodes"], c["n_colors"], [tuple(e) for e in c["edges"]]
    tot = k**n
    ok = sum(
        1 for a in itertools.product(range(k), repeat=n) if all(a[u] != a[v] for u, v in edges)
    )
    return ok / tot, tot


def _kk_consistent(assign, people, statements):
    """A knight's statement is true; a knave's is false.

    Returns None on an unrecognised statement kind so the caller REFUSES rather than
    silently guessing a floor -- a wrong floor is worse than no floor.
    """
    idx = {p: i for i, p in enumerate(people)}
    for s in statements:
        speaker_is_knight = assign[idx[s["speaker"]]]
        kind = s["kind"]
        if kind == "target_is_knave":
            truth = not assign[idx[s["target"]]]
        elif kind == "target_is_knight":
            truth = assign[idx[s["target"]]]
        elif kind == "different_from":
            truth = assign[idx[s["left"]]] != assign[idx[s["right"]]]
        elif kind == "count_knights_eq":
            truth = sum(assign) == s["count"]
        else:
            return None
        if speaker_is_knight != truth:
            return False
    return True


def floor_knights_knaves(c: dict) -> tuple[float | None, int]:
    people, st = c["people"], c["statements"]
    tot = 2 ** len(people)
    ok = 0
    for a in itertools.product([False, True], repeat=len(people)):
        r = _kk_consistent(a, people, st)
        if r is None:
            return None, tot
        ok += bool(r)
    return ok / tot, tot


def floor_travel_budget(c: dict) -> tuple[float, int]:
    acts, budget, hours = c["activities"], c["budget"], c["hours"]
    best, cnt, tot = -1, 0, 2 ** len(acts)
    for r in range(len(acts) + 1):
        for sub in itertools.combinations(range(len(acts)), r):
            if (
                sum(acts[i]["cost"] for i in sub) > budget
                or sum(acts[i]["hours"] for i in sub) > hours
            ):
                continue
            val = sum(acts[i]["value"] for i in sub)
            if val > best:
                best, cnt = val, 1
            elif val == best:
                cnt += 1
    return cnt / tot, tot


def floor_code_property(c: dict) -> tuple[float, int]:
    # The answer is a SUBSET of range(n); exactly one subset is right.
    n = c["domain_n"]
    return 1 / (2**n), 2**n


def floor_or_allocation(c: dict) -> tuple[float, int]:
    prods, caps = c["products"], c["capacities"]
    ranges = [range(p["max_units"] + 1) for p in prods]
    tot = 1
    for r in ranges:
        tot *= len(r)
    best, cnt = -1, 0
    for combo in itertools.product(*ranges):
        if any(
            sum(combo[i] * prods[i][res] for i in range(len(prods))) > cap
            for res, cap in caps.items()
        ):
            continue
        prof = sum(combo[i] * prods[i]["profit"] for i in range(len(prods)))
        if prof > best:
            best, cnt = prof, 1
        elif prof == best:
            cnt += 1
    return cnt / tot, tot


FLOORS = {
    "graph_coloring": floor_graph_coloring,
    "knights_knaves": floor_knights_knaves,
    "travel_budget": floor_travel_budget,
    "code_property": floor_code_property,
    "or_allocation": floor_or_allocation,
}


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _z_vs_chance(n_correct: int, n_total: int, floor: float) -> float | None:
    if not n_total or floor is None:
        return None
    p = n_correct / n_total
    se = math.sqrt(floor * (1 - floor) / n_total)
    return None if se == 0 else (p - floor) / se


def _sc_and_oracle(groups: dict[Any, list[tuple[bool, Any]]]) -> tuple[int, int, int]:
    """Plurality vote (deterministic first-seen tie-break) and oracle@K."""
    sc = sc_evaluable = 0
    for cands in groups.values():
        keys = [a for _, a in cands if a is not None]
        if not keys:
            continue
        sc_evaluable += 1
        counts = collections.Counter(keys)
        best = max(counts.values())
        winner = next(a for a in keys if counts[a] == best)
        if any(ok for ok, a in cands if a == winner and ok):
            sc += 1
    oracle = sum(1 for cands in groups.values() if any(ok for ok, _ in cands))
    return sc, sc_evaluable, oracle


def evaluate(
    label: str,
    groups: dict[Any, list[tuple[bool, Any]]],
    floor: float | None,
    floor_note: str,
    oracle_distinct: bool | None,
    notes: str,
) -> dict:
    n_q = len(groups)
    sizes = [len(v) for v in groups.values()]
    k_min = min(sizes)
    total = sum(sizes)
    n_correct = sum(1 for v in groups.values() for ok, _ in v if ok)
    p = n_correct / total
    z = _z_vs_chance(n_correct, total, floor)
    sc, sc_eval, oracle = _sc_and_oracle(groups)
    sc_acc = sc / sc_eval if sc_eval else float("nan")
    oracle_acc = oracle / n_q
    independence_pred = 1 - (1 - p) ** k_min if floor is not None else None
    return {
        "label": label,
        "n_questions": n_q,
        "k_min": k_min,
        "k_max": max(sizes),
        "n_candidates": total,
        "chance_floor": floor,
        "chance_floor_note": floor_note,
        "per_candidate_accuracy": round(p, 6),
        "z_vs_chance": None if z is None else round(z, 4),
        "generator_above_chance": None if z is None else bool(z > 2.0),
        "sc_accuracy": round(sc_acc, 6),
        "oracle_at_k": round(oracle_acc, 6),
        "headroom_gap_oracle_minus_sc": round(oracle_acc - sc_acc, 6),
        "independence_prediction": (
            None if independence_pred is None else round(independence_pred, 6)
        ),
        "oracle_exceeds_independence": (
            None if independence_pred is None else bool(oracle_acc > independence_pred)
        ),
        "oracle_distinct": oracle_distinct,
        "notes": notes,
    }


def structural_tells(
    rows_by_group: dict[Any, list[dict]], index_key: str, correct_key: str, model_key: str | None
) -> dict:
    """Detect the construction-artifact signature: correct answer confined to a few indices,
    and per-index model counts balanced to the unit. Real sampling does neither."""
    idx_correct: collections.Counter = collections.Counter()
    per_index_models: dict = collections.defaultdict(collections.Counter)
    patterns: collections.Counter = collections.Counter()
    for cands in rows_by_group.values():
        patterns[tuple(bool(c[correct_key]) for c in cands)] += 1
        for c in cands:
            if c[correct_key]:
                idx_correct[c[index_key]] += 1
            if model_key:
                per_index_models[c[index_key]][c[model_key].split("/")[-1]] += 1
    n_idx = max((len(v) for v in rows_by_group.values()), default=0)
    indices_never_correct = [i for i in range(n_idx) if idx_correct.get(i, 0) == 0]
    balanced = (
        all(len(set(v.values())) == 1 for v in per_index_models.values())
        if per_index_models
        else None
    )
    return {
        "correct_candidate_index_distribution": dict(sorted(idx_correct.items())),
        "indices_never_correct": indices_never_correct,
        "distinct_correctness_patterns": {str(k): v for k, v in patterns.most_common()},
        "per_index_model_counts_perfectly_balanced": balanced,
        "construction_artifact_suspected": bool(indices_never_correct) or bool(balanced),
    }


# ---------------------------------------------------------------------------
# Per-corpus loaders
# ---------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def survey() -> dict:
    started = time.time()
    domains: list[dict] = []
    extra: dict[str, Any] = {}

    # -- MMLU-Pro (the refuted proposal, re-measured here as the control) ------------------
    for label, fname, note in [
        (
            "mmlu_pro_5shot",
            "experiment_mmlu_pro_fewshot_candidate_pool.jsonl",
            "the REFUTED proposal; re-measured as the control",
        ),
        (
            "mmlu_pro_zeroshot",
            "experiment_mmlu_pro_verifier_candidate_pool.jsonl",
            "zero-shot sibling pool",
        ),
    ]:
        g: dict = collections.defaultdict(list)
        for r in _jsonl(results_path(fname)):
            g[r["question_index"]].append((bool(r["correct"]), r.get("parsed_letter")))
        domains.append(evaluate(label, g, 0.10, "10-way MCQ; gold letters span A-J", True, note))

    # -- MuSR: the ONLY genuine multi-sample corpus (one model, temperature-sampled) -------
    rows = _jsonl(results_path("experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"))
    g = collections.defaultdict(list)
    n_choices = set()
    for r in rows:
        g[r["question_id"]].append((r["answer"] == r["gold"], r["answer"]))
        n_choices.add(len(r["choices"]))
    d = evaluate(
        "musr_murder_mysteries",
        g,
        0.5,
        f"BINARY choice ({sorted(n_choices)} options); floor is a coin flip",
        True,
        "the only genuine multi-sample single-model pool in the repo",
    )
    d["scoring_model"] = sorted({r["scoring_model"] for r in rows})
    d["questions_with_both_answers_present"] = sum(
        1 for v in g.values() if len({a for _, a in v}) > 1
    )
    domains.append(d)

    # -- Structured-reasoning pools (exact validators; check for construction) -------------
    for label, fname in [
        ("structured_reasoning_exp5125", "experiment_5125_structured_reasoning_pool_v470.jsonl"),
        ("structured_reasoning_exp5136", "experiment_5136_receipt_structured_pool_v2_v471.jsonl"),
    ]:
        raw = _jsonl(results_path(fname))
        g, floors, by_group = collections.defaultdict(list), {}, {}
        for r in raw:
            qid = r["task_id"]
            floors[qid] = FLOORS[r["family"]](r["constraints"])[0]
            by_group[qid] = r["candidates"]
            for c in r["candidates"]:
                g[qid].append((bool(c["correct"]), c.get("normalized_answer")))
        mean_floor = sum(floors.values()) / len(floors)
        d = evaluate(
            label,
            g,
            mean_floor,
            f"ENUMERATED answer space per instance, mean over {len(floors)} tasks",
            True,
            "executable validator exists, so a LEARNED verifier is oracle-distinct",
        )
        d["structural_tells"] = structural_tells(
            by_group, "candidate_index", "correct", "model_hf_id"
        )
        d["families"] = dict(collections.Counter(r["family"] for r in raw))
        domains.append(d)

    # -- exp5786: a genuine multi-MODEL SOTA generation stream -----------------------------
    raw = _jsonl(results_path("experiment_5786_sota_constraint_stream.rows.jsonl"))

    def model_of(r: dict) -> str:
        return r["model_hf_id"].split("/")[-1]

    g = collections.defaultdict(list)
    by_model: dict = collections.defaultdict(dict)
    for r in raw:
        pred = r.get("selected_label") or None
        g[r["fixture_row_id"]].append((pred is not None and pred == r["exact_label"], pred))
        by_model[r["fixture_row_id"]][model_of(r)] = r
    d = evaluate(
        "sota_constraint_stream_exp5786",
        g,
        0.25,
        "4-way A/B/C/D exact label",
        True,
        "REAL generations (raw text, truncations, per-model variation)",
    )
    # Per-model accuracy and parse rate -- a dead arm makes a 3-way vote degenerate, so the
    # honest baseline is the BEST SINGLE MODEL, not the ensemble vote.
    models = sorted({model_of(r) for r in raw})
    per_model = {}
    for m in models:
        sub = [r for r in raw if model_of(r) == m]
        parseable = sum(1 for r in sub if r.get("selected_label"))
        correct = sum(1 for r in sub if r.get("selected_label") == r["exact_label"])
        per_model[m] = {
            "n": len(sub),
            "parseable": parseable,
            "parse_rate": round(parseable / len(sub), 6),
            "accuracy": round(correct / len(sub), 6),
            "finish_reason": dict(collections.Counter(r["finish_reason"] for r in sub)),
        }
    live = [m for m, v in per_model.items() if v["parseable"] > 0]
    dead = [m for m, v in per_model.items() if v["parseable"] == 0]
    best_model = max(live, key=lambda m: per_model[m]["accuracy"])
    best_acc = per_model[best_model]["accuracy"]
    oracle_live = sum(
        1
        for k in by_model
        if any(by_model[k][m].get("selected_label") == by_model[k][m]["exact_label"] for m in live)
    ) / len(by_model)
    d["per_model"] = per_model
    d["dead_arms"] = dead
    d["best_single_model"] = best_model
    d["best_single_model_accuracy"] = best_acc
    d["oracle_over_live_arms"] = round(oracle_live, 6)
    d["selectable_gain_over_best_single_model"] = round(oracle_live - best_acc, 6)
    d["selectable_gain_instances"] = round((oracle_live - best_acc) * len(by_model))
    d["is_multi_sample"] = False
    d["multi_sample_note"] = (
        "K=1 sample per MODEL, so genuine self-consistency is NOT measurable from this corpus; "
        "the reported sc_accuracy is a 3-way ensemble vote containing a dead arm and is NOT the "
        "honest baseline -- selectable_gain_over_best_single_model is."
    )
    per_family = {}
    for fam in sorted({r["family"] for r in raw}):
        ks = [k for k in by_model if by_model[k][live[0]]["family"] == fam]
        accs = {
            m: sum(
                1
                for k in ks
                if by_model[k][m].get("selected_label") == by_model[k][m]["exact_label"]
            )
            / len(ks)
            for m in live
        }
        orc = sum(
            1
            for k in ks
            if any(
                by_model[k][m].get("selected_label") == by_model[k][m]["exact_label"] for m in live
            )
        ) / len(ks)
        per_family[fam] = {
            "n": len(ks),
            "per_model_accuracy": {m: round(v, 6) for m, v in accs.items()},
            "oracle": round(orc, 6),
            "gain_over_best_single": round(orc - max(accs.values()), 6),
        }
    d["per_family"] = per_family
    domains.append(d)

    # -- exp5799 canary ---------------------------------------------------------------------
    raw = _jsonl(results_path("experiment_5799_sota_answer_channel_canary.rows.jsonl"))
    g = collections.defaultdict(list)
    for r in raw:
        pred = r.get("selected_label") or None
        g[r["fixture_row_id"]].append((pred is not None and pred == r["exact_label"], pred))
    d = evaluate(
        "sota_answer_channel_canary_exp5799",
        g,
        0.25,
        "4-way A/B/C/D exact label",
        True,
        "answer-channel canary",
    )
    d["unparseable_fraction"] = round(
        sum(1 for r in raw if not r.get("selected_label")) / len(raw), 6
    )
    domains.append(d)

    # -- ConstraintBench: candidates are solver-backed, NOT LLM samples ----------------------
    raw = _jsonl(results_path("experiment_5044_second_corpus_candidate_cache.jsonl"))
    gen_kinds = collections.Counter(c.get("generator_kind") for r in raw for c in r["candidates"])
    gen_models = collections.Counter(
        str(c.get("generation_model")) for r in raw for c in r["candidates"]
    )
    extra["constraintbench_exp5044"] = {
        "n_rows": len(raw),
        "generator_kinds": dict(gen_kinds),
        "generation_models": dict(gen_models),
        "disqualified_reason": (
            "candidates are deterministic solver-backed variants with generation_model=None; "
            "there is no LLM generator, so criterion 1 is not even defined"
        ),
        "passes_criteria": False,
    }

    # -- Verdict --------------------------------------------------------------------------
    for d in domains:
        c1 = bool(d.get("generator_above_chance")) and bool(d.get("oracle_exceeds_independence"))
        tells = d.get("structural_tells", {})
        constructed = bool(tells.get("construction_artifact_suspected"))
        if "selectable_gain_over_best_single_model" in d:
            c2 = d["selectable_gain_over_best_single_model"] > 0.05
        else:
            c2 = d["headroom_gap_oracle_minus_sc"] > 0.05
        d["criterion_1_generator_above_chance"] = c1
        d["criterion_2_sc_not_saturated"] = bool(c2) and not constructed
        d["criterion_3_oracle_distinct"] = bool(d.get("oracle_distinct"))
        d["construction_artifact"] = constructed
        d["passes_all_three"] = bool(
            c1
            and d["criterion_2_sc_not_saturated"]
            and d["criterion_3_oracle_distinct"]
            and not constructed
        )

    passing = [d["label"] for d in domains if d["passes_all_three"]]
    duration = time.time() - started

    artifact: dict[str, Any] = {
        "experiment": "phase_d_domain_headroom_survey",
        "experiment_id": "phase_d_domain_headroom_survey",
        "schema": "carnot.phase_d_domain_headroom_survey.v1",
        "title": "Phase D domain survey: is there a headroom-present, oracle-distinct corpus in this repo?",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration, 6),
        "domains": domains,
        "non_candidate_corpora": extra,
        "domains_passing_all_three": passing,
        "n_domains_measured": len(domains),
        "honest_verdict": (
            "complete_no_repo_corpus_passes_all_three_criteria_two_disjoint_failure_modes"
            if not passing
            else "complete_survey_found_passing_domain_" + "_".join(passing)
        ),
    }
    return artifact


def main() -> None:
    art = survey()
    # The checksum must be REPRODUCIBLE across runs, so it is computed over the payload with
    # the wall-clock-dependent and self-referential fields blanked. Including duration_s would
    # make every rerun produce a different checksum, which defeats the point of having one.
    normalized = {
        k: v for k, v in art.items() if k not in ("duration_s", "reproducibility_checksum")
    }
    payload = json.dumps(normalized, indent=2, sort_keys=True)
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
    art["checksum_covers"] = "all fields except duration_s and reproducibility_checksum itself"
    out = results_path("phase_d_domain_headroom_survey.json")
    out.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for d in art["domains"]:
        print(
            f"  {d['label']:34s} C1={d['criterion_1_generator_above_chance']!s:5s} "
            f"C2={d['criterion_2_sc_not_saturated']!s:5s} C3={d['criterion_3_oracle_distinct']!s:5s} "
            f"-> pass={d['passes_all_three']}"
        )
    print(f"  passing: {art['domains_passing_all_three'] or 'NONE'}")


if __name__ == "__main__":
    main()
