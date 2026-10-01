"""Measure the original response event under a fixed information ablation.

REQ-VERIFY-7958. Human response targets stay unchanged when source evidence is
erased. Strict decoding gives transport evidence, not a correctness certificate.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import qwen_completion_7932 as transport
from carnot.verify import sentence_labels_7942 as custody
from carnot.verify.source_interventions import sentence_offsets

Json = dict[str, Any]
MODEL = "unsloth/Qwen3.8-27B-GGUF"
ARMS = ("full_source", "source_erased")
SEED = 69058
SYSTEM = (
    "Estimate the probability that the complete original answer contains any span "
    "unsupported by its original source. The target always refers to the ORIGINAL "
    "source, even when source evidence has been erased. Erasure removes information "
    "and does not change the answer's correctness. Return only a JSON object with "
    "unsupported_probability and source_sentence_id. Use null when no visible "
    "source sentence is a proposed witness."
)


def config() -> Json:
    """Freeze the decoder and resource budget before any outcome is observed."""
    return dict(
        seed=SEED,
        temperature=0,
        max_tokens=96,
        input_tokens=6000,
        call_limit=128,
        output_tokens=12288,
        intended_families=64,
        measured_work_cap_s=3000,
        load_timeout_s=300,
        call_timeout_s=120,
        duration_floor_s=10,
        bootstrap_draws=10000,
        grammar_sha256=custody.digest(transport.GRAMMAR.encode()),
    )


def freeze(public: list[Json], count: Callable[[str], int]) -> list[Json]:
    """Preserve complete public bytes; evaluator fields cannot enter admission."""
    custody.index_unique(public, "family_id")
    rows = []
    for row in public:
        if set(row) != custody.PUBLIC_KEYS:
            raise ValueError("public_fields")
        source, answer = (bytes.fromhex(row[k]).decode() for k in ("source_bytes", "answer_bytes"))
        ids = [r["source_sentence_id"] for r in sentence_offsets(source.encode())]
        requests, sizes = {}, {}
        for arm in ARMS:
            body = dict(
                complete_source=source if arm == ARMS[0] else "",
                original_answer=answer,
                target="any_unsupported_span_in_original_response",
                visible_source_sentence_ids=ids if arm == ARMS[0] else [],
                source_evidence_erased=arm == ARMS[1],
            )
            messages = [
                dict(role="system", content=SYSTEM),
                dict(role="user", content=json.dumps(body, ensure_ascii=False)),
            ]
            requests[arm] = dict(
                model=MODEL,
                messages=messages,
                grammar=transport.GRAMMAR,
                temperature=0,
                top_p=1,
                seed=SEED,
                max_tokens=96,
                chat_template_kwargs=dict(enable_thinking=False),
            )
            sizes[arm] = count(json.dumps(messages, ensure_ascii=False))
        order = list(ARMS)
        if int(canonical_hash(row).split(":")[1][-1], 16) % 2:
            order.reverse()
        rows.append(
            dict(
                row,
                requests=requests,
                input_tokens=sizes,
                order=order,
                visible_ids=ids,
                source_cluster_id=custody.digest(source.encode()),
                eligible=bool(source and answer) and max(sizes.values()) <= 6000,
            )
        )
    return rows


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    *,
    token_budget: int = 12288,
    deadline_s: float = 3000,
    started: float | None = None,
) -> list[Json]:
    """Visit every cell once and checkpoint complete pairs without replacement."""
    started = time.monotonic() if started is None else started
    rows: list[Json] = []
    used = 0
    for i, family in enumerate(frozen):
        pair = []
        for position, arm in enumerate(family["order"]):
            ids = family["visible_ids"] if arm == ARMS[0] else []
            row = dict(
                family_id=family["family_id"],
                source_cluster_id=family["source_cluster_id"],
                arm=arm,
                seed=SEED,
                order=position,
                started=False,
                raw_response={},
                visible_ids=ids,
                input_tokens=family["input_tokens"][arm],
                status="excluded",
                request=family["requests"][arm],
                duration_s=0.0,
            )
            if family["eligible"]:
                row["status"] = "censored"
                if time.monotonic() - started < deadline_s - 120 and used + 96 <= token_budget:
                    begin = time.monotonic()
                    row.update(started=True, status="generated")
                    print(
                        f"[exp7958] before_generation pair={i + 1} arm={arm} elapsed_s={begin - started:.3f}",
                        flush=True,
                    )
                    try:
                        row["raw_response"] = runtime.generate(row["request"])
                    except (RuntimeError, OSError, TimeoutError, ValueError) as error:
                        row.update(status="failed", error=f"{type(error).__name__}:{error}")
                    row["duration_s"] = time.monotonic() - begin
                    used += row["raw_response"].get("usage", {}).get("completion_tokens", 0)
                    print(
                        f"[exp7958] after_generation completed_units={len(rows) + len(pair) + 1} elapsed_s={time.monotonic() - started:.3f}",
                        flush=True,
                    )
            row["parsed"] = transport.parse_response(row["raw_response"], ids)
            pair.append(row)
        atomic_json(
            raw / f"pair-{i:03d}.json",
            dict(
                rows=pair, config_hash=canonical_hash(config()), public_hash=canonical_hash(family)
            ),
        )
        rows.extend(pair)
    return rows


def decision(p: float | None) -> str:
    """Escalate invalid outputs and ties to preserve the registered cost rule."""
    if p is None:
        return "escalate"
    costs = dict(accept=5 * p, reject=1 - p, escalate=0.25)
    best = min(costs.values())
    return "escalate" if costs["escalate"] == best else min(costs, key=lambda k: costs[k])


def intervals(values: np.ndarray[Any, Any]) -> tuple[list[float], float]:
    """Resample whole source clusters; repeated responses do not add samples."""
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    signs = rng.choice([-1, 1], size=(10000, len(values)))
    p = (1 + int(np.sum((signs * values).mean(axis=1) >= values.mean()))) / 10001
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])], p


def reduce(rows: list[Json], labels: list[Json]) -> Json:
    """Cold-reduce probability quality and decision costs against the human union."""
    indexed = custody.index_unique(labels, "family_id")
    cells = {(r["family_id"], r["arm"]): r for r in rows}
    if len(cells) != len(rows):
        raise ValueError("duplicate_cell")
    costs, pairs, censored = [], [], []
    groups: dict[str, list[Json]] = defaultdict(list)
    cost_groups: dict[str, list[Json]] = defaultdict(list)
    counters = dict(
        intended=len(rows),
        eligible=0,
        started=0,
        completed=0,
        failed=0,
        censored=0,
        excluded=0,
        independent=0,
        unit="response_arm",
        independent_unit="original_source_cluster",
    )
    automation, false_accepts = dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0)
    totals = dict.fromkeys(ARMS, 0.0)
    class_counts = {"0": 0, "1": 0}
    for row in rows:
        label = indexed.get(row["family_id"], {})
        y = label.get("y")
        parsed = transport.parse_response(row["raw_response"], row["visible_ids"])
        if row["parsed"] != parsed:
            raise ValueError("parse_drift")
        counters["started"] += int(row["started"])
        counters["completed"] += int(parsed["completed"])
        counters["failed"] += int(row["started"] and not parsed["completed"])
        counters["censored"] += int(row["status"] == "censored")
        counters["excluded"] += int(y is None or row["status"] == "excluded")
        counters["eligible"] += int(row["status"] != "excluded")
        if not parsed["completed"]:
            censored.append(
                dict(
                    family_id=row["family_id"],
                    arm=row["arm"],
                    status=row["status"],
                    reason=parsed["status"],
                )
            )
        if y is None:
            continue
        p = parsed["probability"]
        action = decision(p)
        actual = {"accept": 5 * y, "reject": 1 - y, "escalate": 0.25}[action]
        automation[row["arm"]] += int(action != "escalate")
        false_accepts[row["arm"]] += int(action == "accept" and y == 1)
        totals[row["arm"]] += actual
        costs.append(
            dict(
                family_id=row["family_id"],
                arm=row["arm"],
                seed=SEED,
                y=y,
                p=p,
                decision=action,
                actual_cost=actual,
                expected_cost=None
                if p is None
                else {"accept": 5 * p, "reject": 1 - p, "escalate": 0.25}[action],
                implicit_true_excluded_y=label.get("implicit_true_excluded_y"),
            )
        )
    for fid in sorted({r["family_id"] for r in rows}):
        label = indexed.get(fid, {})
        y = label.get("y")
        pair = [cells.get((fid, arm)) for arm in ARMS]
        if y is not None and all(r is not None for r in pair):
            all_cost = [
                {"accept": 5 * y, "reject": 1 - y, "escalate": 0.25}[
                    decision(r["parsed"]["probability"])
                ]
                for r in pair
                if r is not None
            ]
            cost_groups[label["source_cluster_id"]].append(
                dict(cost_gain=all_cost[1] - all_cost[0])
            )
        if y is None or any(r is None or not r["parsed"]["completed"] for r in pair):
            continue
        full, erased = pair
        assert full is not None and erased is not None
        probabilities = [full["parsed"]["probability"], erased["parsed"]["probability"]]
        brier = [(p - y) ** 2 for p in probabilities]
        actual = [
            {"accept": 5 * y, "reject": 1 - y, "escalate": 0.25}[decision(p)] for p in probabilities
        ]
        item = dict(
            family_id=fid,
            source_cluster_id=label["source_cluster_id"],
            y=y,
            probabilities=probabilities,
            brier=brier,
            costs=actual,
            brier_gain=brier[1] - brier[0],
            cost_gain=actual[1] - actual[0],
            sensitivity=probabilities[1] - probabilities[0],
            annotation_count=label.get("annotation_count", 0),
            annotation_types=label.get("annotation_types", []),
        )
        pairs.append(item)
        groups[item["source_cluster_id"]].append(item)
        class_counts[str(y)] += 1
    counters["independent"] = len(groups)
    sufficient = len(groups) >= 32 and min(class_counts.values()) >= 8
    ci, raw_p, adjusted, means = {}, {}, {}, {}
    for name in ("brier", "cost"):
        values = np.asarray(
            [
                np.mean([r[name + "_gain"] for r in group])
                for group in (groups if name == "brier" else cost_groups).values()
            ]
        )
        means[name] = float(values.mean()) if len(values) else None
        ci[name], raw_p[name] = intervals(values) if sufficient else ([None, None], None)
    if sufficient:
        first, second = sorted(raw_p, key=lambda k: raw_p[k])
        adjusted[first] = min(1.0, 2 * raw_p[first])
        adjusted[second] = max(adjusted[first], raw_p[second])
    else:
        adjusted = dict.fromkeys(("brier", "cost"))
    denominator = sum(r["arm"] == ARMS[0] for r in costs)
    coverage = {arm: automation[arm] / denominator if denominator else 0.0 for arm in ARMS}
    benefit = sufficient and all(
        means[k] >= 0.02 and ci[k][0] > 0 and adjusted[k] < 0.05 for k in ("brier", "cost")
    )
    benefit = (
        benefit and coverage[ARMS[0]] >= 0.20 and false_accepts[ARMS[0]] <= false_accepts[ARMS[1]]
    )
    metrics = dict(
        complete_pairs=len(pairs),
        source_cluster_pairs=len(groups),
        brier_gain=means["brier"],
        cost_gain=means["cost"],
        all_intended_labelled_denominator=denominator,
        all_intended_mean_cost={a: totals[a] / denominator if denominator else None for a in ARMS},
        automated_coverage=coverage,
        false_accepts=false_accepts,
        class_counts=class_counts,
        calibration=dict(
            brier_by_arm={
                a: float(np.mean([r["brier"][i] for r in pairs])) if pairs else None
                for i, a in enumerate(ARMS)
            }
        ),
    )
    metrics["all_intended_cost_source_clusters"] = len(cost_groups)
    metrics["calibration"]["bins"] = {
        arm: [
            dict(
                lower=i / 10,
                upper=(i + 1) / 10,
                n=len(bin_rows),
                mean_p=float(np.mean([r["p"] for r in bin_rows])) if bin_rows else None,
                mean_y=float(np.mean([r["y"] for r in bin_rows])) if bin_rows else None,
            )
            for i in range(10)
            for bin_rows in [
                [
                    r
                    for r in costs
                    if r["arm"] == arm and r["p"] is not None and min(9, int(r["p"] * 10)) == i
                ]
            ]
        ]
        for arm in ARMS
    }
    strata = {
        str(n): [r["family_id"] for r in pairs if bool(r["annotation_count"]) == bool(n)]
        for n in (0, 1)
    }
    strata["annotation_types"] = {
        kind: [r["family_id"] for r in pairs if kind in r["annotation_types"]]
        for kind in sorted({kind for r in pairs for kind in r["annotation_types"]})
    }
    return dict(
        rows=rows,
        sample_size_budget=counters,
        paired_family_rows=pairs,
        probability_metrics=metrics,
        typed_cost_rows=costs,
        censor_rows=censored,
        confidence_intervals=ci,
        adjusted_p_values=adjusted,
        raw_p_values=raw_p,
        comparison_status="registered_comparison" if sufficient else "insufficient_data",
        qwen_response_benefit_score=int(benefit),
        onset_strata=strata,
        syntax_completion=dict(completed=counters["completed"], intended=len(rows)),
        sensitivity_analysis=dict(
            scope="descriptive_only",
            source_delta=[r["sensitivity"] for r in pairs],
            implicit_true_excluded_targets=[r.get("implicit_true_excluded_y") for r in costs],
        ),
    )
