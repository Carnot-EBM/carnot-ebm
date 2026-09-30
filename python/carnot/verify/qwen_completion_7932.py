"""Compare bounded decoder completion without inventing sentence truth.

REQ-VERIFY-7932-V688. Evidence views are frozen before either decoder runs,
so output failures cannot change the cohort or nominate a different witness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Callable

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import source_interventions as source

Json = dict[str, Any]
VIEWS = ("full_source", "witness_only", "witness_neighbors", "witness_filler")
DECODERS = ("plain", "grammar")
GRAMMAR = r"""root ::= "{" ws "\"unsupported_probability\"" ws ":" ws probability ws "," ws "\"source_sentence_id\"" ws ":" ws citation ws "}"
probability ::= "0" ("." [0-9]+)? | "1" ("." "0"+)?
citation ::= "null" | "0" | [1-9] [0-9]*
ws ::= [ \t\n\r]*"""
SYSTEM = source.SYSTEM + " Use null when no visible sentence is a proposed witness."


def freeze_config() -> Json:
    """Keep the decoder budget fixed independently of outcomes or eligibility."""
    return dict(
        n_ctx=8192,
        temperature=0,
        seed=67801,
        decoder_order_seed=68832,
        max_tokens=96,
        output_token_limit=36864,
        call_limit=384,
        intended_families=48,
        load_timeout_s=300,
        call_timeout_s=120,
        latest_launch_s=2400,
        measured_work_cap_s=3000,
        duration_floor_s=10,
        witness_rule="unique casefolded Unicode word overlap; earliest original byte tie",
        filler_rule="disjoint contiguous original spans; neighbor-added tokens within 25 percent",
        grammar_sha256=source.digest(GRAMMAR.encode()),
        bootstrap_draws=10000,
    )


def freeze_views(families: list[Json], count: Callable[[str], int]) -> list[Json]:
    """Preserve original bytes and nominate evidence from label-free words only."""
    frozen = []
    for family in families:
        raw = family["complete_source"].encode()
        answer = family["complete_response"].encode()
        if (
            source.digest(raw) != family["source_sha256"]
            or source.digest(answer) != family["response_sha256"]
        ):
            raise ValueError("custody_drift")
        item = dict(
            family,
            requests={},
            exclusion_reason=None,
            witness_sentence_id=None,
            filler_sentence_ids=[],
            neighbor_added_tokens=None,
            filler_tokens=None,
        )
        if not family["eligible"]:
            item.update(eligible=False, exclusion_reason="incomplete_answer_or_empty_source")
            frozen.append(item)
            continue
        spans = source.sentence_offsets(raw)
        parts = [raw[s["start_byte"] : s["end_byte"]].decode() for s in spans]
        target = source.target_span(answer)
        sentence = answer[: target["end_byte"]].decode().strip()
        words = set(re.findall(r"\w+", sentence.casefold()))
        witness = min(
            range(len(parts)),
            key=lambda i: (
                -len(words & set(re.findall(r"\w+", parts[i].casefold()))),
                spans[i]["start_byte"],
            ),
        )
        neighbors = [i for i in range(len(parts)) if abs(i - witness) <= 1]
        added = [i for i in neighbors if i != witness]
        target_tokens = count("".join(parts[i] for i in added))
        sizes = [count(p) for p in parts]
        candidates = []
        for start in range(len(parts)):
            total = 0
            for end in range(start, len(parts)):
                if end in neighbors:
                    break
                total += sizes[end]
                if target_tokens > 0 and 4 * abs(total - target_tokens) <= target_tokens:
                    candidates.append((abs(total - target_tokens), start, end))
                if total > target_tokens * 1.25:
                    break
        filler: list[int] = []
        filler_tokens = 0
        for _, start, end in sorted(candidates):
            proposed = list(range(start, end + 1))
            actual = count("".join(parts[i] for i in proposed))
            if 4 * abs(actual - target_tokens) <= target_tokens:
                filler, filler_tokens = proposed, actual
                break
        item.update(
            witness_sentence_id=witness,
            filler_sentence_ids=filler,
            neighbor_added_tokens=target_tokens,
            filler_tokens=filler_tokens,
        )
        if not filler:
            item.update(eligible=False, exclusion_reason="no_disjoint_length_matched_filler")
        else:
            selections = [list(range(len(parts))), [witness], neighbors, sorted([witness, *filler])]
            for view, ids in zip(VIEWS, selections, strict=True):
                body = dict(
                    complete_source="".join(parts[i] for i in ids),
                    original_answer=sentence,
                    visible_source_sentence_ids=ids,
                    source_sentence_offsets=[spans[i] for i in ids],
                    target_sentence_sha256=target["text_sha256"],
                )
                messages = [
                    dict(role="system", content=SYSTEM),
                    dict(role="user", content=json.dumps(body, ensure_ascii=False, sort_keys=True)),
                ]
                tokens = count(json.dumps(messages, ensure_ascii=False))
                item["requests"][view] = dict(
                    messages=messages,
                    input_tokens=tokens,
                    visible_ids=ids,
                    source_view_sha256=source.digest(body["complete_source"].encode()),
                )
                if tokens + 96 > 8192:
                    item.update(eligible=False, exclusion_reason="context_budget_no_truncation")
        frozen.append(item)
        print(
            f"[exp7932] frozen_views completed_units={len(frozen)} monotonic_s={time.monotonic():.3f}",
            flush=True,
        )
    return frozen


def payload(item: Json, view: str, decoder: str) -> Json:
    """Change only the backend grammar, keeping model-visible messages identical."""
    if decoder not in DECODERS:
        raise ValueError("unplanned_decoder")
    value = dict(
        model=source.MODEL_ID,
        messages=item["requests"][view]["messages"],
        temperature=0,
        top_p=1,
        seed=67801,
        max_tokens=96,
        chat_template_kwargs=dict(enable_thinking=False),
    )
    if decoder == "grammar":
        value["grammar"] = GRAMMAR
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> Json:
    """Reject duplicate keys because a permissive parser could hide overwritten values."""
    if len(dict(pairs)) != len(pairs):
        raise ValueError("duplicate_key")
    return dict(pairs)


def invalid_constant(value: str) -> Any:
    """Reject non-JSON numeric spellings rather than treat NaN as a probability."""
    raise ValueError(value)


def parse_response(response: Json, visible_ids: list[int]) -> Json:
    """Check transport, syntax, schema and citation bounds as separate conditions."""
    result = dict(
        status="invalid_envelope",
        completed=False,
        syntax_valid=False,
        schema_valid=False,
        citation_valid=False,
        probability=None,
        source_sentence_id=None,
        finish_reason=None,
        usage=response.get("usage", {}),
    )
    try:
        choice = response["choices"][0]
        result["finish_reason"] = choice["finish_reason"]
        value = json.loads(
            choice["message"]["content"],
            object_pairs_hook=strict_object,
            parse_constant=invalid_constant,
        )
        result["syntax_valid"] = True
    except (KeyError, IndexError, TypeError, ValueError):
        result["status"] = "invalid_syntax" if "choices" in response else "invalid_envelope"
        return result
    status = "invalid_schema"
    if response.get("model") != source.MODEL_ID:
        status = "wrong_model"
    elif result["finish_reason"] != "stop" or result["usage"].get("completion_tokens", 97) > 96:
        status = "token_limit"
    elif (
        isinstance(value, dict)
        and set(value) == {"unsupported_probability", "source_sentence_id"}
        and type(value["unsupported_probability"]) in (float, int)
        and math.isfinite(value["unsupported_probability"])
        and 0 <= value["unsupported_probability"] <= 1
        and (value["source_sentence_id"] is None or type(value["source_sentence_id"]) is int)
    ):
        result["schema_valid"] = True
        if value["source_sentence_id"] is None or value["source_sentence_id"] in visible_ids:
            result.update(
                completed=True,
                citation_valid=True,
                probability=value["unsupported_probability"],
                source_sentence_id=value["source_sentence_id"],
            )
            status = "completed"
        else:
            status = "invalid_citation"
    result["status"] = status
    return result


def capture(
    manifest: list[Json],
    runtime: Any,
    raw: Path,
    *,
    latest_launch_s: float = 2400,
    token_budget: int = 36864,
    started_s: float | None = None,
) -> list[Json]:
    """Keep every intended cell, with no replacement or retry after a failed reply."""
    started = time.monotonic() if started_s is None else started_s
    rows: list[Json] = []
    rng = random.Random(68832)
    used = 0
    for item in manifest:
        for view in VIEWS:
            order = list(DECODERS)
            rng.shuffle(order)
            for position, decoder in enumerate(order):
                row = dict(
                    family_id=item["family_id"],
                    source_group=item["source_group"],
                    view=view,
                    decoder=decoder,
                    arm=decoder,
                    paired_order=order,
                    order_position=position,
                    seed=67801,
                    intended=True,
                    eligible=item["eligible"],
                    started=False,
                    completed=False,
                    failed=False,
                    censored=False,
                    excluded=not item["eligible"],
                    status="excluded_ineligible",
                    exclusion_reason=item["exclusion_reason"],
                    probability=None,
                    source_sentence_id=None,
                    syntax_valid=None,
                    schema_valid=None,
                    citation_valid=None,
                    request_bytes=None,
                    response_bytes=None,
                    request_sha256=None,
                    response_sha256=None,
                    usage={},
                    finish_reason=None,
                )
                if item["eligible"]:
                    request = payload(item, view, decoder)
                    request_bytes = json.dumps(
                        request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    row.update(
                        request_bytes=request_bytes,
                        request_sha256=source.digest(request_bytes.encode()),
                        visible_ids=item["requests"][view]["visible_ids"],
                        input_tokens=item["requests"][view]["input_tokens"],
                    )
                    if (
                        time.monotonic() - started >= min(latest_launch_s, 2880)
                        or used + 96 > token_budget
                    ):
                        row["status"] = "unstarted_budget"
                    else:
                        row.update(started=True, started_monotonic_ns=time.monotonic_ns())
                        print(
                            f"[exp7932] before_generation completed_units={len(rows)} elapsed_s={time.monotonic() - started:.3f}",
                            flush=True,
                        )
                        try:
                            response = runtime.generate(request)
                            encoded = json.dumps(
                                response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                            )
                            row.update(
                                parse_response(response, row["visible_ids"]),
                                response_bytes=encoded,
                                response_sha256=source.digest(encoded.encode()),
                            )
                            used += response.get("usage", {}).get("completion_tokens", 96)
                        except (TimeoutError, OSError, RuntimeError) as error:
                            row.update(
                                status="timeout"
                                if isinstance(error, TimeoutError)
                                else "transport_error",
                                censored=isinstance(error, TimeoutError),
                                error=f"{type(error).__name__}:{error}",
                            )
                            used += 96
                        row.update(
                            ended_monotonic_ns=time.monotonic_ns(),
                            failed=not row["completed"] and not row["censored"],
                        )
                        print(
                            f"[exp7932] after_generation completed_units={len(rows) + 1} elapsed_s={time.monotonic() - started:.3f}",
                            flush=True,
                        )
                rows.append(row)
        atomic_json(
            raw / "checkpoint.json",
            dict(
                input_hash=canonical_hash(manifest),
                config_hash=canonical_hash(freeze_config()),
                code_hash=sha256_file(Path(__file__)),
                rows=rows,
            ),
        )
        print(
            f"[exp7932] family_complete completed_units={len(rows) // 8} elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    return rows


def bootstrap(values: list[list[float]]) -> Json:
    """Resample source groups together so decoder and view pairs stay aligned."""
    if not values:
        return dict(estimate=None, ci95=None)
    matrix = np.asarray(values, dtype=float)
    draw = np.random.default_rng(68832).integers(0, len(matrix), size=(10000, len(matrix)))
    sampled = matrix[draw].mean(axis=1)
    interval = np.quantile(sampled, [0.025, 0.975], axis=0).tolist()
    return dict(estimate=matrix.mean(axis=0).tolist(), ci95=interval)


def reduce_rows(rows: list[Json]) -> Json:
    """Recompute completion and descriptive sensitivity without sentence labels."""
    families: Json = {}
    for row in rows:
        fid = row["family_id"]
        if fid in families and families[fid]["group"] != row["source_group"]:
            raise ValueError("group_drift")
        family = families.setdefault(fid, dict(group=row["source_group"], cells={}))
        key = row["decoder"] + ":" + row["view"]
        if key in family["cells"]:
            raise ValueError("duplicate_cell")
        family["cells"][key] = row
    clusters: Json = {}
    sensitivity: Json = {}
    totals = dict.fromkeys(DECODERS, 0)
    both = wins = losses = 0
    for family in families.values():
        cells = family["cells"]
        complete = {
            d: all(cells.get(d + ":" + v, {}).get("completed", False) for v in VIEWS)
            for d in DECODERS
        }
        for d in DECODERS:
            totals[d] += complete[d]
        wins += complete["grammar"] and not complete["plain"]
        losses += complete["plain"] and not complete["grammar"]
        clusters.setdefault(family["group"], []).append(
            float(complete["grammar"]) - float(complete["plain"])
        )
        if all(complete.values()):
            both += 1
            sensitivity.setdefault(family["group"], []).append(
                [
                    cells[d + ":witness_neighbors"]["probability"]
                    - cells[d + ":witness_filler"]["probability"]
                    for d in DECODERS
                ]
            )
    paired = bootstrap([[sum(v) / len(v)] for v in clusters.values()])
    discordant = wins + losses
    p = min(
        1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant
    )
    delta = (totals["grammar"] - totals["plain"]) / 48
    ci = [paired["ci95"][0][0], paired["ci95"][1][0]] if paired["ci95"] else None
    sufficient = both >= 32 and len(sensitivity) >= 32
    secondary = (
        bootstrap([np.mean(v, axis=0).tolist() for v in sensitivity.values()])
        if sufficient
        else dict(estimate=None, ci95=None)
    )
    counts = {
        key: sum(bool(row.get(key)) for row in rows)
        for key in ["eligible", "started", "completed", "failed", "censored", "excluded"]
    }
    counts.update(
        intended=384,
        intended_families=48,
        accounted=len(rows),
        accounted_families=len(families),
        eligible_families=sum(
            any(r["eligible"] for r in f["cells"].values()) for f in families.values()
        ),
        independent=len(sensitivity),
        intended_source_clusters=len(clusters),
        actual_output_tokens=sum(r["usage"].get("completion_tokens", 0) for r in rows),
        call_limit=384,
        output_token_limit=36864,
    )
    return dict(
        sample_size_budget=counts,
        complete_family_fraction={d: totals[d] / 48 for d in DECODERS},
        paired_completion_delta=dict(
            estimate=delta,
            ci95=ci,
            exact_p=p,
            grammar_only=wins,
            plain_only=losses,
            bootstrap_draws=10000,
            interval_method="paired_source_cluster_percentile_95",
            protocol_benefit=bool(delta >= 0.05 and ci and ci[0] > 0 and p < 0.05),
        ),
        semantic_sensitivity=dict(
            complete_in_both=both,
            independent_source_clusters=len(sensitivity),
            powered=sufficient,
            arms={
                d: dict(
                    estimate=secondary["estimate"][i] if sufficient else None,
                    ci95=[secondary["ci95"][0][i], secondary["ci95"][1][i]] if sufficient else None,
                )
                for i, d in enumerate(DECODERS)
            },
            interpretation="descriptive neighbors-minus-filler probability; no entailment or factual benefit",
        ),
        sentence_label_eligibility=dict(eligible=0, reason="no_independent_sentence_labels"),
        natural_brier=None,
        natural_cost=None,
        syntax_valid=all(r["syntax_valid"] for r in rows if r["started"]),
    )
