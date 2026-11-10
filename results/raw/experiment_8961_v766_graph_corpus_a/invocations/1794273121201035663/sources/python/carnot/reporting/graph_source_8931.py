"""REQ-REPORT-8931: freeze graph truth separately from response-only features."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import fcntl
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from carnot.inference.qwen_sufficiency_7920 import bounded
from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
SEED = 764031
SETTINGS = dict(pilot.SETTINGS, seed=SEED, logprobs=True, top_logprobs=2)
FEATURE_SCHEMA = dict(
    version="response_confidence_v1",
    learned_features=["mean_token_nll", "first_answer_top2_margin"],
    mean_token_nll="Negative mean of server logprob for every emitted content token, including whitespace and punctuation; no implicit EOS.",
    first_answer_top2_margin="Sorted top1 minus top2 server logprob at the first non-whitespace content token, including malformed text.",
    finite_checks="One to sixteen content tokens, exact text reconstruction, finite nonpositive logprobs, two distinct alternatives.",
    nuisance_only=["token_count", "parser_category"],
    forbidden=[
        "graph",
        "node_identity",
        "start",
        "distance",
        "answer_set",
        "label",
        "path",
        "split",
    ],
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so collection and supervision share visible progress."""
    print(f"[exp8931] phase={phase} completed={completed} pending={pending}", flush=True)


def distance_nodes(nodes: list[str], edges: list[list[str]], start: str, k: int) -> list[str]:
    """BFS records the first visit, so a longer walk cannot replace a shorter path."""
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return sorted(node for node, distance in distances.items() if distance == k)


def graph_hash(graph: Json) -> str:
    """Graph identity excludes roles, questions and labels so roles cannot hide reuse."""
    return str(canonical_hash(dict(nodes=graph["nodes"], edges=graph["edges"])))


def candidate(seed: int) -> Json:
    """Choose size, names, edges and start before looking at any model response."""
    rng = random.Random(seed)
    n = rng.randint(50, 60)
    nodes = sorted(
        rng.sample(
            [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], n
        )
    )
    edges = [
        [a, b] for i, a in enumerate(nodes) for b in nodes[i + 1 :] if rng.random() < 3 / (n - 1)
    ]
    return dict(nodes=nodes, edges=edges, start=rng.choice(nodes), edge_probability=3 / (n - 1))


def graph_protocol(old_graphs: list[Json]) -> Json:
    """Seal 370 fresh graph identities and label-blind roles with every rejection retained."""
    old_seeds = {g["seed"] for g in old_graphs}
    seen = {graph_hash(g) for g in old_graphs}
    graphs, splits, rejections = [], [], []
    roles = ["train"] * 80 + ["calibration"] * 40 + ["test"] * 40
    random.Random(SEED).shuffle(roles)
    for index in range(370):
        control = index < 50
        shard = "control" if control else "A" if index < 210 else "B"
        role = "control" if control else roles[(index - 50) % 160]
        ks = [2, 3] if control else [2]
        for attempt in range(100):
            seed = 2**40 + SEED * 100000 + index * 100 + attempt
            graph = candidate(seed)
            digest = graph_hash(graph)
            answers = {
                str(k): distance_nodes(graph["nodes"], graph["edges"], graph["start"], k)
                for k in ks
            }
            reason = (
                "duplicate_graph_or_seed"
                if seed in old_seeds or digest in seen
                else "empty_requested_distance"
                if not all(answers.values())
                else None
            )
            if reason is None:
                break
            rejections.append(
                dict(
                    seed=seed,
                    graph_hash=digest,
                    reason=reason,
                    scheduled_index=index,
                    attempt=attempt,
                )
            )
        else:
            raise ValueError("graph_rejection_budget")
        seen.add(digest)
        graph_id = f"graph-{index:03d}"
        questions = []
        for k in ks:
            prompt = (
                "Undirected simple graph. Nodes: "
                + ", ".join(graph["nodes"])
                + ".\nEdges: "
                + ", ".join(a + "-" + b for a, b in graph["edges"])
                + f".\nName a node exactly {k} hops from {graph['start']} (shortest distance). Reply with only its two-letter label."
            )
            questions.append(
                dict(
                    unit_id=f"{graph_id}-k{k}",
                    k=k,
                    expected=answers[str(k)],
                    prompt=prompt,
                    prompt_hash=canonical_hash(prompt),
                )
            )
        graphs.append(
            dict(
                graph,
                graph_id=graph_id,
                seed=seed,
                graph_hash=digest,
                role=role,
                shard=shard,
                questions=questions,
            )
        )
        if not control:
            splits.append(
                dict(graph_id=graph_id, seed=seed, graph_hash=digest, role=role, shard=shard)
            )
        if (index + 1) % 50 == 0:
            progress("graph_freeze", index + 1, 369 - index)
    return dict(
        random_seed=SEED,
        graph_manifest=graphs,
        split_manifest=splits,
        shard_manifest=[
            dict(
                shard=shard,
                graph_ids=[g["graph_id"] for g in graphs if g["shard"] == shard],
                roles=dict(train=80, calibration=40, test=40),
            )
            for shard in ("A", "B")
        ],
        graph_rejections=rejections,
        feature_schema=FEATURE_SCHEMA,
        generator_settings=SETTINGS,
        control_acceptance=dict(
            graphs=50,
            answers=100,
            all_answers=True,
            verified_labels=True,
            usable_signals=True,
            empirical_inequality="error_k3 > error_k2",
            uncertainty="paired differences, normal 95% interval; no statistical proof",
        ),
        collection_acceptance=dict(
            graphs=320,
            independent_errors_at_least=200,
            basis="ceil(1.3*200/0.86)=303; use two balanced 160-graph shards",
        ),
        bounds=dict(
            shard_generation_s=1800,
            per_request_s=30,
            preflight_s=240,
            load_s=240,
            retry_count=0,
            gpu=0,
            reasoning=False,
        ),
    )


def scheduled(protocol: Json, shard: str) -> list[Json]:
    """Expand fixed queries without changing graph-level independence or roles."""
    return [
        dict(g, **q, cluster_id=g["graph_id"])
        for g in protocol["graph_manifest"]
        if g["shard"] == shard
        for q in g["questions"]
    ]


def category(answer: str, nodes: list[str], expected: list[str]) -> str:
    """Reuse the historical strict parser without repairing ambiguous answers."""
    node, reason = pilot.parse_answer(answer, nodes)
    return (
        "malformed"
        if reason == "invalid_single_node"
        else "out_of_graph"
        if reason
        else "correct"
        if node in expected
        else "valid_wrong_distance"
    )


def signals(response: Json) -> Json:
    """Accept only emitted server evidence; this function has no graph or truth input."""
    try:
        choice = response["choices"][0]
        tokens = choice["logprobs"]["content"]
        if (
            not 1 <= len(tokens) <= 16
            or "".join(t["token"] for t in tokens) != choice["message"]["content"]
        ):
            raise ValueError("token_alignment")
        first = next(t for t in tokens if t["token"].strip())
        alternatives = first["top_logprobs"]
        values = [t["logprob"] for t in tokens] + [t["logprob"] for t in alternatives]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v > 0
            for v in values
        ):
            raise ValueError("nonfinite_or_invalid_logprob")
        if len(alternatives) < 2 or len({t["token"] for t in alternatives}) != len(alternatives):
            raise ValueError("top2_missing_or_duplicate")
        top = sorted((t["logprob"] for t in alternatives), reverse=True)
        features = dict(
            mean_token_nll=-sum(t["logprob"] for t in tokens) / len(tokens),
            first_answer_top2_margin=top[0] - top[1],
        )
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError("nonfinite_derived_feature")
        return dict(features=features, token_count=len(tokens))
    except (KeyError, IndexError, TypeError, StopIteration, AttributeError) as error:
        raise ValueError("server_logprobs_missing_or_malformed") from error


def censored(unit: Json, reason: str, duration: float = 0) -> Json:
    """Every scheduled slot remains visible without inventing a reasoning label."""
    return dict(
        unit_id=unit["unit_id"],
        cluster_id=unit["cluster_id"],
        seed=unit["seed"],
        arm=f"qwen_reasoning_off_k{unit['k']}",
        metric="final_answer_error",
        k=unit["k"],
        value=None,
        numerator=None,
        denominator=1,
        status="censored",
        missing_reason=reason,
        parser_category=None,
        answer=None,
        features=None,
        signal_status="blocked",
        duration_s=duration,
    )


def label_response(unit: Json, response: Json) -> Json:
    """Derive labels only after raw persistence; missing signals remain a separate gate."""
    expected = distance_nodes(unit["nodes"], unit["edges"], unit["start"], unit["k"])
    if expected != unit["expected"] or not expected:
        raise ValueError("label_truth")
    try:
        answer = response["choices"][0]["message"]["content"]
        if not isinstance(answer, str):
            raise ValueError("response_content")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response_content") from error
    parsed = category(answer, unit["nodes"], expected)
    row = dict(
        censored(unit, ""),
        status="completed",
        missing_reason=None,
        answer=answer,
        parser_category=parsed,
        value=int(parsed != "correct"),
        numerator=int(parsed != "correct"),
    )
    try:
        row.update(signals(response), signal_status="usable", signal_missing_reason=None)
    except ValueError as error:
        row.update(signal_missing_reason=str(error), token_count=None)
    return row


def collect(
    units: list[Json],
    transport: Callable[[Json, float], Json],
    raw: Path,
    model: Json,
    *,
    budget_s: float = 1800,
    request_timeout_s: float = 30,
) -> list[Json]:
    """Use an already owned transport; durable intent prevents retry selection after a crash."""
    import re

    if len({u["unit_id"] for u in units}) != len(units) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", u["unit_id"]) for u in units
    ):
        raise ValueError("unit_roster")
    if (
        model.get("hf_id") != SETTINGS["model"]
        or not isinstance(model.get("sha256"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", model["sha256"])
    ):
        raise ValueError("model_binding")
    budget_s = min(budget_s, 1800)
    binding = canonical_hash(
        dict(
            model=model,
            settings=SETTINGS,
            feature_schema=FEATURE_SCHEMA,
            units=[
                dict(
                    unit_id=u["unit_id"],
                    graph=graph_hash(u),
                    prompt=u["prompt"],
                    k=u["k"],
                    start=u["start"],
                    role=u["role"],
                    shard=u["shard"],
                )
                for u in units
            ],
        )
    )
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract = raw / "contract.json"
        if contract.exists() and json.loads(contract.read_text()) != dict(binding=binding):
            raise ValueError("checkpoint_binding")
        atomic_json(contract, dict(binding=binding))
        spent = sum(
            json.loads(p.read_text()).get("duration_s", 0) for p in raw.glob("*/response.json")
        )
        spent += sum(
            json.loads(p.read_text())["row"]["duration_s"] for p in raw.glob("*/failure.json")
        )
        started = time.monotonic()
        rows = []
        for index, unit in enumerate(units):
            folder = raw / unit["unit_id"]
            intent, response_path, label = (
                folder / "intent.json",
                folder / "response.json",
                folder / "label.json",
            )
            failure = folder / "failure.json"
            progress("before_unit", index, len(units) - index)
            if response_path.exists():
                envelope = json.loads(response_path.read_text())
            elif failure.exists():
                failed = json.loads(failure.read_text())
                row = failed["row"]
                if failed["binding"] != binding or row != censored(
                    unit, row["missing_reason"], row["duration_s"]
                ):
                    raise ValueError("failure_binding")
                rows.append(row)
                progress("reused_censored", index + 1, len(units) - index - 1)
                continue
            elif intent.exists():
                row = censored(unit, "interrupted_after_intent", request_timeout_s)
                atomic_json(failure, dict(binding=binding, row=row))
                spent += request_timeout_s
                rows.append(row)
                progress("interrupted_censored", index + 1, len(units) - index - 1)
                continue
            elif spent + time.monotonic() - started >= budget_s:
                rows.append(censored(unit, "shard_budget"))
                progress("budget_censored", index + 1, len(units) - index - 1)
                continue
            else:
                timeout = min(
                    request_timeout_s, 30, budget_s - spent - (time.monotonic() - started)
                )
                payload = dict(SETTINGS, messages=[dict(role="user", content=unit["prompt"])])
                atomic_json(intent, dict(binding=binding, request=payload))
                began = time.monotonic()
                progress("before_generation", index, len(units) - index)
                try:
                    response = bounded(lambda: transport(payload, timeout), timeout, interval_s=15)
                except (
                    RuntimeError,
                    OSError,
                    ValueError,
                    KeyError,
                    IndexError,
                    TimeoutError,
                ) as error:
                    row = censored(unit, str(error), time.monotonic() - began)
                    atomic_json(failure, dict(binding=binding, row=row))
                    rows.append(row)
                    progress("after_generation_censored", index + 1, len(units) - index - 1)
                    continue
                envelope = dict(
                    binding=binding, raw_response=response, duration_s=time.monotonic() - began
                )
                atomic_json(response_path, envelope)
                progress("after_generation", index + 1, len(units) - index - 1)
            if envelope["binding"] != binding:
                raise ValueError("response_binding")
            try:
                row = label_response(unit, envelope["raw_response"])
            except ValueError as error:
                if str(error) != "response_content":
                    raise
                row = censored(unit, "malformed_transport_response")
            row["duration_s"] = envelope["duration_s"]
            if label.exists() and json.loads(label.read_text()) != row:
                raise ValueError("label_checkpoint_drift")
            atomic_json(label, row)
            if row["features"] is not None:
                atomic_json(folder / "detector.json", row["features"])
            rows.append(row)
            progress("unit_checkpoint", index + 1, len(units) - index - 1)
        return rows


def paired_control(units: list[Json], rows: list[Json]) -> Json:
    """Pair by graph; a positive empirical difference does not prove a population effect."""
    roster = {u["unit_id"]: u for u in units}
    by_id = {r["unit_id"]: r for r in rows}
    if (
        len(units) != 100
        or len(roster) != 100
        or len(by_id) != len(rows)
        or not set(by_id) <= set(roster)
    ):
        raise ValueError("control_roster")
    complete = len(rows) == 100 and all(r["status"] == "completed" for r in rows)
    verified = all(
        r["status"] == "completed"
        and roster[r["unit_id"]]["expected"]
        == distance_nodes(
            roster[r["unit_id"]]["nodes"],
            roster[r["unit_id"]]["edges"],
            roster[r["unit_id"]]["start"],
            roster[r["unit_id"]]["k"],
        )
        and r["parser_category"]
        == category(r["answer"], roster[r["unit_id"]]["nodes"], roster[r["unit_id"]]["expected"])
        and r["value"] == int(r["parser_category"] != "correct")
        and type(r["value"]) is int
        and type(r["denominator"]) is int
        and r["denominator"] == 1
        and r["numerator"] == r["value"]
        and r["cluster_id"] == roster[r["unit_id"]]["cluster_id"]
        and r["seed"] == roster[r["unit_id"]]["seed"]
        for r in rows
    )
    usable = complete and all(
        isinstance(r.get("features"), dict)
        and set(r["features"]) == set(FEATURE_SCHEMA["learned_features"])
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
            for v in r["features"].values()
        )
        for r in rows
    )
    differences = []
    k2 = k3 = None
    interval = None
    if complete and verified:
        k2 = sum(r["value"] for r in rows if roster[r["unit_id"]]["k"] == 2) / 50
        k3 = sum(r["value"] for r in rows if roster[r["unit_id"]]["k"] == 3) / 50
        for graph in sorted({u["cluster_id"] for u in units}):
            pair = {u["k"]: by_id[u["unit_id"]]["value"] for u in units if u["cluster_id"] == graph}
            differences.append(pair[3] - pair[2])
        mean = sum(differences) / 50
        se = math.sqrt(sum((d - mean) ** 2 for d in differences) / 49 / 50)
        interval = dict(
            method="graph-paired normal 95%",
            mean_difference=mean,
            low=max(-1, mean - 1.959963984540054 * se),
            high=min(1, mean + 1.959963984540054 * se),
            standard_error=se,
            denominator=50,
        )
    return dict(
        accepted=bool(
            complete and verified and usable and k3 is not None and k2 is not None and k3 > k2
        ),
        all_answers=complete,
        labels_verified=verified,
        usable_signals=usable,
        error_k2=k2,
        error_k3=k3,
        paired_interval=interval,
        paired_differences=differences,
        independent_count=50 if complete else 0,
        inequality_is_statistical_proof=False,
    )
