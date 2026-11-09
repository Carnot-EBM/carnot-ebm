"""REQ-REPORT-8945: measure opportunity without changing the failed depth verdict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import depth_control_8934 as depth
from carnot.reporting import graph_source_8931 as source
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.graph_source_methods_8931 import CHECKS, reference as reference
from carnot.reporting.v756_branch_qualification import verify_measured

Json = dict[str, Any]
ROOT = pilot.ROOT
NAME = "experiment_8945_v765_depth_opportunity_audit"
TASK = "exp8945-depth-opportunity-audit"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/depth_opportunity_8945.py",
    "python/carnot/reporting/depth_opportunity_runner_8945.py",
    CLI,
]
TEST = "tests/python/test_depth_opportunity_8945.py"
INPUTS = {
    "results/experiment_8931_v764_graph_source_methods.json": "sha256:810b976a4d7883a3a600c9501982eb882d1feb658b0a54cb86afea00320f5b57",
    "results/experiment_8934_v764_depth_control.json": "sha256:7e82e593722459bf4eb5705118925d22c7ce698869d235a80f4bd999baedd47d",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so a supervisor can distinguish work from a stalled process."""
    print(f"[exp8945] phase={phase} completed={completed} pending={pending}", flush=True)


class Rejected(ValueError):
    """Carry the exact failed external check instead of hiding it in a generic block."""

    def __init__(self, receipt: Json):
        super().__init__(receipt["check"])
        self.receipt = receipt


def require(actual: Any, expected: Any, path: Path, field: str, upstream: str) -> None:
    """Type-sensitive comparisons prevent booleans or missing values from passing counts."""
    if actual != expected or type(actual) is not type(expected):
        kind = (
            "missing_file"
            if not path.is_file()
            else "missing_field"
            if actual is None
            else "observed_zero"
            if actual == 0
            else "rejected_producer"
        )
        raise Rejected(
            dict(
                upstream=upstream,
                path=str(path),
                hash=reference(path)["sha256"],
                check="retained_evidence",
                artifact_field=field,
                expected=expected,
                observed=actual,
                failure_kind=kind,
            )
        )


def all_pairs(nodes: list[str], edges: list[list[str]]) -> dict[str, dict[str, int]]:
    """Floyd-Warshall uses matrix relaxation, independently of the producer's BFS queue."""
    infinity = len(nodes) + 1
    distances = {a: {b: 0 if a == b else infinity for b in nodes} for a in nodes}
    for a, b in edges:
        distances[a][b] = distances[b][a] = 1
    for via in nodes:
        for a in nodes:
            for b in nodes:
                distances[a][b] = min(distances[a][b], distances[a][via] + distances[via][b])
    return distances


def authenticate(root: Path) -> Json:
    """Authenticate retained producers directly; retired task gates never run again."""
    result: Json = dict(failures=[], references=[], methods={}, depth={}, work={})
    path, upstream = root, TASK
    try:
        values = []
        for name, pin in INPUTS.items():
            path = root / name
            upstream = {"8931": "exp8931-graph-source-methods", "8934": "exp8934-depth-control"}[
                path.name.split("_")[1]
            ]
            ref = reference(path)
            result["references"].append(ref)
            require(ref["sha256"], pin, path, "sha256", upstream)
            value = json.loads(path.read_text())
            require(
                value.get("required_checks_passed"), True, path, "required_checks_passed", upstream
            )
            for ref in (
                value["raw_shard_hashes"]
                + value["code_config_hashes"]
                + value["source_artifact_hashes"]
            ):
                retained = Path(ref["path"])
                require(reference(retained)["sha256"], ref["sha256"], retained, "sha256", upstream)
            values.append(value)
        methods, old = values
        result.update(methods=methods, depth=old)
        path = Path(old["raw_shard_hashes"][0]["path"])
        work = json.loads(path.read_text())
        units = depth.ordered_units(methods)
        require(work["units"], units, path, "units", upstream)
        require(len(units), 100, path, "answer_count", upstream)
        require(len({u["cluster_id"] for u in units}), 50, path, "graph_count", upstream)
        refs = {r["path"]: r for r in old["raw_shard_hashes"]}
        for i, (unit, row) in enumerate(zip(units, old["rows"], strict=True)):
            distances = all_pairs(unit["nodes"], unit["edges"])[unit["start"]]
            expected = sorted(n for n, d in distances.items() if d == unit["k"])
            require(unit["expected"], expected, path, unit["unit_id"] + ".expected", upstream)
            require(
                source.graph_hash(unit),
                unit["graph_hash"],
                path,
                unit["unit_id"] + ".graph_hash",
                upstream,
            )
            require(
                canonical_hash(unit["prompt"]),
                unit["prompt_hash"],
                path,
                unit["unit_id"] + ".prompt_hash",
                upstream,
            )
            response_path = Path(
                next(p for p in refs if p.endswith("/" + unit["unit_id"] + "/response.json"))
            )
            for retained in [
                response_path.with_name(name)
                for name in ("intent.json", "wire_response.json", "label.json")
            ] + [response_path.parent.parent / "contract.json"]:
                require(
                    reference(retained)["sha256"],
                    refs[str(retained)]["sha256"],
                    retained,
                    "consumed_raw_sha256",
                    upstream,
                )
            envelope = json.loads(response_path.read_text())
            intent = json.loads(response_path.with_name("intent.json").read_text())
            wire = json.loads(response_path.with_name("wire_response.json").read_text())
            contract = json.loads((response_path.parent.parent / "contract.json").read_text())
            require(
                intent,
                dict(
                    binding=contract["binding"],
                    request=dict(
                        source.SETTINGS, messages=[dict(role="user", content=unit["prompt"])]
                    ),
                ),
                response_path,
                "intent.request",
                upstream,
            )
            require(
                envelope["binding"],
                contract["binding"],
                response_path,
                "response.binding",
                upstream,
            )
            require(wire, envelope["raw_response"], response_path, "wire_response", upstream)
            require(
                wire["model"], source.SETTINGS["model"], response_path, "response.model", upstream
            )
            rebuilt = source.label_response(unit, wire)
            rebuilt["duration_s"] = envelope["duration_s"]
            require(
                json.loads(response_path.with_name("label.json").read_text()),
                rebuilt,
                response_path,
                "label",
                upstream,
            )
            rebuilt["actual_token_usage"] = wire["usage"]
            require(row, rebuilt, response_path, "response_label", upstream)
            require(
                row["token_count"],
                wire["usage"]["completion_tokens"],
                response_path,
                "completion_tokens",
                upstream,
            )
            if (i + 1) % 20 == 0:
                progress("authenticated_answers", i + 1, 99 - i)
        reduced = depth.reduce(units, old["rows"])
        require(
            reduced["paired_interval"], old["paired_interval"], path, "paired_interval", upstream
        )
        require(old["depth_source_ready_score"], 0, path, "depth_source_ready_score", upstream)
        require(
            old["honest_verdict"],
            "complete_null_paired_depth_control",
            path,
            "honest_verdict",
            upstream,
        )
        result["work"] = work
    except Rejected as error:
        result["failures"].append(error.receipt)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as error:
        result["failures"].append(
            dict(
                upstream=upstream,
                path=str(path),
                hash=reference(path)["sha256"],
                check="retained_structure",
                artifact_field="producer_or_raw_record",
                expected="complete original evidence",
                observed=str(error),
                failure_kind="rejected_producer",
            )
        )
    return result


def audit(evidence: Json) -> Json:
    """Reduce descriptive opportunity by graph, preserving the old failed source gate."""
    rows, pairs = [], []
    valid = not evidence["failures"]
    if valid:
        units, old = evidence["work"]["units"], evidence["depth"]
        for index in range(0, 100, 2):
            unit = units[index]
            distances = all_pairs(unit["nodes"], unit["edges"])[unit["start"]]
            neighbors = {n for n, d in distances.items() if d == 1}
            pair: Json = dict(
                cluster_id=unit["cluster_id"],
                seed=unit["seed"],
                graph_hash=unit["graph_hash"],
                node_count=len(unit["nodes"]),
                edge_count=len(unit["edges"]),
                start_degree=len(neighbors),
                mean_degree=2 * len(unit["edges"]) / len(unit["nodes"]),
                disconnected_nodes=sorted(
                    n for n, d in distances.items() if d > len(unit["nodes"])
                ),
                status="completed",
                censor_reason=None,
            )
            for i in (index, index + 1):
                u, r = units[i], old["rows"][i]
                expected = sorted(n for n, d in distances.items() if d == u["k"])
                opportunity = dict(
                    valid_set=expected,
                    valid_set_size=len(expected),
                    chance_accuracy=len(expected) / len(u["nodes"]),
                    direct_neighbor_overlap=len(set(expected) & neighbors),
                    actual_answer_category=r["parser_category"],
                    answer=r["answer"],
                    token_count=r["token_count"],
                    error=r["value"],
                    prompt_hash=u["prompt_hash"],
                )
                pair[f"k{u['k']}"] = opportunity
                rows.append(
                    dict(
                        r,
                        **{k: v for k, v in opportunity.items() if k not in r},
                        condition=f"k{u['k']}",
                        censor_reason=None,
                        node_count=len(u["nodes"]),
                    )
                )
            pair.update(
                paired_error=pair["k3"]["error"] - pair["k2"]["error"],
                valid_set_size_difference=pair["k3"]["valid_set_size"]
                - pair["k2"]["valid_set_size"],
                chance_accuracy_difference=pair["k3"]["chance_accuracy"]
                - pair["k2"]["chance_accuracy"],
            )
            pairs.append(pair)
        interval = source.paired_control(units, old["rows"])["paired_interval"]
    else:
        for i in range(50):
            for k in (2, 3):
                rows.append(
                    dict(
                        unit_id=f"graph-{i:03d}-k{k}",
                        cluster_id=f"graph-{i:03d}",
                        arm=f"qwen_reasoning_off_k{k}",
                        condition=f"k{k}",
                        k=k,
                        metric="final_answer_error",
                        value=None,
                        numerator=None,
                        denominator=1,
                        status="censored",
                        censor_reason="external_evidence",
                        missing_reason="external_evidence",
                    )
                )
        interval = None
    means: Json = {
        f"k{k}": sum(r["valid_set_size"] for r in rows if r["k"] == k) / 50 if valid else None
        for k in (2, 3)
    }
    chance = {
        f"k{k}": sum(r["chance_accuracy"] for r in rows if r["k"] == k) / 50 if valid else None
        for k in (2, 3)
    }
    old = evidence["depth"]
    return dict(
        rows=rows,
        paired_opportunity_rows=pairs,
        label_agreement=dict(
            method="independent Floyd-Warshall all-pairs versus stored BFS",
            matched=100 if valid else 0,
            checked=100 if valid else 0,
            intended=100,
            all_agree=valid,
        ),
        mean_valid_set_size=means,
        mean_chance_accuracy=chance,
        opportunity_confounded_score=int(valid and means["k3"] > means["k2"]),
        paired_depth_interval=interval,
        historical_gate_preserved={
            key: old.get(key)
            for key in (
                "honest_verdict",
                "verdict_class",
                "depth_source_ready_score",
                "signal_capture_ready_score",
                "paired_interval",
            )
        },
        causal_explanation=None,
        intended_count=100,
        completed_count=100 if valid else 0,
        failed_count=0,
        censored_count=0 if valid else 100,
        excluded_count=0,
        independent_count=50 if valid else 0,
    )


def qualified(validation: Json) -> bool:
    """Read measured SQLite coverage; a claimed percentage alone cannot license readiness."""
    receipts = validation["receipts"]
    return (
        bool(receipts)
        and {r["name"] for r in receipts} == CHECKS
        and all(r["passed"] and r["exit_code"] == 0 and not r.get("timed_out") for r in receipts)
        and verify_measured(validation["coverage"], OWNED)
    )


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Keep diagnostic completion separate from evidence of a causal depth effect."""
    result = audit(work["evidence"])
    owned = qualified(validation)
    failures = work["evidence"]["failures"]
    cls = "disqualified" if not owned and not fixture else "blocked" if failures else "null"
    value = dict(work["template"])
    value.update(
        result,
        experiment_id=8945,
        task_id=TASK,
        milestone="2026.11.765",
        run_date="20261109",
        random_seed=765045,
        honest_verdict="complete_" + cls + "_depth_opportunity_audit",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        inference_substrate="deterministic_cpu",
        inference_substrate_class="retained_graph_opportunity_audit",
        execution_venue="host_cpu",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        current_generation_calls=0,
        verifier_is_oracle=True,
        exposure_scope="retained_50_graphs_100_final_answers",
        independent_generalization_score=0,
        model_invoked=False,
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"],
        started_at=work["started_at"],
        finished_at=work["finished_at"],
        preconditions_checked=dict(
            retained_inputs_authenticated=not failures,
            no_model_load_or_retry=True,
            historical_runtime_gates=False,
        ),
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            reference(Path(work["raw"]) / "work.json"),
            reference(Path(work["raw"]) / "validation.json"),
        ]
        + [
            dict(ref, scope="retained_producer_no_current_model_work")
            for producer in (work["evidence"]["methods"], work["evidence"]["depth"])
            for ref in producer.get("raw_shard_hashes", [])
        ],
        gate_check_summary=failures,
        acceptance_gates=dict(
            authenticated_50_pairs=not failures,
            independent_labels_agree=result["label_agreement"]["all_agree"],
            owned_checks=owned,
            historical_depth_gate_reopened=False,
        ),
        required_checks_passed=owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        diagnosis_ready_score=int(owned and not failures and not fixture),
        fixture_only=fixture,
        flagged_adversarial=False,
        methodology_note="Authenticate pinned original graphs, requests, responses and BFS labels; independently relax all-pairs distances. Uniform-node chance is valid-set size/n, not a model accuracy prediction. Means and the old error interval use 50 paired graph clusters. Larger k3 opportunity is descriptive; causal explanation is unsupported. No new generation or old collection reopening.",
        next_protocol_license="Separately freeze a matched-cardinality protocol; old collection remains closed.",
        reproduction_commands=[
            "PYTHONPATH=python .venv/bin/python -u " + CLI + " --date 20261109",
            ".venv/bin/python -u " + CLI + " --cold-replay results/" + NAME + ".json",
        ],
    )
    value["field_principles"] = {
        key: "Retained-data CPU diagnostic; no new model work, causal attribution or generalization."
        for key in value
    }
    value["field_principles"].update(
        diagnosis_ready_score="All 50 graph pairs, raw responses and independent labels authenticate, with owned validation passed.",
        opportunity_confounded_score="Mean valid-set size at k3 exceeds k2; structural imbalance only.",
        paired_depth_interval="Original normal 95-percent interval over 50 paired error differences.",
        historical_gate_preserved="Original null disposition and failed depth-source gate remain unchanged.",
    )
    value["schema"] = sorted(set(value) | {"reproducibility_checksum"})
    value["reproducibility_checksum"] = pilot.checksum(value)
    return value
