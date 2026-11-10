"""REQ-REPORT-8963: admit prospective graph evidence without a detector benefit claim."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import graph_corpus_a_8961 as shard
from carnot.reporting import relational_methods_8959 as methods
from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import graph_source_8931 as source
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.corpus_views_8963 import detector_view
from carnot.reporting.depth_opportunity_8945 import all_pairs, require, Rejected as Rejected

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, EXPERIMENT_ID, SEED = (
    "experiment_8963_v766_corpus_admission",
    "exp8963-corpus-admission",
    8963,
    766063,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/corpus_views_8963.py",
    "python/carnot/reporting/corpus_admission_8963.py",
    "python/carnot/reporting/corpus_admission_runner_8963.py",
    CLI,
]
TEST = "tests/python/test_corpus_admission_8963.py"
TESTS = [TEST]
INPUTS = [
    *shard.INPUTS,
    (
        "results/experiment_8961_v766_graph_corpus_a.json",
        "sha256:0ba3322b01052ad632c494186451b5ac29d375547147d55c0470a962bcb1f616",
        "shard_ready_score",
        "exp8961-graph-corpus-a",
    ),
    (
        "results/experiment_8962_v766_graph_corpus_b.json",
        "sha256:e941ec67fc5dbcb3949efd64f704a8a23c6feeb9140aa797b8d774905941960c",
        "shard_ready_score",
        "exp8962-graph-corpus-b",
    ),
]


def authenticate(root: Path) -> Json:
    """Pin original producers and retained bytes without invoking retired task gates."""
    result: Json = dict(failures=[], references=[], producers=[], history={})
    for input_index, (name, pin, field, task) in enumerate(INPUTS):
        source.progress("8963_authenticate_producer", input_index, len(INPUTS) - input_index)
        path = root / name
        ref = calibration.reference(path)
        result["references"].append(ref)
        try:
            require(ref["sha256"], pin, path, "sha256", task)
            value = json.loads(path.read_text())
            for key, expected in ((field, 1), ("task_id", task), ("required_checks_passed", True)):
                require(value.get(key), expected, path, key, task)
            shard.verify_references(value)
            result["references"].extend(
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            )
            result["producers"].append(value)
        except Rejected as error:
            result["failures"].append(dict(error.receipt, upstream=task))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["failures"].append(
                dict(
                    upstream=task,
                    path=str(path),
                    hash=ref["sha256"],
                    check="producer_structure",
                    artifact_field="producer_json",
                    expected="original complete object",
                    observed=str(error),
                    failure_kind="rejected_producer",
                )
            )
    if not result["failures"]:
        history = methods.authenticate(root)
        result["history"] = history
        result["references"].extend(history["references"])
        result["failures"].extend(history["failures"])
        frozen = result["producers"][0]
        if not result["failures"]:
            try:
                require(
                    methods.verify_graphs(
                        dict(graph_manifest=frozen["graph_manifest"], complete=True), history
                    ),
                    True,
                    root / INPUTS[0][0],
                    "source_construction_and_roles",
                    INPUTS[0][3],
                )
            except Rejected as error:
                result["failures"].append(error.receipt)
    return result


def join(evidence: Json) -> tuple[list[Json], list[Json]]:
    """Reconstruct all first responses from their durable request and response custody."""
    protocol = evidence["producers"][0]
    units, rows = [], []
    for primary, kind in zip(evidence["producers"][1:], ("A", "B"), strict=True):
        work = json.loads(Path(primary["raw_shard_hashes"][0]["path"]).read_text())
        expected = shard.ordered_units(protocol, kind)
        require(work["units"], expected, Path(work["raw"]), "frozen_units", primary["task_id"])
        require(
            shard.raw_complete(work),
            True,
            Path(work["raw"]),
            "raw_first_responses",
            primary["task_id"],
        )
        for index, (unit, row) in enumerate(zip(expected, work["runtime"]["rows"], strict=True)):
            folder = shard.capture.folder(Path(work["raw"]) / "runtime/collection", index, unit)
            envelope = json.loads((folder / "response.json").read_text())
            rebuilt = source.label_response(unit, envelope["raw_response"])
            rebuilt.update(
                duration_s=envelope["duration_s"],
                actual_token_usage=envelope["raw_response"].get("usage"),
            )
            require(
                rebuilt,
                row,
                folder / "response.json",
                "first_response_reduction",
                primary["task_id"],
            )
            units.append(unit)
            rows.append(
                dict(
                    row,
                    first_response_reference=calibration.reference(folder / "response.json"),
                    response_id=envelope["raw_response"]["id"],
                )
            )
        source.progress("8963_first_responses_joined", len(units), 1024 - len(units))
    if len({row["response_id"] for row in rows}) != 1024:
        raise ValueError("first_response_identity_overlap")
    return units, rows


def reduce(units: list[Json], responses: list[Json], history: Json) -> Json:
    """Independence and class support certify feasibility, never learned inference quality."""
    if len(units) not in (0, 1024) or [r["unit_id"] for r in responses] != [
        u["unit_id"] for u in units
    ]:
        raise ValueError("corpus_roster")
    graphs = {source.graph_hash(g) for g in history.get("old_graphs", [])}
    seeds = set(history.get("old_seeds", []))
    buckets: dict[Any, list[Any]] = {}
    for graph in history.get("old_graphs", []):
        topology = methods.nx_graph(graph)
        buckets.setdefault(methods.topology_key(topology), []).append(topology)
    manifest, labels, rows = [], [], []
    kinds: Counter[str] = Counter()
    support: Counter[tuple[str, int]] = Counter()
    streams: Counter[str] = Counter()
    usable = True
    for index, (unit, response) in enumerate(zip(units, responses, strict=True)):
        digest, seed = source.graph_hash(unit), unit["seed"]
        topology = methods.nx_graph(unit)
        key = methods.topology_key(topology)
        identity = f"v766-graph-{index:04d}"
        if (
            digest in graphs
            or seed in seeds
            or digest != unit["graph_hash"]
            or unit["cluster_id"] != identity
            or unit["unit_id"] != identity + "-k2"
            or any(unit[k] != v for k, v in methods.role(index).items())
            or any(methods.nx.is_isomorphic(topology, g) for g in buckets.get(key, []))
        ):
            raise ValueError("graph_or_role_overlap")
        graphs.add(digest)
        seeds.add(seed)
        buckets.setdefault(key, []).append(topology)
        distances = all_pairs(unit["nodes"], unit["edges"])[unit["start"]]
        expected = sorted(n for n, distance in distances.items() if distance == 2)
        if expected != unit["expected"] or not expected:
            raise ValueError("all_pairs_truth")
        complete = response["status"] == "completed"
        category = (
            source.category(response["answer"], unit["nodes"], expected)
            if complete
            else "generation_failure"
        )
        label = int(category != "correct") if complete else None
        if complete and (
            response["parser_category"] != category
            or response["value"] != label
            or response["numerator"] != label
        ):
            raise ValueError("independent_label_drift")
        signal = response["features"]
        usable = (
            usable
            and complete
            and response["signal_status"] == "usable"
            and isinstance(signal, dict)
            and set(signal) == set(source.FEATURE_SCHEMA["learned_features"])
            and all(type(v) in (int, float) and methods.np.isfinite(v) for v in signal.values())
        )
        kinds[category] += 1
        if complete:
            support[(unit["role"], int(category != "correct"))] += 1
        if unit["role"] == "online":
            streams[unit["stream_id"]] += 1
        manifest.append(
            dict(
                unit_id=unit["unit_id"],
                cluster_id=unit["cluster_id"],
                graph_hash=digest,
                seed=seed,
                **methods.role(index),
                status=response["status"],
                prompt_hash=unit["prompt_hash"],
                first_response_reference=response.get("first_response_reference"),
                response_id=response.get("response_id"),
            )
        )
        labels.append(
            dict(
                unit_id=unit["unit_id"],
                role=unit["role"],
                stream_id=unit["stream_id"],
                stream_position=unit["stream_position"],
                label=label,
                expected=expected,
                parser_category=category,
                independent_reference_agrees=True,
            )
        )
        rows.append(
            dict(
                unit_id=unit["unit_id"],
                cluster_id=unit["cluster_id"],
                arm="first_response_admission",
                condition=unit["role"],
                metric="response_admitted",
                value=1 if complete else None,
                numerator=1 if complete else None,
                denominator=1,
                status=response["status"],
                censor_reason=response.get("missing_reason"),
            )
        )
        if (index + 1) % 128 == 0:
            source.progress("8963_independent_all_pairs", index + 1, 1023 - index)
    class_rows: list[Json] = [
        dict(role=role, label=label, count=support[(role, label)], minimum=5)
        for role in ("calibration", "test", "online")
        for label in (0, 1)
    ]
    errors = sum(count for (_, label), count in support.items() if label == 1)
    complete_count = len(units) - kinds["generation_failure"]
    if not units:
        rows = [
            dict(
                unit_id=f"v766-graph-{i:04d}-k2",
                cluster_id=f"v766-graph-{i:04d}",
                arm="first_response_admission",
                condition=methods.role(i)["role"],
                metric="response_admitted",
                value=None,
                numerator=None,
                denominator=1,
                status="censored",
                censor_reason="external_evidence",
            )
            for i in range(1024)
        ]
    return dict(
        rows=rows,
        corpus_manifest=manifest,
        evaluation_labels=labels,
        role_counts=dict(Counter(u["role"] for u in units)),
        stream_counts=dict(streams),
        class_support_rows=class_rows,
        class_support_passed=complete_count == 1024
        and errors >= 200
        and all(r["count"] >= 5 for r in class_rows),
        independent_error_count=errors,
        error_type_counts={
            kind: kinds[kind]
            for kind in (
                "correct",
                "valid_wrong_distance",
                "malformed",
                "out_of_graph",
                "generation_failure",
            )
        },
        semantic_error_count=kinds["valid_wrong_distance"],
        format_error_count=kinds["malformed"],
        out_of_graph_error_count=kinds["out_of_graph"],
        generation_failure_count=kinds["generation_failure"],
        labels_verified=bool(units),
        usable_signals=bool(units) and usable,
        intended_count=1024,
        completed_count=complete_count,
        independent_count=complete_count,
        failed_count=0,
        censored_count=1024 - complete_count,
        excluded_count=0,
    )


def isolation(units: list[Json], rows: list[Json], raw: Path) -> list[Json]:
    """Change actual oracle sidecars and metadata while the frozen head scores the same inputs."""
    before = detector_view(units, rows)
    primary = json.loads((ROOT / INPUTS[0][0]).read_text())
    states = json.loads(Path(primary["trained_state_reference"]["path"]).read_text())["states"][
        "relational_energy"
    ]
    source.progress("8963_before_cached_head_scores", 0, len(rows))
    scores = methods.g.ensemble(states, before).tolist()
    controls = []
    for field in ("label", "expected", "distances", "answer_set"):
        sidecar = raw / ("private_oracle_mutation_" + field + ".json")
        changed_units, changed_rows = deepcopy(units), deepcopy(rows)
        atomic_json(sidecar, {field: "changed oracle sidecar"})
        for unit, row in zip(changed_units, changed_rows, strict=True):
            unit[field] = row[field] = "changed oracle metadata"
        after = detector_view(changed_units, changed_rows)
        controls.append(
            dict(
                control=f"mutate_{field}",
                passed=before == after and scores == methods.g.ensemble(states, after).tolist(),
                detector_input_hash=canonical_hash(before),
                prelabel_score_hash=canonical_hash(scores),
                mutated_sidecar=calibration.reference(sidecar),
            )
        )
    source.progress("8963_after_cached_head_scores", len(rows), 0)
    return controls


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Readiness requires the whole custody contract and measured owned qualification."""
    result = {k: v for k, v in work["result"].items() if k != "evaluation_labels"}
    owned = shard.qualified(validation, OWNED) and not validation.get("terminal_rejected", False)
    failures = work["evidence"]["failures"]
    controls = bool(work["controls"]) and all(r["passed"] for r in work["controls"])
    gates = dict(
        authenticated_inputs=not failures,
        complete_first_responses=result["completed_count"] == 1024,
        usable_confidence=result["usable_signals"],
        independent_labels=result["labels_verified"],
        class_support=result["class_support_passed"],
        oracle_separation=controls,
        owned_checks=owned,
    )
    ready = int(not fixture and all(gates.values()))
    failed_owned = not fixture and (not owned or (not failures and not controls))
    cls = (
        "disqualified"
        if failed_owned
        else "blocked"
        if failures
        else "circular_positive"
        if ready
        else "null"
    )
    verdict = f"complete_{cls}_v766_corpus_admission"
    prior = work.get("prior_terminal", {})
    if (
        cls in ("blocked", "disqualified")
        and prior.get("verdict_class") == cls
        and prior.get("gate_check_summary") == failures
    ):
        verdict = prior["honest_verdict"]
    value: Json = dict(
        result,
        experiment_id=EXPERIMENT_ID,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261110",
        random_seed=SEED,
        honest_verdict=verdict,
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if failures else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(failures=failures, historical_tasks_are_retained_inputs=True),
        verifier_is_oracle=True,
        exposure_scope="V765 exposed development; fresh calibration only; final-test sealed; online delay8/block4",
        independent_generalization_score=0,
        corpus_ready_score=ready,
        gate_check_summary=failures,
        source_artifact_hashes=work["input_references"] + work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            calibration.reference(Path(work["raw"]) / "work.json"),
            *work.get("raw_references", []),
        ],
        acceptance_gates={
            k: dict(
                passed=v,
                principle="Every admission gate is required; oracle agreement certifies labels, not learned benefit.",
            )
            for k, v in gates.items()
        },
        required_checks_passed=owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        oracle_separation_rows=work["controls"],
        broker_manifest=work["broker_manifest"],
        inference_views=work["views"],
        fixture_only=fixture,
        flagged_adversarial=False,
        methodology_note="First responses only, no balancing or replenishment. All-pairs truth is evaluation-only. Support minima are not power claims. Cached heads score private isolation operands without fitting or generator calls.",
    )
    principles = {
        "identity": (
            "experiment_id task_id milestone run_date random_seed reproducibility_checksum",
            "Bind the scheduled task to input bytes and reproducible randomness.",
        ),
        "terminal": (
            "honest_verdict verdict_class",
            "A terminal scope and closed class preserve claim strength and rerun retirement.",
        ),
        "compute": (
            "inference_substrate inference_substrate_class execution_venue MODEL_SPECS model_specs model_invocation_counts numerical_training_calls",
            "Separate cached outputs and small heads from current generator invocations.",
        ),
        "clocks": (
            "duration_s phase_spans preconditions_checked",
            "Actual clocks and resource receipts prevent fabricated work or padding.",
        ),
        "oracle": (
            "verifier_is_oracle exposure_scope independent_generalization_score",
            "Exact labels are separate from learned inference and establish no generalization.",
        ),
        "counts": (
            "intended_count completed_count failed_count censored_count excluded_count independent_count rows",
            "Keep every scheduled unit; typed rows determine the independent denominator.",
        ),
        "custody": (
            "gate_check_summary source_artifact_hashes code_config_hashes raw_shard_hashes field_principles",
            "Byte custody and exact field diagnostics distinguish missing, zero and rejected inputs.",
        ),
        "checks": (
            "acceptance_gates required_checks_passed owned_statement_coverage validation_receipts global_test_debt",
            "Owned failures disqualify; global debt cannot waive a failed owned check.",
        ),
        "views": (
            "corpus_ready_score corpus_manifest class_support_rows oracle_separation_rows role_counts broker_manifest",
            "Readiness needs independent graphs and support with labels sealed from inference.",
        ),
    }
    value["field_principles"] = {
        field: principle for fields, principle in principles.values() for field in fields.split()
    }
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
