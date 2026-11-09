"""REQ-REPORT-8951: certify independent labels and support before any live fitting."""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import matched_corpus_a_8949 as shard_a
from carnot.reporting import matched_corpus_b_8950 as shard_b
from carnot.reporting import matched_corpus_runner_8949 as shards
from carnot.reporting import matched_source_methods_8946 as matched
from carnot.reporting import pilot_label_audit_8932 as audit
from carnot.reporting import graph_source_8931 as source
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
ROOT = matched.ROOT
NAME, TASK, EXPERIMENT_ID, SEED = (
    "experiment_8951_v765_matched_corpus_admission",
    "exp8951-matched-corpus-admission",
    8951,
    765051,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/matched_corpus_admission_8951.py",
    "python/carnot/reporting/matched_admission_runner_8951.py",
    CLI,
]
TEST = "tests/python/test_matched_corpus_admission_8951.py"
TESTS = [TEST]
INPUTS = [
    (*shard_a.depth.INPUTS[0],),
    (
        "results/experiment_8937_v764_calibration_methods.json",
        matched.INPUTS["results/experiment_8937_v764_calibration_methods.json"],
        None,
        "exp8937-calibration-methods",
    ),
    (
        f"results/{shard_a.NAME}.json",
        "sha256:84f4c2d4ab4d22aea9e6b61f10ebe07800b719de7c2530d941e48aaf25c9ee02",
        "shard_ready_score",
        shard_a.TASK,
    ),
    (
        f"results/{shard_b.NAME}.json",
        "sha256:fcd9822c71df4dfb85949c239e53839f316e67204a319257aa4f596ba1b6246d",
        "shard_ready_score",
        shard_b.TASK,
    ),
    (matched.branch.HISTORY, matched.branch.PINS[matched.branch.HISTORY], None, "exp8943-capstone"),
]


def authenticate(root: Path) -> Json:
    """Pin producers and raw operands; historical scientific gates are never rerun."""
    failures, refs = [], []
    for name, pin, field, task in INPUTS:
        path = root / name
        refs.append(calibration.reference(path))
        try:
            matched.audit.require(refs[-1]["sha256"], pin, path, "sha256", task)
            value = json.loads(path.read_text())
            if field is not None:
                for key, expected in (
                    (field, 1),
                    ("required_checks_passed", True),
                    ("task_id", task),
                ):
                    matched.audit.require(value.get(key), expected, path, key, task)
            operands = (
                (
                    value["source_artifact_hashes"]
                    + value["code_config_hashes"]
                    + value["raw_shard_hashes"]
                )
                if task != "exp8943-capstone"
                else [
                    r["primary_reference"]
                    for r in value["outcome_rows"]
                    if r["experiment_id"] in (8935, 8936, 8938, 8939)
                ]
            )
            for ref in operands:
                operand = Path(ref["path"])
                matched.audit.require(
                    calibration.reference(operand)["sha256"], ref["sha256"], operand, "sha256", task
                )
                refs.append(ref)
                for module in (source, calibration, calibration.b):
                    executed = Path(module.__file__)
                    if ref.get("source_path", "").endswith("/" + executed.name):
                        matched.audit.require(
                            calibration.reference(executed)["sha256"],
                            ref["sha256"],
                            executed,
                            "executed_helper_bytes",
                            task,
                        )
        except matched.audit.Rejected as error:
            failures.append(error.receipt)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            failures.append(
                dict(
                    upstream=task,
                    **calibration.reference(path),
                    hash=calibration.reference(path)["sha256"],
                    check="producer_structure",
                    artifact_field="producer_json",
                    expected="original complete object",
                    observed=str(error),
                    failure_kind="rejected_producer",
                )
            )
    authority = matched.branch.authority(root)
    failures.extend(
        dict(r, hash=r.get("sha256"), failure_kind=r["observed"]) for r in authority["failures"]
    )
    return dict(failures=failures, references=refs, authority_mode=authority["comparison_mode"])


def join(root: Path) -> tuple[list[Json], list[Json], list[Json]]:
    """Reconstruct first responses without reevaluating an older producer's test qualification."""
    value = json.loads((root / INPUTS[0][0]).read_text())
    work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())
    protocol = work["protocol"]
    matched.audit.require(
        canonical_hash(protocol),
        value["protocol_hash"],
        root / INPUTS[0][0],
        "protocol_hash",
        INPUTS[0][3],
    )
    matched.audit.require(
        protocol["frozen_calibration_protocol"],
        calibration.manifest(),
        root / INPUTS[0][0],
        "frozen_calibration_protocol",
        INPUTS[0][3],
    )
    if not matched.validate_protocol(protocol, work["evidence"]["old_graphs"]):
        raise ValueError("frozen_split_or_rejection_drift")
    units, rows = [], []
    for index, producer in ((2, shard_a), (3, shard_b)):
        path = root / INPUTS[index][0]
        primary = json.loads(path.read_text())
        original = json.loads(Path(primary["raw_shard_hashes"][0]["path"]).read_text())
        expected = source.scheduled(protocol, producer.SHARD)
        matched.audit.require(original["units"], expected, path, "frozen_units", producer.TASK)
        matched.audit.require(
            shard_a.raw_complete(original), True, path, "raw_first_responses", producer.TASK
        )
        raw = Path(original["raw"])
        binding = canonical_hash(
            dict(
                model=original["runtime"]["preconditions_checked"]["model"],
                settings=source.SETTINGS,
                feature_schema=source.FEATURE_SCHEMA,
                units=[
                    dict(
                        unit_id=u["unit_id"],
                        graph=source.graph_hash(u),
                        prompt=u["prompt"],
                        k=u["k"],
                        start=u["start"],
                        role=u["role"],
                        shard=u["shard"],
                    )
                    for u in expected
                ],
            )
        )
        matched.audit.require(
            json.loads((raw / "runtime/collection/contract.json").read_text()),
            dict(binding=binding),
            path,
            "collection_binding",
            producer.TASK,
        )
        rebuilt = []
        for unit in expected:
            folder = raw / "runtime/collection" / unit["unit_id"]
            envelope = json.loads((folder / "response.json").read_text())
            matched.audit.require(
                envelope["binding"], binding, folder / "response.json", "binding", producer.TASK
            )
            matched.audit.require(
                json.loads((folder / "intent.json").read_text()),
                dict(
                    binding=binding,
                    request=dict(
                        source.SETTINGS, messages=[dict(role="user", content=unit["prompt"])]
                    ),
                ),
                folder / "intent.json",
                "request",
                producer.TASK,
            )
            row = source.label_response(unit, envelope["raw_response"])
            row["duration_s"] = envelope["duration_s"]
            matched.audit.require(
                row,
                json.loads((folder / "label.json").read_text()),
                folder / "label.json",
                "label",
                producer.TASK,
            )
            matched.audit.require(
                row["features"],
                json.loads((folder / "detector.json").read_text()),
                folder / "detector.json",
                "features",
                producer.TASK,
            )
            row["actual_token_usage"] = envelope["raw_response"]["usage"]
            rebuilt.append(row)
        matched.audit.require(
            rebuilt, original["runtime"]["rows"], path, "response_rows", producer.TASK
        )
        reduced = shard_a.reduce(expected, rebuilt, producer.SHARD)
        for name, key in (
            ("evaluation_labels.json", "label_sidecar"),
            ("detector_view.json", "detector_view"),
        ):
            matched.audit.require(
                json.loads((raw / name).read_text()),
                dict(rows=reduced[key]),
                raw / name,
                key,
                producer.TASK,
            )
        matched.audit.require(primary["rows"], reduced["rows"], path, "primary_rows", producer.TASK)
        units.extend(expected)
        rows.extend(rebuilt)
    old = work["evidence"]["old_graphs"] + [
        g for g in protocol["graph_manifest"] if g["shard"] == "control"
    ]
    return units, rows, old


def reduce(units: list[Json], rows: list[Json], old: list[Json]) -> Json:
    """Check truth independently and apply feasibility minima without outcome-driven splits."""
    typed, manifest = [], []
    if units:
        shard_a.check_units(units[:160], "A")
        shard_a.check_units(units[160:], "B")
        if (
            len(rows) != 320
            or len({u["unit_id"] for u in units}) != 320
            or len({u["graph_hash"] for u in units}) != 320
            or len({u["seed"] for u in units}) != 320
            or {u["graph_hash"] for u in units} & {source.graph_hash(g) for g in old}
            or {u["seed"] for u in units} & {g["seed"] for g in old}
        ):
            raise ValueError("independent_roster")
        for unit, row in zip(units, rows, strict=True):
            lengths = audit.distances(unit)
            answers = {
                str(k): sorted(n for n, length in lengths.items() if length == k) for k in (2, 3)
            }
            if (
                source.graph_hash(unit) != unit["graph_hash"]
                or answers != unit["valid_sets"]
                or answers["2"] != unit["expected"]
                or not len(answers["2"]) == len(answers["3"]) > 0
                or any(
                    source.distance_nodes(unit["nodes"], unit["edges"], unit["start"], k)
                    != answers[str(k)]
                    for k in (2, 3)
                )
                or row["unit_id"] != unit["unit_id"]
                or row["cluster_id"] != unit["graph_id"]
                or row["status"] != "completed"
                or row["parser_category"]
                != source.category(row["answer"], unit["nodes"], answers["2"])
                or type(row["value"]) is not int
                or row["value"] != row["numerator"]
                or row["value"] != int(row["parser_category"] != "correct")
            ):
                raise ValueError("independent_label_or_opportunity")
            typed.append(
                dict(
                    row,
                    role=unit["role"],
                    shard=unit["shard"],
                    condition="k2",
                    censor_reason=None,
                    independent_reference_agrees=True,
                )
            )
            manifest.append(
                dict(
                    unit_id=unit["unit_id"],
                    cluster_id=unit["graph_id"],
                    graph_hash=unit["graph_hash"],
                    seed=unit["seed"],
                    role=unit["role"],
                    shard=unit["shard"],
                    prompt_hash=unit["prompt_hash"],
                    valid_sets=answers,
                    true_distances=lengths,
                )
            )
    else:
        typed = [
            dict(
                unit_id=f"matched-graph-{i:03d}-k2",
                cluster_id=f"matched-graph-{i:03d}",
                arm="qwen_reasoning_off_k2",
                condition="k2",
                metric="final_answer_error",
                value=None,
                numerator=None,
                denominator=1,
                status="censored",
                censor_reason="external_evidence",
                missing_reason="external_evidence",
                role=None,
                shard="A" if i < 210 else "B",
                parser_category=None,
            )
            for i in range(50, 370)
        ]
    counts = dict(Counter(u["role"] for u in units))
    kinds = {kind: sum(r["parser_category"] == kind for r in typed) for kind in audit.CATEGORIES}
    support = []
    for role in ("train", "calibration", "test"):
        group = [r for r in typed if r["role"] == role]
        for label in (0, 1):
            n = sum(r["value"] == label for r in group)
            support.append(
                dict(
                    split=role,
                    label=label,
                    metric="class_support",
                    value=n,
                    numerator=n,
                    denominator=len(group),
                    minimum_required=5,
                    passed=n >= 5,
                )
            )
    errors = sum(r["value"] == 1 for r in typed)
    minimum = min(r["value"] for r in support)
    return dict(
        rows=typed,
        corpus_manifest=manifest,
        role_counts=counts,
        error_type_counts=kinds,
        independent_error_count=errors,
        semantic_error_count=kinds["valid_wrong_distance"],
        format_error_count=kinds["malformed"],
        out_of_graph_error_count=kinds["out_of_graph"],
        class_support_rows=support,
        minimum_class_support_achieved=minimum,
        minimum_independent_errors_required=200,
        class_support_passed=len(units) == 320 and errors >= 200 and minimum >= 5,
        intended_count=320,
        completed_count=len(units),
        independent_count=len(units),
        failed_count=0,
        censored_count=320 - len(units),
        excluded_count=0,
    )


def detector_view(responses: list[Json]) -> list[Json]:
    """Only server confidence crosses the scorer boundary; graph metadata is inaccessible."""
    return [source.signals(response)["features"] for response in responses]


def development() -> Json:
    """Requalify frozen methods on development data, including oracle-metadata mutations."""
    source.progress("before_admission_development", 0, 6)
    result: Json = calibration.development()
    diagnostic = result["fixture"]["diagnostic"]
    responses: list[Json] = []
    for features in diagnostic["features"]:
        lp = -features["mean_token_nll"]
        responses.append(
            dict(
                choices=[
                    dict(
                        message=dict(content="AA"),
                        logprobs=dict(
                            content=[
                                dict(
                                    token="AA",
                                    logprob=lp,
                                    top_logprobs=[
                                        dict(token="AA", logprob=lp),
                                        dict(
                                            token="BB",
                                            logprob=lp - features["first_answer_top2_margin"],
                                        ),
                                    ],
                                )
                            ]
                        ),
                    )
                ]
            )
        )
    before = detector_view(responses)
    scores = calibration.b.score(result["fit"], before, diagnostic["nuisance"])
    mutations = []
    for field in ("label", "true_distances", "answer_set", "graph_id"):
        changed = deepcopy(responses)
        for response in changed:
            response[field] = {"changed": "oracle metadata must stay outside the view"}
        after = detector_view(changed)
        mutations.append(
            dict(
                field=field,
                features_unchanged=before == after,
                scores_unchanged=scores
                == calibration.b.score(result["fit"], after, diagnostic["nuisance"]),
            )
        )
    tree = ast.parse(Path(calibration.b.__file__).read_text())
    imports = [
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("carnot")
    ]
    imports.extend(
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for alias in n.names
        if alias.name.startswith("carnot")
    )
    isolated = all(name == "carnot.models.gibbs" for name in imports)
    mutation_passed = all(r["features_unchanged"] and r["scores_unchanged"] for r in mutations)
    result["feature_access_controls"] = dict(
        import_isolation_passed=isolated,
        scoring_imports=imports,
        mutation_isolation_passed=mutation_passed,
        mutation_rows=mutations,
        learned_features=list(calibration.b.FEATURES),
        live_fitting=False,
        test_threshold_search=False,
        feature_extraction="graph_source_8931.signals(response) only",
        scoring_boundary="graph_calibration_benchmark.score(state, feature_view, syntax_view)",
    )
    result["control_rows"].extend(
        [
            dict(control="detector_import_isolation", passed=isolated),
            dict(control="oracle_metadata_mutation", passed=mutation_passed),
        ]
    )
    source.progress("after_admission_development", len(result["control_rows"]), 0)
    return result


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Admission is custody and feasibility; oracle agreement never establishes detector value."""
    result, dev = work["result"], work["development"]
    owned = shard_a.qualified(validation, OWNED) and not validation.get("terminal_rejected", False)
    controls = bool(dev) and all(r["passed"] for r in dev["control_rows"])
    failures = work["evidence"]["failures"]
    ready = int(
        owned and controls and not failures and result["class_support_passed"] and not fixture
    )
    cls = (
        "disqualified"
        if (not owned or (dev and not controls)) and not fixture
        else "blocked"
        if failures
        else "circular_positive"
        if ready
        else "null"
    )
    value = dict(work["template"])
    raw = Path(work["raw"])
    value.update(
        result,
        experiment_id=EXPERIMENT_ID,
        task_id=TASK,
        milestone="2026.11.765",
        run_date="20261109",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_matched_corpus_admission",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        execution_venue="host",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=dev.get("numerical_training_calls", 0),
        duration_s=work["ended"] - work["started"],
        started_at=work["started_at"],
        finished_at=work["finished_at"],
        phase_spans=work["phase_spans"],
        preconditions_checked=work["evidence"],
        verifier_is_oracle=True,
        exposure_scope="matched_graph_final_answer_labels_and_support_only; development-only calibration requalification",
        independent_generalization_score=0,
        gate_check_summary=failures,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            calibration.reference(raw / name)
            for name in (
                "work.json",
                "validation.json",
                "development.json",
                "detector_view.json",
                "evaluation_labels.json",
            )
        ],
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            complete_corpus=result["completed_count"] == 320,
            independent_labels=bool(result["corpus_manifest"]),
            class_support=result["class_support_passed"],
            calibration_methods=controls,
            owned_checks=owned,
            no_live_fitting=True,
            frozen_split=True,
        ),
        required_checks_passed=owned and (controls or bool(failures)),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        corpus_ready_score=ready,
        feature_access_controls=dev.get("feature_access_controls", {}),
        calibration_protocol_hash=canonical_hash(calibration.manifest()),
        source_protocol_hash=work["source_protocol_hash"],
        frozen_calibration_protocol=calibration.manifest(),
        historical_scheduler_skips=work["historical_scheduler_skips"],
        development_control_rows=dev.get("control_rows", []),
        development_comparison_rows=[
            dict(r, condition="development_only", censor_reason=None) for r in dev.get("rows", [])
        ],
        fixture_only=fixture,
        flagged_adversarial=bool(validation.get("terminal_rejected", False)),
        methodology_note="Join exactly the frozen 320 first responses. Independent BFS/reference labels certify custody, not detector accuracy. Require 200 errors and 5 of each class per split without rebalancing. These are feasibility minima, not a power guarantee. Only development fixtures requalify calibration; no live fitting or test threshold search.",
        reproduction_commands=[
            f"PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
    )
    value["field_principles"] = {
        key: "Independent final-answer custody and feasibility only; no detector accuracy or inherited compute claim."
        for key in [*value, "reproducibility_checksum"]
    }
    value["field_principles"]["numerical_training_calls"] = (
        "1200 development-only small-head updates; validation/replay are verification work, never live corpus fitting."
    )
    value["schema"] = sorted(set(value) | {"schema", "reproducibility_checksum"})
    value["reproducibility_checksum"] = calibration.checksum(value)
    return value
