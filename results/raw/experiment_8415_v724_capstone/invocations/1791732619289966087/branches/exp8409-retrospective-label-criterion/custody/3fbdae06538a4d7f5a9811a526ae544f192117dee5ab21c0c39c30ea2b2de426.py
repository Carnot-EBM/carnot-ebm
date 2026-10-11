"""REQ-REPORT-8409: fixed factorial agreement measures released evidence only."""

from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import released_question_identity_8403 as custody
from carnot.reporting import human_label_custody_8389 as old
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.human_label_custody_8389 import (
    reference as reference,
    read_reference as read_reference,
)

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8409_v724_retrospective_label_criterion"
TASK = "exp8409-retrospective-label-criterion"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_retrospective_label_criterion_8409.py"
OWNED = [
    "python/carnot/reporting/retrospective_label_criterion_8409.py",
    "python/carnot/reporting/retrospective_label_criterion_runner_8409.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
ARMS = custody.ARMS
SEED = 7238388
BOOTSTRAPS = 10000
PIN = "sha256:13afe888c9f7f516d63adff8eab9eea25e011745db6530d41cd836925661235b"
PROTOCOL_PIN = "sha256:0ed6f87ede5046c2113257c9bff4dacfee88415fddb8d7be18506668f6301e90"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so long reductions cannot hide unfinished work."""
    print(f"[exp8409] phase={phase} completed={completed} pending={pending}", flush=True)


def input_manifest(root: Path) -> Json:
    """An independently fixed producer hash prevents a rehashed join from authorizing itself."""
    return dict(
        primaries={"8403": dict(path=str(root / "results" / (custody.NAME + ".json")), sha256=PIN)}
    )


def kappa_value(pairs: list[tuple[int, int]]) -> float | None:
    """Chance correction is undefined when either agreement denominator is absent."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = (
        sum(sum(a == v for a, _ in pairs) * sum(b == v for _, b in pairs) for v in (0, 1)) / n**2
    )
    return (observed - expected) / (1 - expected) if expected != 1 else None


def metrics(rows: list[Json], target: str) -> Json:
    """Every arm uses the same complete paired answers, preventing selection by condition."""
    paired = [r for r in rows if r.get("qualified") and r.get(target) in (0, 1)]
    counts, kappas = {}, {}
    for arm in ARMS:
        pairs = [(r[target], r["factorial_outputs"][arm]["label"]) for r in paired]
        counts[arm] = {
            name: sum(a == a0 and b == b0 for a, b in pairs)
            for name, a0, b0 in [("FP", 0, 1), ("FN", 1, 0), ("TP", 1, 1), ("TN", 0, 0)]
        }
        kappas[arm] = kappa_value(pairs)
    zeros = sum(r[target] == 0 for r in paired)
    ones = len(paired) - zeros
    fpr = {a: counts[a]["FP"] / zeros if zeros else None for a in ARMS}
    fnr = {a: counts[a]["FN"] / ones if ones else None for a in ARMS}
    contrast = (
        0.5
        * (
            (counts["p1"]["FP"] / zeros - counts["p2t"]["FP"] / zeros)
            + (counts["p1s"]["FP"] / zeros - counts["p2"]["FP"] / zeros)
        )
        if zeros
        else None
    )
    return dict(
        false_positive_rates=fpr,
        false_negative_rates=fnr,
        confusion_counts=counts,
        kappa=kappas,
        primary_contrast=contrast,
        paired_factual_count=zeros,
        paired_hallucination_count=ones,
        paired_count=len(paired),
    )


def bootstrap(rows: list[Json], target: str) -> Json:
    """Resample whole questions so three generator answers cannot count as three sources."""
    groups: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        if row.get("question_hash"):
            groups[row["question_hash"]].append(row)
    vectors = []
    for group in groups.values():
        eligible = [r for r in group if r.get("qualified") and r.get(target) == 0]
        vectors.append(
            [
                len(eligible),
                *[sum(r["factorial_outputs"][a]["label"] for r in eligible) for a in ARMS],
            ]
        )
    if not vectors or not sum(v[0] for v in vectors):
        return dict(ci95=None, defined_replicates=0, undefined_replicates=BOOTSTRAPS)
    matrix = np.asarray(vectors, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    values = []
    for index in range(0, BOOTSTRAPS, 1000):
        progress("bootstrap_" + target, index, BOOTSTRAPS - index)
        totals = matrix[rng.integers(0, len(matrix), size=(1000, len(matrix)))].sum(axis=1)
        valid = totals[:, 0] > 0
        rates = totals[valid, 1:] / totals[valid, :1]
        values.extend((0.5 * (rates[:, 0] + rates[:, 1] - rates[:, 2] - rates[:, 3])).tolist())
    progress("bootstrap_" + target + "_complete", BOOTSTRAPS, 0)
    return dict(
        ci95=np.quantile(values, [0.025, 0.975]).tolist(),
        defined_replicates=len(values),
        undefined_replicates=BOOTSTRAPS - len(values),
    )


def analyze(master: list[Json], audit: list[Json]) -> Json:
    """Keep every intended answer and missing arm while reducing the paired estimand."""
    progress("paired_join_before", 0, 900)
    rows = custody.join_panel(master, audit, [], ["previously_opened_targets"])["rows"]
    primary = metrics(rows, "human_primary")
    sensitivity = metrics(rows, "human_sensitivity")
    primary_boot = bootstrap(rows, "human_primary")
    sensitivity["bootstrap"] = bootstrap(rows, "human_sensitivity")
    condition_rows, missing = [], []
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("question_hash"):
            clusters[row["question_hash"]].append(row["unit_id"])
        for arm in ARMS:
            output = row.get("factorial_outputs", {}).get(arm, {})
            valid = bool(row.get("audit_aligned") and output.get("valid"))
            factual = row.get("human_primary") == 0
            condition_rows.append(
                dict(
                    unit_id=row["unit_id"],
                    arm=arm,
                    question_hash=row.get("question_hash"),
                    generator=row.get("generator"),
                    human_primary=row.get("human_primary"),
                    output=output,
                    false_positive_indicator=int(output["label"] == 1)
                    if valid and factual
                    else None,
                    missing_reason=None
                    if valid
                    else row.get("missing_reason") or "condition_output_invalid",
                )
            )
            if not valid:
                missing.append(
                    dict(
                        unit_id=row["unit_id"],
                        arm=arm,
                        reason=condition_rows[-1]["missing_reason"],
                        observed_error="unavailable_or_invalid",
                    )
                )
    factual_rows = [r for r in rows if r.get("human_primary") == 0]
    bounds = {}
    for arm in ARMS:
        observed = [
            r["factorial_outputs"][arm]["label"]
            for r in factual_rows
            if r.get("audit_aligned") and r["factorial_outputs"][arm]["valid"]
        ]
        total = len(factual_rows)
        bounds[arm] = (
            [sum(observed) / total, (sum(observed) + total - len(observed)) / total]
            if total
            else [0, 1]
        )
    contrast_bounds = [
        0.5 * (bounds["p1"][0] + bounds["p1s"][0] - bounds["p2t"][1] - bounds["p2"][1]),
        0.5 * (bounds["p1"][1] + bounds["p1s"][1] - bounds["p2t"][0] - bounds["p2"][0]),
    ]
    humans = [
        (r["human_primary"], r["human_sensitivity"])
        for r in rows
        if r.get("human_primary") in (0, 1) and r.get("human_sensitivity") in (0, 1)
    ]
    disagreement = sum(a != b for a, b in humans)
    strata = {}
    for field in ("dataset", "generator"):
        strata[field] = {
            str(key): metrics([r for r in rows if r.get(field) == key], "human_primary")
            for key in sorted({r[field] for r in rows if field in r})
        }
    progress("paired_join_after", len(master), 900 - len(master))
    return dict(
        primary,
        rows=rows,
        intended_count=900,
        completed_count=len(master),
        failed_count=0,
        censored_count=max(0, 900 - len(master)),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=len(clusters),
        question_cluster_rows=[
            dict(question_hash=k, item_ids=v, answer_count=len(v))
            for k, v in sorted(clusters.items())
        ],
        primary_ci95=primary_boot["ci95"],
        bootstrap=primary_boot,
        kappa=dict(
            annotator1_vs_annotator2=kappa_value(humans),
            annotator1_vs_conditions=primary["kappa"],
            annotator2_vs_conditions=sensitivity["kappa"],
        ),
        reliability_sensitivity=sensitivity,
        annotation_disagreement=dict(
            comparable_count=len(humans),
            disagreements=disagreement,
            rate=disagreement / len(humans) if humans else None,
            missing_count=900 - len(humans),
        ),
        strata=strata,
        per_item_condition_rows=condition_rows,
        missing_pairs=missing,
        missing_pair_bounds=dict(
            condition_fpr_bounds=bounds,
            contrast_bounds=contrast_bounds,
            factual_answer_count=len(factual_rows),
            missing_factual_answers=len(factual_rows) - primary["paired_factual_count"],
            unknown_target_answers=900 - sum(r.get("human_primary") in (0, 1) for r in rows),
            unknown_target_global_bounds=[-1, 1],
            scope="valid factual targets; absent targets retain uninformative global bounds",
        ),
    )


def release_manifest(primary: Json) -> Json:
    """Traverse declared producer references rather than searching arbitrary nested paths."""
    manifest = json.loads(read_reference(primary["manifest_reference"]))
    upstream = json.loads(read_reference(manifest["primaries"]["8389"]))
    return dict(json.loads(read_reference(upstream["manifest_reference"])))


def reduce_manifest(manifest: Json) -> Json:
    """Reviewed CSV blobs, not rehashed summary rows, authorize retrospective measurement."""
    ref = manifest["primaries"]["8403"]
    if ref["sha256"] != PIN:
        raise ValueError("primary_pin")
    gates, master, audit, primary = [], [], [], {}
    release_files: Json = {}
    try:
        primary = json.loads(read_reference(ref))
        read_reference(
            dict(
                path=str(ROOT / "openspec/change-proposals/v723-service-and-label-protocol.json"),
                sha256=PROTOCOL_PIN,
            )
        )
        if primary.get("release_criterion_ready_score") != 1:
            gates.append(
                old.gate(
                    ref["path"],
                    "release_criterion_ready_score",
                    1,
                    primary.get("release_criterion_ready_score"),
                    PIN,
                    upstream=custody.TASK,
                )
            )
        else:
            release = release_manifest(primary)
            if release["release_commit"] != old.COMMIT or release["target_mapping"] != old.TARGET:
                raise ValueError("release_mapping")
            release_files = release["release_files"]
            operands = {
                name: old.authenticate(name, release_files[name]) for name in old.RELEASE_FILES
            }
            master = list(csv.DictReader(io.StringIO(operands[old.MASTER].decode("utf-8-sig"))))
            audit = list(csv.DictReader(io.StringIO(operands[old.AUDIT].decode("utf-8-sig"))))
    except OSError as error:
        gates.append(
            old.gate(
                str(error.filename),
                "external_operand_available",
                True,
                None,
                ref["sha256"],
                upstream=custody.TASK,
            )
        )
    value = analyze(master, audit)
    value.update(
        gate_check_summary=gates,
        release_criterion_ready_score=int(not gates),
        disjoint_eval_ready_score=primary.get("disjoint_eval_ready_score", 0),
        unknown_exposure_obligations=primary.get("exposure_ledger", {}).get(
            "unresolved_obligations", ["upstream_absent"]
        ),
        release_file_hashes=release_files,
        target_mapping=old.TARGET,
        human_targets_previously_opened=True,
        retrospective_replication=True,
    )
    if len(master) != 900 or value["excluded_count"] or value["primary_contrast"] is None:
        value["gate_check_summary"].append(
            old.gate(
                ref["path"],
                "complete_paired_factorial_panel",
                900,
                value["paired_count"],
                PIN,
                upstream=custody.TASK,
            )
        )
    return value


def build(work: Json, receipts: list[Json]) -> Json:
    """Audit qualification and the scientific effect are separate terminal decisions."""
    value = reduce_manifest(json.loads(read_reference(work["manifest_reference"])))
    gates = value["gate_check_summary"] + work["gate_check_summary"]
    findings = work.get("adversarial_findings", [])
    passed = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    positive = bool(value["primary_ci95"] and value["primary_ci95"][0] > 0)
    kind = (
        "disqualified"
        if not passed
        else "blocked"
        if gates
        else "circular_positive"
        if positive
        else "null"
    )
    ready = int(passed and not gates and not findings)
    value.update(
        experiment_id=8409,
        task_id=TASK,
        milestone="2026.10.724",
        run_date="20261011",
        honest_verdict=f"complete_{kind}_retrospective_label_criterion",
        verdict_class=kind,
        gate_check_summary=gates,
        criterion_audit_ready_score=ready,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=work["model_invocation_counts"],
        historical_model_provenance=[
            dict(generator=m, kind="cached_third_party_answers", current_calls=0)
            for m in old.GENERATORS
        ]
        + [dict(judge="GPT-5-mini", kind="cached_factorial_outputs", current_calls=0)],
        sample_size_budget=dict(
            intended_responses=900,
            intended_questions=300,
            bootstrap_replicates=BOOTSTRAPS,
            bootstrap_seed=SEED,
            confidence_level=0.95,
            repeats_are_independent=False,
            independent_evaluation_required=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="released previously opened A1 targets; retrospective within-release agreement only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed or bool(findings),
        acceptance_gates=dict(
            release_criterion_ready=value["release_criterion_ready_score"] == 1,
            complete_factorial_panel=not value["gate_check_summary"],
            current_authority=not work["gate_check_summary"],
            owned_validation=passed,
            no_unresolved_adversarial_findings=not findings,
            positive_primary_lower_bound=positive,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=work["terminal_validation_sidecar_path"],
        adversarial_findings=findings,
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        source_artifact_hashes=work["source_artifact_hashes"]
        + list(value["release_file_hashes"].values()),
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[work["manifest_reference"], work["primitive_reference"]],
        manifest_reference=work["manifest_reference"],
        primitive_reference=work["primitive_reference"],
        work_reference=reference(Path(work["work_path"])),
        invocation_id=work["invocation_id"],
        authority=work["authority"],
        repository_health=work["repository_health"],
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields="release_criterion_ready_score, disjoint_eval_ready_score, exposure_ledger, authenticated released CSV operands or task authority",
            )
            for r in work["source_artifact_hashes"]
        ],
        reproducibility_checksum=canonical_hash(
            [
                work["manifest_reference"],
                work["source_artifact_hashes"],
                work["code_config_hashes"],
                SEED,
            ]
        ),
        methodology_note="Frozen 0.5*((FPR_p1-FPR_p2t)+(FPR_p1s-FPR_p2)); A1 primary, A2 sensitivity; 10000 whole normalized-question resamples, seed7238388. Positive means agreement with released A1, not world truth or independent Carnot detector superiority. No fitting, prompt selection, regeneration or current Qwen inference. H1=-0.00390625 and H2=0 remain closed.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to authenticated operands, all intended units, fixed descriptive statistics and preserved exposed scope."
        for k in value
    }
    value["field_principles"].update(
        criterion_audit_ready_score="Qualified complete audit, including a null or negative effect; never a generalization score.",
        primary_contrast="Frozen factorial contrast on the identical paired A1 factual answers.",
        primary_ci95="Percentile 95 percent interval from 10000 whole-question bootstrap draws; all generators stay together.",
        missing_pair_bounds="Assign each absent condition indicator zero or one coherently with fixed contrast signs; unknown targets retain global bounds.",
        kappa="Chance-corrected agreement with human labels; undefined marginals remain null, never proof of truth.",
        annotation_disagreement="A1 versus A2 on jointly observed labels, with missing targets named.",
        independent_count="Unique normalized released questions, not unseen evaluation sources or bootstrap repetitions.",
        verifier_is_oracle="A1 is the criterion authority, so a positive agreement replication remains circular_positive.",
        retrospective_replication="Previously opened released evidence cannot become a new unseen test.",
        unknown_exposure_obligations="Copy every unresolved Exp8403 exposure obligation without relaxing its separate independent gate.",
        field_principles="Explain every typed field without replacing evidence.",
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild from author CSV blobs so even self-consistently rehashed reports must match."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(read_reference(value["work_reference"]))
        for ref in work["source_artifact_hashes"] + work["code_config_hashes"]:
            read_reference(ref)
        manifest = json.loads(read_reference(work["manifest_reference"]))
        if json.loads(read_reference(work["primitive_reference"])) != reduce_manifest(manifest):
            raise ValueError("primitive_reduction_drift")
        if value != build(work, value["validation_receipts"]):
            raise ValueError("reduction_drift")
        validate_primary(value, ROOT / "results" / (NAME + ".json"))
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        progress("replay_rejected_" + type(error).__name__)
        return False
