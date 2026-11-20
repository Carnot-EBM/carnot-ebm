"""REQ-REPORT-9141-V779: original human agreement is descriptive, never a truth ceiling."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.v779_contract_evidence import check as check

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9141_v779_human_contrast_audit", "exp9141-human-contrast-audit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_human_contrast_9141.py"
OWNED = [
    "python/carnot/reporting/human_contrast_9141.py",
    "python/carnot/reporting/human_contrast_runner_9141.py",
    CLI,
]
UPSTREAM = "results/experiment_9138_v779_contract.json"
UPSTREAM_PIN = "sha256:65528ba505f6ab790c80ef7bba257c1bbe30e2a63502cad0be0d247f88125169"
SEED = 779141
ROLES = dict(TRAIN=150, CAL=50, TEST=100)
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flushed boundaries distinguish cached reduction from absent model work."""
    print(f"[exp9141] phase={phase}", flush=True)


def protocol() -> Json:
    """Freeze descriptive decisions before labels or future model outcomes are reduced."""
    return dict(
        version="v779-human-contrast-v1",
        seed=SEED,
        draws=10000,
        exclusions=[],
        scope="exposed_development; no training selection, model selection or promotion",
        target="A1:0 correct,1 error; A2:-1/null missing, never imputed or consensus",
        agreement=dict(
            estimand="mean within-qid observed A1/A2 agreement fraction",
            unit="paired whole qid with all siblings",
            ci95=[0.025, 0.975],
            empty="unavailable",
            missing="omit unobserved pairs; report observed qid and sibling support",
            truth_ceiling=False,
        ),
        static=dict(
            slices=["mixed", "homogeneous"],
            definition="mixed iff 0<A1 error siblings<sibling count",
            access="evaluator_only_after_prediction_seal",
            unit="paired whole qid",
        ),
        online=dict(
            probability_change_gte=0.05,
            classification_threshold=0.5,
            slices=["changed", "unchanged"],
            reference_arm="adaptive_spline",
            initial_reference="same row initial frozen_spline probability",
            membership="one reference-adaptive sealed mask shared by every arm, never arm-specific subsets",
            threshold_flips="(p_adaptive>=0.5)!=(p_initial>=0.5), reported separately in both slices",
            future_unit="paired whole stream preserving order, siblings and membership",
            retention_unit="paired whole stream conditional on fixed TEST100; repeated retention never new qids",
        ),
        metrics=dict(Brier="mean((p-y)^2)", signed_calibration_residual="mean(p-y)"),
        empty_slice_rule="unavailable with numerator/denominator/counts; never a win",
        interval_scope="descriptive conditional on original fit/guard/retention; no independent generalization",
        resolution=dict(
            qid_rate_grid="1/n",
            sibling_rate_grid="1/(3*n), not independent",
            maximum_normal95_width="1.96/sqrt(n), Bernoulli approximation",
            best_case_two_sided_sign_test="6/n at alpha0.05: six concordant nonzero qid changes have p=2^(1-6)=0.03125; not guaranteed power",
        ),
        static_bounds=[
            "combined Brier improvement lower>0 vs probability_length",
            "combined Brier improvement lower>0 vs combined_donor",
        ],
        online_bounds=[
            "frozen_spline minus adaptive_spline future Brier lower>0",
            "bias_only_spline minus adaptive_spline future Brier lower>0",
            "adaptive_linear minus adaptive_spline future Brier lower>0",
            "donor_label_spline minus adaptive_spline future Brier lower>0",
            "temperature_only_spline minus adaptive_spline future Brier lower>0",
            "adaptive_spline retained Brier increase upper<=0.005",
            "adaptive_spline retained classification-error increase upper<=0",
        ],
        acceptance="complete authenticated custody and executed checks, independent of favorable counts; preserve original two static/seven online bounds",
    )


def group_rows(labels: list[Json]) -> list[Json]:
    """Keep siblings together so a hard question cannot become three independent successes."""
    groups: dict[str, list[Json]] = defaultdict(list)
    seen: set[str] = set()
    for r in labels:
        if (
            r["unit_id"] in seen
            or type(r["annotator1"]) is not int
            or r["annotator1"] not in [0, 1]
            or (
                r["annotator2"] is not None
                and (type(r["annotator2"]) is not int or r["annotator2"] not in [-1, 0, 1])
            )
        ):
            raise ValueError("duplicate_or_label_encoding")
        seen.add(r["unit_id"])
        groups[r["qid"]].append(r)
    rows = []
    for i, (qid, siblings) in enumerate(sorted(groups.items())):
        if i % 50 == 0:
            progress(f"group_reduction_{i}_of_{len(groups)}")
        if (
            len(siblings) != 3
            or len({(r["role"], r["group_id"]) for r in siblings}) != 1
            or siblings[0]["role"] not in ROLES
        ):
            raise ValueError("whole_qid_partition")
        observed = [r for r in siblings if r["annotator2"] in [0, 1]]
        errors = sum(r["annotator1"] for r in siblings)
        disagreements = sum(r["annotator1"] != r["annotator2"] for r in observed)
        rows.append(
            dict(
                unit_id=qid,
                qid=qid,
                group_id=siblings[0]["group_id"],
                role=siblings[0]["role"],
                sibling=None,
                siblings=[
                    dict(
                        unit_id=r["unit_id"],
                        annotator1=r["annotator1"],
                        annotator2=-1 if r["annotator2"] is None else r["annotator2"],
                    )
                    for r in siblings
                ],
                sibling_count=3,
                a1_error_count=errors,
                a1_correct_count=3 - errors,
                a2_observed_count=len(observed),
                a2_missing_count=3 - len(observed),
                disagreement_count=disagreements,
                agreement_count=len(observed) - disagreements,
                slice="all_correct" if errors == 0 else "all_wrong" if errors == 3 else "mixed",
                arm="human_annotation",
                condition="original_role",
                seed=SEED,
                metric="A1_error_fraction",
                absolute_metric=errors / 3,
                numerator=errors,
                denominator=3,
                status="completed",
                exclusion_reason=None,
                censor_reason=None,
            )
        )
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate_group_identity")
    return rows


def agreement(rows: list[Json]) -> Json:
    """Bootstrap whole paired qids; unknown A2 provides no vote or artificial agreement."""
    paired = [r for r in rows if r["a2_observed_count"]]
    values = np.array([r["agreement_count"] / r["a2_observed_count"] for r in paired])
    ci, estimate = None, None
    if len(values):
        rng = np.random.default_rng(SEED)
        draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        ci = list(map(float, np.quantile(draws, [0.025, 0.975])))
        estimate = float(values.mean())
    return dict(
        estimate=estimate,
        ci95=ci,
        width=None if ci is None else ci[1] - ci[0],
        numerator=sum(r["agreement_count"] for r in paired),
        denominator=sum(r["a2_observed_count"] for r in paired),
        intended_qids=len(rows),
        observed_qids=len(paired),
        missing_qids=len(rows) - len(paired),
        status="available" if paired else "unavailable",
        draws=10000 if paired else 0,
        undefined_draws=0,
        resampling_unit="paired_whole_qid",
        estimand="equal_weight_observed_qid_agreement",
    )


def reduce(inspected: Json) -> Json:
    """Recompute every descriptive count from actual independent qid rows."""
    rows = inspected["rows"]
    panels = {
        "ALL": rows,
        **{role: [r for r in rows if r["role"] == role] for role in ROLES},
        "future40": [r for r in rows if r["qid"] in inspected["future_qids"]],
    }
    intervals = {name: agreement(panel) for name, panel in panels.items()}
    resolution = {}
    for name, panel, n in [("TEST100", panels["TEST"], 100), ("future40", panels["future40"], 40)]:
        resolution[name] = dict(
            intended_qids=n,
            completed_qids=len(panel),
            sibling_count=sum(r["sibling_count"] for r in panel),
            observed_agreement_qids=intervals["TEST" if n == 100 else name]["observed_qids"],
            qid_rate_grid=1 / n,
            sibling_rate_grid=1 / (3 * n),
            siblings_independent=False,
            maximum_normal95_width=1.96 / n**0.5,
            best_case_sign_test_minimum_changed_qids=6,
            best_case_sign_test_rate=6 / n,
            best_case_sign_test_p=0.03125,
            empirical_agreement_width=intervals["TEST" if n == 100 else name]["width"],
            guaranteed_power=False,
        )
    return dict(
        group_counts={
            k: sum(r["slice"] == k for r in rows) for k in ["mixed", "all_correct", "all_wrong"]
        },
        role_group_counts={role: dict(Counter(r["slice"] for r in panels[role])) for role in ROLES},
        agreement_intervals=intervals,
        panel_resolution=resolution,
    )


def inspect(root: Path) -> Json:
    """Authenticate original bytes before dependent fields, with explicit failed operands."""
    progress("authenticate_before")
    path = root / UPSTREAM
    result: Json = dict(rows=[], future_qids=[], checks=[], source_references=[], references={})
    checks = result["checks"]

    def bind(reference: Json, fields: list[str]) -> Path:
        target = Path(reference["path"])
        checks.append(
            check(
                target,
                "sha256",
                reference["sha256"],
                ref(target)["sha256"] or "missing_file",
                "immutable_bytes",
            )
        )
        result["source_references"].append(
            dict(
                reference,
                producer_path=reference.get("producer_path", reference["path"]),
                imported_fields=fields,
            )
        )
        return target

    try:
        bind(
            dict(path=str(path), sha256=UPSTREAM_PIN),
            [
                "source_ready_score",
                "required_checks_passed",
                "protocol_references",
                "historical_model_provenance",
            ],
        )
        if not checks[-1]["passed"]:
            return result
        contract = json.loads(path.read_bytes())
        for field, expected in [("source_ready_score", 1), ("required_checks_passed", True)]:
            checks.append(
                check(path, field, expected, contract.get(field, "missing_field"), "upstream_gate")
            )
        if not all(r["passed"] for r in checks):
            return result
        operands = {}
        for key in [
            "evaluator_label_reference",
            "source_manifest_reference",
            "initial_roster_reference",
            "static_protocol_reference",
            "online_protocol_reference",
        ]:
            target = bind(contract[key], [key])
            result["references"][key] = contract[key]
            if checks[-1]["passed"]:
                operands[key] = json.loads(target.read_bytes())
        if not all(r["passed"] for r in checks):
            return result
        manifest = operands["source_manifest_reference"]
        master_ref = manifest["release_files"]["data/human_judge_release_master.csv"]
        master = bind(master_ref, ["item_id", "qid", "annotator1_label", "annotator2_label"])
        if not checks[-1]["passed"]:
            return result
        with master.open(encoding="utf-8-sig", newline="") as stream:
            original = {
                r["item_id"]: (int(r["annotator1_label"]), int(r["annotator2_label"]))
                for r in csv.DictReader(stream)
            }
        labels = operands["evaluator_label_reference"]["rows"]
        paired = {
            r["unit_id"]: (r["annotator1"], -1 if r["annotator2"] is None else r["annotator2"])
            for r in labels
        }
        checks.append(
            check(
                master,
                "original_A1_A2_exact_match",
                True,
                paired == original and len(labels) == len(original),
                "original_labels",
            )
        )
        rows = group_rows(labels)
        checks.append(
            check(
                path,
                "original_qid_roles",
                ROLES,
                dict(Counter(r["role"] for r in rows)),
                "original_partition",
            )
        )
        roster = operands["initial_roster_reference"]
        checks.append(
            check(
                Path(contract["initial_roster_reference"]["path"]),
                "sibling_roster",
                True,
                all(
                    {r["unit_id"] for r in row["siblings"]}
                    == set(roster["sibling_roster"][row["qid"]])
                    for row in rows
                ),
                "original_siblings",
            )
        )
        future = [qid for stream in roster["stream_roster"] for qid in stream[1:]]
        checks.append(
            check(
                path,
                "future40_unique_original_TRAIN_qids",
                True,
                len(future) == len(set(future)) == 40
                and set(future) <= {r["qid"] for r in rows if r["role"] == "TRAIN"},
                "future_roster",
            )
        )
        checks.append(
            check(
                path,
                "static_bounds",
                protocol()["static_bounds"],
                operands["static_protocol_reference"]["bounds"],
                "fixed_bounds",
            )
        )
        checks.append(
            check(
                path,
                "online_bound_count",
                7,
                len(operands["online_protocol_reference"]["benefit_bounds"]["bounds"]),
                "fixed_bounds",
            )
        )
        if all(r["passed"] for r in checks):
            result.update(rows=rows, future_qids=sorted(future))
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append(
            check(
                path,
                "original_source_contract",
                "authenticated_complete_fields",
                "missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field"
                if isinstance(error, KeyError)
                else str(error),
                "source_contract",
            )
        )
    progress("authenticate_after")
    return result
