"""REQ-REPORT-9211-V784: target identifiability and exact label-ambiguity bounds without model work.

Authenticate original Exp9097 human custody and Exp9126/9132 role references
independently of administrative readiness. Freeze the November 11 v2 taxonomy
rubric and exact affine Brier contrast endpoint reducer before predictions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import human_label_custody_8389 as old
from carnot.reporting import logit_methods_runner_9126 as static_runner
from carnot.reporting import online_methods_runner_9132 as online_runner
from carnot.reporting.logit_methods_9126 import ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9211_v784_label_ambiguity", "exp9211-label-ambiguity"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_label_ambiguity_9211.py"
OWNED = [
    "python/carnot/reporting/label_ambiguity_9211.py",
    "python/carnot/reporting/label_ambiguity_runner_9211.py",
    CLI,
]
SEED = 784211
ROLES = dict(TRAIN=150, CAL=50, TEST=100)
GENERATORS = ("gemma-2-9b", "llama3-8b", "mistral-7b")
TARGETS = ("factual_error", "expressed_uncertainty", "reelicitation")
MODEL_SPECS: list[Json] = []
PINS = {
    "results/experiment_9097_v776_human_authority.json": (
        "sha256:24c9e0983d9a4eb0aa5a488b5467386e57d6f13b79447f0f3a2f3d1c68e25532"
    ),
    "results/experiment_9126_v778_logit_methods.json": (
        "sha256:16cd3742f2ec4dea0d67b6503e3d9e4dd84f7a3898fdbe15087065208ed1a1fb"
    ),
    "results/experiment_9132_v778_online_methods.json": (
        "sha256:f59a4197962a8591757d094b480082a9abcad40e81ecd2e1bf6244218802bd55"
    ),
    "results/experiment_9137_v778_capstone.json": (
        "sha256:4cc5a692cfdefdbe72073ee03eed2a16bd9674a2ca56e3a7d0bf41e9e2bb842e"
    ),
    "results/experiment_9141_v779_human_contrast_audit.json": (
        "sha256:b2aa70be88fc641c93452909cfb8f754b894958aa8d4a10ed8b64634ec214d5e"
    ),
    "results/experiment_9180_v782_contract.json": (
        "sha256:42c35a172e548872dbbc82c737ce2c3ee3a624d4a0fec2495225c2f7f38e92f2"
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so source authentication and CPU controls stay observable."""
    print(f"[exp9211] phase={phase} completed={completed} pending={pending}", flush=True)


def check(
    path: Path, field: str, expected: Any, observed: Any, name: str, upstream: str = TASK
) -> Json:
    """Keep strict operand types so missing_file, missing_field and observed zero differ."""
    return dict(
        check=name,
        upstream=upstream,
        **ref(path),
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def rubric() -> Json:
    """Freeze the November 11 v2 taxonomy evidence rubric before any model prediction."""
    return dict(
        version="v784-target-identifiability-v2",
        paper="Kim et al., What Hallucinations Have We Measured? Semantic Illusion, Confabulation, and Expressed Uncertainty, arXiv:2610.38520v2 (2026-11-11)",
        factual_error="original human criterion and factual-error evidence",
        expressed_uncertainty="explicit original human annotation",
        reelicitation="same-generator semantically equivalent original prompts",
        targets=dict(
            factual_error=dict(
                column="annotator1_label",
                original_criterion="Primary human label: 1=hallucination, 0=correct, -1=missing/uncertain (factual_correctness)",
                qualifies="binary factual-error target when authenticated annotator1_label is in {0, 1}",
            ),
            expressed_uncertainty=dict(
                column=None,
                original_criterion="Explicit human annotation of expressed uncertainty distinct from missingness",
                qualifies="requires dedicated human uncertainty label; -1 sentinel conflates missing/uncertain or not annotated",
            ),
            reelicitation=dict(
                column="question",
                original_criterion="Same-generator semantically equivalent prompt re-elicitations with human factuality labels",
                qualifies="requires >=2 equivalent prompts per (qid, generator) with human verification to separate SI from CF",
            ),
        ),
        statuses=["present", "absent", "ambiguous"],
        decisions=[
            "source_evidence_present",
            "source_evidence_absent",
            "source_evidence_ambiguous",
        ],
        absent_is_negative=False,
        cross_generator_substitution=False,
        keyword_or_llm_inference_allowed=False,
        predictions_read=False,
        SI_CF_unavailable_without_evidence=True,
    )


def protocol() -> Json:
    """Freeze the exact affine Brier ambiguity protocol and weights before predictions."""
    return dict(
        version="v784-label-ambiguity-v1",
        seed=SEED,
        primary_target="original A1 factual-error binary label in {0, 1}; A2=-1 remains missing",
        contrast="d(y)=(p_control-y)^2-(p_treatment-y)^2=p_control^2-p_treatment^2-2*y*(p_control-p_treatment)",
        scenarios=dict(
            observed_disagreement="observed agreement (A1==A2, A2 in {0,1}) and missing A2 (A2==-1) fix [A1]; observed disagreement (A1!=A2, A2 in {0,1}) permits [0, 1]",
            missing_annotation_envelope="observed agreement (A1==A2, A2 in {0,1}) fixes [A1]; observed disagreement or missing A2 permits [0, 1]",
        ),
        repeated_retention="one binary label variable per unique response (unit_id); sum weighted coefficients across repeated appearances before optimizing endpoints",
        weights=dict(
            static_TEST_qid="1/100",
            sibling="1/3",
            stream="1/10",
            future_qid_within_stream="1/4",
            retention_TEST_qid="1/100",
            retention_measurement="1/3000 per stream-qid-sibling appearance; 1/300 per unique TEST response across 10 streams",
        ),
        pairwise_only=True,
        joint_simultaneous_worst_case_claimed=False,
        primary_gate_rescue=False,
        selection="no mask, A2, endpoint or sensitivity result selects training, temperature, promotion or primary acceptance",
    )


def evaluate_target_evidence(
    record: Json,
    target: str,
    *,
    provenance_valid: bool = True,
    same_generator_prompts: int = 1,
    cross_generator_substitution: bool = False,
    synthetic_category: bool = False,
    conflicting_criterion: bool = False,
) -> Json:
    """Classify target evidence availability without keyword heuristics or model inference."""
    if target not in TARGETS:
        raise ValueError("unknown_target")
    if not provenance_valid or synthetic_category or cross_generator_substitution:
        reason = (
            "missing_source_provenance"
            if not provenance_valid
            else "synthetic_or_inferred_category_rejected"
            if synthetic_category
            else "cross_generator_sibling_substitution_rejected"
        )
        return dict(
            target=target,
            status="absent",
            decision="source_evidence_absent",
            identifiable=False,
            si_cf_identifiable=False,
            reason=reason,
        )
    if conflicting_criterion:
        return dict(
            target=target,
            status="ambiguous",
            decision="source_evidence_ambiguous",
            identifiable=False,
            si_cf_identifiable=False,
            reason="conflicting_criterion_definition",
        )
    a1 = record.get("annotator1")
    if target == "factual_error":
        if a1 in (0, 1):
            return dict(
                target=target,
                status="present",
                decision="source_evidence_present",
                identifiable=True,
                si_cf_identifiable=False,
                reason="Authenticated annotator1_label in {0,1} under original factual_correctness criterion",
            )
        if a1 == -1:
            return dict(
                target=target,
                status="ambiguous",
                decision="source_evidence_ambiguous",
                identifiable=False,
                si_cf_identifiable=False,
                reason="annotator1_label=-1 conflates missing and uncertain in original dictionary",
            )
        return dict(
            target=target,
            status="absent",
            decision="source_evidence_absent",
            identifiable=False,
            si_cf_identifiable=False,
            reason="missing_factual_error_annotation",
        )
    if target == "expressed_uncertainty":
        if a1 == -1:
            return dict(
                target=target,
                status="ambiguous",
                decision="source_evidence_ambiguous",
                identifiable=False,
                si_cf_identifiable=False,
                reason="annotator1_label=-1 conflates missing and uncertain in original dictionary",
            )
        return dict(
            target=target,
            status="absent",
            decision="source_evidence_absent",
            identifiable=False,
            si_cf_identifiable=False,
            reason="No explicit human expressed-uncertainty annotation column in LPHB release; -1 in annotator2_label denotes not annotated",
        )
    if same_generator_prompts >= 2 and bool(record.get("reelicitation_human_verified")):
        return dict(
            target=target,
            status="present",
            decision="source_evidence_present",
            identifiable=True,
            si_cf_identifiable=True,
            reason="Authenticated same-generator equivalent-prompt re-elicitations with human factuality verification",
        )
    if same_generator_prompts >= 2:
        return dict(
            target=target,
            status="ambiguous",
            decision="source_evidence_ambiguous",
            identifiable=False,
            si_cf_identifiable=False,
            reason="Multiple same-generator prompts lack authenticated human factuality labels per prompt",
        )
    return dict(
        target=target,
        status="absent",
        decision="source_evidence_absent",
        identifiable=False,
        si_cf_identifiable=False,
        reason="Single prompt per (qid, generator) in release; three siblings are distinct generators (gemma-2-9b, llama3-8b, mistral-7b), not same-generator equivalent-prompt re-elicitations",
    )


def build_masks_and_targets(
    labels: list[Json],
    master_rows: dict[str, Json],
    master_ref: Json,
    dict_ref: Json,
    *,
    expected_qids: int | None = 300,
) -> Json:
    """Build immutable per-response admissible label sets and v2 taxonomy rows."""
    groups: dict[str, list[Json]] = defaultdict(list)
    seen: set[str] = set()
    for rec in labels:
        uid = rec.get("unit_id")
        a1 = rec.get("annotator1")
        raw_a2 = rec.get("annotator2")
        a2 = -1 if raw_a2 is None else raw_a2
        if (
            not isinstance(uid, str)
            or uid in seen
            or type(a1) is not int
            or a1 not in (0, 1)
            or type(a2) is not int
            or a2 not in (-1, 0, 1)
            or uid not in master_rows
        ):
            raise ValueError("duplicate_or_label_encoding")
        seen.add(uid)
        groups[str(rec["qid"])].append(dict(rec, annotator1=a1, annotator2=a2))
    if expected_qids is not None and len(groups) != expected_qids:
        raise ValueError("whole_qid_count")
    rows: list[Json] = []
    mask_rows: list[Json] = []
    target_rows: list[Json] = []
    rubric_defs = rubric()["targets"]
    for idx, (qid, siblings) in enumerate(sorted(groups.items())):
        if idx % 100 == 0:
            progress("build_masks_and_targets", idx, len(groups) - idx)
        if (
            len(siblings) != 3
            or len({(s["role"], s["group_id"]) for s in siblings}) != 1
            or siblings[0]["role"] not in ROLES
            or {master_rows[s["unit_id"]]["model"] for s in siblings} != set(GENERATORS)
        ):
            raise ValueError("whole_qid_partition")
        for s in sorted(siblings, key=lambda item: str(item["unit_id"])):
            uid = str(s["unit_id"])
            mrec = master_rows[uid]
            sibling = str(mrec["model"])
            role = str(s["role"])
            a1, a2 = int(s["annotator1"]), int(s["annotator2"])
            if a2 == -1:
                a2_status = "missing_a2"
                obs_adm = [a1]
                env_adm = [0, 1]
            elif a1 == a2:
                a2_status = "observed_agreement"
                obs_adm = [a1]
                env_adm = [a1]
            else:
                a2_status = "observed_disagreement"
                obs_adm = [0, 1]
                env_adm = [0, 1]
            t_evals = {
                t: evaluate_target_evidence(s, t, provenance_valid=True, same_generator_prompts=1)
                for t in TARGETS
            }
            for t in TARGETS:
                ev = t_evals[t]
                target_rows.append(
                    dict(
                        unit_id=f"{uid}:{t}",
                        response_unit_id=uid,
                        qid=qid,
                        sibling=sibling,
                        role=role,
                        target=t,
                        evidence_path=master_ref["path"],
                        evidence_sha256=master_ref["sha256"],
                        dictionary_path=dict_ref["path"],
                        dictionary_sha256=dict_ref["sha256"],
                        source_row=int(mrec["original_row"]),
                        source_column=rubric_defs[t]["column"],
                        original_criterion=rubric_defs[t]["original_criterion"],
                        status=ev["status"],
                        decision=ev["decision"],
                        identifiable=ev["identifiable"],
                        si_cf_identifiable=ev["si_cf_identifiable"],
                        reason=ev["reason"],
                    )
                )
            mask_entry = dict(
                unit_id=uid,
                qid=qid,
                group_id=s["group_id"],
                sibling=sibling,
                role=role,
                annotator1=a1,
                annotator2=a2,
                a2_status=a2_status,
                observed_disagreement_admissible=obs_adm,
                missing_annotation_envelope_admissible=env_adm,
            )
            mask_rows.append(mask_entry)
            rows.append(
                dict(
                    **mask_entry,
                    group=qid,
                    arm="human_label_ambiguity",
                    condition=role,
                    seed=SEED,
                    factual_error_status=t_evals["factual_error"]["status"],
                    expressed_uncertainty_status=t_evals["expressed_uncertainty"]["status"],
                    reelicitation_status=t_evals["reelicitation"]["status"],
                    identifiable_si_cf=False,
                    metric="A1_factual_error",
                    absolute_metric=a1,
                    numerator=a1,
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                    censor_reason=None,
                )
            )
    if len({r["group_id"] for r in rows}) != len(groups):
        raise ValueError("duplicate_group_identity")
    return dict(rows=rows, mask_rows=mask_rows, target_identifiability_rows=target_rows)


def reduce_contrast_endpoints(
    observations: list[Json],
    masks: dict[str, Json] | list[Json],
    scenario: str = "observed_disagreement",
) -> Json:
    """Compute exact affine Brier contrast extrema with one shared label per response."""
    if scenario not in ("observed_disagreement", "missing_annotation_envelope"):
        raise ValueError("unknown_scenario")
    if not observations:
        raise ValueError("empty_observations")
    mask_index = {str(m["unit_id"]): m for m in masks} if isinstance(masks, list) else dict(masks)
    key = (
        "observed_disagreement_admissible"
        if scenario == "observed_disagreement"
        else "missing_annotation_envelope_admissible"
    )
    default_weight = 1.0 / len(observations)
    const_by_unit: dict[str, float] = defaultdict(float)
    slope_by_unit: dict[str, float] = defaultdict(float)
    weight_by_unit: dict[str, float] = defaultdict(float)
    count_by_unit: Counter[str] = Counter()
    for obs in observations:
        uid = str(obs.get("unit_id", ""))
        if uid not in mask_index:
            raise ValueError("unknown_unit_id")
        c = float(obs["p_control"])
        t = float(obs["p_treatment"])
        w = float(obs.get("weight", default_weight))
        if (
            not (math.isfinite(c) and math.isfinite(t) and math.isfinite(w))
            or not (0.0 <= c <= 1.0 and 0.0 <= t <= 1.0)
            or w <= 0.0
        ):
            raise ValueError("invalid_probability_or_weight")
        const_by_unit[uid] += w * (c * c - t * t)
        slope_by_unit[uid] += -2.0 * w * (c - t)
        weight_by_unit[uid] += w
        count_by_unit[uid] += 1
    total_weight = math.fsum(weight_by_unit.values())
    a1_terms: list[float] = []
    min_terms: list[float] = []
    max_terms: list[float] = []
    min_assignment: dict[str, int] = {}
    max_assignment: dict[str, int] = {}
    unit_terms: list[Json] = []
    variable_count = 0
    for uid in sorted(count_by_unit):
        m = mask_index[uid]
        a1 = m.get("annotator1")
        admissible = m.get(key)
        if (
            type(a1) is not int
            or a1 not in (0, 1)
            or not isinstance(admissible, list)
            or not admissible
            or any(type(y) is not int or y not in (0, 1) for y in admissible)
        ):
            raise ValueError("invalid_mask_entry")
        allowed = sorted(set(admissible))
        if len(allowed) > 1:
            variable_count += 1
        const_val = const_by_unit[uid] / total_weight
        slope_val = slope_by_unit[uid] / total_weight
        a1_val = const_val + slope_val * a1
        candidates = [(const_val + slope_val * y, abs(y - a1), y) for y in allowed]
        min_val, _, y_min = min(candidates, key=lambda item: (item[0], item[1]))
        max_val, _, y_max = max(candidates, key=lambda item: (item[0], -item[1]))
        a1_terms.append(a1_val)
        min_terms.append(min_val)
        max_terms.append(max_val)
        min_assignment[uid] = y_min
        max_assignment[uid] = y_max
        unit_terms.append(
            dict(
                unit_id=uid,
                appearances=count_by_unit[uid],
                weight=weight_by_unit[uid] / total_weight,
                constant_term=const_val,
                linear_coefficient=slope_val,
                admissible=allowed,
                a1=a1,
                y_min=y_min,
                y_max=y_max,
                a1_contribution=a1_val,
                min_contribution=min_val,
                max_contribution=max_val,
            )
        )
    a1_contrast = math.fsum(a1_terms)
    min_contrast = math.fsum(min_terms)
    max_contrast = math.fsum(max_terms)
    return dict(
        scenario=scenario,
        a1_contrast=a1_contrast,
        min_contrast=min_contrast,
        max_contrast=max_contrast,
        width=max_contrast - min_contrast,
        observation_count=len(observations),
        unique_response_count=len(count_by_unit),
        variable_response_count=variable_count,
        fixed_response_count=len(count_by_unit) - variable_count,
        repeated_response_count=sum(v > 1 for v in count_by_unit.values()),
        weight_sum=total_weight,
        min_assignment=min_assignment,
        max_assignment=max_assignment,
        unit_terms=unit_terms,
        shared_response_labels_enforced=True,
        joint_simultaneous_worst_case_claimed=False,
    )


def reduce_both_scenarios(observations: list[Json], masks: dict[str, Json] | list[Json]) -> Json:
    """Evaluate both frozen ambiguity scenarios on identical weighted observations."""
    obs = reduce_contrast_endpoints(observations, masks, "observed_disagreement")
    env = reduce_contrast_endpoints(observations, masks, "missing_annotation_envelope")
    return dict(
        observed_disagreement=obs,
        missing_annotation_envelope=env,
        a1_contrast=obs["a1_contrast"],
        envelope_contains_observed=(
            env["min_contrast"] <= obs["min_contrast"] + 1e-15
            and obs["max_contrast"] <= env["max_contrast"] + 1e-15
        ),
    )


def controls(mask_rows: list[Json], rosters: Json) -> Json:
    """Prove exact endpoint reduction, shared retention labels and support counts."""
    progress("controls_before")
    panel_masks = [
        dict(
            unit_id="ctrl_agree",
            qid="ctrl_q0",
            sibling="llama3-8b",
            role="TEST",
            annotator1=0,
            annotator2=0,
            a2_status="observed_agreement",
            observed_disagreement_admissible=[0],
            missing_annotation_envelope_admissible=[0],
        ),
        dict(
            unit_id="ctrl_disagree_a",
            qid="ctrl_q0",
            sibling="gemma-2-9b",
            role="TEST",
            annotator1=1,
            annotator2=0,
            a2_status="observed_disagreement",
            observed_disagreement_admissible=[0, 1],
            missing_annotation_envelope_admissible=[0, 1],
        ),
        dict(
            unit_id="ctrl_disagree_b",
            qid="ctrl_q0",
            sibling="mistral-7b",
            role="TEST",
            annotator1=0,
            annotator2=1,
            a2_status="observed_disagreement",
            observed_disagreement_admissible=[0, 1],
            missing_annotation_envelope_admissible=[0, 1],
        ),
        dict(
            unit_id="ctrl_missing",
            qid="ctrl_q1",
            sibling="llama3-8b",
            role="TEST",
            annotator1=1,
            annotator2=-1,
            a2_status="missing_a2",
            observed_disagreement_admissible=[1],
            missing_annotation_envelope_admissible=[0, 1],
        ),
    ]
    base_obs = [
        dict(unit_id="ctrl_agree", p_control=0.8, p_treatment=0.2, weight=0.25),
        dict(unit_id="ctrl_disagree_a", p_control=0.7, p_treatment=0.3, weight=0.25),
        dict(unit_id="ctrl_disagree_b", p_control=0.2, p_treatment=0.6, weight=0.25),
        dict(unit_id="ctrl_missing", p_control=0.9, p_treatment=0.1, weight=0.25),
    ]
    both = reduce_both_scenarios(base_obs, panel_masks)
    endpoint_rows: list[Json] = []
    for scenario, key in [
        ("observed_disagreement", "observed_disagreement_admissible"),
        ("missing_annotation_envelope", "missing_annotation_envelope_admissible"),
    ]:
        grid = [m[key] for m in panel_masks]
        exhaustive = []
        for assignment in itertools.product(*grid):
            ymap = {m["unit_id"]: y for m, y in zip(panel_masks, assignment, strict=True)}
            total = math.fsum(
                o["weight"]
                * (
                    (o["p_control"] - ymap[o["unit_id"]]) ** 2
                    - (o["p_treatment"] - ymap[o["unit_id"]]) ** 2
                )
                for o in base_obs
            )
            exhaustive.append((total, ymap))
        exact_min = min(v[0] for v in exhaustive)
        exact_max = max(v[0] for v in exhaustive)
        red = both[scenario]
        ok = math.isclose(red["min_contrast"], exact_min, abs_tol=1e-15) and math.isclose(
            red["max_contrast"], exact_max, abs_tol=1e-15
        )
        endpoint_rows.append(
            dict(
                control_id="exhaustive_small_panel_" + scenario,
                scenario=scenario,
                probabilities=[
                    dict(
                        unit_id=o["unit_id"],
                        p_control=o["p_control"],
                        p_treatment=o["p_treatment"],
                        weight=o["weight"],
                    )
                    for o in base_obs
                ],
                admissible_assignments=len(exhaustive),
                exact_min=exact_min,
                exact_max=exact_max,
                reducer_min=red["min_contrast"],
                reducer_max=red["max_contrast"],
                a1_contrast=red["a1_contrast"],
                min_assignment=red["min_assignment"],
                max_assignment=red["max_assignment"],
                passed=ok,
            )
        )
    # Repeated retention across 10 streams with opposite-sign stream coefficients.
    ten_stream_obs = [
        dict(
            unit_id="ctrl_disagree_a",
            p_control=0.8 if s < 6 else 0.1,
            p_treatment=0.2 if s < 6 else 0.9,
            weight=0.1,
        )
        for s in range(10)
    ]
    ret_red = reduce_contrast_endpoints(ten_stream_obs, panel_masks, "observed_disagreement")
    ret_y0 = math.fsum(
        o["weight"] * (o["p_control"] ** 2 - o["p_treatment"] ** 2) for o in ten_stream_obs
    )
    ret_y1 = math.fsum(
        o["weight"] * ((o["p_control"] - 1.0) ** 2 - (o["p_treatment"] - 1.0) ** 2)
        for o in ten_stream_obs
    )
    per_obs_min = math.fsum(
        o["weight"]
        * min(
            o["p_control"] ** 2 - o["p_treatment"] ** 2,
            (o["p_control"] - 1.0) ** 2 - (o["p_treatment"] - 1.0) ** 2,
        )
        for o in ten_stream_obs
    )
    endpoint_rows.append(
        dict(
            control_id="ten_stream_shared_label_retention",
            scenario="observed_disagreement",
            probabilities=ten_stream_obs,
            admissible_assignments=2,
            exact_min=min(ret_y0, ret_y1),
            exact_max=max(ret_y0, ret_y1),
            reducer_min=ret_red["min_contrast"],
            reducer_max=ret_red["max_contrast"],
            a1_contrast=ret_red["a1_contrast"],
            min_assignment=ret_red["min_assignment"],
            max_assignment=ret_red["max_assignment"],
            forbidden_per_observation_min=per_obs_min,
            passed=(
                math.isclose(ret_red["min_contrast"], min(ret_y0, ret_y1), abs_tol=1e-15)
                and math.isclose(ret_red["max_contrast"], max(ret_y0, ret_y1), abs_tol=1e-15)
                and per_obs_min < ret_red["min_contrast"] - 0.1
            ),
        )
    )
    perm_red = reduce_contrast_endpoints(
        list(reversed(base_obs)), panel_masks, "missing_annotation_envelope"
    )
    dup_red = reduce_contrast_endpoints(
        [dict(o, weight=o["weight"] / 10.0) for o in base_obs * 10],
        panel_masks,
        "missing_annotation_envelope",
    )
    swap_red = reduce_contrast_endpoints(
        [dict(o, p_control=o["p_treatment"], p_treatment=o["p_control"]) for o in base_obs],
        panel_masks,
        "missing_annotation_envelope",
    )
    const_red = reduce_contrast_endpoints(
        [dict(o, p_treatment=o["p_control"]) for o in base_obs],
        panel_masks,
        "missing_annotation_envelope",
    )
    bound_obs = [
        dict(unit_id="ctrl_disagree_a", p_control=1.0, p_treatment=0.0, weight=0.5),
        dict(unit_id="ctrl_missing", p_control=0.0, p_treatment=1.0, weight=0.5),
    ]
    bound_red = reduce_contrast_endpoints(bound_obs, panel_masks, "missing_annotation_envelope")
    env_ref = both["missing_annotation_envelope"]
    for cid, red, exp_min, exp_max, probs in [
        (
            "permutation_invariance",
            perm_red,
            env_ref["min_contrast"],
            env_ref["max_contrast"],
            list(reversed(base_obs)),
        ),
        (
            "duplicate_ten_stream_scaling",
            dup_red,
            env_ref["min_contrast"],
            env_ref["max_contrast"],
            base_obs,
        ),
        (
            "sign_reversal_symmetry",
            swap_red,
            -env_ref["max_contrast"],
            -env_ref["min_contrast"],
            [dict(o, p_control=o["p_treatment"], p_treatment=o["p_control"]) for o in base_obs],
        ),
        (
            "constant_prediction_zero_width",
            const_red,
            0.0,
            0.0,
            [dict(o, p_treatment=o["p_control"]) for o in base_obs],
        ),
        ("boundary_ZeroOne_probabilities", bound_red, -1.0, 1.0, bound_obs),
    ]:
        endpoint_rows.append(
            dict(
                control_id=cid,
                scenario="missing_annotation_envelope",
                probabilities=probs,
                admissible_assignments=8 if len(probs) == 4 else 4,
                exact_min=exp_min,
                exact_max=exp_max,
                reducer_min=red["min_contrast"],
                reducer_max=red["max_contrast"],
                a1_contrast=red["a1_contrast"],
                min_assignment=red["min_assignment"],
                max_assignment=red["max_assignment"],
                passed=(
                    math.isclose(red["min_contrast"], exp_min, abs_tol=1e-15)
                    and math.isclose(red["max_contrast"], exp_max, abs_tol=1e-15)
                ),
            )
        )
    # Empirical TEST100 and online future40 / retention100 mask fixture checks (method only).
    test_masks = [m for m in mask_rows if m["role"] == "TEST"]
    test_fixture = [
        dict(unit_id=m["unit_id"], p_control=0.55, p_treatment=0.45, weight=1.0 / 300.0)
        for m in test_masks
    ]
    test_both = reduce_both_scenarios(test_fixture, test_masks)
    endpoint_rows.append(
        dict(
            control_id="empirical_TEST100_mask_fixture",
            scenario="both",
            probabilities=test_fixture[:3],
            admissible_assignments=2**7,
            exact_min=test_both["observed_disagreement"]["min_contrast"],
            exact_max=test_both["observed_disagreement"]["max_contrast"],
            reducer_min=test_both["observed_disagreement"]["min_contrast"],
            reducer_max=test_both["observed_disagreement"]["max_contrast"],
            a1_contrast=test_both["a1_contrast"],
            min_assignment=test_both["observed_disagreement"]["min_assignment"],
            max_assignment=test_both["observed_disagreement"]["max_assignment"],
            passed=bool(
                test_both["envelope_contains_observed"]
                and test_both["observed_disagreement"]["variable_response_count"] == 7
                and test_both["missing_annotation_envelope"]["variable_response_count"] == 214
            ),
        )
    )
    sample_rec = dict(
        unit_id="q__llama3-8b",
        qid="q",
        sibling="llama3-8b",
        role="TEST",
        annotator1=1,
        annotator2=-1,
    )
    target_controls = [
        dict(
            control_id="missing_provenance_rejected",
            passed=not evaluate_target_evidence(
                sample_rec, "factual_error", provenance_valid=False
            )["identifiable"],
        ),
        dict(
            control_id="synthetic_category_rejected",
            passed=not evaluate_target_evidence(
                sample_rec, "expressed_uncertainty", synthetic_category=True
            )["identifiable"],
        ),
        dict(
            control_id="cross_generator_sibling_substitution_rejected",
            passed=not evaluate_target_evidence(
                sample_rec, "reelicitation", cross_generator_substitution=True
            )["si_cf_identifiable"],
        ),
        dict(
            control_id="conflicting_criterion_ambiguous",
            passed=evaluate_target_evidence(
                sample_rec, "factual_error", conflicting_criterion=True
            )["decision"]
            == "source_evidence_ambiguous",
        ),
        dict(
            control_id="sentinel_minus_one_ambiguous",
            passed=evaluate_target_evidence(
                dict(sample_rec, annotator1=-1), "expressed_uncertainty"
            )["decision"]
            == "source_evidence_ambiguous",
        ),
    ]
    stream_qids = {q for s in rosters["stream_roster"] for q in s}
    future_qids = {q for s in rosters["stream_roster"] for q in s[1:]}
    supports: Json = {}
    for name, subset, stream_factor in [
        ("ALL", mask_rows, 1),
        ("TRAIN", [m for m in mask_rows if m["role"] == "TRAIN"], 1),
        ("CAL", [m for m in mask_rows if m["role"] == "CAL"], 1),
        ("TEST", [m for m in mask_rows if m["role"] == "TEST"], 1),
        (
            "initial100",
            [m for m in mask_rows if m["qid"] in set(rosters["initial_group_roster"])],
            1,
        ),
        ("stream50", [m for m in mask_rows if m["qid"] in stream_qids], 1),
        ("future40", [m for m in mask_rows if m["qid"] in future_qids], 1),
        ("guard50", [m for m in mask_rows if m["qid"] in set(rosters["guard_roster"])], 1),
        (
            "retention100",
            [m for m in mask_rows if m["qid"] in set(rosters["retention_roster"])],
            10,
        ),
    ]:
        supports[name] = dict(
            independent_qids=len({m["qid"] for m in subset}),
            unique_responses=len(subset),
            repeated_stream_appearances=len(subset) * stream_factor,
            a1_observed_responses=sum(m["annotator1"] in (0, 1) for m in subset),
            a1_missing_responses=0,
            a1_error_responses=sum(m["annotator1"] == 1 for m in subset),
            a1_correct_responses=sum(m["annotator1"] == 0 for m in subset),
            a2_observed_responses=sum(m["annotator2"] in (0, 1) for m in subset),
            a2_missing_responses=sum(m["annotator2"] == -1 for m in subset),
            observed_agreement_responses=sum(
                m["a2_status"] == "observed_agreement" for m in subset
            ),
            observed_disagreement_responses=sum(
                m["a2_status"] == "observed_disagreement" for m in subset
            ),
            observed_disagreement_variable_responses=sum(
                len(m["observed_disagreement_admissible"]) == 2 for m in subset
            ),
            observed_disagreement_fixed_responses=sum(
                len(m["observed_disagreement_admissible"]) == 1 for m in subset
            ),
            missing_annotation_envelope_variable_responses=sum(
                len(m["missing_annotation_envelope_admissible"]) == 2 for m in subset
            ),
            missing_annotation_envelope_fixed_responses=sum(
                len(m["missing_annotation_envelope_admissible"]) == 1 for m in subset
            ),
        )
    target_summary = dict(
        evaluated_primary_target="original_human_factual_error_A1",
        by_role={
            role: dict(
                responses=len([m for m in mask_rows if role == "ALL" or m["role"] == role]),
                factual_error_present=len(
                    [m for m in mask_rows if role == "ALL" or m["role"] == role]
                ),
                expressed_uncertainty_present=0,
                expressed_uncertainty_absent=len(
                    [m for m in mask_rows if role == "ALL" or m["role"] == role]
                ),
                reelicitation_present=0,
                reelicitation_absent=len(
                    [m for m in mask_rows if role == "ALL" or m["role"] == role]
                ),
                identifiable_si_cf_rows=0,
            )
            for role in ("TRAIN", "CAL", "TEST", "ALL")
        },
        identifiable_si_cf_rows=0,
        identifiable_expressed_uncertainty_rows=0,
        identifiable_factual_error_rows=len(mask_rows),
        category_metrics="unavailable: zero identifiable SI/CF or expressed-uncertainty rows in original human release",
    )
    progress("controls_after", len(endpoint_rows), 0)
    return dict(
        endpoint_control_rows=endpoint_rows,
        target_control_rows=target_controls,
        support_counts=supports,
        target_summary=target_summary,
    )


def inspect(root: Path) -> Json:
    """Independently authenticate Exp9097 custody and Exp9126/9132 role authority."""
    progress("authenticate_before")
    result: Json = dict(
        checks=[],
        source_references=[],
        protocol_references={},
        rows=[],
        mask_rows=[],
        target_identifiability_rows=[],
        rosters={},
        controls={},
        reused_health_receipt={},
    )
    checks = result["checks"]
    values: dict[str, Json] = {}
    try:
        for rel_path, pin in PINS.items():
            path = root / rel_path
            upstream = "exp" + Path(rel_path).name.split("_")[1]
            gate = check(
                path,
                "sha256",
                pin,
                ref(path)["sha256"] or "missing_file",
                "immutable_bytes",
                upstream,
            )
            checks.append(gate)
            result["source_references"].append(
                dict(
                    path=str(path),
                    producer_path=str(path),
                    sha256=pin,
                    imported_fields=["original_authority", "readiness", "protocol_references"],
                )
            )
            if gate["passed"]:
                values[Path(rel_path).name.split("_")[1]] = json.loads(path.read_bytes())
        if len(values) != len(PINS):
            return result
        for num, expected_fields in [
            ("9097", dict(source_ready_score=1, required_checks_passed=True)),
            ("9126", dict(methods_ready_score=1, required_checks_passed=True)),
            ("9132", dict(online_methods_ready_score=1, required_checks_passed=True)),
            ("9137", dict(reconciliation_ready_score=1, required_checks_passed=True)),
            ("9141", dict(contrast_protocol_ready_score=1, required_checks_passed=True)),
            (
                "9180",
                dict(
                    source_ready_score=1,
                    static_methods_ready_score=1,
                    online_methods_ready_score=1,
                    required_checks_passed=True,
                ),
            ),
        ]:
            rel_path = next(p for p in PINS if p.startswith(f"results/experiment_{num}_"))
            path = root / rel_path
            for field, expected in expected_fields.items():
                checks.append(
                    check(
                        path,
                        field,
                        expected,
                        values[num].get(field, "missing_field"),
                        "upstream_gate",
                        f"exp{num}",
                    )
                )
        if not all(c["passed"] for c in checks):
            return result
        cap_work_ref = values["9137"]["work_reference"]
        cap_work_path = Path(cap_work_ref["path"])
        checks.append(
            check(
                cap_work_path,
                "sha256",
                cap_work_ref["sha256"],
                ref(cap_work_path)["sha256"] or "missing_file",
                "capstone_work",
                "exp9137",
            )
        )
        saved_operands = json.loads(cap_work_path.read_bytes())["operands"]

        def bind(reference: Json, upstream: str, fields: list[str]) -> Path:
            original = str(reference["path"])
            retained = saved_operands.get(original, reference)
            target = Path(retained["path"])
            checks.append(
                check(
                    target,
                    "sha256",
                    reference["sha256"],
                    ref(target)["sha256"] or "missing_file",
                    "retained_producer_binding",
                    upstream,
                )
            )
            result["source_references"].append(
                dict(
                    path=str(target),
                    producer_path=reference.get("producer_path", original),
                    sha256=reference["sha256"],
                    imported_fields=fields,
                )
            )
            return target

        a9097 = values["9097"]
        for r in [*a9097["source_references"], *a9097["current_work_receipts"]]:
            if "path" in r and "sha256" in r:
                bind(r, "exp9097", ["human_custody_chain"])
        manifest_path = bind(
            a9097["source_manifest_reference"], "exp9097", ["release_files", "target_mapping"]
        )
        manifest = json.loads(manifest_path.read_bytes())
        for name, file_ref in manifest["release_files"].items():
            bound_file = bind(file_ref, "exp8389", [name])
            old.authenticate(name, dict(file_ref, path=str(bound_file)))
        for key, ref_obj in sorted(values["9180"]["protocol_references"].items()):
            bound_p = bind(ref_obj, "exp9180", [key])
            result["protocol_references"][key] = dict(
                ref(bound_p), producer_path=ref_obj.get("producer_path", ref_obj["path"])
            )
        for num, runner, label in [
            ("9126", static_runner, "static"),
            ("9132", online_runner, "online"),
        ]:
            m_work_path = bind(
                values[num]["work_reference"], f"exp{num}", ["method_work_and_receipts"]
            )
            m_work = json.loads(m_work_path.read_bytes())
            m_val = dict(
                checks=values[num]["validation_receipts"],
                owned_statement_coverage=values[num]["owned_statement_coverage"],
            )
            m_inputs = {
                r["path"]: bind(r, f"exp{num}", ["method_code_or_test"])
                for r in m_work["scope"]["code_test_hashes"]
            }
            with patch.object(
                runner.d,
                "ref",
                lambda p, mapping=m_inputs: dict(
                    ref(p), sha256=ref(mapping.get(str(p), p))["sha256"]
                ),
            ):
                receipt_ok = bool(runner.receipts_pass(m_work, m_val))
            checks.append(
                check(
                    m_work_path,
                    f"{label}_method_receipts_pass",
                    True,
                    receipt_ok,
                    "method_receipt_replay",
                    f"exp{num}",
                )
            )
        master_ref_raw = manifest["release_files"][old.MASTER]
        dict_ref_raw = manifest["release_files"]["data/human_judge_release_data_dictionary.csv"]
        master_path = Path(saved_operands.get(master_ref_raw["path"], master_ref_raw)["path"])
        dict_path = Path(saved_operands.get(dict_ref_raw["path"], dict_ref_raw)["path"])
        with master_path.open(encoding="utf-8-sig", newline="") as stream:
            master_rows = {
                row["item_id"]: dict(row, original_row=idx + 2)
                for idx, row in enumerate(csv.DictReader(stream))
            }
        eval_labels = json.loads(
            Path(result["protocol_references"]["evaluator_label_reference"]["path"]).read_bytes()
        )["rows"]
        inf_view = json.loads(
            Path(result["protocol_references"]["inference_view_reference"]["path"]).read_bytes()
        )["rows"]
        rosters = json.loads(
            Path(result["protocol_references"]["initial_roster_reference"]["path"]).read_bytes()
        )
        view_clean = len(inf_view) == 900 and all(
            set(r) == {"unit_id", "qid", "sibling", "question", "answer_text"}
            and "sha256:" + hashlib.sha256(r["question"].encode()).hexdigest()
            == "sha256:"
            + hashlib.sha256(master_rows[r["unit_id"]]["question"].encode()).hexdigest()
            and "sha256:" + hashlib.sha256(r["answer_text"].encode()).hexdigest()
            == "sha256:"
            + hashlib.sha256(master_rows[r["unit_id"]]["answer_text"].encode()).hexdigest()
            for r in inf_view
        )
        checks.append(
            check(
                Path(result["protocol_references"]["inference_view_reference"]["path"]),
                "label_free_inference_view_900",
                True,
                view_clean,
                "inference_view_separation",
                "exp9124",
            )
        )
        shards = [
            json.loads(
                Path(
                    result["protocol_references"][f"shard_{s}_roster_reference"]["path"]
                ).read_bytes()
            )["qids"]
            for s in ("a", "b", "c")
        ]
        qids = sorted({r["qid"] for r in eval_labels})
        checks.append(
            check(
                master_path,
                "lexical_shards_equal_300_qids",
                [qids[:100], qids[100:200], qids[200:300]],
                shards,
                "lexical_shard_partition",
                "exp9180",
            )
        )
        built = build_masks_and_targets(
            eval_labels,
            master_rows,
            dict(path=str(master_path), sha256=master_ref_raw["sha256"]),
            dict(path=str(dict_path), sha256=dict_ref_raw["sha256"]),
        )
        ctrl = controls(built["mask_rows"], rosters)
        checks.append(
            check(
                master_path,
                "all_endpoint_and_target_controls_passed",
                True,
                all(r["passed"] for r in ctrl["endpoint_control_rows"])
                and all(r["passed"] for r in ctrl["target_control_rows"]),
                "method_controls",
                TASK,
            )
        )
        if all(c["passed"] for c in checks):
            result.update(
                rows=built["rows"],
                mask_rows=built["mask_rows"],
                target_identifiability_rows=built["target_identifiability_rows"],
                rosters=rosters,
                controls=ctrl,
                reused_health_receipt=values["9180"].get("repository_health", {}),
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append(
            check(
                root,
                "authenticated_source_custody",
                "complete_valid_source",
                str(error),
                "source_custody_exception",
                TASK,
            )
        )
    progress("authenticate_after", len(result["rows"]), 0)
    return result
