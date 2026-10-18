"""REQ-REPORT-8074: independently audit cached development source decisions.

Public predictions precede human-target access in separate exited processes.
Fixed-model interventions diagnose dependence, without new generator evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot import experiment_8072_v699_sealed_methods as methods
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import evidence_features_7980 as features

Json = dict[str, Any]
Array = NDArray[np.float64]
ROOT = methods.ROOT
NAME = "experiment_8074_v699_interaction_decision_audit"
TASK = "exp8074-interaction-decision-audit"
MODULE, CLI, TEST = (
    f"python/carnot/{NAME}.py",
    f"scripts/experiments/{NAME}.py",
    "tests/python/test_interaction_decision_8074.py",
)
OWNED = [MODULE, CLI]
ARMS = methods.methods()["source"]["arms"]
SEED = 6998074
CONFIG = dict(
    seed=SEED,
    draws=10000,
    margin=0.02,
    groups=72,
    per_class=8,
    beneficial=5,
    brier_increase=0.01,
    alpha=0.05,
    permutation="source-hash sorted cyclic successor; all eight features; q fixed",
    ablations="all three coefficients zero; each separately zero; no retraining",
)
PINS = {
    "experiment_8073_v699_interaction_energy_fit": "sha256:7bb65301fd326fdbffb9c3ddc318729f2cafb0019adf04b34e8867142bbb3689",
    "experiment_8072_v699_sealed_methods": "sha256:159df4b8b393c5a4feaeb71292b81b7b1c7219de39bdfd34d47a8bd1ba026d9e",
    "experiment_7995_v693_qwen_development_capture": "sha256:df6eb8b559181455348b1d806f23c36d13c7f49202c2d49b2233aa80edebe02d",
}
PUBLIC = "results/raw/experiment_8019_v695_eligible_targets/public/543f3435b76890308279f715cca49cc94301f01b8cf88aeaf212cef7c567fe89.json"
LABELS = "results/raw/experiment_8019_v695_eligible_targets/evaluator/737dc62554bbed00082171a976714f08c4720b92f77f24a1a6152b1e64d36794.json"
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real completed counts let operators detect stalled children without guessing."""
    print(
        f"[exp8074] {phase} elapsed_s={time.monotonic() - START:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


class InputBlock(ValueError):
    """Exact failed operands distinguish absent evidence from a measured zero."""

    def __init__(self, path: Path, field: str, expected: Any, observed: Any):
        self.check = dict(
            check=field,
            upstream=path.stem,
            path=str(path.absolute()),
            hash=sha256_file(path) if path.is_file() else None,
            field=field,
            op="==",
            expected=expected,
            observed=observed,
        )
        super().__init__(field)


def require(path: Path, field: str, expected: Any, observed: Any) -> None:
    """Fail before dependent work can consume an unauthenticated operand."""
    if observed != expected:
        raise InputBlock(path, field, expected, observed)


def bind(
    path: Path,
    raw: Path,
    refs: list[Json],
    digest: str | None = None,
    *,
    fields: tuple[str, ...] | None = None,
) -> Json:
    """Preserve exact source bytes so later readers can replay the original inputs."""
    require(path, "resource_exists", True, path.is_file())
    actual = sha256_file(path)
    require(path, "sha256", digest or actual, actual)
    destination = raw / "inputs" / (actual[7:] + path.suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(path.read_bytes())
    refs.append(dict(reference(destination), original_path=str(path)))
    if fields is not None:
        encoded = destination.read_text()
        decoder = json.JSONDecoder()
        selected = {}
        for field in fields:
            marker = "\n  " + json.dumps(field) + ":"
            require(path, "public_metadata_field." + field, True, marker in encoded)
            offset = encoded.index(marker) + len(marker)
            selected[field] = decoder.raw_decode(encoded[offset:].lstrip())[0]
        return selected
    return dict(json.loads(destination.read_text()))


def action(p: float | None) -> str:
    """Escalation wins boundary ties; absent predictions cannot become accepts."""
    if p is None:
        return "escalate"
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    return "accept" if p < 0.1 else "reject" if p > 0.5 else "escalate"


def predict(head: Json, x: Array, *, logistic: bool = False) -> Array:
    """Rebuild the registered basis without importing the producer's predictor."""
    g, arm = head["geometry"], head["arm"]
    if g["feature_names"] != methods.methods()["source"]["features"] or x.shape[1] != 9:
        raise ValueError("feature_columns")
    lo, hi = np.asarray(g["scaler"]["minimum"]), np.asarray(g["scaler"]["maximum"])
    u = np.clip((x - lo) / np.where(hi > lo, hi - lo, 1.0), 0, 1)
    q = np.clip(x[:, 0], 0.0001, 0.9999)
    z = (np.log(q / (1 - q)) - g["logit_center"]) / g["logit_scale"]
    matrix = np.ones((len(x), 1))
    if arm != "intercept":
        matrix = np.column_stack((matrix, z))
    if arm == "linear":
        matrix = np.column_stack((matrix, u[:, 1:]))
    if arm in {"additive", "interaction"}:
        splines = [BSpline.design_matrix(u[:, j], g["knots"], 3).toarray() for j in range(9)]
        matrix = np.column_stack((matrix, *splines))
    if arm == "interaction":
        centered = u - np.asarray(g["feature_centers"])
        terms = np.column_stack(
            (z * centered[:, 2], z * centered[:, 8], centered[:, 6] * centered[:, 4])
        )
        matrix = np.column_stack((matrix, terms - np.asarray(g["interaction_centers"])))
    b, a = head["calibration"]
    f = b + a * (matrix @ np.asarray(head["parameters"]))
    if logistic:
        return np.asarray(expit(f), dtype=float)
    supported, unsupported = np.exp(-np.maximum(f, 0)), np.exp(f - np.maximum(f, 0))
    return np.asarray(unsupported / (supported + unsupported), dtype=float)


def public_inputs(
    root: Path, raw: Path, *, absent_head: bool = False, future_label: bool = False
) -> Json:
    """Only public rows and capture receipts enter the prediction process."""
    refs: list[Json] = []
    plan: Json = dict(
        failures=[],
        references=refs,
        public=[],
        heads=[],
        head_refs=[],
        labels_opened=0,
        original_model_capture_refs=[],
    )
    try:
        for name in methods.prior.INPUTS:
            path = root / name
            require(path, "resource_exists", True, path.is_file())
            refs.append(reference(path))
        for name in ("python", "pytest", "coverage", "ruff", "mypy"):
            path = ROOT / ".venv/bin" / name
            require(path, "resource_exists", True, path.is_file())
        require(Path(sys.executable), "python_version>=3.11", True, sys.version_info >= (3, 11))
        parents = {}
        for name, digest in PINS.items():
            path = root / "results" / (name + ".json")
            metadata = (
                "flagged_adversarial",
                "verdict_class",
                "terminal_validation_sidecar_path",
            ) + {
                list(PINS)[0]: ("head_checkpoints", "interaction_fit_ready_score"),
                list(PINS)[1]: ("method_freeze", "role_manifests", "source_protocol_ready_score"),
                list(PINS)[2]: ("MODEL_SPECS", "raw_response_shards", "rows"),
            }[name]
            value = bind(path, raw, refs, digest, fields=metadata)
            require(path, "flagged_adversarial", False, value["flagged_adversarial"])
            require(path, "eligible_verdict", True, value["verdict_class"] in {"null", "positive"})
            publication = bind(Path(value["terminal_validation_sidecar_path"]), raw, refs)
            pub = publication.get("publication", publication)
            side_path = Path(pub["sidecar_path"])
            bind(side_path, raw, refs)
            report = read_bound_sidecar(path, side_path)
            require(side_path, "terminal_primary_sha256", digest, pub["primary_sha256"])
            require(side_path, "primary_path", str(path.absolute()), report["primary_path"])
            require(side_path, "report.passed", True, report["report"]["passed"])
            parents[name] = value
        fit, protocol, captures = (parents[n] for n in PINS)
        require(
            root / "results" / (list(PINS)[-1] + ".json"),
            "public_capture_has_outcomes",
            False,
            any(set(r) & {"y", "eligible_y", "human_label"} for r in captures["rows"]),
        )
        require(
            root / "results", "interaction_fit_ready_score", 1, fit["interaction_fit_ready_score"]
        )
        require(
            root / "results",
            "source_protocol_ready_score",
            1,
            protocol["source_protocol_ready_score"],
        )
        require(
            root / "results",
            "sealed_source_methods",
            methods.methods()["source"],
            protocol["method_freeze"]["methods"]["source"],
        )
        role_ref = protocol["role_manifests"]["evaluation"]
        originals = bind(Path(role_ref["path"]), raw, refs, role_ref["sha256"])["rows"]
        require(
            Path(role_ref["path"]),
            "original_slots",
            list(range(96)),
            [r["slot"] for r in originals],
        )
        public = bind(root / PUBLIC, raw, refs, "sha256:" + Path(PUBLIC).stem)["rows"]
        lookup = {r["family_id"]: r for r in public}
        for ref in fit["head_checkpoints"]:
            path = Path(ref["path"]) if not absent_head else raw / "absent_head.json"
            progress("numerical_head_load_before", len(plan["heads"]), 5 - len(plan["heads"]))
            plan["heads"].append(bind(path, raw, refs, ref["sha256"]))
            plan["head_refs"].append(refs[-1])
            progress("numerical_head_load_after", len(plan["heads"]), 5 - len(plan["heads"]))
        require(root / "results", "head_roster", ARMS, [h["arm"] for h in plan["heads"]])
        capture_map = {
            r["family_id"]: ref
            for r, ref in zip(captures["rows"], captures["raw_response_shards"], strict=True)
        }
        for original in originals:
            p = dict(lookup[original["family_id"]])
            if future_label:
                p["y"] = 1
            require(
                root / PUBLIC,
                "future_label_fields",
                [],
                sorted(set(p) & {"y", "eligible_y", "label", "labels", "decision"}),
            )
            for key in ("source_bytes", "answer_bytes", "source_cluster_id"):
                require(root / PUBLIC, key, original[key], p[key])
            source = features.normalized(bytes.fromhex(p["source_bytes"]))
            require(root / PUBLIC, "source_identity", p["source_cluster_id"], source)
            fresh = features.extract(
                {k: p[k] for k in ("source_bytes", "answer_bytes", "family_id")}
            )
            require(root / PUBLIC, "feature_reconstruction", p["features"], fresh["values"])
            ref = capture_map[p["family_id"]]
            capture = bind(Path(ref["path"]), raw, refs, ref["sha256"])
            require(
                Path(ref["path"]),
                "capture_identity",
                p["capture_identity"],
                capture["capture_identity"],
            )
            require(Path(ref["path"]), "family_id", p["family_id"], capture["family_id"])
            scalar = capture["parsed"].get("probability")
            if scalar is not None:
                generated = json.loads(capture["raw_response"]["choices"][0]["message"]["content"])
                require(
                    Path(ref["path"]),
                    "original_generated_scalar",
                    scalar,
                    generated["unsupported_probability"],
                )
            require(Path(ref["path"]), "original_Qwen_scalar", p["q"], scalar)
            plan["public"].append(
                dict(
                    p,
                    source=source,
                    unit=f"evaluation/{original['slot']}",
                    slot=original["slot"],
                    capture_ref=refs[-1],
                )
            )
        progress("source_reconstruction_after", len(plan["public"]), 0)
        labels = root / LABELS
        require(labels, "resource_exists", True, labels.is_file())
        require(labels, "sha256", "sha256:" + labels.stem, sha256_file(labels))
        plan["label_reference"] = reference(labels)
        plan["original_model_capture_refs"] = [
            dict(
                reference(root / "results" / (list(PINS)[-1] + ".json")),
                scope="historical cached Qwen generation; zero current pretrained calls",
                MODEL_SPECS=captures["MODEL_SPECS"],
            )
        ]
    except InputBlock as error:
        plan["failures"].append(error.check)
    return plan


def permute(public: list[Json]) -> tuple[Array, list[str]]:
    """A cyclic source-hash permutation changes features while retaining scalar and identity."""
    groups = sorted(
        {r["source"] for r in public}, key=lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    donors = dict(zip(groups, groups[1:] + groups[:1], strict=True))
    values = {r["source"]: r["features"] for r in public}
    return (
        np.asarray([[r["q"], *values[donors[r["source"]]]] for r in public], dtype=float),
        [donors[r["source"]] for r in public],
    )


def predictions(plan: Json) -> list[Json]:
    """Seal every diagnostic too, so evaluator labels cannot select interventions."""
    public = plan["public"]
    usable = [r for r in public if r["public_eligible"] and r["q"] is not None]
    x = np.asarray([[r["q"], *r["features"]] for r in usable], dtype=float)
    shifted, donors = permute(usable)
    variants = [(h["arm"], h, x, {}) for h in plan["heads"]]
    head = plan["heads"][-1]
    variants.append(
        (
            "feature_permutation",
            head,
            shifted,
            dict(zip([r["unit"] for r in usable], donors, strict=True)),
        )
    )
    for name, indexes in [
        ("zero_interactions", [-3, -2, -1]),
        *[(f"zero_interaction_{i}", [i - 3]) for i in range(3)],
    ]:
        ablated = deepcopy(head)
        for index in indexes:
            ablated["parameters"][index] = 0.0
        variants.append((name, ablated, x, {}))
    rows = []
    for arm, h, vector, donor in variants:
        ps = dict(zip([r["unit"] for r in usable], predict(h, vector).tolist(), strict=True))
        for r in public:
            p = ps.get(r["unit"])
            rows.append(
                dict(
                    unit=r["unit"],
                    source=r["source"],
                    family_id=r["family_id"],
                    arm=arm,
                    seed=SEED,
                    condition="evaluation" if arm in ARMS else "diagnostic",
                    probability=p,
                    decision=action(p),
                    status="completed" if p is not None else "excluded",
                    exclusion_reason=None
                    if p is not None
                    else r["exclusion_reason"] or "missing_scalar",
                    donor_source=donor.get(r["unit"]),
                )
            )
    return rows


def payload(plan: Json) -> Json:
    """The seal binds input, head and prediction bytes, excluding mutable timing."""
    return {k: plan[k] for k in ("public", "heads", "head_refs", "prediction_rows", "references")}


def seal(root: Path, raw: Path, **options: bool) -> Json:
    """Exit the public process with durable predictions before targets are decoded."""
    progress("public_preconditions_before")
    plan = public_inputs(root, raw, **options)
    progress("public_preconditions_after", len(plan["public"]), 0)
    plan["prediction_rows"] = [] if plan["failures"] else predictions(plan)
    plan["prediction_seal"] = dict(
        payload_hash=canonical_hash(payload(plan)),
        sealed_monotonic_ns=time.monotonic_ns(),
        labels_opened=0,
    )
    atomic_json(raw / "predictions.json", plan)
    progress("public_prediction_seal_written", len(plan["prediction_rows"]), 0)
    return plan


def reduce(predicted: list[Json], targets: list[Json]) -> Json:
    """Pair original source groups, keeping repeated arms out of the independent count."""
    labels = {r["family_id"]: r for r in targets}
    if (
        len(labels) != len(targets)
        or set(labels) != {r["family_id"] for r in predicted}
        or any(
            r["y"] is not None and (type(r["y"]) is not int or r["y"] not in (0, 1))
            for r in targets
        )
    ):
        raise ValueError("target_contract")
    rows = []
    for r in predicted:
        target, p = labels[r["family_id"]], r["probability"]
        y = target["y"]
        known = y is not None and p is not None
        cost = (
            (
                0.5
                if r["decision"] == "escalate"
                else float(5 * y)
                if r["decision"] == "accept"
                else float(1 - y)
            )
            if known
            else None
        )
        rows.append(
            dict(
                r,
                y=y,
                actual_cost=cost,
                brier=(p - y) ** 2 if known else None,
                metric="typed_cost",
                numerator=cost if known else 0,
                denominator=int(known),
                status="completed" if known else "excluded",
                exclusion_reason=None
                if known
                else target["exclusion_reason"] or r["exclusion_reason"] or "unknown_target",
            )
        )
    primary = [r for r in rows if r["arm"] in ARMS]
    groups = sorted({r["source"] for r in primary if r["denominator"]})
    summaries = []
    for source in groups:
        for arm in ARMS:
            selected = [
                r for r in primary if r["source"] == source and r["arm"] == arm and r["denominator"]
            ]
            summaries.append(
                dict(
                    source=source,
                    arm=arm,
                    numerator=sum(r["actual_cost"] for r in selected),
                    denominator=len(selected),
                    cost=float(np.mean([r["actual_cost"] for r in selected])),
                    brier=float(np.mean([r["brier"] for r in selected])),
                    labels=sorted({r["y"] for r in selected}),
                    decisions=[r["decision"] for r in selected],
                )
            )
    lookup = {(r["arm"], r["family_id"]): r for r in primary}
    paired = []
    for source in groups:
        interactions = [
            r
            for r in primary
            if r["source"] == source and r["arm"] == "interaction" and r["denominator"]
        ]
        baseline = [lookup[("additive", r["family_id"])] for r in interactions]
        gains = [
            b["actual_cost"] - r["actual_cost"] for b, r in zip(baseline, interactions, strict=True)
        ]
        paired.append(
            dict(
                source=source,
                y=interactions[0]["y"],
                gain=float(np.mean(gains)),
                brier_increase=float(
                    np.mean(
                        [
                            r["brier"] - b["brier"]
                            for b, r in zip(baseline, interactions, strict=True)
                        ]
                    )
                ),
                beneficial_changed=any(
                    g > 0 and b["decision"] != r["decision"]
                    for g, b, r in zip(gains, baseline, interactions, strict=True)
                ),
                added_false_accept=any(
                    r["y"] == 1 and r["decision"] == "accept" and b["decision"] != "accept"
                    for b, r in zip(baseline, interactions, strict=True)
                ),
                denominator=len(interactions),
            )
        )
    gains = np.asarray([r["gain"] for r in paired], dtype=float)
    draw_means: list[float] = []
    rng = np.random.default_rng(SEED)
    progress("bootstrap_benchmark_before", 0, CONFIG["draws"])
    if len(gains):
        for start in range(0, CONFIG["draws"], 1000):
            draw_means.extend(
                gains[rng.integers(0, len(gains), (1000, len(gains)))].mean(axis=1).tolist()
            )
            progress("bootstrap", len(draw_means), CONFIG["draws"] - len(draw_means))
    progress("bootstrap_benchmark_after", len(draw_means), CONFIG["draws"] - len(draw_means))
    gain = float(gains.mean()) if len(gains) else None
    classes = {str(y): len({r["source"] for r in paired if r["y"] == y}) for y in (0, 1)}
    support = len(paired) >= CONFIG["groups"] and min(classes.values()) >= CONFIG["per_class"]
    beneficial = sum(r["beneficial_changed"] for r in paired)
    false_accepts = [r["source"] for r in paired if r["added_false_accept"]]
    brier = float(np.mean([r["brier_increase"] for r in paired])) if paired else None
    safety = bool(
        paired
        and not false_accepts
        and brier <= CONFIG["brier_increase"]
        and beneficial >= CONFIG["beneficial"]
    )
    centered = (
        np.asarray(draw_means) - gain + CONFIG["margin"] if gain is not None else np.array([])
    )
    exceedances = int(np.sum(centered >= gain)) if gain is not None else 0
    raw_p = (1 + exceedances) / (1 + len(draw_means)) if draw_means else None
    h1 = dict(
        id="H1_source_interactions",
        observed_gain=gain,
        margin=CONFIG["margin"],
        raw_p_value=raw_p,
        family_p_value=raw_p if support and safety else 1,
        support_passed=support,
        safety_passed=safety,
        complete_source_groups=len(paired),
        class_counts=classes,
        beneficial_changed_sources=beneficial,
        added_false_accept_sources=false_accepts,
        brier_increase=brier,
        completed_draws=len(draw_means),
        censored_draws=CONFIG["draws"] - len(draw_means),
        margin_test_exceedances=exceedances,
        descriptive_interval=np.quantile(draw_means, [0.025, 0.975]).tolist()
        if draw_means
        else None,
        capstone_raw_p_value=raw_p,
        local_holm_family=dict(H1=raw_p if support and safety else 1, H2=1),
        benefit_passed=bool(support and safety and gain >= CONFIG["margin"] and raw_p <= 0.025),
    )
    diagnostics = [
        dict(
            r,
            intervention_cost_change=r["actual_cost"]
            - lookup[("interaction", r["family_id"])]["actual_cost"]
            if r["denominator"]
            else None,
            decision_changed=r["decision"] != lookup[("interaction", r["family_id"])]["decision"],
            retrained=False,
            primary_hypothesis=False,
        )
        for r in rows
        if r["arm"] not in ARMS
    ]
    counts = dict(
        intended_count=480,
        eligible_count=sum(r["denominator"] for r in primary),
        independent_count=len(paired),
        completed_count=sum(r["denominator"] for r in primary),
        excluded_count=sum(not r["denominator"] for r in primary),
        censored_count=480 - len(primary),
        failed_count=0,
    )
    return dict(
        rows=primary,
        H1=h1,
        paired_source_rows=paired,
        bootstrap_draws=draw_means,
        per_source_cost_brier_rows=summaries,
        source_intervention_rows=diagnostics,
        descriptive_arm_metrics=[
            dict(
                arm=arm,
                source_groups=len(groups),
                typed_cost=float(np.mean([r["cost"] for r in summaries if r["arm"] == arm]))
                if groups
                else None,
                brier=float(np.mean([r["brier"] for r in summaries if r["arm"] == arm]))
                if groups
                else None,
                primary_contrast=arm in {"additive", "interaction"},
            )
            for arm in ARMS
        ],
        **counts,
    )


def evaluate(raw: Path) -> Json:
    """Reconstruct before opening complete human annotations in this evaluator child."""
    plan = json.loads((raw / "predictions.json").read_text())
    if plan["failures"]:
        return dict(failures=plan["failures"])
    if canonical_hash(payload(plan)) != plan["prediction_seal"]["payload_hash"]:
        raise ValueError("prediction_seal")
    for ref in plan["references"]:
        checked(ref)
    public_ref = next(
        ref for ref in plan["references"] if ref["sha256"] == "sha256:" + Path(PUBLIC).stem
    )
    original_public = {
        r["family_id"]: r for r in json.loads(checked(public_ref).read_text())["rows"]
    }
    rebuilt = deepcopy(plan)
    rebuilt["heads"] = [json.loads(checked(ref).read_text()) for ref in plan["head_refs"]]
    for row in rebuilt["public"]:
        original = original_public[row["family_id"]]
        if any(row[k] != v for k, v in original.items()) or row["source"] != features.normalized(
            bytes.fromhex(row["source_bytes"])
        ):
            raise ValueError("independent_prediction")
        fresh = features.extract({k: row[k] for k in ("source_bytes", "answer_bytes", "family_id")})
        capture = json.loads(checked(row["capture_ref"]).read_text())
        if fresh["values"] != row["features"] or capture["parsed"].get("probability") != row["q"]:
            raise ValueError("independent_prediction")
    independent = predictions(rebuilt)
    if independent != plan["prediction_rows"] or rebuilt["heads"] != plan["heads"]:
        raise ValueError("independent_prediction")
    x = np.asarray(
        [
            [r["q"], *r["features"]]
            for r in plan["public"]
            if r["public_eligible"] and r["q"] is not None
        ]
    )
    error = max(
        float(np.max(np.abs(predict(h, x) - predict(h, x, logistic=True))))
        for h in rebuilt["heads"]
    )
    if error >= 1e-10:
        raise ValueError("equivalent_logistic_parity")
    opened = time.monotonic_ns()
    refs: list[Json] = []
    label_ref = plan["label_reference"]
    labels = bind(Path(label_ref["path"]), raw, refs, label_ref["sha256"])
    require(Path(label_ref["path"]), "access_policy", "evaluator_only", labels["access_policy"])
    targets = complete_targets(plan["public"], labels)
    result = reduce(independent, targets)
    result.update(
        independent_prediction_rows=independent,
        targets=targets,
        equivalent_logistic_parity=dict(maximum_error=error, passed=True, independent_method=False),
        label_access_receipt=dict(
            prediction_file=reference(raw / "predictions.json"),
            opened_monotonic_ns=opened,
            opened_after_seal=opened > plan["prediction_seal"]["sealed_monotonic_ns"],
            opened_roles=["stream_manifest"],
            selected_role="evaluation96",
            decoded_manifest_rows=len(labels["rows"]),
            selected_rows=len(plan["public"]),
            label_reference=refs[0],
            evaluator_pid=os.getpid(),
        ),
        failures=[],
    )
    return result


def complete_targets(public: list[Json], labels: Json) -> list[Json]:
    """Unsupported spans define y only when the complete response has verified custody."""
    lookup = {r["family_id"]: r for r in labels["rows"]}
    targets = []
    for row in public:
        target = lookup[row["family_id"]]
        answer = bytes.fromhex(row["answer_bytes"])
        digest = "sha256:" + hashlib.sha256(answer).hexdigest()
        spans = [a for a in labels["annotation_rows"] if a["family_id"] == row["family_id"]]
        complete = (
            target["custody_passed"] is True
            and target["completely_annotated"] is True
            and target["quality"] == "good"
            and target["annotation_count"] == len(spans)
            and target["response_sha256"] == digest
            and target["response_id"] == row["response_id"]
        )
        for span in spans:
            start, end = span["start_byte"], span["end_byte"]
            complete = complete and (
                span["response_sha256"] == digest
                and span["text_equal"] is True
                and span["response_id"] == target["response_id"]
                and 0 <= start < end <= len(answer)
                and answer[start:end].decode() == span["text"]
            )
        y = int(bool(spans)) if complete and target["y"] is not None else None
        require(Path(LABELS), "complete_response_target", y, target["eligible_y"])
        if y is not None:
            require(Path(LABELS), "unsupported_orientation", y, target["y"])
        targets.append(
            dict(
                family_id=row["family_id"],
                y=y,
                complete_annotation=complete,
                exclusion_reason=None if y is not None else "incomplete_or_unknown_target",
            )
        )
    return targets


def build(work: Json, raw: Path, receipts: list[Json], coverage: Json, fixture: bool) -> Json:
    """Current owned checks qualify an audit; scientific losses remain complete nulls."""
    plan, measured = work["plan"], work["measurement"]
    required = [r for r in receipts if r.get("classification", "required") == "required"]
    checks = (
        not fixture
        and [r["name"] for r in required] == work["validation_manifest"]
        and all(r["passed"] for r in required)
        and all(
            p in coverage
            and coverage[p]["summary"]["num_statements"] > 0
            and coverage[p]["summary"]["missing_lines"] == 0
            for p in OWNED
        )
    )
    failures = deepcopy(plan["failures"] + measured.get("failures", []))
    h1 = measured.get("H1", {})
    if h1 and not h1["support_passed"]:
        failures.append(
            dict(
                check="evaluation_support",
                upstream=TASK,
                path=str(raw / "evaluation.json"),
                hash=sha256_file(raw / "evaluation.json"),
                field="H1.complete_source_groups_and_classes",
                op=">=",
                expected=dict(groups=72, per_class=8),
                observed=dict(groups=h1["complete_source_groups"], classes=h1["class_counts"]),
            )
        )
    kind = (
        "blocked"
        if failures
        else "null"
        if fixture
        else "disqualified"
        if not checks
        else "positive"
        if h1.get("benefit_passed")
        else "null"
    )
    ready = int(checks and not failures and bool(h1))
    if kind == "disqualified":
        failures.append(
            dict(
                check="owned_validation",
                upstream=TASK,
                path=str(raw / "validation.json"),
                hash=sha256_file(raw / "validation.json"),
                field="required_checks_passed",
                op="==",
                expected=True,
                observed=False,
            )
        )
    v: Json = dict(
        experiment_id=8074,
        task_id=TASK,
        milestone="2026.10.699",
        run_date="20261003",
        schema="carnot.v699.interaction_decision_audit.v1",
        verdict_class=kind,
        honest_verdict="complete_blocked_" + Path(failures[0]["path"]).stem
        if kind == "blocked"
        else "complete_" + kind + "_interaction_decision_audit",
        verifier_is_oracle=fixture,
        flagged_adversarial=False,
        required_checks_passed=checks,
        decision_audit_ready_score=ready,
        source_interaction_benefit_score=int(ready and h1.get("benefit_passed", False)),
        generalized_learning_benefit_score=0,
        claim_scope="Historically exposed cached development source-feature dependence and typed decisions. No live Qwen, generic hallucination, source-text removal or independent generalization result.",
        rows=measured.get("rows", []),
        independent_prediction_rows=measured.get("independent_prediction_rows", []),
        prediction_seal=plan["prediction_seal"],
        label_access_receipt=measured.get("label_access_receipt", {}),
        H1=h1,
        paired_source_rows=measured.get("paired_source_rows", []),
        per_source_cost_brier_rows=measured.get("per_source_cost_brier_rows", []),
        descriptive_arm_metrics=measured.get("descriptive_arm_metrics", []),
        source_intervention_rows=measured.get("source_intervention_rows", []),
        equivalent_logistic_parity=measured.get("equivalent_logistic_parity", {}),
        intended_count=480,
        eligible_count=measured.get("eligible_count", 0),
        independent_count=measured.get("independent_count", 0),
        completed_count=measured.get("completed_count", 0),
        censored_count=measured.get("censored_count", 480),
        excluded_count=measured.get("excluded_count", 0),
        failed_count=0,
        sample_size_budget=dict(
            source_groups_planned=96,
            primary_arms=5,
            diagnostic_arms=5,
            count_unit="primary source/arm rows; independent_count counts source groups once",
            support=dict(groups=72, per_class=8),
            independent_environments=0,
        ),
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if h1
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        substrate_declaration=dict(
            reduction="verifier_ensemble_against_cached_candidates"
            if h1
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
        ),
        trained_head_specs=[
            dict(
                arm=h["arm"],
                parameter_count=len(h["parameters"]),
                current_training_steps=0,
                checkpoint=ref,
                scope="sealed imported small numerical head",
            )
            for h, ref in zip(plan["heads"], plan["head_refs"], strict=True)
        ],
        oracle_distinct_scope="No exact oracle enters natural evidence; private tests grant no science credit.",
        original_model_capture_refs=plan["original_model_capture_refs"],
        conditional_development_intervals=dict(
            H1=h1.get("descriptive_interval"), scope="conditional exposed development uncertainty"
        ),
        random_seed=SEED,
        source_artifact_hashes=plan["references"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        duration_s=work["duration_s"],
        validation_receipts=receipts,
        validation_command_manifest=work["commands"],
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
        preserved_v695_failed_validation=work["historical_failures"],
        methodology_note="Independent source bytes, original generated scalar receipts and sealed checkpoint coefficients. Costs5/1/.5/0; source-group bootstrap10000; fixed interventions without retraining. Exact logistic parity prohibits energy-unique superiority.",
    )
    v["raw_shard_hashes"] = [
        reference(raw / p)
        for p in ("predictions.json", "evaluation.json", "work.json", "configuration.json")
    ]
    v["reproducibility_checksum"] = canonical_hash(
        dict(
            config=CONFIG,
            sources=v["source_artifact_hashes"],
            code=v["code_config_hashes"],
            raw=v["raw_shard_hashes"],
        )
    )
    v["field_principles"] = {
        k: "Bind "
        + k
        + " to primitive current evidence; prevent cached development results from receiving independent scientific credit."
        for k in v
    }
    v["field_principles"].update(
        H1="Raw nonzero-margin p reaches capstone; unsupported or safety-failing H1 receives family p=1; diagnostics never replace H1/H2.",
        decision_audit_ready_score="A current validated supported audit is usable even when benefit is null.",
        source_interaction_benefit_score="Benefit requires support, safety, useful changed decisions and conservative two-hypothesis significance.",
        preserved_v695_failed_validation="Clean current validation cannot erase older failed validation bytes.",
        source_intervention_rows="Fixed-model coefficient/feature diagnostics do not remove source text from cached Qwen or establish retrained controls.",
        independent_count="Arms, coefficients, repeats and resamples cannot create independent source groups.",
    )
    return v


def manifest(private: Path) -> list[Json]:
    """Reuse qualified checks while limiting coverage to this audit's new statements."""
    specs = methods.manifest(private)
    replacements = {methods.MODULE: MODULE, methods.CLI: CLI, methods.TEST: TEST}
    for spec in specs:
        spec["argv"] = [
            replacements.get(a, a)
            .replace(str(ROOT / methods.MODULE), str(ROOT / MODULE))
            .replace(str(ROOT / methods.CLI), str(ROOT / CLI))
            for a in spec["argv"]
        ]
        if spec["name"] == "repository_full_suite":
            spec.update(argv=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"], deadline_s=180)
        if spec["name"] == "focused_unit_and_cli":
            spec["deadline_s"] = 360
    (private / "coverage.ini").write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    specs.insert(
        6,
        dict(
            name="E2E-019",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "--basetemp=" + str(private / "e2e019"),
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
            ],
            deadline_s=120,
            expected_exit=0,
            classification="required",
        ),
    )
    return specs


def run_phase(spec: Json, private: Path, raw: Path) -> Json:
    """Bracket bounded children and retain their actual exits and sealed log hashes."""
    progress("subprocess_before_" + spec["name"], 0, 1)
    receipt = run_check(ROOT, spec, private, raw / "validation_logs")
    progress("subprocess_after_" + spec["name"], 1, 0)
    return receipt


def replay(path: Path) -> bool:
    """Cold reconstruction rejects altered predictions, costs, validation or code bytes."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            checked(ref)
        if any(
            sha256_file(ROOT / p) != digest for p, digest in value["code_config_hashes"].items()
        ):
            return False
        for receipt in value["validation_receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        work = json.loads((raw / "work.json").read_text())
        if json.loads((raw / "configuration.json").read_text())["config"] != CONFIG:
            return False
        if work["measurement"].get("H1"):
            reduced = evaluate(raw)
            for field in reduced:
                if field != "label_access_receipt" and reduced[field] != work["measurement"][field]:
                    return False
        validation = json.loads((raw / "validation.json").read_text())
        return (
            build(work, raw, validation["receipts"], validation["coverage"], validation["fixture"])
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal_manifest(path: Path) -> list[Json]:
    """Freeze the final readers before measurement, using the same argv at publication."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    return [dict(name=n, argv=a, deadline_s=60, expected_exit=0) for n, a in commands]


def terminal(path: Path) -> Json:
    """Independent cold replay and unchanged artifact auditors check candidate bytes."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    with tempfile.TemporaryDirectory(prefix="carnot-8074-terminal-") as temp:
        receipts = [run_phase(spec, Path(temp), raw) for spec in terminal_manifest(path)]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Publish only after separate prediction and evaluator children exit and checks finish."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("start_no_pretrained_model_loads_or_generation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--predict-child", type=Path)
    parser.add_argument("--evaluate-child", type=Path)
    parser.add_argument("--absent-head", action="store_true")
    parser.add_argument("--future-label", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            return 0 if replay(args.cold_replay) else 1
        if args.predict_child:
            seal(
                args.root,
                args.predict_child,
                absent_head=args.absent_head,
                future_label=args.future_label,
            )
            return 0
        if args.evaluate_child:
            try:
                measured = evaluate(args.evaluate_child)
            except InputBlock as error:
                measured = dict(failures=[error.check])
            atomic_json(args.evaluate_child / "evaluation.json", measured)
            progress("evaluator_normal_exit", measured.get("completed_count", 0), 0)
            return 0
        output = (args.fixture_output or args.output).absolute()
        raw = output.parent / "raw" / output.stem
        if (raw / "work.json").exists():
            raise ValueError("immutable_invocation_exists")
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8074-") as temp:
            private = Path(temp)
            specs = manifest(private)
            py = str(ROOT / ".venv/bin/python")
            children = [
                dict(
                    name=n,
                    argv=[py, "-u", str(ROOT / CLI), "--root", str(args.root), flag, str(raw)]
                    + (["--absent-head"] if args.absent_head else [])
                    + (["--future-label"] if args.future_label else []),
                    deadline_s=120,
                    expected_exit=0,
                    classification="required",
                )
                for n, flag in [
                    ("prediction_child_normal_exit", "--predict-child"),
                    ("evaluator_child_normal_exit", "--evaluate-child"),
                ]
            ]
            config = os.environ.get("CARNOT_8074_COVERAGE_CONFIG")
            if config:
                for child in children:
                    child["argv"] = [
                        py,
                        "-m",
                        "coverage",
                        "run",
                        "--rcfile=" + config,
                        "--data-file=" + str(Path(config).parent / ".coverage"),
                        *child["argv"][2:],
                    ]
            codes = {
                p: sha256_file(ROOT / p)
                for p in [
                    *OWNED,
                    TEST,
                    methods.MODULE,
                    "python/carnot/verify/evidence_features_7980.py",
                    "python/carnot/reporting/primary_publication.py",
                    "python/carnot/reporting/current_work_receipt.py",
                    "python/carnot/reporting/v686_contract_validation.py",
                ]
            }
            atomic_json(
                raw / "configuration.json",
                dict(
                    config=CONFIG,
                    commands=children + specs,
                    terminal_commands=terminal_manifest(raw / "terminal_candidate.json"),
                    code_config_hashes=codes,
                    diagnostic_amendment="V699 task fixes source-hash cyclic permutation and joint zero-three ablation; individual sealed ablations also retained",
                ),
            )
            frozen = time.monotonic()
            progress("methods_and_validation_commands_frozen")
            receipts = [run_phase(c, private, raw) for c in children]
            plan = json.loads((raw / "predictions.json").read_text())
            measured = (
                json.loads((raw / "evaluation.json").read_text()) if receipts[-1]["passed"] else {}
            )
            measured_at = time.monotonic()
            os.environ["CARNOT_8074_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            for spec in [] if args.fixture_output else specs:
                receipts.append(run_phase(spec, private, raw))
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            history = [
                reference(p)
                for p in (
                    ROOT / "results/raw/experiment_8021_v695_typed_decision_test/validation_logs"
                ).rglob("*.log")
            ]
            history += [
                reference(p) for p in (raw / "preserved_failed_attempts").rglob("*") if p.is_file()
            ]
            ended = time.monotonic()
            work = dict(
                plan=plan,
                measurement=measured,
                commands=children + specs,
                code_config_hashes=codes,
                validation_manifest=[
                    s["name"] for s in children + specs if s["classification"] == "required"
                ],
                historical_failures=history,
                duration_s=ended - began,
                phase_spans=[
                    dict(phase="freeze", duration_s=frozen - began),
                    dict(phase="predict_and_evaluate", duration_s=measured_at - frozen),
                    dict(phase="validation", duration_s=ended - measured_at),
                ],
            )
            atomic_json(raw / "work.json", work)
            atomic_json(
                raw / "validation.json",
                dict(receipts=receipts, coverage=coverage, fixture=bool(args.fixture_output)),
            )
            value = build(work, raw, receipts, coverage, bool(args.fixture_output))
            atomic_json(
                raw / "primitive_rows.json",
                dict(rows=value["rows"], diagnostics=value["source_intervention_rows"]),
            )
            progress("publication_before", value["completed_count"], 1)
            publication = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, measurement_exit_receipts=receipts[:2]),
            )
            progress("complete", value["completed_count"], 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("rejected_" + str(error))
        return 1
