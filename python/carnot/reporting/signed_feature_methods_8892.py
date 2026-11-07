"""REQ-REPORT-8892: freeze signed features using development evidence only.

The two pilots check execution and fitting. The successor's24 paired streams
remain sealed; a good development fit cannot establish future benefit.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from carnot.memory import signed_memory_8892 as signed
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.memory import causal_memory_8802 as primitive
from carnot.memory import categorical_memory_8830 as categorical
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import learning_signal_diagnosis_8815 as learning
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8892_v761_signed_feature_methods", "exp8892-signed-feature-methods"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/memory/signed_memory_8892.py",
    "python/carnot/reporting/signed_feature_methods_8892.py",
    "python/carnot/reporting/signed_feature_runner_8892.py",
    CLI,
]
TESTS = ["tests/python/test_signed_feature_methods_8892.py"]
SEED = 7618892


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose completed work without changing the finite wall-clock budget."""
    print(f"[exp8892] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Change world seeds, retain topology families, and freeze multiplicity before fitting."""
    runtime = signed.engine()
    runtime.SEED = SEED
    p = runtime.protocol()
    for stream in p["cohort_manifest"] + p["pilot_manifest"]:
        stream["stream_id"] = stream["stream_id"].replace("v760", "v761")
        for split in ("online", "retention"):
            for row in stream[split]:
                row["unit_id"] = row["unit_id"].replace("v760", "v761")
    p.update(
        feature_schema=dict(
            name=signed.SCHEMA,
            legacy_coordinates=list(primitive.COUNTS),
            extra_coordinates=signed.EXTRA,
            empty_pair_overlap=1,
            missing_candidates_overlap=0,
            overlap="Jaccard on shared-variable tuples",
            refutation="any matching explicit negative record, including conflicts",
            scaling="development maxima only; zero maxima replaced by one",
            atom_order="ascending positive count, negative count, variable-role tuple",
        ),
        feature_arms=["old", "signed"],
        multiplicity=dict(
            family_intervals=27,
            family_error=0.01,
            per_tail=0.01 / 54,
            allocation="nine original gates per arm plus nine paired signed-minus-old contrasts",
        ),
        evaluation_opened=False,
        pilot_only=True,
    )
    p["configuration"]["features"] = list(primitive.COUNTS) + signed.EXTRA + ["bias"]
    p["arm_definitions"] = {
        feature + ":" + control: dict(
            definition,
            feature_arm=feature,
            control=control,
            mask_extra_coordinates=feature == "old",
        )
        for feature in ("old", "signed")
        for control, definition in p["arm_definitions"].items()
    }
    p["finite_roster"].update(arms=8, fit_heads=10)
    p["expected_row_counts"] = dict(online=9216, retention=27648, total=36864)
    estimands = [
        dict(
            split="online",
            checkpoint=None,
            endpoint="brier",
            other=control,
            upper=0 if control == "matched_permuted" else -0.01,
            strict=control == "matched_permuted",
            diagnostic=control == "frozen_query_only",
        )
        for control in ("frozen_verified", "matched_permuted", "frozen_query_only")
    ]
    estimands += [
        dict(
            split="retention",
            checkpoint=checkpoint,
            endpoint=endpoint,
            other="frozen_verified",
            upper=0.01,
            strict=False,
            diagnostic=False,
        )
        for checkpoint in (0, 24, 48)
        for endpoint in ("brier", "error")
    ]
    p["simultaneous_estimands"] = [
        dict(e, feature_arm=feature) for feature in ("old", "signed") for e in estimands
    ]
    p["simultaneous_estimands"] += [
        dict(
            e,
            feature_arm="signed_minus_old",
            definition="difference of matched_true-minus-other contrasts",
            upper=0,
            diagnostic=True,
        )
        for e in estimands
    ]
    return dict(p)


def development(p: Json) -> Json:
    """Tabulate collisions before fitting; labels remain separate from the observation payload."""
    rows = []
    for j, query in enumerate(p["topology_manifest"]["development"]):
        for error in old.ERRORS:
            value = old.world(query, f"v761-fit/{j}/{error}/", error, SEED + j)
            for admitted in (False, True):
                observed = signed.observe(value, [old.topology(query)] if admitted else [])
                rows.append(
                    dict(
                        unit_id=f"development/{j}/{error}/{admitted}",
                        input=value,
                        admitted=admitted,
                        structure=old.topology(query),
                        truth=str(symbolic.decide(value)["decision"]),
                        **observed,
                    )
                )
    tables: list[Json] = []
    for arm, width in (("old", 6), ("signed", 8)):
        for admitted in (False, True):
            selected = [
                dict(r, counts=r["counts"][:width]) for r in rows if r["admitted"] == admitted
            ]
            table, floor = learning.collisions(selected, str(admitted))
            tables.extend(dict(r, arm=arm, empirical_brier_floor=floor) for r in table)
    scale = [max((r["counts"][i] for r in rows), default=0) or 1.0 for i in (6, 7)]
    return dict(
        rows=rows,
        scale=scale,
        feature_collision_rows=tables,
        manifest=dict(
            scope="development only",
            worlds=48,
            observations=96,
            data_hash=canonical_hash(rows),
            scaling_data_hash=canonical_hash(rows),
        ),
    )


def fit(dev: Json) -> Json:
    """Keep all five initial seeds and240 updates; no pilot outcome selects parameters."""
    rows = [r for r in dev["rows"] if r["admitted"]]
    heads: Json = {}
    fits: list[Json] = []
    for arm in ("old", "signed"):
        heads[arm] = {}
        for seed in old.SEEDS:
            progress("fit_before_" + arm, len(fits), 10 - len(fits))
            params = signed.head(seed, arm, dev["scale"])
            order = np.random.default_rng(seed).permutation(len(rows))
            for step in range(240):
                row = rows[int(order[step % len(order)])]
                params, _ = signed.update(params, row["counts"], row["truth"])
            brier = float(
                np.mean(
                    [
                        sum(
                            (v - float(i == primitive.LABELS.index(r["truth"]))) ** 2
                            for i, v in enumerate(signed.probabilities(params, r["counts"]))
                        )
                        for r in rows
                    ]
                )
            )
            heads[arm][str(seed)] = params
            fits.append(dict(arm=arm, seed=seed, brier=brier, optimizer_steps=240))
            progress("fit_after_" + arm, len(fits), 10 - len(fits))
    return dict(heads=heads, fit_rows=fits, numerical_training_calls=2400)


def pilot(raw: Path, stream: Json, fitted: Json, p: Json) -> Json:
    """Reuse complete delayed streams and restart checks with isolated signed head functions."""
    rows: list[Json] = []
    controls: list[Json] = []
    refs: list[Json] = []
    for feature in ("old", "signed"):
        runtime = signed.engine()
        runtime.SEED, runtime.progress = SEED, progress
        observations = []

        def observed(value: Json, templates: list[str]) -> Json:
            observation = signed.observe(value, templates)
            observations.append(dict(evidence_hash=canonical_hash(value), **observation))
            return observation

        runtime.observe = observed
        progress("pilot_before_" + feature, 0, 48)
        result = runtime.run_stream(raw / feature, stream, fitted["heads"][feature], p)
        rows.extend(
            dict(r, arm=feature + ":" + r["arm"], feature_arm=feature) for r in result["rows"]
        )
        controls.extend(dict(r, feature_arm=feature) for r in result["controls"])
        refs.append(dict(feature_arm=feature, observations=observations, **result))
        progress("pilot_after_" + feature, 48, 0)
    for i in range(48):
        a, b = [
            next(
                r
                for r in rows
                if r["arm"] == f + ":matched_true"
                and r["split"] == "online"
                and r["ordinal"] == i + 1
            )
            for f in ("old", "signed")
        ]
        require(a["sealed"]["counts"] == b["sealed"]["counts"], "feature_work_mismatch")
    return dict(rows=rows, controls=controls, arms=refs, numerical_training_calls=192)


def controls() -> list[Json]:
    """Deliberate label leaks must be rejected even when they would make fitting easy."""
    value = old.world(old.graphs()["development"][0], "leak/", "support", SEED)
    rows = []
    for field in ("truth", "label", "source_hidden_label", "oracle_decision"):
        try:
            signed.observe(dict(value, **{field: "support"}), [])
            rejected = False
        except ValueError:
            rejected = True
        rows.append(dict(check="reject_leaked_" + field, passed=rejected))
    try:
        signed.vector(categorical.head(11, "linear"), [0.0] * 8)
        rejected = False
    except ValueError:
        rejected = True
    rows.append(dict(check="reject_old_checkpoint", passed=rejected))
    return rows
