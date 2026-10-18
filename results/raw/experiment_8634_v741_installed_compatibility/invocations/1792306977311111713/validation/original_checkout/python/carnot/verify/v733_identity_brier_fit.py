"""REQ-REPORT-8528: authenticate frozen question roles before fitting small Brier heads.

The shipped measurement and publication contracts retain their resource checks.
Current fitting never opens the evaluator target shard or changes a historical head.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from carnot.reporting import v733_contract_methods as methods
from carnot.reporting import v733_source_consumer_qualification as custody
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v733_source_diagnostics import diagnostic_rows
from carnot.verify import v726_criterion_energy_fit as old
from carnot.verify import v730_source_control_fit as inherited
from carnot.verify import v733_identity_brier as n

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8528_v733_identity_brier_fit", "exp8528-identity-brier-fit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v733_identity_brier_8528.py"
OWNED = [f"python/carnot/verify/v733_identity_brier{s}.py" for s in ["", "_fit"]] + [
    "python/carnot/reporting/v733_identity_brier_runner.py",
    CLI,
]
UPSTREAM = f"results/{custody.NAME}.json"
UPSTREAM_PIN = "sha256:81e5281bc9b115eb32e8327d0c7f3a92ff638903bbc2c6991d70f09d81c2204c"
MANIFEST_REF = dict(
    path=str(
        ROOT
        / "results/raw/experiment_8527_v733_source_consumer_qualification/invocations/1792027088475173897/comparator_manifest.json"
    ),
    sha256="sha256:547cd8b2eea2060455764c44ac075a8459eb20c6efde6cc5be0ef70846ba536e",
)
ARMS, SEEDS = custody.COMPARATORS, old.SEEDS
CONFIG = methods.protocol()["identity_brier"]
MODEL_SPECS: list[Json] = []
read, reference, checksum = methods.read, methods.reference, methods.checksum
_BUILD, _REPLAY = old.build, old.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Visible phase counts let supervisors distinguish CPU work from a stall."""
    print(f"[exp8528] phase={phase} completed={completed} pending={pending}", flush=True)


def operands(root: Path) -> tuple[Json, Json]:
    """A pinned qualified producer binds original roles, scaler, donors and fit-only labels."""
    upstream = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(root / UPSTREAM, Path(terminal["publication"]["sidecar_path"]))
    consumer = reader_receipt(custody.TASK, root / "results", field="comparator_inputs_ready_score")
    if not (
        upstream["required_checks_passed"]
        and not upstream["flagged_adversarial"]
        and upstream["comparator_inputs_ready_score"] == 1
        and bound["report"]["passed"]
        and consumer["passed"]
    ):
        raise ValueError("comparator_custody_unqualified")
    manifest = read(upstream["comparator_manifest_reference"])
    if manifest["identity_brier"] != CONFIG or manifest["operators"]["knots"] != old.sparse.KNOTS:
        raise ValueError("identity_registration_drift")
    refs = [manifest["basis_reference"], *manifest["original_authorities"]["V726"]]
    for ref in refs:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("original_method_authority")
    wrapper = read(manifest["original_trainer_reference"])
    view = read(wrapper["reference"])
    panel = read(manifest["original_feature_reference"])
    legacy_ref = next(
        r
        for r in upstream["source_artifact_hashes"]
        if "experiment_8487" in r["path"] and r["path"].endswith("frozen_heads.json")
    )
    legacy = read(legacy_ref)
    if legacy["scaler"] != manifest["operators"]["scaler"]:
        raise ValueError("original_scaler")
    return dict(
        upstream,
        trainer_view=wrapper["reference"],
        stratum_roster=[],
        additional_refs=[
            upstream["comparator_manifest_reference"],
            manifest["original_trainer_reference"],
            manifest["original_feature_reference"],
            legacy_ref,
            *refs,
        ],
    ), dict(view, manifest=manifest, panel=panel, legacy=legacy)


def train(view: Json, strata: list[Json]) -> Json:
    """Only three residual arms train; seed slots retain deterministic duplicate accounting."""
    if any(r["role"] not in {"fit", "tune"} for r in view["rows"]):
        raise ValueError("test_label_boundary")
    heads = dict(
        raw_p2=deepcopy(view["legacy"]["heads"]["raw_p2"]),
        legacy_full_source=deepcopy(view["legacy"]["heads"]["full_spline9"]),
    )
    scaler, checks = view["legacy"]["scaler"], {}
    for arm in n.TRAINABLE:
        source = "source_permuted_spline9" if arm == n.TRAINABLE[-1] else "full_spline9"
        vectors = {r["answer_id"]: r["x"] for r in view["panel"]["rows"] if r["arm"] == source}
        groups = [
            [dict(r, x=vectors[r["unit_id"]]) for r in view["rows"] if r["role"] == role]
            for role in ("fit", "tune")
        ]
        heads[arm] = []
        for seed in SEEDS:
            head = n.fit(arm, seed, scaler, *groups)
            check = n.gradient_checks(head)
            checks[f"{arm}:{seed}"] = check
            if not check["passed"]:
                raise ValueError("brier_gradient_check")
            heads[arm].append(head)
    hashes = {
        arm: [canonical_hash({k: v for k, v in h.items() if k != "seed"}) for h in hs]
        for arm, hs in heads.items()
    }
    return dict(
        heads=heads,
        scaler=scaler,
        optimizer_steps=3000,
        test_label_reads=0,
        parameter_counts=dict(zip(ARMS, [0, 109, 13, 109, 109], strict=True)),
        alpha_choices={a: {str(h["seed"]): h.get("alpha") for h in hs} for a, hs in heads.items()},
        training_prefix_rows={
            a: {str(h["seed"]): h["prefixes"] for h in heads[a]} for a in n.TRAINABLE
        },
        duplicate_seed_heads={
            a: dict(
                unique_heads=len(set(hs)),
                duplicate_slots=len(hs) - len(set(hs)),
                independent_replications=1,
            )
            for a, hs in hashes.items()
        },
        head_hashes=hashes,
        fit_tune_losses={
            a: [
                dict(
                    seed=h["seed"],
                    initial=h["initial_loss"],
                    final=h["final_loss"],
                    tune_alpha_losses=h["tune_alpha_losses"],
                )
                for h in heads[a]
            ]
            for a in n.TRAINABLE
        },
        fit_questions=150,
        tune_questions=50,
        gradient_checks=dict(passed=True, rows=checks),
    )


def predictions(result: Json, panel: Json) -> list[Json]:
    """Prediction receives public vectors only after coefficients and alphas are sealed."""
    rows = []
    for arm in ARMS:
        source = "source_permuted_spline9" if arm == n.TRAINABLE[-1] else "full_spline9"
        units = [r for r in panel["rows"] if r["role"] == "test" and r["arm"] == source]
        x = np.asarray([r["x"] for r in units], dtype=float)
        p: n.Array
        for head in result["heads"][arm]:
            if arm in n.TRAINABLE:
                p = n.mix(
                    x[:, 0],
                    expit(
                        n.offset(x)
                        + n.matrix(arm, x, head["scaler"]) @ np.asarray(head["parameters"])
                    ),
                    head["alpha"],
                )
            else:
                p = old.predict(head, x)
            for r, probability in zip(units, p, strict=True):
                rows.append(
                    dict(
                        unit_id=f"{r['answer_id']}:{arm}:{head['seed']}",
                        answer_id=r["answer_id"],
                        question_hash=r["question_hash"],
                        family=r["question_hash"],
                        arm=arm,
                        seed=head["seed"],
                        status="completed",
                        censored=False,
                        excluded=False,
                        absolute_metric=float(probability),
                        raw_numerator=float(probability),
                        raw_denominator=1,
                        missing_reason=None,
                        energy=n.energy(float(probability)),
                        **inherited.decision(float(probability)),
                    )
                )
        progress("predictions_" + arm, len(rows), 7500 - len(rows))
    return rows


@contextmanager
def binding() -> Iterator[None]:
    """Scoped adapters preserve existing resource gates and historical producer behavior."""
    proxy = SimpleNamespace(**dict(vars(methods), legacy=old.methods.legacy))
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        UPSTREAM=UPSTREAM,
        UPSTREAM_PIN=UPSTREAM_PIN,
        ARMS=ARMS,
        CONFIG=CONFIG,
        methods=proxy,
        operands=operands,
        train=train,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path) -> Json:
    """Retain real resource receipts and seal both heads and predictions without evaluator reads."""
    fitting_spans = []

    def timed_train(view: Json, strata: list[Json]) -> Json:
        """Keep actual fitting cost outside deterministic head state so replay stays exact."""
        start = time.monotonic_ns()
        try:
            return train(view, strata)
        finally:
            fitting_spans.append(
                dict(name="numeric_fit", duration_s=(time.monotonic_ns() - start) / 1e9)
            )

    with binding(), patch.object(old, "train", timed_train):
        work = dict(old.measure(root, raw))
    work["fitting_spans"] = fitting_spans
    work["phase_spans"].extend(fitting_spans)
    work["preconditions"].update(measurement_cap_s=3600, optional_cuda_and_boards_gate_cpu=False)
    for failure in work["failures"]:
        if (
            failure.get("field", failure.get("artifact_field"))
            == "feature_panel_ready_score_and_sealed_fit_tune"
        ):
            failure.update(
                check="qualified_comparator_custody",
                upstream=custody.TASK,
                upstream_id=custody.TASK,
                field="comparator_inputs_ready_score_and_original_seals",
                artifact_field="comparator_inputs_ready_score_and_original_seals",
                sha256=UPSTREAM_PIN,
            )
    work["decisions"], work["seal_clock"] = [], dict(heads_sealed_ns=time.monotonic_ns())
    if work["result"]:
        upstream, view = operands(root)
        work["source_refs"] += upstream["additional_refs"]
        work["decisions"] = predictions(work["result"], view["panel"])
    atomic_json(raw / "predictions.json", dict(rows=work["decisions"], test_label_reads=0))
    work["seal_clock"].update(predictions_sealed_ns=time.monotonic_ns(), evaluator_opened=False)
    work["primitive_files"].append(reference(raw / "predictions.json"))
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["failures"] = diagnostic_rows(work["failures"])
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness depends on complete qualified predictions, including valid identity fallbacks."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    result = work["result"] or {}
    ready = bool(value["head_ready_score"] and len(work["decisions"]) == 7500)
    verdict = (
        "disqualified" if not value["required_checks_passed"] else "null" if ready else "blocked"
    )
    rows = work["decisions"] or [
        dict(
            unit_id=f"{r['answer_id']}:{r['arm']}:{r['seed']}",
            answer_id=r["answer_id"],
            question_hash=r["question_hash"],
            family=r["question_hash"],
            arm=r["arm"],
            seed=r["seed"],
            status="censored",
            censored=True,
            excluded=False,
            absolute_metric=None,
            raw_numerator=0,
            raw_denominator=1,
            missing_reason="qualified_comparator_input_unavailable_or_owned_fit_failed",
        )
        for r in read(MANIFEST_REF)["slots"]
        if r["role"] == "test"
    ]
    value.update(
        experiment_id=8528,
        task_id=TASK,
        milestone="2026.10.733",
        run_date="20261015",
        honest_verdict=f"complete_{verdict}_v733_identity_brier_fit",
        verdict_class=verdict,
        identity_heads_ready_score=int(ready),
        decision_audit_ready_score=int(ready),
        head_ready_score=int(ready),
        rows=rows,
        intended_count=7500,
        completed_count=len(work["decisions"]),
        censored_count=7500 - len(work["decisions"]),
        independent_count=100 if ready else 0,
        gate_check_summary=diagnostic_rows(value["gate_check_summary"]),
        frozen_heads_reference=reference(raw / "frozen_heads.json"),
        predictions_reference=work["primitive_files"][-1],
        seal_clock=work["seal_clock"],
        sample_size_budget=dict(
            intended_test_questions=100,
            intended_test_answers=300,
            intended_arm_seed_answer_slots=7500,
            fit_questions=150,
            tune_questions=50,
            seeds_are_duplicates=True,
            siblings_are_not_independent=True,
        ),
        optimization_accounting=dict(
            measured_fit_duration_s=sum(r["duration_s"] for r in work["fitting_spans"]),
            full_batch_updates=3000 if result else 0,
            coefficients_per_update=[13, 109, 109],
            coefficient_update_operations=231000 if result else 0,
            frozen_control_updates=0,
            current_llm_calls=0,
        ),
        methodology_note="Fixed-budget Brier residual fitting on exposed fit roles; tune-only identity-admissible mixtures; frozen test predictions without evaluator reads. Prefixes diagnose optimization, not sample efficiency or generalization.",
    )
    for key in ["alpha_choices", "training_prefix_rows", "parameter_counts"]:
        value[key] = result.get(key, {})
    value["field_principles"].update(
        {
            k: "Frozen roles and original producer hashes bind computation; readiness is distinct from benefit."
            for k in value
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute primitives from pinned upstream bytes so local rehashing cannot hide drift."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if work["result"]:
            _, view = operands(Path(work["upstream_reference"]["path"]).parents[1])
            if predictions(work["result"], view["panel"]) != work["decisions"]:
                return False
        if read(value["predictions_reference"])["rows"] != work["decisions"]:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    with binding(), patch.object(old, "build", build):
        return bool(_REPLAY(path))
