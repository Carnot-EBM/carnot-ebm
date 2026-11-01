"""REQ-VERIFY-8431: train small numeric heads on exposed historical judgments.

The shipped cubic fitter supplies the mathematics. Independent upstream seals
and fit-only scaling keep a deterministic retrospective fit separate from test
labels and from any current language model inference or factuality claim.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
from scipy.special import expit

from carnot.reporting import v726_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.human_label_custody_8389 import gate
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8431_v726_criterion_energy_fit", "exp8431-criterion-energy-fit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v726_criterion_energy_fit_8431.py"
OWNED = [
    "python/carnot/verify/v726_criterion_energy_fit.py",
    "python/carnot/reporting/v726_criterion_energy_runner.py",
    CLI,
]
UPSTREAM = "results/experiment_8430_v726_criterion_feature_panel.json"
UPSTREAM_PIN = "sha256:c5783590d946a9500c335a7d39e0fb154dd5943c0efc2c1355dde885c3712989"
ARMS = ("raw_p2", "scalar_calibrated", "logistic9", "spline9", "source_permuted_spline9")
SEEDS = sparse.SEEDS
MODEL_SPECS: list[Json] = []
reference, read, checksum = methods.reference, methods.read, methods.checksum
CONFIG = dict(
    sparse.CONFIG,
    arms=list(ARMS),
    scalar_parameters=2,
    permutation_seeds=dict(fit=725, tune=7252, test=7251),
    temperature_ties=[1.0, 0.5, 2.0],
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Visible completed counts let supervisors distinguish computation from a stall."""
    print(f"[exp8431] phase={phase} completed={completed} pending={pending}", flush=True)


def operands(root: Path) -> tuple[Json, Json]:
    """Only the independently sealed fit/tune view is opened; evaluator bytes stay shut."""
    upstream = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    if upstream["feature_panel_ready_score"] != 1 or not upstream["required_checks_passed"]:
        raise ValueError("feature_panel_not_ready")
    view = read(upstream["trainer_view"])
    return dict(upstream), dict(view)


def choose_temperature(losses: Json) -> float:
    """A declared tie order prevents silent preference for the first grid value."""
    return min((1.0, 0.5, 2.0), key=lambda t: losses[str(t)])


def donor_mappings(strata: list[Json]) -> Json:
    """Move whole question bundles within datasets, matching each generator position."""
    groups: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in strata:
        key = row["role"], row["dataset"]
        groups[key].setdefault(row["question_hash"], {})[row["generator"]] = row["unit_id"]
    mappings: Json = {role: [] for role in CONFIG["permutation_seeds"]}
    for role, seed in CONFIG["permutation_seeds"].items():
        rng = np.random.default_rng(seed)
        for (group_role, dataset), questions in sorted(groups.items()):
            if role != group_role:
                continue
            recipients = sorted(questions)
            donors = rng.permutation(recipients).tolist()
            for recipient, donor in zip(recipients, donors, strict=True):
                if set(questions[recipient]) != set(questions[donor]):
                    raise ValueError("generator_slots")
                for generator, uid in sorted(questions[recipient].items()):
                    mappings[role].append(
                        dict(
                            recipient_question=recipient,
                            donor_question=donor,
                            recipient_unit_id=uid,
                            donor_unit_id=questions[donor][generator],
                            dataset=dataset,
                            generator=generator,
                        )
                    )
    return mappings


def permute_rows(rows: list[Json], mapping: list[Json]) -> list[Json]:
    """Only eight public source numbers change; q, targets and source IDs stay fixed."""
    original = {row["unit_id"]: row for row in rows}
    donors = {row["recipient_unit_id"]: row["donor_unit_id"] for row in mapping}
    result = deepcopy(rows)
    for row in result:
        if row["unit_id"] in donors:
            donor = original[donors[row["unit_id"]]]
            row["x"] = [row["x"][0], *donor["x"][1:]]
    return result


def predict(head: Json, x: sparse.Array) -> sparse.Array:
    """A scalar head receives q alone even when its caller holds all nine inputs."""
    if head["arm"] == "raw_p2":
        return np.asarray(x[:, 0], dtype=float)
    if head["arm"] == "platt":
        return np.asarray(scalar.predict(head, x[:, 0]), dtype=float)
    return np.asarray(sparse.predict(head, x), dtype=float)


def scalar_fit(
    seed: int, x: sparse.Array, y: sparse.Array, tx: sparse.Array, ty: sparse.Array
) -> Json:
    """Two logistic coefficients use the same zero start, loss penalty and batch budget."""
    theta: sparse.Array = np.zeros(2)
    head = dict(
        arm="platt",
        seed=seed,
        parameters=theta.tolist(),
        temperature=1.0,
        parameter_count=2,
        optimizer_steps=200,
    )
    initial = float(np.log(2))
    for step in range(200):
        z, jac = scalar.logits_jacobian("platt", theta, x[:, 0])
        theta -= 0.01 * (jac.T @ (expit(z) - y) / len(x) + 0.002 * theta)
        if step % 50 == 0:
            progress(f"scalar_seed{seed}_steps{step + 1}/200")
    head["parameters"] = theta.tolist()
    z, _ = scalar.logits_jacobian("platt", theta, x[:, 0])
    head.update(
        initial_loss=initial,
        final_loss=float(np.mean(np.logaddexp(0, z) - y * z) + 0.001 * np.sum(theta**2)),
    )
    losses = {
        str(t): scalar.bce(
            np.asarray(expit(scalar.logits_jacobian("platt", theta, tx[:, 0])[0] / t)), ty
        )
        for t in (0.5, 1.0, 2.0)
    }
    head.update(temperature=choose_temperature(losses), tune_temperature_losses=losses)
    return head


def numeric_checks(head: Json, probe: sparse.Array) -> Json:
    """Independent finite differences and binary normalization check frozen arithmetic."""
    if head["arm"] == "platt":
        theta: sparse.Array = np.asarray(head["parameters"], dtype=float)
        z, jac = scalar.logits_jacobian("platt", theta, probe[:, 0])
        errors = []
        for i in range(2):
            delta = np.zeros(2)
            delta[i] = 1e-6
            up = scalar.logits_jacobian("platt", theta + delta, probe[:, 0])[0]
            down = scalar.logits_jacobian("platt", theta - delta, probe[:, 0])[0]
            errors.append(float(np.max(np.abs((up - down) / 2e-6 - jac[:, i]))))
        fd = max(errors)
        z = z / head["temperature"]
        locality: Json = dict(passed=True, scope="two_parameter_scalar")
    else:
        fd = sparse.finite_difference(head, probe[0])
        z = sparse.logits_jacobian(head, probe)[0] / head["temperature"]
        locality = (
            sparse.audit(head, probe[0])
            if head["arm"] == "spline"
            else dict(passed=True, scope="dense_logistic")
        )
        for row in locality.get("rows", []):
            row.pop("duration_s", None)
    weights = np.exp(np.column_stack((np.zeros(len(z)), z)) - np.maximum(0, z)[:, None])
    normalized = weights[:, 1] / weights.sum(axis=1)
    parity = float(np.max(np.abs(normalized - expit(z))))
    return dict(
        finite_difference_error=fd,
        normalization_error=parity,
        sparse_locality=locality,
        passed=bool(fd < 1e-7 and parity < 1e-12 and locality["passed"]),
    )


def train(view: Json, strata: list[Json]) -> Json:
    """Fit all declared slots without ever taking test targets or scoring test inputs."""
    rows = view["rows"]
    if any(row["role"] not in {"fit", "tune"} for row in rows):
        raise ValueError("test_label_boundary")
    data = {
        role: [
            dict(
                family_id=r["unit_id"],
                source_cluster_id=r["question_hash"],
                q=r["x"][0] if r["x"] else None,
                features=r["x"][1:] if r["x"] else None,
                y=r["y"],
                status="completed",
            )
            for r in rows
            if r["role"] == role
        ]
        for role in ("fit", "tune")
    }
    sparse.validate_data(data)
    groups: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        groups[row["question_hash"]].append(row)
    usable = [
        r
        for r in rows
        if len(groups[r["question_hash"]]) == 3
        and all(
            peer["x"] is not None and peer["y"] is not None for peer in groups[r["question_hash"]]
        )
    ]
    mappings = donor_mappings(strata)
    permuted = permute_rows(usable, mappings["fit"] + mappings["tune"])
    f, t = ([r for r in usable if r["role"] == role] for role in ("fit", "tune"))
    x, tx = (np.asarray([r["x"] for r in part], dtype=float) for part in (f, t))
    y, ty = (np.asarray([r["y"] for r in part], dtype=float) for part in (f, t))
    scaler = sparse.fit_scaler(x)
    zero = dict(
        arm="spline", parameters=[0.0] * 109, decay_scale=1.0, scaler=scaler, temperature=1.0
    )
    if not sparse.audit(zero, x[0])["passed"]:
        raise ValueError("numeric_check_before_fit")
    heads: Json = {arm: [] for arm in ARMS}
    checks, losses, hashes, choices, changes = {}, {}, {}, {}, {}
    completed = 0
    for arm in ARMS:
        progress("fit_begin_" + arm, completed, 25 - completed)
        px, ptx = (
            (
                np.asarray([r["x"] for r in permuted if r["role"] == role], dtype=float)
                for role in ("fit", "tune")
            )
            if arm == "source_permuted_spline9"
            else (x, tx)
        )
        for seed in SEEDS:
            if arm == "raw_p2":
                head = dict(
                    arm="raw_p2",
                    seed=seed,
                    parameters=[],
                    parameter_count=0,
                    optimizer_steps=0,
                    temperature=1.0,
                    initial_loss=scalar.bce(px[:, 0], y),
                    final_loss=scalar.bce(px[:, 0], y),
                    tune_temperature_losses={"1.0": scalar.bce(ptx[:, 0], ty)},
                )
            elif arm == "scalar_calibrated":
                head = scalar_fit(seed, px, y, ptx, ty)
            else:
                head = sparse.fit_one(
                    "logistic" if arm == "logistic9" else "spline", seed, scaler, px, y, ptx, ty
                )
                head.pop("duration_s")
                head["temperature"] = choose_temperature(head["tune_temperature_losses"])
            head["research_arm"] = arm
            if arm != "raw_p2":
                check = numeric_checks(head, px[:2])
                if not check["passed"]:
                    raise ValueError("numeric_check_after_fit")
                checks[f"{arm}:{seed}"] = check
            heads[arm].append(head)
            completed += 1
            progress(f"fit_end_{arm}_seed{seed}", completed, 25 - completed)
        hashes[arm] = [
            canonical_hash({k: v for k, v in h.items() if k != "seed"}) for h in heads[arm]
        ]
        choices[arm] = {str(h["seed"]): h["temperature"] for h in heads[arm]}
        losses[arm] = [
            dict(
                seed=h["seed"],
                initial_fit_objective=h["initial_loss"],
                final_fit_objective=h["final_loss"],
                fit_log_loss=scalar.bce(predict(h, px), y),
                tune_temperature_losses=h["tune_temperature_losses"],
                chosen_tune_log_loss=h["tune_temperature_losses"][str(h["temperature"])],
            )
            for h in heads[arm]
        ]
        theta = np.asarray(heads[arm][0]["parameters"])
        changes[arm] = dict(
            changed=bool(np.any(theta != 0)),
            changed_count=int(np.sum(theta != 0)),
            maximum_absolute_change=float(np.max(np.abs(theta))) if len(theta) else 0.0,
            initial_parameters=[0.0] * len(theta),
            fitted_parameters=theta.tolist(),
        )
    return dict(
        heads=heads,
        scaler=scaler,
        clip_counts=dict(fit=sparse.scale(x, scaler)[1], tune=sparse.scale(tx, scaler)[1]),
        head_hashes=hashes,
        duplicate_seed_heads={
            arm: dict(
                unique_heads=len(set(v)),
                duplicate_slots=len(v) - len(set(v)),
                independent_replications=1,
            )
            for arm, v in hashes.items()
        },
        fit_tune_losses=losses,
        temperature_choices=choices,
        coefficient_changes=changes,
        gradient_checks=dict(passed=True, rows=checks),
        normalization_parity=dict(
            passed=True,
            maximum_error=max(c["normalization_error"] for c in checks.values()),
            exact_energy_definition="E0=0,E1=-eta/T; sigmoid identity control",
        ),
        donor_mappings=mappings,
        test_label_reads=0,
        fit_rows=len(f),
        tune_rows=len(t),
        fit_questions=len({r["question_hash"] for r in f}),
        tune_questions=len({r["question_hash"] for r in t}),
        optimizer_steps=4000,
        config=CONFIG,
    )


def measure(root: Path, raw: Path) -> Json:
    """Authenticate only real CPU dependencies before fitting and retain missing operands."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 25)
    resources = dict(
        methods.legacy.resources(raw),
        free_disk_bytes=shutil.disk_usage(raw).free,
        tool_paths={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        task_cap_s=4800,
        measurement_cap_s=3900,
        validation_reserve_s=900,
        model_loads=0,
        generations=0,
    )
    authority = methods.authority(root, raw / "authority")
    failures = list(authority["gate_check_summary"])
    for failure in failures:
        failure["upstream_id"] = TASK
    if (
        not resources["private_disk_backed_scratch"]
        or resources["available_memory_bytes"] < resources["minimum_memory_bytes"]
        or resources["free_disk_bytes"] < 104857600
        or not all(resources["tool_paths"].values())
    ):
        failures.append(
            gate(str(raw), "private_disk_memory_tools_deadline", True, resources, upstream=TASK)
        )
    sources: list[Json] = []
    upstream: Json = {}
    view: Json = dict(rows=[])
    try:
        upstream, view = operands(root)
        sources = [
            reference(root / UPSTREAM),
            upstream["trainer_view"],
            dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN),
        ]
        methods.protocol()
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                str(root / UPSTREAM),
                "feature_panel_ready_score_and_sealed_fit_tune",
                1,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream="exp8430-criterion-feature-panel",
            )
        )
    result, owned_failure = None, False
    if not failures:
        progress("numeric_learning_begin", 0, 25)
        try:
            result = train(view, upstream["stratum_roster"])
        except (ValueError, KeyError, FloatingPointError) as error:
            owned_failure = True
            failures.append(
                gate(str(raw), "required_numeric_checks", True, str(error), upstream=TASK)
            )
        progress("numeric_learning_end", 25 if result else 0, 0 if result else 25)
    files = dict(
        trainer_operands=dict(view=view, strata=upstream.get("stratum_roster", [])),
        frozen_heads=result
        or dict(heads={}, unavailable_reason="required_inputs_or_numeric_checks_failed"),
    )
    for name, value in files.items():
        atomic_json(raw / (name + ".json"), value)
        (raw / (name + ".json")).chmod(0o600)
    work = dict(
        authority=authority,
        failures=failures,
        owned_failure=owned_failure,
        result=result,
        preconditions=resources,
        source_refs=sources,
        upstream_reference=dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN),
        code_refs=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                "python/carnot/verify/sparse_energy_7996.py",
                "python/carnot/verify/multivariate_energy_7982.py",
                "python/carnot/verify/qwen_energy_calibration_7972.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
        primitive_files=[reference(raw / (n + ".json")) for n in files],
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[
            dict(name="authenticate_and_fit", duration_s=(time.monotonic_ns() - start) / 1e9)
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness qualifies fitted computations; exposed-label benefit remains unmeasured."""
    qualified = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and not work.get("adversarial_findings")
        and not work["owned_failure"]
    )
    result = work["result"] or {}
    ready = qualified and not work["failures"] and bool(result) and work["authority"]["activated"]
    verdict = "disqualified" if not qualified else "circular_positive" if ready else "blocked"
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            head = next(
                (h for h in result.get("heads", {}).get(arm, []) if h["seed"] == seed), None
            )
            rows.append(
                dict(
                    unit_id=f"{arm}:{seed}",
                    family=arm,
                    arm=arm,
                    seed=seed,
                    status="completed" if head else "censored",
                    censored=head is None,
                    excluded=False,
                    absolute_metric=int(head is not None),
                    raw_numerator=int(head is not None),
                    raw_denominator=1,
                    missing_reason=None
                    if head
                    else "required_input_unavailable_or_numeric_check_failed",
                    metric=None
                    if head is None
                    else dict(
                        final_fit_objective=head["final_loss"], temperature=head["temperature"]
                    ),
                )
            )
    frozen = reference(raw / "frozen_heads.json")
    value = dict(
        experiment_id=8431,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261011",
        honest_verdict=f"complete_{verdict}_v726_criterion_energy_fit",
        verdict_class=verdict,
        gate_check_summary=[
            *work["failures"],
            *[
                gate(
                    str(r.get("stdout_path", raw)),
                    "required_owned_check",
                    True,
                    False,
                    upstream=TASK,
                )
                for r in receipts
                if not r["passed"]
            ],
        ],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(methods.legacy.ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[
            dict(
                producer="exp8430-criterion-feature-panel",
                model="GPT-5-mini",
                scope="released cached p2 judgments; generator names are reporting provenance only",
            )
        ],
        rows=rows,
        intended_count=25,
        completed_count=sum(not r["censored"] for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=result.get("fit_questions", 0) + result.get("tune_questions", 0),
        sample_size_budget=dict(
            intended_arm_seed_slots=25,
            intended_fit_answers=450,
            intended_tune_answers=150,
            intended_fit_questions=150,
            intended_tune_questions=50,
            independent_generalization_questions=0,
            seeds_are_duplicates=True,
            sibling_answers_are_not_independent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="retrospective public human labels; old overlap unknown; no current Qwen or hidden-game inference",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            owned_validation=qualified,
            frozen_fitted_computations=bool(ready),
            scientific_benefit_measured=False,
            independent_generalization=False,
        ),
        validation_receipts=receipts,
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=17,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=[*work["code_refs"], dict(config=CONFIG, sha256=canonical_hash(CONFIG))],
        raw_shard_hashes=[
            reference(raw / "measurement.json"),
            *work["primitive_files"],
            *work.get("coverage_shards", []),
        ],
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields=[
                    "feature_panel_ready_score",
                    "trainer_view",
                    "stratum_roster",
                    "feature_schema",
                ],
            )
            for ref in work["source_refs"]
        ],
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        head_ready_score=int(ready),
        frozen_heads_path=frozen["path"],
        frozen_heads_sha256=frozen["sha256"],
        arm_roster=list(ARMS),
        feature_schema=dict(
            names=[
                "p2_hallucination_probability",
                "overlap_mean",
                "overlap_maximum",
                "negation_mismatch_mean",
                "negation_mismatch_maximum",
                "numeric_mismatch_mean",
                "numeric_mismatch_maximum",
                "uncovered_fraction_mean",
                "uncovered_fraction_maximum",
            ],
            scalar_receives_q_only=True,
            source_ids_are_predictors=False,
            scaling="fit_only_shared",
        ),
        test_label_reads=0,
        methodology_note="Twenty deterministic full-batch numeric fits and five raw identity slots; fit-only scaling; tune log loss temperatures; no test scoring or independent factuality claim.",
    )
    for key in [
        "head_hashes",
        "duplicate_seed_heads",
        "fit_tune_losses",
        "temperature_choices",
        "coefficient_changes",
        "gradient_checks",
        "normalization_parity",
        "donor_mappings",
        "clip_counts",
    ]:
        value[key] = result.get(key, {})
    value["field_principles"] = {
        key: "Bound to authenticated fit/tune operands and actual invocation; retrospective computation qualification only."
        for key in value
    }
    value["field_principles"].update(
        head_ready_score="Valid frozen fitted computations, never improved decision quality.",
        independent_count="Fit/tune whole questions only; repeated seeds and siblings never multiply support.",
        donor_mappings="Whole-question within-dataset source bundles; q, labels, generator position and original IDs stay fixed.",
        acceptance_gates="Owned checks and fitting readiness are separate from scientific benefit.",
        test_label_reads="Zero evaluator-target opens; test donor metadata carries no labels.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold refitting uses independent upstream seals, so repaired local hashes cannot help."""
    from tempfile import TemporaryDirectory

    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in [*work["source_refs"], *work["code_refs"], *work["primitive_files"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("operand_drift")
        with TemporaryDirectory(prefix="exp8431-cold-", dir="/var/tmp") as directory:
            root = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                snap = work["authority"]["authority_snapshots"][role]
                if "snapshot_path" in snap:
                    source = Path(snap["snapshot_path"])
                    if sha256_file(source) != snap["sha256"]:
                        raise ValueError("authority_snapshot_drift")
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            if (
                methods.authority(root, root / "cold")["activated"]
                != work["authority"]["activated"]
            ):
                raise ValueError("authority_drift")
        raw = Path(value["primitive_reference"]["path"]).parent
        frozen = read(reference(raw / "frozen_heads.json"))
        if work["result"]:
            ref = work["upstream_reference"]
            if ref["sha256"] != UPSTREAM_PIN:
                raise ValueError("independent_upstream_pin")
            upstream, view = operands(Path(ref["path"]).parents[1])
            expected = train(view, upstream["stratum_roster"])
            if expected != work["result"] or expected != frozen:
                raise ValueError("frozen_fit_drift")
            if read(reference(raw / "trainer_operands.json")) != dict(
                view=view, strata=upstream["stratum_roster"]
            ):
                raise ValueError("trainer_operand_drift")
        elif frozen.get("heads"):
            raise ValueError("missing_operand_fitted_head")
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    raise ValueError("receipt_drift")
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
