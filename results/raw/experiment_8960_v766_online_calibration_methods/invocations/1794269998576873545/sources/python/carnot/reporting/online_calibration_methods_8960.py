"""REQ-REPORT-8960: freeze causal calibration methods without a benefit claim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from carnot.memory import online_calibration_8960 as adapter
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, SEED = (
    "experiment_8960_v766_online_calibration_methods",
    "exp8960-online-calibration-methods",
    766060,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/memory/online_calibration_8960.py",
    "python/carnot/reporting/online_calibration_methods_8960.py",
    "python/carnot/reporting/online_calibration_runner_8960.py",
    CLI,
]
TESTS = ["tests/python/test_online_calibration_methods_8960.py"]
PINS = {
    "experiment_8959_v766_relational_methods": "sha256:eaa01cdf08424373c0ea12b703a09099bf489cd015593111a5142c1c2f25c966",
    "experiment_8954_v765_gradient_benefit_audit": "sha256:90a5716c9c3f640d3060c5981fbb73509b9d0929a03b4cc26fa4cb9fe2c74740",
}
CHECKS = {
    "owned_unit_child",
    "coverage_combine",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018",
    "e2e021",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts make each bounded phase visible to the supervisor."""
    print(f"[exp8960] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Absent evidence has a named path rather than an invented content digest."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def authenticate(root: Path) -> Json:
    """Pin retained evidence directly; never invoke a retired scheduler task."""
    result: Json = dict(failures=[], references=[], relational={})
    for name, pin in PINS.items():
        path = root / "results" / (name + ".json")
        ref = reference(path)
        expected: Any = pin
        field, observed = "producer_bytes", ref["sha256"]
        check = "missing_file" if observed is None else "rejected_producer"
        try:
            if observed != expected:
                raise ValueError(check)
            value = json.loads(path.read_text())
            if "8959" in name:
                for field, expected in (
                    ("relational_methods_ready_score", 1),
                    ("required_checks_passed", True),
                ):
                    observed = value.get(field)
                    check = (
                        "missing_field"
                        if field not in value
                        else "observed_zero"
                        if observed == 0
                        else "rejected_producer"
                    )
                    if type(observed) is not type(expected) or observed != expected:
                        raise ValueError(check)
                result["relational"] = value
            result["references"].append(ref)
            refs = (
                (
                    value["source_artifact_hashes"]
                    + value["code_config_hashes"]
                    + value["raw_shard_hashes"]
                )
                if "8959" in name
                else []
            )
            for i, operand in enumerate(refs):
                ref = reference(Path(operand["path"]))
                field, expected, observed = "sha256", operand["sha256"], ref["sha256"]
                check = "missing_file" if observed is None else "rejected_producer"
                if observed != expected:
                    raise ValueError(check)
                result["references"].append(ref)
                if i % 200 == 0:
                    progress("authenticate_operands", i, len(refs) - i)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["failures"].append(
                dict(
                    upstream=name,
                    path=ref["path"],
                    hash=ref["sha256"],
                    check="missing_field" if isinstance(error, KeyError) else check,
                    artifact_field=error.args[0] if isinstance(error, KeyError) else field,
                    expected="present retained input field"
                    if isinstance(error, KeyError)
                    else expected,
                    observed=None if isinstance(error, KeyError) else observed,
                )
            )
    return result


def protocol(evidence: Json) -> Json:
    """Freeze future response roles and nine bounds without opening their labels."""
    source = evidence["relational"]
    roles = source.get("role_manifest", [])
    streams = [
        dict(
            stream_id=name,
            unit_ids=[
                r["unit_id"]
                for r in sorted(
                    [r for r in roles if r["stream_id"] == name], key=lambda r: r["stream_position"]
                )
            ],
        )
        for name in sorted({r["stream_id"] for r in roles if r["role"] == "online"})
    ]
    bounds = [
        dict(
            split="online",
            metric="brier",
            ordinals=[17, 32],
            against=arm,
            upper=-0.01 if arm == "frozen" else 0,
            strict=arm != "frozen",
        )
        for arm in ("frozen", "bias_only", "permuted")
    ]
    bounds += [
        dict(
            split="retention",
            metric=metric,
            checkpoint=q,
            against="frozen",
            upper=0.01,
            strict=False,
        )
        for q in (0, 16, 32)
        for metric in ("brier", "error")
    ]
    guards = [
        r for r in source.get("development_guard_rows", []) if r["arm"] == "relational_energy"
    ]
    return dict(
        streams=streams,
        retention_ids=[r["unit_id"] for r in roles if r["role"] == "test"],
        adapter_equations=[
            "z=logit(clamp(p_base,1e-6,1-1e-6)); p_error=sigmoid(a*z+b)",
            "a0=1; b0=0; L=mean(softplus(a*z+b)-y*(a*z+b))+0.01*((a-1)^2+b^2)",
            "one block SGD: (a,b)<-(a,b)-0.05*gradient(L); a in[0.25,4]; b in[-2,2]",
        ],
        arm_table=[
            dict(
                arm=a,
                slope=a not in ("frozen", "bias_only"),
                bias=a != "frozen",
                permutation=list(adapter.PERMUTATION) if a == "permuted" else None,
            )
            for a in adapter.ARMS
        ],
        delay_protocol=dict(
            block_size=4,
            releases=[
                dict(after_query=q, labels=list(range(q - 7, q - 3))) for q in range(8, 33, 4)
            ],
            delay_queries=[4, 7],
            pending_at_stop=8,
            prediction_seal="atomic fsync journal before broker read; all arms share releases",
            label_authority="LabelBroker only; adapters receive numeric released blocks; no sidecar paths",
        ),
        guard_manifest=dict(
            rows=guards,
            count=64,
            baseline=[1, 0],
            brier_increase_max=0.005,
            error_increase_max=0,
            scope="repeated exposed historical development safeguard; no statistical guarantee",
            temperature_policy="recompute historical guard probabilities at the same fresh-calibration temperature frozen for online scoring",
        ),
        multiplicity_ledger=dict(
            bounds=bounds,
            draws=10000,
            seed=SEED,
            family_alpha=0.01,
            alpha_per_bound=0.01 / 9,
            candidate="adaptive",
            unit="whole paired stream",
            method="one-sided percentile upper; linear quantile; Bonferroni nine bounds",
            shared_retention_panel=True,
            shared_retention_count=128,
            retention_resampling="independent stream and graph-ID draws; same graph draw for all sampled streams and checkpoints",
            seed_rule="numpy default_rng(seed); each draw:24 stream indices, then128 graph indices; replacement",
            evaluation_status="frozen_only; future real-response evaluation is downstream",
            retention_checkpoint_order="read-only snapshot after that query's released-block update and guard; query0 is initial",
        ),
        frozen_score=dict(
            weights=source.get("trained_state_reference"),
            scorer="carnot.autoresearch.relational_graph_encoder.ensemble",
            arm="relational_energy",
            seeds=[71, 72, 73],
            representation_frozen=True,
            temperature="select once on128 fresh calibration graphs using existing ascending grid; freeze before online predictions",
            ensemble_logit="logit of mean seed probabilities after common temperature; never mean of logits",
        ),
        hardware_path=dict(
            current="host CPU",
            service_work=[
                "feature extraction",
                "frozen energy score",
                "prediction seal",
                "feedback release",
                "64 guard evaluations",
                "bounded two-scalar SGD",
                "durable writes",
                "restart",
            ],
            batch_map="GPU/NPU matrix work",
            scalar_map="Rust/SIMD",
            measured_removable_cost=None,
            speedup_claim=None,
        ),
        prior_synthetic_claim="V765 eleven-bound synthetic claim remains failed; exact signed-head mechanism closed",
        motivation="Continual Calibration and TOOD separate calibration from representation; their guarantees do not transfer",
    )


def development(raw: Path) -> Json:
    """Constructed probabilities exercise causal mechanics, never real Qwen benefit."""
    labels = raw / "sealed" / "labels.json"
    labels.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_json(labels, dict(labels=[0, 1, 0, 1] * 8))
    labels.chmod(0o600)
    p, guards = [0.2, 0.7, 0.3, 0.8] * 8, [(0.1, 0), (0.9, 1)] * 32
    progress("private_mechanics_before", 0, 3)
    full = adapter.trajectory(raw / "full", p, labels, guards)
    resumed = adapter.trajectory(raw / "restart", p, labels, guards, restart=True)
    atomic_json(labels, dict(labels=[0, 1, 0, 1] * 3 + [1, 0, 1, 0] * 5))
    future = adapter.trajectory(raw / "future", p, labels, guards)
    refused = False
    try:
        adapter.LabelBroker(labels).release(32, raw / "full/state.json", role="retention")
    except PermissionError:
        refused = True
    checks = dict(
        nonzero_parameter_updates=full["params"]["adaptive"] != [1, 0],
        actual_arm_disagreement=any(
            len({r["value"] for r in full["rows"] if r["ordinal"] == q}) >= 3 for q in range(9, 33)
        ),
        future_label_invariance=full["rows"][:80] == future["rows"][:80],
        retention_label_refusal=refused,
        pending_feedback=full["pending"] == list(range(25, 33)),
        exact_restart_query16=full == resumed,
    )
    progress("private_mechanics_after", 3, 0)
    return dict(
        rows=full["rows"],
        control_rows=[
            dict(control=k, passed=v, condition="private_mechanics", independent_count=0)
            for k, v in checks.items()
        ],
        state=full,
        independent_count=0,
        numerical_training_calls=sum(
            s["numerical_training_calls"] for s in (full, resumed, future)
        ),
    )


def checksum(value: Json) -> str:
    """Bind claims and randomness while leaving the digest outside its own input."""
    return cast(
        str, canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})
    )


def qualified(validation: Json) -> bool:
    """Actual SQLite statement membership is required; a percentage string is insufficient."""
    from carnot.reporting.v756_branch_qualification import verify_measured

    receipts = validation["receipts"]
    return (
        bool(receipts)
        and {r["name"] for r in receipts} == CHECKS
        and all(r["passed"] and r["exit_code"] == 0 and not r["timed_out"] for r in receipts)
        and verify_measured(validation["coverage"], OWNED)
    )


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Readiness requires validated mechanics, while scientific benefit stays unmeasured."""
    p, result, failures = protocol(work["evidence"]), work["result"], work["evidence"]["failures"]
    valid = qualified(validation)
    controls = bool(result) and all(c["passed"] for c in result["control_rows"])
    ready = int(valid and controls and not failures and not fixture)
    cls = (
        "disqualified"
        if not valid and not fixture
        else "blocked"
        if failures
        else "null"
        if controls
        else "disqualified"
    )
    rows = result.get("rows", [])
    value: Json = dict(work["template"])
    value.update(
        experiment_id=8960,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261109",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_online_calibration_methods",
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=work["current_work_receipt"]["invocation_counts"],
        model_invoked=False,
        numerical_training_calls=result.get("numerical_training_calls", 0),
        duration_s=work["current_work_receipt"]["duration_s"],
        phase_spans=work["current_work_receipt"]["phase_spans"],
        current_work_receipt=work["current_work_receipt"],
        preconditions_checked=work["evidence"],
        verifier_is_oracle=False,
        exposure_scope="private constructed probability mechanics and retained exposed guard metadata; fresh online labels unopened",
        independent_generalization_score=0,
        intended_count=128,
        completed_count=len(rows),
        failed_count=0,
        censored_count=128 - len(rows),
        excluded_count=0,
        independent_count=0,
        rows=rows,
        gate_check_summary=failures,
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"],
        acceptance_gates=dict(
            authenticated_inputs=dict(
                passed=not failures,
                principle="Custody precedes work; historical tasks are retained inputs.",
            ),
            causal_mechanics=dict(
                passed=controls,
                principle="Private causal exercise qualifies methods regardless of development gain.",
            ),
            owned_checks=dict(
                passed=valid,
                principle="Zero-exclusion unit and child coverage and every owned check must pass.",
            ),
        ),
        required_checks_passed=valid,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        online_methods_ready_score=ready,
        continuous_self_learning_task=True,
        generator_weights_frozen=True,
        protocol=p,
        protocol_hash=canonical_hash(p),
        **p,
        control_rows=result.get("control_rows", []),
        development_evidence=result,
        fixture_only=fixture,
        flagged_adversarial=False,
        scientific_benefit_evaluated=False,
        scientific_benefit_score=0,
        methodology_note="Two-scalar calibration only. Frozen graph encoder, head, generator and selected temperature. Private fixtures are not real-response evidence. Repeated guard use supplies no guarantee. No 100x claim.",
        reproduction_commands=[f"PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109"],
        upstream_compute_scope="Retained relational weights and guards own their original training; current counts contain only63 private scalar SGD steps.",
    )
    value["field_principles"] = {
        k: "Frozen method, retained bytes or actual current administrative receipt; no real-response benefit claim."
        for k in value
    }
    value["field_principles"].update(
        rows="Every private probability has typed operands and status; zero independent scientific samples.",
        online_methods_ready_score="Causal mechanics plus all owned checks; no development performance threshold.",
        numerical_training_calls="Count actual bounded scalar SGD steps separately from zero model loads and generator calls.",
        multiplicity_ledger="Nine joint bounds; shared128-graph retention panel does not multiply independent samples.",
        reproducibility_checksum="Bind scheduled identity, source bytes, configuration and seeded randomness.",
        gate_check_summary="Name upstream, exact field, expected and observed evidence for every external refusal.",
        hardware_path="Complete service work is frozen; unmeasured hardware maps confer no speedup.",
    )
    for fields, principle in [
        (
            ("experiment_id", "task_id", "milestone", "run_date", "random_seed"),
            "Bind the result to the scheduled task, retained input bytes and reproducible randomness.",
        ),
        (
            ("honest_verdict",),
            "An honest terminal disposition avoids wasted retries and preserves exact rerun retirement.",
        ),
        (
            ("verdict_class",),
            "Closed claim strength; oracle exposure or a failed acceptance gate forbids positive.",
        ),
        (
            ("inference_substrate", "inference_substrate_class", "execution_venue"),
            "Record the actual legal substrate and host venue; no_model_load never claims a new GGUF run.",
        ),
        (
            ("MODEL_SPECS", "model_specs", "model_invocation_counts"),
            "Separate private scalar training, retained model outputs and zero current generator loads or calls.",
        ),
        (
            ("duration_s", "phase_spans", "preconditions_checked"),
            "Actual clocks and authenticated prerequisites prevent fabricated execution and duration padding.",
        ),
        (
            ("verifier_is_oracle", "exposure_scope", "independent_generalization_score"),
            "A learned graph score on executable-oracle labels cannot establish the no-cheap-oracle north star.",
        ),
        (
            (
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
            ),
            "Retain every scheduled private unit; repeated controls and probabilities contribute zero independent science samples.",
        ),
        (
            ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"),
            "Exact content custody binds retained producers, executed code, configuration and raw operands.",
        ),
        (
            (
                "acceptance_gates",
                "required_checks_passed",
                "owned_statement_coverage",
                "validation_receipts",
                "global_test_debt",
            ),
            "Require every owned gate and measured statements with zero exclusions; readiness differs from benefit and global debt.",
        ),
        (
            ("continuous_self_learning_task", "generator_weights_frozen"),
            "FR-11 learning changes the small decision adapter only; mandated generator weights remain frozen.",
        ),
        (
            ("adapter_equations", "arm_table", "delay_protocol", "guard_manifest", "control_rows"),
            "Freeze a reproducible real-response calibration claim distinct from the retired synthetic signed-head replay.",
        ),
    ]:
        value["field_principles"].update({field: principle for field in fields})
    value["reproducibility_checksum"] = checksum(value)
    return value
